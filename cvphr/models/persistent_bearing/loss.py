import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentBearingLoss(nn.Module):
    def __init__(
        self,
        spatial_sigma_cells: float = 0.75,
        angle_kappa: float = 20.0,
        auxiliary_weight: float = 0.2,
        reliability_weight: float = 0.01,
        reliability_target: float = 0.75,
        heading_mixture_weight: float = 0.5,
        circular_weight: float = 0.1,
        unit_circle_weight: float = 0.05,
    ):
        super().__init__()
        self.spatial_sigma_cells = spatial_sigma_cells
        self.angle_kappa = angle_kappa
        self.auxiliary_weight = auxiliary_weight
        self.reliability_weight = reliability_weight
        self.reliability_target = reliability_target
        self.heading_mixture_weight = heading_mixture_weight
        self.circular_weight = circular_weight
        self.unit_circle_weight = unit_circle_weight

    @staticmethod
    def _total_variation(mask: torch.Tensor) -> torch.Tensor:
        vertical = (mask[..., 1:, :] - mask[..., :-1, :]).abs().mean()
        horizontal = (mask[..., :, 1:] - mask[..., :, :-1]).abs().mean()
        return vertical + horizontal

    def _soft_target(
        self,
        logits: torch.Tensor,
        coords: torch.Tensor,
        theta_degrees: torch.Tensor,
        angle_centers: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = logits.shape
        dtype = logits.dtype
        device = logits.device

        x_grid = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        y_grid = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing="ij")

        x_scale = max(width - 1, 1) / 2.0
        y_scale = max(height - 1, 1) / 2.0
        dx = (x_grid[None] - coords[:, 0, None, None]) * x_scale
        dy = (y_grid[None] - coords[:, 1, None, None]) * y_scale
        spatial_log_target = -0.5 * (dx.square() + dy.square()) / (
            self.spatial_sigma_cells ** 2
        )

        theta = torch.deg2rad(theta_degrees.reshape(-1)).to(dtype=dtype)
        delta = angle_centers[None] - theta[:, None]
        circular_log_target = self.angle_kappa * (torch.cos(delta) - 1.0)

        log_target = (
            circular_log_target[:, :, None, None]
            + spatial_log_target[:, None]
        )
        return torch.softmax(log_target.flatten(start_dim=1), dim=1).reshape_as(
            logits
        )

    @staticmethod
    def _circular_mixture_losses(
        output: Dict[str, torch.Tensor],
        direction_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = output["logits"]
        zero = logits.sum() * 0.0
        mixture = output.get("heading_mixture")
        if mixture is None:
            return {
                "heading_mixture_nll": zero,
                "circular": zero,
                "unit_circle": zero,
            }

        # Evaluate the circular likelihood in fp32 for stable Bessel functions.
        weights = mixture["weights"].float().clamp_min(1e-9)
        directions = mixture["directions"].float()
        raw_directions = mixture["raw_directions"].float()
        kappa = mixture["kappa"].float().clamp(1e-4, 100.0)
        target = F.normalize(direction_target.float(), dim=-1, eps=1e-6)

        cosine_delta = (
            directions * target[:, None, :]
        ).sum(dim=-1).clamp(-1.0, 1.0)
        log_i0 = torch.log(torch.special.i0e(kappa).clamp_min(1e-12)) + kappa
        log_density = (
            kappa * cosine_delta - math.log(2.0 * math.pi) - log_i0
        )
        log_joint = weights.log() + log_density
        mixture_nll = -torch.logsumexp(log_joint, dim=1).mean()

        responsibility = torch.softmax(log_joint, dim=1)
        circular = (
            responsibility * (1.0 - cosine_delta)
        ).sum(dim=1).mean()
        raw_norm_squared = raw_directions.square().sum(dim=-1)
        unit_circle = (
            weights.detach() * (raw_norm_squared - 1.0).square()
        ).sum(dim=1).mean()
        return {
            "heading_mixture_nll": mixture_nll,
            "circular": circular,
            "unit_circle": unit_circle,
        }

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        theta_degrees: torch.Tensor,
        direction_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = output["logits"]
        angle_centers = output["angle_centers"].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        target = self._soft_target(
            logits,
            coords,
            theta_degrees,
            angle_centers,
        )
        log_probabilities = torch.log_softmax(
            logits.flatten(start_dim=1), dim=1
        ).reshape_as(logits)
        pose_nll = -(target * log_probabilities).sum(dim=(1, 2, 3)).mean()

        probabilities = torch.softmax(
            logits.flatten(start_dim=1), dim=1
        ).reshape_as(logits)
        _, _, height, width = logits.shape
        x_grid = torch.linspace(
            -1.0, 1.0, width, device=logits.device, dtype=logits.dtype
        )
        y_grid = torch.linspace(
            -1.0, 1.0, height, device=logits.device, dtype=logits.dtype
        )
        spatial_probability = probabilities.sum(dim=1)
        expected_x = (
            spatial_probability.sum(dim=1) * x_grid[None]
        ).sum(dim=1)
        expected_y = (
            spatial_probability.sum(dim=2) * y_grid[None]
        ).sum(dim=1)
        expected_position = torch.stack((expected_x, expected_y), dim=-1)

        angle_probability = probabilities.sum(dim=(2, 3))
        expected_direction = torch.stack(
            (
                (angle_probability * torch.cos(angle_centers)[None]).sum(dim=1),
                (angle_probability * torch.sin(angle_centers)[None]).sum(dim=1),
            ),
            dim=-1,
        )
        expected_direction = F.normalize(expected_direction, dim=1, eps=1e-6)
        auxiliary_position = F.smooth_l1_loss(expected_position, coords)
        circular_losses = self._circular_mixture_losses(
            output, direction_target
        )
        if "heading_mixture" in output:
            auxiliary_direction = circular_losses["circular"]
        else:
            auxiliary_direction = F.smooth_l1_loss(
                expected_direction, direction_target
            )
        auxiliary = 0.8 * auxiliary_position + 0.2 * auxiliary_direction

        satellite_reliability = output["satellite_reliability"]
        uav_reliability = output["uav_reliability"]
        reliability_mean = (
            satellite_reliability.mean() - self.reliability_target
        ).square() + (
            uav_reliability.mean() - self.reliability_target
        ).square()
        reliability_smoothness = self._total_variation(
            satellite_reliability
        ) + self._total_variation(uav_reliability)
        reliability = reliability_mean + 0.1 * reliability_smoothness

        total = (
            pose_nll
            + self.auxiliary_weight * auxiliary
            + self.reliability_weight * reliability
            + self.heading_mixture_weight
            * circular_losses["heading_mixture_nll"]
            + self.circular_weight * circular_losses["circular"]
            + self.unit_circle_weight * circular_losses["unit_circle"]
        )
        return {
            "loss": total,
            "pose_nll": pose_nll,
            "auxiliary": auxiliary,
            "position_aux": auxiliary_position,
            "direction_aux": auxiliary_direction,
            "reliability": reliability,
            "heading_mixture_nll": circular_losses["heading_mixture_nll"],
            "circular": circular_losses["circular"],
            "unit_circle": circular_losses["unit_circle"],
            "expected_position": expected_position,
            "expected_direction": expected_direction,
            "target_distribution": target,
        }
