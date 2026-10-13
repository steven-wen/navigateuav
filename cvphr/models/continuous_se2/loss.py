"""Losses for continuous cross-view SE(2) alignment."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousSE2Loss(nn.Module):
    def __init__(
        self,
        position_weight: float = 0.8,
        direction_weight: float = 0.2,
        intermediate_weight: float = 0.5,
        correspondence_weight: float = 0.2,
        reliability_weight: float = 0.01,
        reliability_target: float = 0.75,
    ):
        super().__init__()
        self.position_weight = float(position_weight)
        self.direction_weight = float(direction_weight)
        self.intermediate_weight = float(intermediate_weight)
        self.correspondence_weight = float(correspondence_weight)
        self.reliability_weight = float(reliability_weight)
        self.reliability_target = float(reliability_target)

    @staticmethod
    def _direction_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1, eps=1e-6)
        target = F.normalize(target, dim=-1, eps=1e-6)
        return (1.0 - (prediction * target).sum(dim=-1)).mean()

    @staticmethod
    def _target_correspondences(
        query_coords: torch.Tensor,
        coords: torch.Tensor,
        direction: torch.Tensor,
        crop_scale: float,
    ) -> torch.Tensor:
        direction = F.normalize(direction, dim=1, eps=1e-6)
        cosine, sine = direction[:, 0:1], direction[:, 1:2]
        x = query_coords[:, 0].unsqueeze(0)
        y = query_coords[:, 1].unsqueeze(0)
        rotated_x = cosine * x - sine * y
        rotated_y = sine * x + cosine * y
        translation = coords * (1.0 - crop_scale)
        return torch.stack(
            (
                translation[:, 0:1] + crop_scale * rotated_x,
                translation[:, 1:2] + crop_scale * rotated_y,
            ),
            dim=-1,
        )

    def _reliability_loss(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        satellite = output["satellite_reliability"]
        uav = output["uav_reliability"]
        return (satellite.mean() - self.reliability_target).square() + (
            uav.mean() - self.reliability_target
        ).square()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        direction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        position = F.smooth_l1_loss(output["position"], coords)
        heading = self._direction_loss(output["direction"], direction)
        pose_stages = output["pose_stages"]
        if pose_stages.shape[1] > 1:
            stage_position = F.smooth_l1_loss(
                pose_stages[:, :-1, :2],
                coords[:, None, :].expand_as(pose_stages[:, :-1, :2]),
            )
            stage_direction = self._direction_loss(
                pose_stages[:, :-1, 2:],
                direction[:, None, :].expand_as(pose_stages[:, :-1, 2:]),
            )
            intermediate = (
                self.position_weight * stage_position
                + self.direction_weight * stage_direction
            )
        else:
            intermediate = position * 0.0

        query_height = int(round(output["query_coords"].shape[0] ** 0.5))
        map_width = int(round(output["match_probability"].shape[-1] ** 0.5))
        crop_scale = query_height / map_width
        target_matches = self._target_correspondences(
            output["query_coords"], coords, direction, crop_scale
        )
        valid = (target_matches.abs() <= 1.0).all(dim=-1)
        correspondence_error = F.smooth_l1_loss(
            output["matched_coords"], target_matches, reduction="none"
        ).mean(dim=-1)
        correspondence = (
            correspondence_error * valid.to(correspondence_error.dtype)
        ).sum() / valid.sum().clamp_min(1)
        reliability = self._reliability_loss(output)
        final_pose = (
            self.position_weight * position
            + self.direction_weight * heading
        )
        total = (
            final_pose
            + self.intermediate_weight * intermediate
            + self.correspondence_weight * correspondence
            + self.reliability_weight * reliability
        )
        return {
            "loss": total,
            "position": position,
            "direction": heading,
            "intermediate": intermediate,
            "correspondence": correspondence,
            "reliability": reliability,
        }
