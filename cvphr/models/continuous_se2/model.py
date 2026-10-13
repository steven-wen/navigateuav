"""Continuous SE(2) alignment built on the PersistentBearing frontend."""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvphr.models.persistent_bearing.model import PersistentBearing


class PoseResidualRefiner(nn.Module):
    """Predict a bounded continuous pose update from aligned dense features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        max_position_step: float,
        max_angle_step_degrees: float,
    ):
        super().__init__()
        self.max_position_step = float(max_position_step)
        self.max_angle_step = math.radians(float(max_angle_step_degrees))
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_dim * 4, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(self._groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(self._groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _groups(channels: int) -> int:
        for groups in (16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(
        self,
        uav_feature: torch.Tensor,
        aligned_satellite: torch.Tensor,
        position: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.cat(
            (
                uav_feature,
                aligned_satellite,
                uav_feature - aligned_satellite,
                uav_feature * aligned_satellite,
            ),
            dim=1,
        )
        context = self.encoder(pair)
        raw = self.head(torch.cat((context, position, direction), dim=1))
        position_delta = self.max_position_step * torch.tanh(raw[:, :2])
        angle_delta = self.max_angle_step * torch.tanh(raw[:, 2:3])
        return torch.cat((position_delta, angle_delta), dim=1)


class ContinuousSE2Bearing(PersistentBearing):
    """Estimate and iteratively refine a continuous planar pose.

    Dense UAV-to-satellite attention produces soft point correspondences. A
    differentiable weighted Procrustes solve converts those correspondences to
    a continuous translation and unit-circle heading. The current pose then
    warps the satellite feature map with ``grid_sample`` and a shared residual
    head performs a small number of continuous SE(2) updates.
    """

    _V1_FRONTEND_PREFIXES = (
        "backbone.",
        "satellite_adapter.",
        "uav_adapter.",
        "shared_metric_mlp.",
    )

    def __init__(
        self,
        backbone_name: str = "dinov3_convnext_base",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        feature_dim: int = 256,
        adapter_mlp_depth: int = 3,
        adapter_mlp_ratio: float = 4.0,
        shared_mlp_depth: int = 3,
        shared_mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.05,
        match_dim: int = 128,
        refinement_dim: int = 128,
        refinement_steps: int = 3,
        max_position_step: float = 0.25,
        max_angle_step_degrees: float = 30.0,
        attention_temperature: float = 0.07,
        use_reliability_gate: bool = True,
        procrustes_epsilon: float = 1e-4,
        check_finite: bool = False,
    ):
        if refinement_steps < 0:
            raise ValueError("refinement_steps must be non-negative")
        if match_dim < 2 or refinement_dim < 2:
            raise ValueError("match_dim and refinement_dim must be at least 2")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_path=weights_path,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
            num_angle_bins=36,
            topk=1,
            adapter_mlp_depth=adapter_mlp_depth,
            adapter_mlp_ratio=adapter_mlp_ratio,
            shared_mlp_depth=shared_mlp_depth,
            shared_mlp_ratio=shared_mlp_ratio,
            mlp_dropout=mlp_dropout,
            heading_distribution="legacy",
        )
        self.logit_scale.requires_grad_(False)
        self.model_name = "continuous_se2_bearing"
        self.match_dim = int(match_dim)
        self.refinement_steps = int(refinement_steps)
        self.attention_temperature = float(attention_temperature)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.procrustes_epsilon = float(procrustes_epsilon)
        self.check_finite = bool(check_finite)

        self.match_projection = nn.Sequential(
            nn.Conv2d(feature_dim, match_dim, 1, bias=False),
            nn.GroupNorm(PoseResidualRefiner._groups(match_dim), match_dim),
            nn.GELU(),
        )
        self.refinement_projection = nn.Sequential(
            nn.Conv2d(feature_dim, refinement_dim, 1, bias=False),
            nn.GroupNorm(
                PoseResidualRefiner._groups(refinement_dim), refinement_dim
            ),
            nn.GELU(),
        )
        self.refiner = PoseResidualRefiner(
            feature_dim=refinement_dim,
            hidden_dim=refinement_dim,
            max_position_step=max_position_step,
            max_angle_step_degrees=max_angle_step_degrees,
        )

    @staticmethod
    def _pixel_center_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5)
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5)
        y = 2.0 * y / height - 1.0
        x = 2.0 * x / width - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _ensure_finite(self, name: str, value: torch.Tensor) -> None:
        if self.check_finite and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite tensor produced by {name}")

    def _dense_features(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self._extract_domain_features(patches)
        satellite = features["satellite_features"]
        uav = features["uav_features"]
        satellite_reliability = features["satellite_reliability"]
        uav_reliability = features["uav_reliability"]
        if self.use_reliability_gate:
            satellite = satellite * satellite_reliability
            uav = uav * uav_reliability
        return {
            "satellite_map": self._stitch_tiles(satellite),
            "uav_feature": uav,
            "satellite_reliability": self._stitch_tiles(
                satellite_reliability
            ),
            "uav_reliability": uav_reliability,
        }

    def _soft_correspondences(
        self,
        uav_feature: torch.Tensor,
        satellite_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        query = self.match_projection(uav_feature).flatten(2).transpose(1, 2)
        key = self.match_projection(satellite_map).flatten(2).transpose(1, 2)
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
        with torch.cuda.amp.autocast(enabled=False):
            logits = torch.bmm(query.float(), key.float().transpose(1, 2))
            logits = logits / self.attention_temperature
            row_probability = torch.softmax(logits, dim=-1)
            column_probability = torch.softmax(logits, dim=1)
            assignment = row_probability * column_probability
            assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )

        map_height, map_width = satellite_map.shape[-2:]
        query_height, query_width = uav_feature.shape[-2:]
        satellite_coords = self._pixel_center_grid(
            map_height, map_width, logits.device, logits.dtype
        )
        query_coords = self._pixel_center_grid(
            query_height, query_width, logits.device, logits.dtype
        )
        matched_coords = torch.matmul(assignment, satellite_coords)
        entropy = -(
            row_probability * row_probability.clamp_min(1e-9).log()
        ).sum(dim=-1) / math.log(row_probability.shape[-1])
        confidence = (1.0 - entropy).clamp_min(1e-4)
        return {
            "match_logits": logits,
            "match_probability": row_probability,
            "assignment": assignment,
            "query_coords": query_coords,
            "satellite_coords": satellite_coords,
            "matched_coords": matched_coords,
            "match_confidence": confidence,
        }

    @staticmethod
    def _solve_similarity_transform(
        query_coords: torch.Tensor,
        matched_coords: torch.Tensor,
        confidence: torch.Tensor,
        crop_scale: float,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = matched_coords.shape[0]
        source = query_coords.unsqueeze(0).expand(batch_size, -1, -1)
        weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(
            epsilon
        )
        source_mean = (weights.unsqueeze(-1) * source).sum(dim=1)
        target_mean = (weights.unsqueeze(-1) * matched_coords).sum(dim=1)
        source_centered = source - source_mean.unsqueeze(1)
        target_centered = matched_coords - target_mean.unsqueeze(1)
        covariance = torch.einsum(
            "bn,bni,bnj->bij",
            weights,
            source_centered,
            target_centered,
        )
        cosine_score = covariance[:, 0, 0] + covariance[:, 1, 1]
        sine_score = covariance[:, 0, 1] - covariance[:, 1, 0]
        direction = torch.stack(
            (cosine_score + epsilon, sine_score), dim=1
        )
        direction = F.normalize(direction, dim=1, eps=epsilon)

        cosine, sine = direction[:, 0], direction[:, 1]
        rotated_mean = torch.stack(
            (
                cosine * source_mean[:, 0] - sine * source_mean[:, 1],
                sine * source_mean[:, 0] + cosine * source_mean[:, 1],
            ),
            dim=1,
        )
        map_translation = target_mean - crop_scale * rotated_mean
        valid_center_extent = max(1.0 - crop_scale, epsilon)
        position = map_translation / valid_center_extent
        return position, direction

    @staticmethod
    def _warp_satellite(
        satellite_map: torch.Tensor,
        position: torch.Tensor,
        direction: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        map_height, map_width = satellite_map.shape[-2:]
        output_height, output_width = output_size
        scale_x = output_width / map_width
        scale_y = output_height / map_height
        translation_x = position[:, 0] * (1.0 - scale_x)
        translation_y = position[:, 1] * (1.0 - scale_y)
        cosine, sine = direction[:, 0], direction[:, 1]
        affine = torch.stack(
            (
                scale_x * cosine,
                -scale_y * sine,
                translation_x,
                scale_x * sine,
                scale_y * cosine,
                translation_y,
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        grid = F.affine_grid(
            affine,
            size=(
                satellite_map.shape[0],
                satellite_map.shape[1],
                output_height,
                output_width,
            ),
            align_corners=False,
        )
        aligned = F.grid_sample(
            satellite_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return aligned, grid

    @staticmethod
    def _update_pose(
        position: torch.Tensor,
        direction: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_position = position + residual[:, :2]
        delta = residual[:, 2]
        delta_cosine = torch.cos(delta)
        delta_sine = torch.sin(delta)
        cosine, sine = direction[:, 0], direction[:, 1]
        next_direction = torch.stack(
            (
                cosine * delta_cosine - sine * delta_sine,
                sine * delta_cosine + cosine * delta_sine,
            ),
            dim=1,
        )
        return next_position, F.normalize(next_direction, dim=1, eps=1e-6)

    def forward_full(
        self,
        patches: torch.Tensor,
        return_attention: bool = True,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dense = self._dense_features(patches)
        satellite_map = dense["satellite_map"]
        uav_feature = dense["uav_feature"]
        matching = self._soft_correspondences(uav_feature, satellite_map)

        crop_scale = uav_feature.shape[-1] / satellite_map.shape[-1]
        position, direction = self._solve_similarity_transform(
            matching["query_coords"],
            matching["matched_coords"],
            matching["match_confidence"],
            crop_scale=crop_scale,
            epsilon=self.procrustes_epsilon,
        )
        pose_stages = [torch.cat((position, direction), dim=1)]
        aligned_stages = []
        satellite_refinement = self.refinement_projection(satellite_map)
        uav_refinement = self.refinement_projection(uav_feature)
        sampling_grid = None
        for _ in range(self.refinement_steps):
            aligned, sampling_grid = self._warp_satellite(
                satellite_refinement,
                position,
                direction,
                output_size=uav_refinement.shape[-2:],
            )
            residual = self.refiner(
                uav_refinement, aligned, position, direction
            )
            position, direction = self._update_pose(
                position, direction, residual
            )
            pose_stages.append(torch.cat((position, direction), dim=1))
            aligned_stages.append(aligned)

        self._ensure_finite("position", position)
        self._ensure_finite("direction", direction)
        output = {
            "position": position,
            "direction": direction,
            "pose_stages": torch.stack(pose_stages, dim=1),
            "matched_coords": matching["matched_coords"],
            "query_coords": matching["query_coords"],
            "match_confidence": matching["match_confidence"],
            "satellite_reliability": dense["satellite_reliability"],
            "uav_reliability": dense["uav_reliability"],
        }
        if return_attention:
            output["match_logits"] = matching["match_logits"]
            output["match_probability"] = matching["match_probability"]
            output["satellite_coords"] = matching["satellite_coords"]
        if aligned_stages:
            output["aligned_satellite"] = aligned_stages[-1]
            output["sampling_grid"] = sampling_grid
        if return_features:
            output["satellite_map_feature"] = satellite_map
            output["uav_feature"] = uav_feature
        return output

    def forward(
        self, patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_full(patches, return_attention=False)
        return output["position"], output["direction"]

    def load_v1_frontend(
        self,
        checkpoint: Union[str, Path, Dict[str, torch.Tensor]],
    ) -> Dict[str, object]:
        if isinstance(checkpoint, (str, Path)):
            payload = torch.load(Path(checkpoint), map_location="cpu")
        else:
            payload = checkpoint
        source = payload.get("model_state_dict", payload)
        target = self.state_dict()
        compatible = {}
        skipped = []
        for key, value in source.items():
            if not key.startswith(self._V1_FRONTEND_PREFIXES):
                continue
            if key not in target or target[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value
        if not any(key.startswith("satellite_adapter.") for key in compatible):
            raise RuntimeError("V1 checkpoint contains no compatible frontend")
        self.load_state_dict(compatible, strict=False)
        return {
            "loaded_tensors": len(compatible),
            "loaded_parameters": sum(value.numel() for value in compatible.values()),
            "skipped": skipped,
        }
