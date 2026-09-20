import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_dense_backbone


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualChannelMLP(nn.Module):
    """ConvNeXt-style point-wise MLP that preserves dense geometry."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer_scale * self.mlp(self.norm(x))


def _build_channel_mlp(
    channels: int,
    depth: int,
    ratio: float,
    dropout: float,
) -> nn.Module:
    if depth < 0:
        raise ValueError("MLP depth must be non-negative")
    if ratio <= 0:
        raise ValueError("MLP ratio must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("MLP dropout must be in [0, 1)")
    if depth == 0:
        return nn.Identity()
    hidden_channels = max(channels, int(round(channels * ratio)))
    return nn.Sequential(
        *[
            ResidualChannelMLP(
                channels=channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
            )
            for _ in range(depth)
        ]
    )


class DomainFeatureAdapter(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, ...],
        feature_dim: int,
        mlp_depth: int = 0,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        groups = _group_count(feature_dim)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_dim, feature_dim, 1, bias=False),
                    nn.GroupNorm(groups, feature_dim),
                    nn.GELU(),
                )
                for in_dim in in_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        self.channel_mlp = _build_channel_mlp(
            channels=feature_dim,
            depth=mlp_depth,
            ratio=mlp_ratio,
            dropout=mlp_dropout,
        )
        self.reliability = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, 1, 1),
        )

    def forward(
        self,
        stages: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = stages[2].shape[-2:]
        projected = []
        for projection, stage in zip(self.projections, stages):
            feature = projection(stage)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)

        fused = self.fuse(torch.stack(projected, dim=0).mean(dim=0))
        fused = self.channel_mlp(fused)
        reliability = 0.1 + 0.9 * torch.sigmoid(self.reliability(fused))
        return fused, reliability


class CircularConv1d(nn.Module):
    """One-dimensional convolution with periodic angle-axis padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CircularConv1d requires an odd kernel size")
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), mode="circular")
        return self.conv(x)


class SymmetryAwareCircularHead(nn.Module):
    """Refine circular evidence into a multimodal von Mises distribution."""

    def __init__(
        self,
        angle_centers: torch.Tensor,
        hidden_dim: int = 64,
        top_modes: int = 3,
        initial_kappa: float = 20.0,
        score_scale: float = 2.0,
    ):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("heading hidden_dim must be at least 4")
        if top_modes < 1:
            raise ValueError("heading top_modes must be positive")
        if initial_kappa <= 0:
            raise ValueError("initial_kappa must be positive")

        self.num_angle_bins = int(angle_centers.numel())
        self.top_modes = min(int(top_modes), self.num_angle_bins)
        self.initial_kappa = float(initial_kappa)
        self.score_scale = float(score_scale)

        bin_width = 2.0 * math.pi / self.num_angle_bins
        self.direction_residual_scale = 2.0 * math.sin(bin_width / 4.0)
        base_directions = torch.stack(
            (torch.cos(angle_centers), torch.sin(angle_centers)), dim=-1
        )
        tangent_directions = torch.stack(
            (-torch.sin(angle_centers), torch.cos(angle_centers)), dim=-1
        )
        self.register_buffer(
            "base_directions", base_directions, persistent=False
        )
        self.register_buffer(
            "tangent_directions", tangent_directions, persistent=False
        )

        self.refiner = nn.Sequential(
            CircularConv1d(1, hidden_dim, 3),
            nn.GELU(),
            CircularConv1d(hidden_dim, hidden_dim, 3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 4, 1),
        )
        final = self.refiner[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        kappa_bias_target = max(initial_kappa - 1.0, 1e-4)
        final.bias.data[3] = math.log(math.expm1(kappa_bias_target))

    @staticmethod
    def _circular_peaks(weights: torch.Tensor, topk: int) -> torch.Tensor:
        circular = torch.cat((weights[:, -1:], weights, weights[:, :1]), dim=1)
        pooled = F.max_pool1d(circular.unsqueeze(1), 3, stride=1).squeeze(1)
        peak_weights = torch.where(
            weights >= pooled,
            weights,
            torch.full_like(weights, -1.0),
        )
        return peak_weights.topk(min(topk, weights.shape[1]), dim=1).indices

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.gather(1, indices)
        expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return values.gather(1, expanded)

    def forward(self, angle_log_evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        if angle_log_evidence.ndim != 2:
            raise ValueError(
                "Expected angle evidence [B,A], got "
                f"{tuple(angle_log_evidence.shape)}"
            )
        if angle_log_evidence.shape[1] != self.num_angle_bins:
            raise ValueError(
                f"Expected {self.num_angle_bins} angle bins, got "
                f"{angle_log_evidence.shape[1]}"
            )

        log_evidence = angle_log_evidence - torch.logsumexp(
            angle_log_evidence, dim=1, keepdim=True
        )
        network_input = log_evidence.exp().unsqueeze(1) * self.num_angle_bins
        refinement = self.refiner(network_input)

        score_delta = self.score_scale * torch.tanh(refinement[:, 0])
        mixture_logits = log_evidence + score_delta
        mixture_weights = torch.softmax(mixture_logits, dim=1)

        radial_residual = 0.1 * torch.tanh(refinement[:, 1]).unsqueeze(-1)
        tangent_residual = (
            self.direction_residual_scale
            * torch.tanh(refinement[:, 2]).unsqueeze(-1)
        )
        base = self.base_directions.to(
            device=refinement.device, dtype=refinement.dtype
        ).unsqueeze(0)
        tangent = self.tangent_directions.to(
            device=refinement.device, dtype=refinement.dtype
        ).unsqueeze(0)
        raw_directions = (
            (1.0 + radial_residual) * base + tangent_residual * tangent
        )
        directions = F.normalize(raw_directions, dim=-1, eps=1e-6)
        kappa = 1.0 + F.softplus(refinement[:, 3])
        kappa = kappa.clamp(max=100.0)

        mode_indices = self._circular_peaks(
            mixture_weights, self.top_modes
        )
        mode_weights = self._gather(mixture_weights, mode_indices)
        mode_directions = self._gather(directions, mode_indices)
        mode_kappa = self._gather(kappa, mode_indices)
        mode_angles = torch.atan2(
            mode_directions[..., 1], mode_directions[..., 0]
        )

        entropy = -(
            mixture_weights * mixture_weights.clamp_min(1e-9).log()
        ).sum(dim=1) / math.log(self.num_angle_bins)
        resultant = (
            mixture_weights.unsqueeze(-1) * directions
        ).sum(dim=1).norm(dim=-1)

        if self.num_angle_bins % 2 == 0:
            half = self.num_angle_bins // 2
            pair_overlap = torch.minimum(
                mixture_weights[:, :half], mixture_weights[:, half:]
            )
            symmetry_score = 2.0 * pair_overlap.max(dim=1).values
        else:
            opposite = torch.roll(
                mixture_weights, shifts=self.num_angle_bins // 2, dims=1
            )
            symmetry_score = 2.0 * torch.minimum(
                mixture_weights, opposite
            ).max(dim=1).values
        symmetry_score = symmetry_score.clamp(0.0, 1.0)

        dominant_weight = mode_weights[:, 0]
        dominant_kappa = mode_kappa[:, 0]
        concentration_confidence = 1.0 - torch.exp(-dominant_kappa / 10.0)
        confidence = (
            dominant_weight
            * (1.0 - entropy)
            * (1.0 - 0.5 * symmetry_score)
            * concentration_confidence
        ).clamp(0.0, 1.0)

        return {
            "logits": mixture_logits,
            "weights": mixture_weights,
            "raw_directions": raw_directions,
            "directions": directions,
            "kappa": kappa,
            "mode_indices": mode_indices,
            "mode_weights": mode_weights,
            "mode_directions": mode_directions,
            "mode_angles": mode_angles,
            "mode_kappa": mode_kappa,
            "entropy": entropy,
            "resultant_length": resultant,
            "symmetry_score": symmetry_score,
            "confidence": confidence,
        }


class PersistentBearing(nn.Module):
    """Dense joint position-heading observation model for Bearing-UAV."""

    def __init__(
        self,
        backbone_name: str = "dinov3_convnext_small",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        feature_dim: int = 64,
        num_angle_bins: int = 36,
        rotation_sign: float = 1.0,
        topk: int = 5,
        adapter_mlp_depth: int = 0,
        adapter_mlp_ratio: float = 4.0,
        shared_mlp_depth: int = 0,
        shared_mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        heading_distribution: str = "legacy",
        heading_hidden_dim: int = 64,
        heading_top_modes: int = 3,
        heading_initial_kappa: float = 20.0,
    ):
        super().__init__()
        if num_angle_bins < 4:
            raise ValueError("num_angle_bins must be at least 4")
        if topk < 1:
            raise ValueError("topk must be positive")

        large_variant = (
            backbone_name == "dinov3_convnext_base"
            or feature_dim > 64
            or adapter_mlp_depth > 0
            or shared_mlp_depth > 0
        )
        self.model_name = (
            "persistent_bearing_large"
            if large_variant
            else "persistent_bearing"
        )
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.num_angle_bins = num_angle_bins
        self.rotation_sign = float(rotation_sign)
        self.default_topk = topk
        if heading_distribution not in ("legacy", "von_mises_mixture"):
            raise ValueError(
                "heading_distribution must be 'legacy' or "
                "'von_mises_mixture'"
            )
        self.heading_distribution = heading_distribution

        self.backbone = build_dense_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_path=weights_path,
            freeze=freeze_backbone,
        )
        in_dims = tuple(self.backbone.out_dims)
        adapter_kwargs = {
            "in_dims": in_dims,
            "feature_dim": feature_dim,
            "mlp_depth": adapter_mlp_depth,
            "mlp_ratio": adapter_mlp_ratio,
            "mlp_dropout": mlp_dropout,
        }
        self.satellite_adapter = DomainFeatureAdapter(**adapter_kwargs)
        self.uav_adapter = DomainFeatureAdapter(**adapter_kwargs)
        self.shared_metric_mlp = _build_channel_mlp(
            channels=feature_dim,
            depth=shared_mlp_depth,
            ratio=shared_mlp_ratio,
            dropout=mlp_dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        angle_centers = (
            torch.arange(num_angle_bins, dtype=torch.float32)
            * (2.0 * math.pi / num_angle_bins)
            - math.pi
        )
        self.register_buffer("angle_centers", angle_centers, persistent=True)
        self.heading_head = None
        if heading_distribution == "von_mises_mixture":
            self.heading_head = SymmetryAwareCircularHead(
                angle_centers=angle_centers,
                hidden_dim=heading_hidden_dim,
                top_modes=heading_top_modes,
                initial_kappa=heading_initial_kappa,
            )
            self.model_name = f"{self.model_name}_sac"

    @classmethod
    def get_model_name(cls) -> str:
        return "persistent_bearing"

    @property
    def resolved_weights_path(self) -> Optional[str]:
        path = getattr(self.backbone, "weights_path", None)
        return str(path) if path is not None else None

    @staticmethod
    def _stitch_tiles(tiles: torch.Tensor) -> torch.Tensor:
        """Stitch p1=00, p2=10, p3=01, p4=11 into a continuous map."""
        if tiles.ndim != 5 or tiles.shape[1] != 4:
            raise ValueError(f"Expected tile features [B,4,C,H,W], got {tiles.shape}")
        top = torch.cat((tiles[:, 0], tiles[:, 1]), dim=-1)
        bottom = torch.cat((tiles[:, 2], tiles[:, 3]), dim=-1)
        return torch.cat((top, bottom), dim=-2)

    def _extract_domain_features(
        self,
        patches: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if patches.ndim != 5 or patches.shape[1] != 5:
            raise ValueError(f"Expected patches [B,5,3,H,W], got {patches.shape}")

        batch_size, patch_count = patches.shape[:2]
        flat_patches = patches.reshape(
            batch_size * patch_count, *patches.shape[2:]
        )
        flat_stages = self.backbone(flat_patches)

        satellite_stages = []
        uav_stages = []
        for stage in flat_stages:
            stage = stage.reshape(batch_size, patch_count, *stage.shape[1:])
            satellite_stages.append(
                stage[:, :4].reshape(batch_size * 4, *stage.shape[2:])
            )
            uav_stages.append(stage[:, 4])

        satellite_features, satellite_reliability = self.satellite_adapter(
            satellite_stages
        )
        uav_features, uav_reliability = self.uav_adapter(uav_stages)
        satellite_features = self.shared_metric_mlp(satellite_features)
        uav_features = self.shared_metric_mlp(uav_features)

        feature_shape = satellite_features.shape[1:]
        reliability_shape = satellite_reliability.shape[1:]
        satellite_features = satellite_features.reshape(
            batch_size, 4, *feature_shape
        )
        satellite_reliability = satellite_reliability.reshape(
            batch_size, 4, *reliability_shape
        )

        return {
            "satellite_features": satellite_features,
            "satellite_reliability": satellite_reliability,
            "uav_features": uav_features,
            "uav_reliability": uav_reliability,
        }

    def _rotate_queries(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = query.shape
        angle = (
            self.rotation_sign
            * self.angle_centers.to(device=query.device, dtype=query.dtype)
        )
        angle = angle.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        zeros = torch.zeros_like(cosine)
        affine = torch.stack(
            (cosine, -sine, zeros, sine, cosine, zeros),
            dim=-1,
        ).reshape(-1, 2, 3)

        expanded = query.unsqueeze(1).expand(
            -1, self.num_angle_bins, -1, -1, -1
        )
        expanded = expanded.reshape(
            batch_size * self.num_angle_bins,
            channels,
            height,
            width,
        )
        grid = F.affine_grid(affine, expanded.shape, align_corners=False)
        rotated = F.grid_sample(
            expanded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rotated.reshape(
            batch_size,
            self.num_angle_bins,
            channels,
            height,
            width,
        )

    def _correlate(
        self,
        map_feature: torch.Tensor,
        query_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, query_h, query_w = query_feature.shape
        map_h, map_w = map_feature.shape[-2:]
        if map_h < query_h or map_w < query_w:
            raise ValueError(
                "Stitched map must not be smaller than the UAV feature map: "
                f"map={map_feature.shape}, query={query_feature.shape}"
            )

        normalized_map = F.normalize(map_feature, p=2, dim=1, eps=1e-6)
        rotated_query = self._rotate_queries(query_feature)
        rotated_query = F.normalize(rotated_query, p=2, dim=2, eps=1e-6)

        map_patches = F.unfold(
            normalized_map,
            kernel_size=(query_h, query_w),
        )
        map_patches = F.normalize(map_patches, p=2, dim=1, eps=1e-6)
        query_vectors = rotated_query.flatten(start_dim=2)
        query_vectors = F.normalize(query_vectors, p=2, dim=2, eps=1e-6)

        logits = torch.einsum("bad,bdl->bal", query_vectors, map_patches)
        out_h = map_h - query_h + 1
        out_w = map_w - query_w + 1
        scale = self.logit_scale.exp().clamp(max=100.0)
        return (scale * logits).reshape(
            batch_size,
            self.num_angle_bins,
            out_h,
            out_w,
        )

    def _decode_map(
        self,
        probabilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = probabilities.shape
        indices = probabilities.flatten(start_dim=1).argmax(dim=1)
        angle_indices = torch.div(
            indices, height * width, rounding_mode="floor"
        )
        spatial_indices = indices.remainder(height * width)
        row_indices = torch.div(spatial_indices, width, rounding_mode="floor")
        col_indices = spatial_indices.remainder(width)

        x = (
            2.0 * col_indices.to(probabilities.dtype) / max(width - 1, 1)
            - 1.0
        )
        y = (
            2.0 * row_indices.to(probabilities.dtype) / max(height - 1, 1)
            - 1.0
        )
        angle = self.angle_centers[angle_indices]
        position = torch.stack((x, y), dim=-1)
        direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        return position, direction

    def _extract_topk(
        self,
        probabilities: torch.Tensor,
        topk: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size, angle_bins, height, width = probabilities.shape
        circular = torch.cat(
            (
                probabilities[:, -1:],
                probabilities,
                probabilities[:, :1],
            ),
            dim=1,
        )
        pooled = F.max_pool3d(
            circular.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=(0, 1, 1),
        ).squeeze(1)
        peak_scores = torch.where(
            probabilities >= pooled,
            probabilities,
            torch.zeros_like(probabilities),
        )
        k = min(topk, angle_bins * height * width)
        scores, indices = peak_scores.flatten(start_dim=1).topk(k, dim=1)

        angle_indices = torch.div(
            indices, height * width, rounding_mode="floor"
        )
        spatial_indices = indices.remainder(height * width)
        row_indices = torch.div(spatial_indices, width, rounding_mode="floor")
        col_indices = spatial_indices.remainder(width)
        x = 2.0 * col_indices.to(probabilities.dtype) / max(width - 1, 1) - 1.0
        y = 2.0 * row_indices.to(probabilities.dtype) / max(height - 1, 1) - 1.0
        angle = self.angle_centers[angle_indices]
        poses = torch.stack((x, y, angle), dim=-1)

        flat_probabilities = probabilities.flatten(start_dim=1)
        entropy = -(
            flat_probabilities
            * flat_probabilities.clamp_min(1e-9).log()
        ).sum(dim=1)
        entropy = entropy / math.log(flat_probabilities.shape[1])
        top_two = flat_probabilities.topk(min(2, flat_probabilities.shape[1]), dim=1).values
        margin = (
            top_two[:, 0] - top_two[:, 1]
            if top_two.shape[1] == 2
            else top_two[:, 0]
        )
        confidence = ((1.0 - entropy) * (0.5 + 0.5 * margin)).clamp(0.0, 1.0)

        return {
            "poses": poses,
            "scores": scores,
            "entropy": entropy,
            "margin": margin,
            "confidence": confidence,
        }

    def forward_full(
        self,
        patches: torch.Tensor,
        topk: Optional[int] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self._extract_domain_features(patches)
        satellite_map = self._stitch_tiles(features["satellite_features"])
        satellite_reliability = self._stitch_tiles(
            features["satellite_reliability"]
        )
        uav_feature = features["uav_features"]
        uav_reliability = features["uav_reliability"]

        weighted_map = satellite_map * satellite_reliability
        weighted_query = uav_feature * uav_reliability
        logits = self._correlate(weighted_map, weighted_query)
        probabilities = torch.softmax(logits.flatten(start_dim=1), dim=1)
        probabilities = probabilities.reshape_as(logits)
        position, direction = self._decode_map(probabilities)

        heading_mixture = None
        if self.heading_head is not None:
            angle_log_evidence = torch.logsumexp(logits, dim=(2, 3))
            heading_mixture = self.heading_head(angle_log_evidence)
            direction = heading_mixture["mode_directions"][:, 0]

        output = {
            "logits": logits,
            "probabilities": probabilities,
            "position": position,
            "direction": direction,
            "angle_centers": self.angle_centers,
            "satellite_reliability": satellite_reliability,
            "uav_reliability": uav_reliability,
        }
        if heading_mixture is not None:
            output["heading_mixture"] = heading_mixture
            output["heading_confidence"] = heading_mixture["confidence"]
            output["heading_symmetry_score"] = heading_mixture[
                "symmetry_score"
            ]
        requested_topk = self.default_topk if topk is None else topk
        if requested_topk > 0:
            output["topk"] = self._extract_topk(
                probabilities, requested_topk
            )
        if return_features:
            output["uav_feature"] = uav_feature
            output["satellite_map_feature"] = satellite_map
            output["uav_descriptor"] = F.normalize(
                F.adaptive_avg_pool2d(uav_feature, 1).flatten(1),
                dim=1,
            )
        return output

    def forward(
        self,
        patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_full(patches, topk=0, return_features=False)
        return output["position"], output["direction"]

    def checkpoint_state_dict(
        self,
        include_backbone: bool = False,
    ) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        if include_backbone or not self.freeze_backbone:
            return state
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("backbone.encoder.")
        }
