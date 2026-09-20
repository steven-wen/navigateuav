from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import convnext_base, convnext_small


_DINOV3_WEIGHT_GLOBS = {
    "dinov3_convnext_small": (
        "dinov3_convnext_small_pretrain_lvd1689m-*.pth",
    ),
    "dinov3_convnext_base": (
        "dinov3_convnext_base_pretrain_lvd1689m-*.pth",
    ),
}


def _candidate_hub_dirs() -> List[Path]:
    candidates = []
    torch_home = Path(torch.hub.get_dir()).expanduser()
    candidates.append(torch_home / "checkpoints")
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def find_dinov3_weights(
    backbone_name: str,
    weights_path: Optional[str] = None,
) -> Optional[Path]:
    if weights_path:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {path}")
        return path

    patterns = _DINOV3_WEIGHT_GLOBS.get(backbone_name, ())
    for checkpoint_dir in _candidate_hub_dirs():
        for pattern in patterns:
            matches = sorted(checkpoint_dir.glob(pattern))
            if matches:
                return matches[-1]
    return None


def _translate_dinov3_convnext_state(
    source_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Translate Meta DINOv3 ConvNeXt names to torchvision ConvNeXt names."""
    translated = {}
    block_name_map = {
        "dwconv": "block.0",
        "norm": "block.2",
        "pwconv1": "block.3",
        "pwconv2": "block.5",
    }

    for key, value in source_state.items():
        parts = key.split(".")
        target_key = None

        if parts[0] == "downsample_layers":
            stage_idx = int(parts[1])
            target_key = ".".join(
                ["features", str(2 * stage_idx), parts[2]] + parts[3:]
            )
        elif parts[0] == "stages":
            stage_idx = int(parts[1])
            block_idx = parts[2]
            component = parts[3]
            suffix = parts[4:]
            stage_prefix = ["features", str(2 * stage_idx + 1), block_idx]
            if component == "gamma":
                target_key = ".".join(stage_prefix + ["layer_scale"])
                value = value.reshape(-1, 1, 1)
            elif component in block_name_map:
                target_key = ".".join(
                    stage_prefix + block_name_map[component].split(".") + suffix
                )
        elif key in ("norm.weight", "norm.bias"):
            target_key = key.replace("norm.", "classifier.0.")

        if target_key is not None:
            translated[target_key] = value

    return translated


class DinoV3ConvNeXt(nn.Module):
    """Python 3.9-compatible DINOv3 ConvNeXt dense feature extractor."""

    def __init__(
        self,
        backbone_name: str,
        builder,
        out_dims: Tuple[int, int, int, int],
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_dims = out_dims
        self.freeze = freeze
        self.weights_path = find_dinov3_weights(backbone_name, weights_path)
        self.encoder = builder(
            weights=None,
            stochastic_depth_prob=0.0,
        )

        if pretrained:
            if self.weights_path is None:
                searched = ", ".join(str(path) for path in _candidate_hub_dirs())
                raise FileNotFoundError(
                    f"{backbone_name} weights were not found. "
                    "Set --weights or DINOV3_WEIGHTS. Searched: "
                    f"{searched}"
                )
            source_state = torch.load(self.weights_path, map_location="cpu")
            translated = _translate_dinov3_convnext_state(source_state)
            incompatible = self.encoder.load_state_dict(translated, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            missing_features = [
                key
                for key in incompatible.missing_keys
                if key.startswith("features.")
            ]
            if unexpected or missing_features:
                raise RuntimeError(
                    "Failed to map DINOv3 ConvNeXt weights: "
                    f"unexpected={unexpected}, missing_features={missing_features}"
                )

        if self.freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def _forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for stage_idx in range(4):
            x = self.encoder.features[2 * stage_idx](x)
            x = self.encoder.features[2 * stage_idx + 1](x)
            if stage_idx == 3:
                x = self.encoder.classifier[0](x)
            outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.freeze:
            with torch.no_grad():
                return self._forward_stages(x)
        return self._forward_stages(x)


class DinoV3ConvNeXtSmall(DinoV3ConvNeXt):
    out_dims: Tuple[int, int, int, int] = (96, 192, 384, 768)

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__(
            backbone_name="dinov3_convnext_small",
            builder=convnext_small,
            out_dims=self.out_dims,
            pretrained=pretrained,
            weights_path=weights_path,
            freeze=freeze,
        )


class DinoV3ConvNeXtBase(DinoV3ConvNeXt):
    out_dims: Tuple[int, int, int, int] = (128, 256, 512, 1024)

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__(
            backbone_name="dinov3_convnext_base",
            builder=convnext_base,
            out_dims=self.out_dims,
            pretrained=pretrained,
            weights_path=weights_path,
            freeze=freeze,
        )


class TinyDenseBackbone(nn.Module):
    """Small random backbone used only for CPU tests and interface checks."""

    out_dims: Tuple[int, int, int, int] = (32, 64, 128, 256)

    def __init__(self):
        super().__init__()
        dims = self.out_dims
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, dims[0], 4, stride=4),
                    nn.GroupNorm(8, dims[0]),
                    nn.GELU(),
                ),
                self._stage(dims[0], dims[1]),
                self._stage(dims[1], dims[2]),
                self._stage(dims[2], dims[3]),
            ]
        )
        self.weights_path = None
        self.freeze = False

    @staticmethod
    def _stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def build_dense_backbone(
    backbone_name: str,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    freeze: bool = True,
) -> nn.Module:
    if backbone_name == "dinov3_convnext_small":
        return DinoV3ConvNeXtSmall(
            pretrained=pretrained,
            weights_path=weights_path,
            freeze=freeze,
        )
    if backbone_name == "dinov3_convnext_base":
        return DinoV3ConvNeXtBase(
            pretrained=pretrained,
            weights_path=weights_path,
            freeze=freeze,
        )
    if backbone_name == "tiny_cnn":
        return TinyDenseBackbone()
    raise ValueError(f"Unsupported persistent backbone: {backbone_name}")
