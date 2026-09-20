import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config.base_info import UNI_PIXEL, d_merge_rsis
from config.paths import rsi_dir_city8_25pp_4096bc
from cvphr.models.persistent_bearing import PersistentBearing
from cvphr.models.posaglreg.models import RSBlockDatasetPA_v3q


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = dict(checkpoint["model_kwargs"])
    weights_path = model_kwargs.get("weights_path")
    if weights_path and not Path(weights_path).is_file():
        model_kwargs["weights_path"] = None
    model = PersistentBearing(**model_kwargs)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    missing_non_backbone = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.encoder.")
    ]
    if unexpected or missing_non_backbone:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"unexpected={unexpected}, missing={missing_non_backbone}"
        )
    return model.to(device).eval(), checkpoint


def _test_indices(size: int, ratio: Tuple[float, float, float], seed: int):
    train_size = int(size * ratio[0])
    val_size = int(size * ratio[1])
    permutation = torch.randperm(
        size, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return permutation[train_size + val_size :]


def _city_id(target_path: str) -> str:
    match = re.search(r"254k_([^_]{2,4})_", target_path)
    return match.group(1) if match else "unknown"


def _meter_scales() -> Dict[str, Tuple[float, float]]:
    scales = {}
    for city_id, image_name in d_merge_rsis["merge_c4_254k"].items():
        json_path = Path(rsi_dir_city8_25pp_4096bc) / image_name
        json_path = json_path.with_suffix(".json")
        with json_path.open() as handle:
            metadata = json.load(handle)
        scales[city_id] = (
            UNI_PIXEL * float(metadata["lngm_per_pixel"]),
            UNI_PIXEL * float(metadata["latm_per_pixel"]),
        )
    return scales


def _heading_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction / np.maximum(
        np.linalg.norm(prediction, axis=1, keepdims=True), 1e-8
    )
    target = target / np.maximum(
        np.linalg.norm(target, axis=1, keepdims=True), 1e-8
    )
    cosine = np.clip((prediction * target).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _ece(confidence: np.ndarray, success: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if mask.any():
            result += (
                mask.mean()
                * abs(float(success[mask].mean()) - float(confidence[mask].mean()))
            )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PersistentBearing checkpoint on the independent test split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = _load_model(args.checkpoint, device)
    training_config = checkpoint["training_config"]
    dataset_dir = args.dataset_dir or Path(training_config["dataset_dir"])
    metadata_csv = dataset_dir / "metadata" / "metadata.csv"
    dataset = RSBlockDatasetPA_v3q(str(metadata_csv), is_train=False)
    ratio = tuple(float(item) for item in training_config["split_ratio"])
    indices = _test_indices(
        len(dataset), ratio, int(training_config["seed"])
    )
    test_set = Subset(dataset, indices)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(test_set, **loader_kwargs)

    positions = []
    directions = []
    position_targets = []
    direction_targets = []
    target_paths = []
    confidences = []
    topk_poses = []
    topk_scores = []
    heading_confidences = []
    heading_symmetry_scores = []
    heading_entropies = []
    heading_mode_angles = []
    heading_mode_weights = []
    heading_mode_kappas = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Persistent test")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            patches = batch["patches"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model.forward_full(
                    patches,
                    topk=args.topk,
                    return_features=False,
                )
            positions.append(output["position"].cpu())
            directions.append(output["direction"].cpu())
            position_targets.append(batch["coords"])
            direction_targets.append(batch["agl_coords"])
            confidences.append(output["topk"]["confidence"].cpu())
            topk_poses.append(output["topk"]["poses"].cpu())
            topk_scores.append(output["topk"]["scores"].cpu())
            if "heading_mixture" in output:
                mixture = output["heading_mixture"]
                heading_confidences.append(mixture["confidence"].cpu())
                heading_symmetry_scores.append(
                    mixture["symmetry_score"].cpu()
                )
                heading_entropies.append(mixture["entropy"].cpu())
                heading_mode_angles.append(mixture["mode_angles"].cpu())
                heading_mode_weights.append(mixture["mode_weights"].cpu())
                heading_mode_kappas.append(mixture["mode_kappa"].cpu())
            target_paths.extend(batch["target_path"])

    positions_np = torch.cat(positions).numpy()
    directions_np = torch.cat(directions).numpy()
    position_targets_np = torch.cat(position_targets).numpy()
    direction_targets_np = torch.cat(direction_targets).numpy()
    confidence_np = torch.cat(confidences).numpy()
    topk_poses_np = torch.cat(topk_poses).numpy()
    topk_scores_np = torch.cat(topk_scores).numpy()
    has_heading_mixture = bool(heading_confidences)
    if has_heading_mixture:
        heading_confidence_np = torch.cat(heading_confidences).numpy()
        heading_symmetry_np = torch.cat(heading_symmetry_scores).numpy()
        heading_entropy_np = torch.cat(heading_entropies).numpy()
        heading_mode_angles_np = torch.cat(heading_mode_angles).numpy()
        heading_mode_weights_np = torch.cat(heading_mode_weights).numpy()
        heading_mode_kappas_np = torch.cat(heading_mode_kappas).numpy()

    scales = _meter_scales()
    distance_errors = []
    topk_distance_errors = []
    for sample_index, target_path in enumerate(target_paths):
        scale_x, scale_y = scales.get(_city_id(target_path), (32.0, 32.0))
        delta = positions_np[sample_index] - position_targets_np[sample_index]
        distance_errors.append(
            math.hypot(delta[0] * scale_x, delta[1] * scale_y)
        )
        topk_delta = (
            topk_poses_np[sample_index, :, :2]
            - position_targets_np[sample_index][None]
        )
        topk_distance_errors.append(
            np.sqrt(
                (topk_delta[:, 0] * scale_x) ** 2
                + (topk_delta[:, 1] * scale_y) ** 2
            )
        )
    distance_errors = np.asarray(distance_errors)
    topk_distance_errors = np.asarray(topk_distance_errors)
    heading_errors = _heading_error(directions_np, direction_targets_np)

    gt_theta = np.arctan2(
        direction_targets_np[:, 1], direction_targets_np[:, 0]
    )
    topk_heading_errors = np.abs(
        np.arctan2(
            np.sin(topk_poses_np[:, :, 2] - gt_theta[:, None]),
            np.cos(topk_poses_np[:, :, 2] - gt_theta[:, None]),
        )
    )
    topk_heading_errors = np.degrees(topk_heading_errors)
    joint_15 = (distance_errors <= 15.0) & (heading_errors <= 15.0)
    topk_joint_15 = (
        (topk_distance_errors <= 15.0) & (topk_heading_errors <= 15.0)
    ).any(axis=1)
    sign_recall = (
        np.sign(positions_np) == np.sign(position_targets_np)
    ).all(axis=1)

    metrics = {
        "samples": int(len(distance_errors)),
        "model_name": model.model_name,
        "backbone": model.backbone_name,
        "mean_location_error_m": float(distance_errors.mean()),
        "mean_heading_error_deg": float(heading_errors.mean()),
        "median_heading_error_deg": float(np.median(heading_errors)),
        "heading_p90_deg": float(np.percentile(heading_errors, 90)),
        "location_p95_m": float(np.percentile(distance_errors, 95)),
        "heading_p95_deg": float(np.percentile(heading_errors, 95)),
        "max_heading_error_deg": float(heading_errors.max()),
        "heading_error_over_45_pct": float(
            100.0 * (heading_errors > 45.0).mean()
        ),
        "heading_error_over_90_pct": float(
            100.0 * (heading_errors > 90.0).mean()
        ),
        "recall_at_1": float(100.0 * sign_recall.mean()),
        "lsr_at_15": float(100.0 * (distance_errors <= 15.0).mean()),
        "hsr_at_15": float(100.0 * (heading_errors <= 15.0).mean()),
        "joint_sr_15m_15deg": float(100.0 * joint_15.mean()),
        "topk_joint_oracle_15m_15deg": float(100.0 * topk_joint_15.mean()),
        "joint_success_ece": float(_ece(confidence_np, joint_15)),
        "mean_confidence": float(confidence_np.mean()),
    }
    if has_heading_mixture:
        heading_success_15 = heading_errors <= 15.0
        metrics.update(
            {
                "heading_success_ece": float(
                    _ece(heading_confidence_np, heading_success_15)
                ),
                "mean_heading_confidence": float(
                    heading_confidence_np.mean()
                ),
                "mean_heading_symmetry_score": float(
                    heading_symmetry_np.mean()
                ),
                "high_symmetry_rate_pct": float(
                    100.0 * (heading_symmetry_np >= 0.25).mean()
                ),
                "mean_heading_entropy": float(heading_entropy_np.mean()),
                "mean_dominant_kappa": float(
                    heading_mode_kappas_np[:, 0].mean()
                ),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"persistent_test_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    metadata = pd.read_csv(metadata_csv).iloc[indices[: len(distance_errors)]].copy()
    metadata["x_pred"] = positions_np[:, 0]
    metadata["y_pred"] = positions_np[:, 1]
    metadata["cos_pred"] = directions_np[:, 0]
    metadata["sin_pred"] = directions_np[:, 1]
    metadata["distance_error_m"] = distance_errors
    metadata["heading_error_deg"] = heading_errors
    metadata["confidence"] = confidence_np
    metadata["joint_success_15"] = joint_15
    metadata["topk_joint_success_15"] = topk_joint_15
    metadata["topk_poses"] = [
        json.dumps(value.tolist()) for value in topk_poses_np
    ]
    metadata["topk_scores"] = [
        json.dumps(value.tolist()) for value in topk_scores_np
    ]
    if has_heading_mixture:
        metadata["heading_confidence"] = heading_confidence_np
        metadata["heading_symmetry_score"] = heading_symmetry_np
        metadata["heading_entropy"] = heading_entropy_np
        metadata["heading_mode_angles_deg"] = [
            json.dumps(np.degrees(value).tolist())
            for value in heading_mode_angles_np
        ]
        metadata["heading_mode_weights"] = [
            json.dumps(value.tolist()) for value in heading_mode_weights_np
        ]
        metadata["heading_mode_kappas"] = [
            json.dumps(value.tolist()) for value in heading_mode_kappas_np
        ]
    metadata.to_csv(output_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
