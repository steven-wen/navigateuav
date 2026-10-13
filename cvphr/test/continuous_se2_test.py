"""Evaluate continuous SE(2) alignment with baseline pose metrics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cvphr.models.continuous_se2 import ContinuousSE2Bearing, ContinuousSE2Loss
from cvphr.models.posaglreg.models import RSBlockDatasetPA_v3q
from cvphr.test.baseline_pose_metrics import evaluate_pose_predictions
from cvphr.train.persistent_train import _split_lengths


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = dict(checkpoint["model_kwargs"])
    weights_path = model_kwargs.get("weights_path")
    if weights_path and not Path(weights_path).is_file():
        model_kwargs["weights_path"] = None
    model = ContinuousSE2Bearing(**model_kwargs)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.encoder.")
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch: unexpected={unexpected}, missing={missing}"
        )
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous cross-view SE(2) alignment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = _load_model(args.checkpoint, device)
    config = checkpoint["training_config"]
    dataset_dir = args.dataset_dir or Path(config["dataset_dir"])
    metadata_csv = dataset_dir / "metadata" / "metadata.csv"
    dataset = RSBlockDatasetPA_v3q(str(metadata_csv), is_train=False)
    ratio = tuple(float(item) for item in config["split_ratio"])
    train_size, val_size, _ = _split_lengths(len(dataset), ratio)
    permutation = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(config["seed"])),
    ).tolist()
    test_indices = permutation[train_size + val_size :]
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(Subset(dataset, test_indices), **loader_kwargs)

    weights = config.get("loss_weights", {})
    criterion = ContinuousSE2Loss(
        position_weight=float(weights.get("position", 0.8)),
        direction_weight=float(weights.get("direction", 0.2)),
        intermediate_weight=float(weights.get("intermediate", 0.5)),
        correspondence_weight=float(weights.get("correspondence", 0.2)),
        reliability_weight=float(weights.get("reliability", 0.01)),
    )
    positions = []
    directions = []
    initial_poses = []
    position_targets = []
    direction_targets = []
    target_paths = []
    match_confidences = []
    sat_reliability = []
    uav_reliability = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(loader, desc="Continuous SE2 test")
        ):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            direction = batch["agl_coords"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model.forward_full(patches, return_attention=True)
                losses = criterion(output, coords, direction)
            if not all(
                bool(torch.isfinite(output[key]).all())
                for key in ("position", "direction")
            ):
                raise FloatingPointError(
                    f"Non-finite prediction at batch {batch_index}"
                )
            batch_size = patches.shape[0]
            loss_sum += float(losses["loss"]) * batch_size
            positions.append(output["position"].cpu())
            directions.append(output["direction"].cpu())
            initial_poses.append(output["pose_stages"][:, 0].cpu())
            position_targets.append(batch["coords"])
            direction_targets.append(batch["agl_coords"])
            target_paths.extend(batch["target_path"])
            match_confidences.append(output["match_confidence"].mean(dim=1).cpu())
            sat_reliability.append(
                output["satellite_reliability"].mean(dim=(1, 2, 3)).cpu()
            )
            uav_reliability.append(
                output["uav_reliability"].mean(dim=(1, 2, 3)).cpu()
            )

    positions_np = torch.cat(positions).numpy()
    directions_np = torch.cat(directions).numpy()
    initial_np = torch.cat(initial_poses).numpy()
    position_targets_np = torch.cat(position_targets).numpy()
    direction_targets_np = torch.cat(direction_targets).numpy()
    confidence_np = torch.cat(match_confidences).numpy()
    sat_reliability_np = torch.cat(sat_reliability).numpy()
    uav_reliability_np = torch.cat(uav_reliability).numpy()
    evaluation = evaluate_pose_predictions(
        positions_np,
        position_targets_np,
        directions_np,
        direction_targets_np,
        target_paths,
    )
    initial_evaluation = evaluate_pose_predictions(
        initial_np[:, :2],
        position_targets_np,
        initial_np[:, 2:],
        direction_targets_np,
        target_paths,
    )
    metrics = dict(evaluation.metrics)
    metrics.update(
        {
            "model_name": model.model_name,
            "model_class": type(model).__name__,
            "backbone": model.backbone_name,
            "dataset": str(dataset_dir),
            "split_ratio": ratio,
            "samples": len(positions_np),
            "test_loss": loss_sum / len(positions_np),
            "initial_mean_location_error_m": float(
                initial_evaluation.metrics["mean_location_error_m"]
            ),
            "initial_mean_heading_error_deg": float(
                initial_evaluation.metrics["mean_heading_error_deg"]
            ),
            "mean_match_confidence": float(confidence_np.mean()),
            "mean_satellite_reliability": float(sat_reliability_np.mean()),
            "mean_uav_reliability": float(uav_reliability_np.mean()),
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"continuous_se2_test_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename in ("metrics.json", "test_mae.json"):
        with (output_dir / filename).open("w") as handle:
            json.dump(metrics, handle, indent=2)

    metadata = pd.read_csv(metadata_csv).iloc[
        test_indices[: len(positions_np)]
    ].copy()
    metadata["x_pred"] = positions_np[:, 0]
    metadata["y_pred"] = positions_np[:, 1]
    metadata["cos_pred"] = directions_np[:, 0]
    metadata["sin_pred"] = directions_np[:, 1]
    metadata["initial_x_pred"] = initial_np[:, 0]
    metadata["initial_y_pred"] = initial_np[:, 1]
    metadata["initial_cos_pred"] = initial_np[:, 2]
    metadata["initial_sin_pred"] = initial_np[:, 3]
    metadata["distance_error_m"] = evaluation.distance_errors
    metadata["heading_error_deg"] = evaluation.heading_errors
    metadata["match_confidence"] = confidence_np
    metadata.to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
