import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from config.paths import proj_dir
from cvphr.models.persistent_bearing import (
    PersistentBearing,
    PersistentBearingLoss,
)
from cvphr.models.posaglreg.models import RSBlockDatasetPA_v3q


def _parse_ratio(value: str) -> Tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in value.split(","))
    if len(values) != 3 or any(item < 0 for item in values):
        raise ValueError("split ratio must contain three non-negative numbers")
    total = sum(values)
    if total <= 0:
        raise ValueError("split ratio sum must be positive")
    return tuple(item / total for item in values)


def _split_lengths(size: int, ratio: Tuple[float, float, float]):
    train_size = int(size * ratio[0])
    val_size = int(size * ratio[1])
    test_size = size - train_size - val_size
    return train_size, val_size, test_size


def build_loaders(
    metadata_csv: Path,
    batch_size: int,
    num_workers: int,
    split_ratio: Tuple[float, float, float],
    seed: int,
):
    augmented = RSBlockDatasetPA_v3q(str(metadata_csv), is_train=True)
    normalized = RSBlockDatasetPA_v3q(str(metadata_csv), is_train=False)
    lengths = _split_lengths(len(augmented), split_ratio)
    permutation = torch.randperm(
        len(augmented), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_end = lengths[0]
    val_end = train_end + lengths[1]
    train_set = Subset(augmented, permutation[:train_end])
    val_set = Subset(normalized, permutation[train_end:val_end])
    test_set = Subset(normalized, permutation[val_end:])

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _limited(loader: Iterable, max_batches: int):
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        yield batch_index, batch


def _load_partial_state(model: PersistentBearing, state: Dict[str, torch.Tensor]):
    incompatible = model.load_state_dict(state, strict=False)
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


def _save_checkpoint(
    path: Path,
    model: PersistentBearing,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    model_kwargs: Dict,
    training_config: Dict,
    include_backbone: bool,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.checkpoint_state_dict(
                include_backbone=include_backbone
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_kwargs": model_kwargs,
            "training_config": training_config,
            "backbone_included": include_backbone,
        },
        path,
    )


def _heading_error_degrees(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction, dim=1, eps=1e-6)
    target = torch.nn.functional.normalize(target, dim=1, eps=1e-6)
    cosine = (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def run_epoch(
    model: PersistentBearing,
    criterion: PersistentBearingLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    training: bool,
    max_batches: int,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
) -> Dict[str, float]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train(training)
    totals = {
        "loss": 0.0,
        "pose_nll": 0.0,
        "heading_mixture_nll": 0.0,
        "circular": 0.0,
        "unit_circle": 0.0,
        "auxiliary": 0.0,
        "reliability": 0.0,
        "position_mae": 0.0,
        "heading_mae": 0.0,
        "heading_symmetry": 0.0,
        "heading_entropy": 0.0,
        "samples": 0,
    }

    context = torch.enable_grad if training else torch.no_grad
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    if training:
        optimizer.zero_grad(set_to_none=True)

    with context():
        for batch_index, batch in _limited(loader, max_batches):
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            direction = batch["agl_coords"].to(device, non_blocking=True)
            theta = batch["theta"].to(device, non_blocking=True)
            batch_samples = patches.shape[0]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model.forward_full(patches, topk=0)
                losses = criterion(output, coords, theta, direction)

            if training:
                group_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps,
                    total_batches - group_start,
                )
                scaler.scale(losses["loss"] / group_size).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ),
                            max_grad_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            position_error = (output["position"] - coords).abs().mean(dim=1)
            heading_error = _heading_error_degrees(
                output["direction"], direction
            )
            for key in (
                "loss",
                "pose_nll",
                "heading_mixture_nll",
                "circular",
                "unit_circle",
                "auxiliary",
                "reliability",
            ):
                totals[key] += float(losses[key].detach().item()) * batch_samples
            totals["position_mae"] += float(position_error.sum().item())
            totals["heading_mae"] += float(heading_error.sum().item())
            if "heading_mixture" in output:
                mixture = output["heading_mixture"]
                totals["heading_symmetry"] += float(
                    mixture["symmetry_score"].sum().item()
                )
                totals["heading_entropy"] += float(
                    mixture["entropy"].sum().item()
                )
            totals["samples"] += batch_samples

    sample_count = max(totals.pop("samples"), 1)
    return {key: value / sample_count for key, value in totals.items()}


def main():
    default_dataset = (
        Path(proj_dir)
        / "Bearing_UAV_90K"
        / "c4m_254k_96bc_b15_s100_v3d"
    )
    parser = argparse.ArgumentParser(
        description="Train the PersistentBearing joint SE(2) observation model."
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-root", type=Path, default=Path("results/persistent_bearing"))
    parser.add_argument("--backbone", default="dinov3_convnext_small")
    parser.add_argument("--weights", default=os.environ.get("DINOV3_WEIGHTS"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--angle-bins", type=int, default=36)
    parser.add_argument("--rotation-sign", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--adapter-mlp-depth", type=int, default=0)
    parser.add_argument("--adapter-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--shared-mlp-depth", type=int, default=0)
    parser.add_argument("--shared-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument(
        "--heading-distribution",
        choices=("legacy", "von_mises_mixture"),
        default="legacy",
    )
    parser.add_argument("--heading-hidden-dim", type=int, default=64)
    parser.add_argument("--heading-top-modes", type=int, default=3)
    parser.add_argument("--heading-initial-kappa", type=float, default=20.0)
    parser.add_argument("--heading-mixture-weight", type=float, default=0.5)
    parser.add_argument("--circular-weight", type=float, default=0.1)
    parser.add_argument("--unit-circle-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split-ratio", default="0.7,0.2,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--include-backbone-in-checkpoint", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    metadata_csv = args.dataset_dir / "metadata" / "metadata.csv"
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_csv}")

    split_ratio = _parse_ratio(args.split_ratio)
    train_loader, val_loader, _ = build_loaders(
        metadata_csv=metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=split_ratio,
        seed=args.seed,
    )

    model_kwargs = {
        "backbone_name": args.backbone,
        "pretrained": not args.no_pretrained,
        "weights_path": args.weights,
        "freeze_backbone": not args.unfreeze_backbone,
        "feature_dim": args.feature_dim,
        "num_angle_bins": args.angle_bins,
        "rotation_sign": args.rotation_sign,
        "topk": args.topk,
        "adapter_mlp_depth": args.adapter_mlp_depth,
        "adapter_mlp_ratio": args.adapter_mlp_ratio,
        "shared_mlp_depth": args.shared_mlp_depth,
        "shared_mlp_ratio": args.shared_mlp_ratio,
        "mlp_dropout": args.mlp_dropout,
        "heading_distribution": args.heading_distribution,
        "heading_hidden_dim": args.heading_hidden_dim,
        "heading_top_modes": args.heading_top_modes,
        "heading_initial_kappa": args.heading_initial_kappa,
    }
    model = PersistentBearing(**model_kwargs).to(device)
    model_kwargs["weights_path"] = model.resolved_weights_path
    criterion = PersistentBearingLoss(
        heading_mixture_weight=args.heading_mixture_weight,
        circular_weight=args.circular_weight,
        unit_circle_weight=args.unit_circle_weight,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"persistent_{args.backbone}_a{args.angle_bins}_d{args.feature_dim}_"
        f"am{args.adapter_mlp_depth}_sm{args.shared_mlp_depth}_"
        f"{args.heading_distribution}_"
        f"b{args.batch_size}x{args.gradient_accumulation_steps}_{timestamp}"
    )
    result_dir = args.output_root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    training_config = {
        "model_class": "PersistentBearing",
        "model_kwargs": model_kwargs,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "metadata": str(metadata_csv.resolve()),
        "split_ratio": split_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_weights": {
            "heading_mixture": args.heading_mixture_weight,
            "circular": args.circular_weight,
            "unit_circle": args.unit_circle_weight,
        },
        "scheduler": "CosineAnnealingLR",
        "max_grad_norm": args.max_grad_norm,
        "result_dir": str(result_dir.resolve()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
    }
    with (result_dir / "training_configure.json").open("w") as handle:
        json.dump(training_config, handle, indent=2)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        _load_partial_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    history: List[Dict] = []
    print(f"device={device}", flush=True)
    print(f"dataset={args.dataset_dir}", flush=True)
    print(f"result_dir={result_dir}", flush=True)
    print(f"weights={model.resolved_weights_path}", flush=True)
    print(
        f"trainable_parameters={training_config['trainable_parameters']}",
        flush=True,
    )
    print(
        f"micro_batch_size={args.batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={training_config['effective_batch_size']}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()
        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            training=True,
            max_batches=args.max_train_batches,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        val_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            training=False,
            max_batches=args.max_val_batches,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=1,
        )
        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.time() - start_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            ),
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        with (result_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)

        improved = val_metrics["loss"] < best_val_loss
        scheduler.step()
        if improved:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(
                result_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                model_kwargs,
                training_config,
                args.include_backbone_in_checkpoint,
            )
        _save_checkpoint(
            result_dir / "last_checkpoint.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            model_kwargs,
            training_config,
            args.include_backbone_in_checkpoint,
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_pos_mae={val_metrics['position_mae']:.4f} "
            f"val_heading_mae={val_metrics['heading_mae']:.2f} "
            f"val_heading_nll={val_metrics['heading_mixture_nll']:.4f} "
            f"val_symmetry={val_metrics['heading_symmetry']:.3f} "
            f"peak_mem={record['peak_gpu_memory_gb']:.2f}GB "
            f"best={best_val_loss:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
