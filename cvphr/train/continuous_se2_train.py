"""Train continuous cross-view SE(2) alignment on the Bearing-UAV split."""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from config.paths import proj_dir
from cvphr.models.continuous_se2 import ContinuousSE2Bearing, ContinuousSE2Loss
from cvphr.train.persistent_train import _parse_ratio, build_loaders


def _limited(loader: Iterable, max_batches: int):
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        yield index, batch


def _heading_error_degrees(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction, dim=1, eps=1e-6)
    target = torch.nn.functional.normalize(target, dim=1, eps=1e-6)
    cosine = (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _parameter_groups(
    model: ContinuousSE2Bearing,
    alignment_lr: float,
    frontend_lr: float,
    backbone_lr: float,
):
    definitions = (
        (
            "alignment",
            (model.match_projection, model.refinement_projection, model.refiner),
            alignment_lr,
        ),
        (
            "frontend",
            (
                model.satellite_adapter,
                model.uav_adapter,
                model.shared_metric_mlp,
            ),
            frontend_lr,
        ),
        ("backbone", (model.backbone,), backbone_lr),
    )
    groups = []
    seen = set()
    for name, modules, learning_rate in definitions:
        parameters = []
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if parameters:
            groups.append(
                {"params": parameters, "lr": learning_rate, "group_name": name}
            )
    remaining = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in seen
    ]
    if remaining:
        names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        raise RuntimeError(f"Unassigned trainable parameters: {names}")
    return groups


def run_epoch(
    model: ContinuousSE2Bearing,
    criterion: ContinuousSE2Loss,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    training: bool,
    max_batches: int,
    accumulation_steps: int,
    max_grad_norm: float,
    log_interval: int,
) -> Dict[str, float]:
    model.train(training)
    keys = (
        "loss",
        "position",
        "direction",
        "intermediate",
        "correspondence",
        "reliability",
    )
    totals = {key: 0.0 for key in keys}
    totals.update({"position_mae": 0.0, "heading_mae": 0.0, "samples": 0})
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.no_grad
    started = time.time()

    with context():
        for batch_index, batch in _limited(loader, max_batches):
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            direction = batch["agl_coords"].to(device, non_blocking=True)
            batch_size = patches.shape[0]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model.forward_full(patches, return_attention=True)
                losses = criterion(output, coords, direction)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_index}: {losses}"
                )

            if training:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, total_batches - group_start)
                scaler.scale(losses["loss"] / group_size).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        max_grad_norm,
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update(max(scaler.get_scale() / 2.0, 1.0))
                        print(
                            f"warning=non_finite_gradient step={batch_index + 1}",
                            flush=True,
                        )
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

            for key in keys:
                totals[key] += float(losses[key].detach()) * batch_size
            totals["position_mae"] += float(
                (output["position"] - coords).abs().mean(dim=1).sum()
            )
            totals["heading_mae"] += float(
                _heading_error_degrees(output["direction"], direction).sum()
            )
            totals["samples"] += batch_size
            if log_interval > 0 and (
                batch_index == 0 or (batch_index + 1) % log_interval == 0
            ):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"phase={'train' if training else 'val'} "
                    f"step={batch_index + 1}/{total_batches} "
                    f"loss={totals['loss'] / totals['samples']:.6f} "
                    f"samples_s={totals['samples'] / elapsed:.1f}",
                    flush=True,
                )

    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def _save_checkpoint(
    path: Path,
    model: ContinuousSE2Bearing,
    optimizer,
    scheduler,
    scaler,
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
            "scaler_state_dict": scaler.state_dict(),
            "model_kwargs": model_kwargs,
            "training_config": training_config,
            "backbone_included": include_backbone,
        },
        path,
    )


def _load_model_state(model: ContinuousSE2Bearing, state: Dict) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.encoder.")
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint incompatible: unexpected={unexpected}, missing={missing}"
        )


def main():
    default_dataset = (
        Path(proj_dir) / "Bearing_UAV_90K" / "c4m_254k_96bc_b15_s100"
    )
    parser = argparse.ArgumentParser(
        description="Train continuous SE(2) cross-view alignment."
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/continuous_se2_2d")
    )
    parser.add_argument("--backbone", default="dinov3_convnext_base")
    parser.add_argument("--weights", default=os.environ.get("DINOV3_WEIGHTS"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--v1-checkpoint", type=Path)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--adapter-mlp-depth", type=int, default=3)
    parser.add_argument("--adapter-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--shared-mlp-depth", type=int, default=3)
    parser.add_argument("--shared-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.05)
    parser.add_argument("--match-dim", type=int, default=128)
    parser.add_argument("--refinement-dim", type=int, default=128)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--attention-temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--alignment-lr", type=float, default=2e-4)
    parser.add_argument("--frontend-lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--position-weight", type=float, default=0.8)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--intermediate-weight", type=float, default=0.5)
    parser.add_argument("--correspondence-weight", type=float, default=0.2)
    parser.add_argument("--reliability-weight", type=float, default=0.01)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--split-ratio", default="0.7,0.2,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--include-backbone-in-checkpoint", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
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
        metadata_csv,
        args.batch_size,
        args.num_workers,
        split_ratio,
        args.seed,
    )

    model_kwargs = {
        "backbone_name": args.backbone,
        "pretrained": not args.no_pretrained,
        "weights_path": args.weights,
        "freeze_backbone": not args.unfreeze_backbone,
        "feature_dim": args.feature_dim,
        "adapter_mlp_depth": args.adapter_mlp_depth,
        "adapter_mlp_ratio": args.adapter_mlp_ratio,
        "shared_mlp_depth": args.shared_mlp_depth,
        "shared_mlp_ratio": args.shared_mlp_ratio,
        "mlp_dropout": args.mlp_dropout,
        "match_dim": args.match_dim,
        "refinement_dim": args.refinement_dim,
        "refinement_steps": args.refinement_steps,
        "attention_temperature": args.attention_temperature,
    }
    model = ContinuousSE2Bearing(**model_kwargs)
    model_kwargs["weights_path"] = model.resolved_weights_path
    frontend_report = None
    if args.v1_checkpoint:
        if not args.v1_checkpoint.is_file():
            raise FileNotFoundError(
                f"V1 checkpoint not found: {args.v1_checkpoint}"
            )
        frontend_report = model.load_v1_frontend(args.v1_checkpoint)
        print(f"v1_frontend_load={frontend_report}", flush=True)
    model = model.to(device)
    criterion = ContinuousSE2Loss(
        position_weight=args.position_weight,
        direction_weight=args.direction_weight,
        intermediate_weight=args.intermediate_weight,
        correspondence_weight=args.correspondence_weight,
        reliability_weight=args.reliability_weight,
    )
    groups = _parameter_groups(
        model, args.alignment_lr, args.frontend_lr, args.backbone_lr
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda", init_scale=args.amp_init_scale
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"continuous_se2_{args.backbone}_d{args.feature_dim}_m{args.match_dim}_"
        f"r{args.refinement_steps}_b{args.batch_size}x"
        f"{args.gradient_accumulation_steps}_{timestamp}"
    )
    result_dir = args.output_root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    group_info = {
        group["group_name"]: {
            "lr": group["lr"],
            "parameters": sum(item.numel() for item in group["params"]),
        }
        for group in optimizer.param_groups
    }
    training_config = {
        "model_class": "ContinuousSE2Bearing",
        "model_kwargs": model_kwargs,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "metadata": str(metadata_csv.resolve()),
        "split_ratio": split_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size
        * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "optimizer_groups": group_info,
        "scheduler": "CosineAnnealingLR",
        "loss_weights": {
            "position": args.position_weight,
            "direction": args.direction_weight,
            "intermediate": args.intermediate_weight,
            "correspondence": args.correspondence_weight,
            "reliability": args.reliability_weight,
        },
        "v1_checkpoint": str(args.v1_checkpoint) if args.v1_checkpoint else None,
        "v1_frontend_load": frontend_report,
        "result_dir": str(result_dir.resolve()),
    }
    with (result_dir / "training_configure.json").open("w") as handle:
        json.dump(training_config, handle, indent=2)

    start_epoch = 0
    best_val_loss = math.inf
    history: List[Dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        _load_model_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    print(f"device={device}", flush=True)
    print(f"dataset={args.dataset_dir}", flush=True)
    print(f"result_dir={result_dir}", flush=True)
    print(f"optimizer_groups={group_info}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer,
            scaler,
            True,
            args.max_train_batches,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            args.log_interval,
        )
        val_metrics = run_epoch(
            model,
            criterion,
            val_loader,
            device,
            optimizer,
            scaler,
            False,
            args.max_val_batches,
            1,
            args.max_grad_norm,
            args.log_interval,
        )
        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.time() - started,
            "learning_rates": {
                group["group_name"]: group["lr"]
                for group in optimizer.param_groups
            },
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3
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
        if improved:
            best_val_loss = val_metrics["loss"]
        scheduler.step()
        if improved:
            _save_checkpoint(
                result_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                scaler,
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
            scaler,
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
            f"val_pos_mae={val_metrics['position_mae']:.5f} "
            f"val_heading_mae={val_metrics['heading_mae']:.3f} "
            f"peak_mem={record['peak_gpu_memory_gb']:.2f}GB "
            f"best={best_val_loss:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
