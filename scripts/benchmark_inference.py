#!/usr/bin/env python3
"""Benchmark a Bearing-UAV checkpoint with its five-image model input."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cvphr.models.posaglreg.models import load_config_and_model
from cvphr.models.persistent_bearing import PersistentBearing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind", choices=("baseline", "persistent"), default="baseline"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    checkpoint_path = args.model_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_kind == "persistent":
        model_kwargs = dict(checkpoint["model_kwargs"])
        model = PersistentBearing(**model_kwargs).to(device)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("backbone.encoder.")
        ]
        if incompatible.unexpected_keys or missing:
            raise RuntimeError(
                f"Incompatible checkpoint: unexpected={incompatible.unexpected_keys}, "
                f"missing={missing}"
            )
    else:
        model_class, model_kwargs = load_config_and_model(str(args.model_dir))
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inputs = torch.randn(
        args.batch_size,
        5,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    autocast = torch.cuda.amp.autocast

    def forward():
        if args.model_kind == "persistent":
            return model.forward_full(inputs, topk=5, return_features=False)
        return model(inputs)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast(enabled=args.amp):
                forward()
        torch.cuda.synchronize(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        for start, end in zip(starts, ends):
            start.record()
            with autocast(enabled=args.amp):
                forward()
            end.record()
        torch.cuda.synchronize(device)

    times_ms = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)])
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = {
        "model_dir": str(args.model_dir.resolve()),
        "model_kind": args.model_kind,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": torch.cuda.get_device_name(device),
        "precision": "AMP" if args.amp else "FP32",
        "input_shape": list(inputs.shape),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "mean_ms": float(times_ms.mean()),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "std_ms": float(times_ms.std()),
        "fps_at_batch_1": float(1000.0 / times_ms.mean()) if args.batch_size == 1 else None,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
