"""Baseline-compatible pose metrics shared by V1-family evaluators."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.base_info import UNI_PIXEL, d_merge_rsis
from config.paths import rsi_dir_city8_25pp_4096bc


@dataclass(frozen=True)
class PoseEvaluation:
    metrics: Dict[str, object]
    distance_errors: np.ndarray
    heading_errors: np.ndarray
    meter_scales: np.ndarray
    city_ids: np.ndarray


def city_id_from_path(target_path: str) -> str:
    match = re.search(r"254k_([^_]{2,4})_", target_path)
    if match is None:
        raise ValueError(f"Cannot extract city id from path: {target_path}")
    return match.group(1)


def load_city_scales() -> Dict[str, Tuple[float, float, float, float]]:
    """Return x/y meter and longitude/latitude scales for normalized offsets."""
    scales = {}
    for city_id, image_name in d_merge_rsis["merge_c4_254k"].items():
        metadata_path = Path(rsi_dir_city8_25pp_4096bc) / image_name
        with metadata_path.with_suffix(".json").open() as handle:
            metadata = json.load(handle)
        scales[city_id] = (
            UNI_PIXEL * float(metadata["lngm_per_pixel"]),
            UNI_PIXEL * float(metadata["latm_per_pixel"]),
            UNI_PIXEL * float(metadata["lng_per_pixel"]),
            UNI_PIXEL * float(metadata["lat_per_pixel"]),
        )
    return scales


def _heading_errors(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    dot_product = (prediction * target).sum(axis=1)
    denominator = (
        np.linalg.norm(prediction, axis=1)
        * np.linalg.norm(target, axis=1)
        + 1e-8
    )
    cosine = np.clip(dot_product / denominator, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def evaluate_pose_predictions(
    positions: np.ndarray,
    position_targets: np.ndarray,
    directions: np.ndarray,
    direction_targets: np.ndarray,
    target_paths: Sequence[str],
    city_scales: Optional[
        Mapping[str, Tuple[float, float, float, float]]
    ] = None,
) -> PoseEvaluation:
    """Apply the baseline MLE/MHE/Recall/LSR/HSR definitions."""
    positions = np.asarray(positions, dtype=np.float32)
    position_targets = np.asarray(position_targets, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    direction_targets = np.asarray(direction_targets, dtype=np.float32)
    sample_count = positions.shape[0]
    expected_shape = (sample_count, 2)
    for name, value in (
        ("positions", positions),
        ("position_targets", position_targets),
        ("directions", directions),
        ("direction_targets", direction_targets),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )
    if len(target_paths) != sample_count:
        raise ValueError("target_paths must align with the prediction arrays")

    scales = city_scales or load_city_scales()
    city_ids = np.asarray([city_id_from_path(path) for path in target_paths])
    unknown = sorted(set(city_ids) - set(scales))
    if unknown:
        raise ValueError(f"Missing map scales for cities: {unknown}")
    scale_array = np.asarray([scales[city_id] for city_id in city_ids])

    position_delta = positions - position_targets
    meter_delta = position_delta * scale_array[:, :2]
    distance_errors = np.linalg.norm(meter_delta, axis=1)
    heading_errors = _heading_errors(directions, direction_targets)

    per_city_mle = {
        city_id: float(distance_errors[city_ids == city_id].mean())
        for city_id in sorted(set(city_ids))
    }
    per_city_lonlat = []
    for city_id in sorted(per_city_mle):
        mask = city_ids == city_id
        per_city_lonlat.append(
            (position_delta[mask] * scale_array[mask, 2:]).mean(axis=0)
        )
    mean_lonlat = np.abs(np.stack(per_city_lonlat).mean(axis=0))

    sign_recall = (
        np.sign(positions) == np.sign(position_targets)
    ).all(axis=1)
    metrics: Dict[str, object] = {
        "samples": int(sample_count),
        "mae_pos": np.abs(position_delta).mean(axis=0).tolist(),
        "mae_lonlat": mean_lonlat.tolist(),
        "mae_dir": np.abs(directions - direction_targets).mean(axis=0).tolist(),
        "MLE_each_rsi": list(per_city_mle.values()),
        "mle_by_city_m": per_city_mle,
        "mae_dis": float(np.mean(list(per_city_mle.values()))),
        "mae_agl": float(heading_errors.mean()),
        "recall_at_1": float(100.0 * sign_recall.mean()),
    }
    for threshold in (5, 10, 15, 20, 25, 30):
        metrics[f"lsr_at_{threshold}"] = float(
            100.0 * (distance_errors <= threshold).mean()
        )
        metrics[f"hsr_at_{threshold}"] = float(
            100.0 * (heading_errors <= threshold).mean()
        )
    metrics.update(
        {
            "mean_distance_error": float(distance_errors.mean()),
            "std_distance_error": float(distance_errors.std()),
            "median_distance_error": float(np.median(distance_errors)),
            "min_distance_error": float(distance_errors.min()),
            "max_distance_error": float(distance_errors.max()),
            "distance_error_95th_percentile": float(
                np.percentile(distance_errors, 95)
            ),
            "mean_angle_error": float(heading_errors.mean()),
            "std_angle_error": float(heading_errors.std()),
            "median_angle_error": float(np.median(heading_errors)),
            "min_angle_error": float(heading_errors.min()),
            "max_angle_error": float(heading_errors.max()),
            "angle_error_90th_percentile": float(
                np.percentile(heading_errors, 90)
            ),
            "angle_error_95th_percentile": float(
                np.percentile(heading_errors, 95)
            ),
            # Compatibility aliases used by the V1 evaluator.
            "mean_location_error_m": float(
                np.mean(list(per_city_mle.values()))
            ),
            "mean_heading_error_deg": float(heading_errors.mean()),
        }
    )
    return PoseEvaluation(
        metrics=metrics,
        distance_errors=distance_errors,
        heading_errors=heading_errors,
        meter_scales=scale_array[:, :2],
        city_ids=city_ids,
    )
