#!/usr/bin/env python3
"""Aggregate Bearing-Naver route outputs into SR, SPL, and NE."""

import argparse
import csv
import glob
import json
import re
from pathlib import Path


ROUTE_RE = re.compile(r"nav_(\d+bc)(\d+)_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--step-m", type=float, required=True)
    parser.add_argument("--threshold-m", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_route(route_dir, project_dir, step_m, threshold_m):
    match = ROUTE_RE.search(route_dir.name)
    if not match:
        raise ValueError(f"Cannot parse route ID from {route_dir.name}")

    city, trajectory = match.groups()
    route_id = f"{city}_{trajectory}"
    record_files = list(route_dir.glob("*_uav_traj_records.csv"))
    if len(record_files) != 1:
        raise ValueError(f"Expected one records CSV in {route_dir}, got {len(record_files)}")

    with record_files[0].open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"No navigation records in {record_files[0]}")

    waypoint_path = project_dir / "loc2traj" / "traj_wps_gcs" / f"wps{route_id}.json"
    with waypoint_path.open() as handle:
        waypoints = json.load(handle)

    reference_length_m = float(next(iter(waypoints.values()))["total_length"])
    steps = len(records)
    actual_length_m = steps * step_m
    final_distance_m = float(records[-1]["distance_ep"])
    success = final_distance_m <= threshold_m
    route_spl = (
        reference_length_m / max(reference_length_m, actual_length_m)
        if success
        else 0.0
    )

    return {
        "route": route_id,
        "success": success,
        "steps": steps,
        "reference_length_m": reference_length_m,
        "actual_length_m": actual_length_m,
        "final_distance_m": final_distance_m,
        "route_spl": route_spl,
        "output_dir": str(route_dir),
    }


def main():
    args = parse_args()
    project_dir = args.project_dir.resolve()
    route_dirs = [Path(path) for path in sorted(glob.glob(str(project_dir / args.pattern)))]
    if not route_dirs:
        raise FileNotFoundError(f"No route directories match {args.pattern}")

    routes = [
        read_route(path, project_dir, args.step_m, args.threshold_m)
        for path in route_dirs
    ]
    successful = sum(route["success"] for route in routes)
    count = len(routes)
    summary = {
        "protocol": {
            "route_count": count,
            "step_m": args.step_m,
            "arrival_threshold_m": args.threshold_m,
            "sr_definition": "mean(final_distance_m <= arrival_threshold_m)",
            "spl_definition": "mean(success * L / max(P, L))",
            "ne_definition": "mean(final_distance_m)",
        },
        "metrics": {
            "successful_routes": successful,
            "sr_at_20_percent": 100.0 * successful / count,
            "spl_percent": 100.0 * sum(route["route_spl"] for route in routes) / count,
            "navigation_error_m": sum(route["final_distance_m"] for route in routes) / count,
        },
        "routes": routes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
