import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config.base_info import (
    MAX_DISTANCE,
    UNI_PIXEL,
    generate_grid_blocks,
    id_image_map,
    rsijson2info,
    rsi_dir_city8_25pp_4096bc,
)
from cvphr.models.persistent_bearing import (
    HypothesisAlignedGeoMemory,
    MemoryConfig,
    PersistentBearing,
)
from naver.runners.nav import UAVNavigation


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


class PersistentUAVNavigation(UAVNavigation):
    """Bearing-UAV navigation runner with route-level multi-hypothesis memory."""

    def __init__(
        self,
        *args,
        memory_config: Optional[MemoryConfig] = None,
        **kwargs,
    ):
        self.geo_memory = HypothesisAlignedGeoMemory(memory_config)
        self._last_command_heading = None
        super().__init__(*args, **kwargs)

    def _load_model(self):
        self.device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        checkpoint_path = Path(self.posreg_model_dir)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PersistentBearing checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_kwargs = dict(checkpoint["model_kwargs"])
        weights_path = model_kwargs.get("weights_path")
        if weights_path and not Path(weights_path).is_file():
            model_kwargs["weights_path"] = None
        self.model_kwargs = model_kwargs
        self.model = PersistentBearing(**model_kwargs)
        _load_partial_state(self.model, checkpoint["model_state_dict"])
        self.model = self.model.to(self.device).eval()
        print(f"Using device: {self.device}")
        print(f"PersistentBearing weights loaded: {checkpoint_path}")

    def _prepare_patches(self, patches):
        tensors = []
        for patch in patches:
            if not isinstance(patch, np.ndarray):
                patch = np.asarray(patch)
            tensors.append(self.nav_transform(patch))
        return torch.stack(tensors).unsqueeze(0).to(self.device)

    def _command_motion(self, fly_angle_ccs: float) -> Optional[torch.Tensor]:
        heading = math.radians(float(fly_angle_ccs))
        if self._last_command_heading is None:
            self._last_command_heading = heading
            return None

        x_scale_m = UNI_PIXEL * float(self.drsi["lngm_per_pixel"])
        y_scale_m = UNI_PIXEL * self.latm_per_pixel
        delta_heading = math.atan2(
            math.sin(heading - self._last_command_heading),
            math.cos(heading - self._last_command_heading),
        )
        self._last_command_heading = heading
        return torch.tensor(
            [
                self.uav_step * math.cos(heading) / x_scale_m,
                -self.uav_step * math.sin(heading) / y_scale_m,
                delta_heading,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _local_to_global(
        local_poses: torch.Tensor,
        block_indices,
    ) -> torch.Tensor:
        block_x, block_y = block_indices
        global_poses = local_poses.clone()
        global_poses[:, 0] += 2.0 * float(block_x) + 2.0
        global_poses[:, 1] += 2.0 * float(block_y) + 2.0
        return global_poses

    @staticmethod
    def _global_to_local(
        global_pose: torch.Tensor,
        block_indices,
    ):
        block_x, block_y = block_indices
        x = float(global_pose[0] - (2.0 * float(block_x) + 2.0))
        y = float(global_pose[1] - (2.0 * float(block_y) + 2.0))
        angle = float(global_pose[2])
        direction = np.asarray(
            [math.cos(angle), math.sin(angle)],
            dtype=np.float32,
        )
        return np.asarray([x, y], dtype=np.float32), direction

    def phreg(
        self,
        uav_frame_id,
        fly_angle_ccs,
        cur_point_real,
        block_center,
        block_indices,
    ):
        patches, patches_fdirs, record = self.get_patches(
            uav_frame_id,
            fly_angle_ccs,
            cur_point_real,
            block_center,
            block_indices,
        )
        if not record or record.get("flag_out_of_map", True) or len(patches) != 5:
            return None, None, True, patches_fdirs, record

        patches_tensor = self._prepare_patches(patches)
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.device.type == "cuda"
        ):
            output = self.model.forward_full(
                patches_tensor,
                topk=self.geo_memory.config.max_hypotheses,
                return_features=True,
            )

        local_candidates = output["topk"]["poses"][0].detach().cpu()
        global_candidates = self._local_to_global(
            local_candidates,
            block_indices,
        )
        memory_output = self.geo_memory.step(
            poses=global_candidates,
            scores=output["topk"]["scores"][0],
            descriptor=output["uav_descriptor"][0],
            motion=self._command_motion(fly_angle_ccs),
        )
        pos_pred, dir_pred = self._global_to_local(
            memory_output["pose"],
            block_indices,
        )
        self.nav_step["persistent_topk_global"] = global_candidates.tolist()
        self.nav_step["persistent_weights"] = memory_output["weights"].tolist()
        self.nav_step["persistent_risk"] = float(memory_output["risk"])
        self.nav_step["persistent_lost"] = bool(memory_output["lost"])
        self.nav_step["persistent_memory_write"] = bool(
            memory_output["wrote_memory"]
        )
        self.nav_step["persistent_confirmed_entries"] = int(
            memory_output["confirmed_entries"]
        )
        return pos_pred, dir_pred, False, patches_fdirs, record


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run Bearing-UAV navigation with PersistentBearing memory."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rsi-id", default="37bc")
    parser.add_argument("--traj-id", type=int, default=50)
    parser.add_argument("--uav-step", type=float, default=25.0)
    parser.add_argument("--arrival-threshold", type=float, default=20.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-root", type=Path, default=repo_root / "loc2traj")
    parser.add_argument("--position-sigma", type=float, default=1.0)
    parser.add_argument("--memory-overlap-sigma", type=float, default=1.0)
    parser.add_argument("--write-risk-threshold", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rsi_id not in id_image_map:
        raise KeyError(f"Unknown RSI id: {args.rsi_id}")
    rsi_image_path = Path(rsi_dir_city8_25pp_4096bc) / id_image_map[args.rsi_id]
    rsi_json_path = rsi_image_path.with_suffix(".json")
    rs_traj_id = f"{args.rsi_id}_{args.traj_id:02d}"
    waypoint_path = repo_root / "loc2traj" / "traj_wps_gcs" / f"wps{rs_traj_id}.json"
    if not waypoint_path.is_file():
        raise FileNotFoundError(f"Waypoint file not found: {waypoint_path}")

    with waypoint_path.open(encoding="utf-8") as handle:
        waypoint_data = json.load(handle)
    waypoints = [value["lnglat"] for value in waypoint_data.values()]
    start_point = waypoints[0]
    end_point = waypoints[-1]

    map_metadata = rsijson2info(str(rsi_json_path))
    block_centers = generate_grid_blocks(
        map_metadata["width_pixel"],
        map_metadata["height_pixel"],
        map_metadata["lng"],
        map_metadata["lat"],
        map_metadata["lng_per_pixel"],
        map_metadata["lat_per_pixel"],
        n_block=15,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_root
        / f"persistent_nav_{rs_traj_id}_s{args.uav_step:g}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / f"{args.rsi_id}_block_cnt_point_dicts.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(block_centers, handle, ensure_ascii=False, indent=2)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    navigation = PersistentUAVNavigation(
        start_point=start_point,
        end_point=end_point,
        waypoints=waypoints,
        block_cnt_point_dict=block_centers,
        uav_2d3d="3d",
        uav_step=args.uav_step,
        max_steps=args.max_steps
        or int(MAX_DISTANCE / args.uav_step),
        output_dir=str(output_dir),
        th_arrive=args.arrival_threshold,
        rs_traj_id=rs_traj_id,
        rs_image_dir=str(rsi_image_path),
        rsi_type="254k",
        device_id=args.device_id,
        posreg_model_dir=str(args.checkpoint),
        model_class=PersistentBearing,
        model_kwargs=checkpoint["model_kwargs"],
        dataset_kwargs={},
        memory_config=MemoryConfig(
            max_hypotheses=int(checkpoint["model_kwargs"].get("topk", 5)),
            position_sigma=args.position_sigma,
            memory_overlap_sigma=args.memory_overlap_sigma,
            write_risk_threshold=args.write_risk_threshold,
        ),
    )
    print(f"output_dir={output_dir}")
    if args.dry_run:
        print("dry_run=ok")
        return
    navigation.fly()


if __name__ == "__main__":
    main()
