"""An excavator digs a trench in the reconstructed road surface (deformable terrain).

The excavator world of ``simulate_excavator.py`` (MuJoCo arm, look-alike
barriers as rigid bodies) stands on the road side of a barrier, which only
defines where the work zone is.  Between the machine and the barriers a
``SoilField`` height field (``sim_splat.terrain``) replaces the ground.
Each pass:

1. the bucket comes down at the far end of the trench, to the pass depth;
2. drags toward the machine, cutting everything above the bucket's bottom
   into the bucket (up to its capacity);
3. lifts, moves over the spoil pile beyond the trench (this arm has no
   swing joint), and dumps: a mound of the same volume, relaxed to the
   angle of repose.

The bucket does not collide with the soil (the cut is kinematic), and the
barriers keep their rigid-body physics.  The initial surface comes from the
scene's LiDAR (seed points), which is reliable on the road where the
rendered depth is not.

Output (``torch.save``, replayed by ``render_physics.py``): the excavator
payload plus ``terrain``: grid origin and resolution in frame E,
``world_from_e``, the initial heights and the heights at every video frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim_splat.config import load_scene_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.scripts.simulate_excavator import BUCKET, ExcavatorWorld
from sim_splat.terrain import SoilField


def _lidar_ground(scene_path: Path, world_from_e: np.ndarray, soil: SoilField) -> tuple[np.ndarray, int]:
    """Initial heights (frame E) from static LiDAR seed points: per cell, the median of the
    returns within 0.3 m of the region's ground level; empty cells take the region median."""

    scene = parse_scene(load_scene_config(scene_path))
    points = scene.points_xyz.numpy().astype(np.float64)
    owners = scene.metadata["dynamic_point_track_indices"].numpy()
    local = (points[owners < 0] - world_from_e[:3, 3]) @ world_from_e[:3, :3]
    rows, cols = soil.heights.shape
    cx = np.round((local[:, 0] - soil.origin[0]) / soil.resolution).astype(int)
    cy = np.round((local[:, 1] - soil.origin[1]) / soil.resolution).astype(int)
    inside = (cx >= 0) & (cx < cols) & (cy >= 0) & (cy < rows) & (np.abs(local[:, 2]) < 1.0)
    if not inside.any():
        return soil.heights, 0
    level = float(np.median(local[inside, 2]))
    ground = inside & (np.abs(local[:, 2] - level) < 0.3)
    heights = np.full((rows, cols), np.nan)
    for r, c, z in zip(cy[ground], cx[ground], local[ground, 2]):
        heights[r, c] = z if np.isnan(heights[r, c]) else 0.5 * (heights[r, c] + z)
    heights = np.where(np.isnan(heights), level, heights)
    return heights, int(ground.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--track", required=True, help="barrier that marks the work zone (look-alike asset)")
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="other look-alike assets as rigid bodies")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-offset", type=float, default=0.0)
    parser.add_argument("--standoff", type=float, default=5.2, help="excavator base to barrier, m")
    parser.add_argument("--trench", type=float, nargs=2, default=(1.9, 2.9), help="trench x range in frame E, m")
    parser.add_argument("--dump-x", type=float, default=3.8, help="spoil pile beyond the trench (the arm cannot swing)")
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--pass-depth", type=float, default=0.12, help="m deeper per pass")
    parser.add_argument("--capacity", type=float, default=0.35, help="bucket capacity, m^3")
    parser.add_argument("--speed", type=float, default=0.8, help="m/s")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    import mujoco

    neighbours = sorted(path.parent for path in args.asset_root.glob("*/asset.json")) if args.asset_root else ()
    world = ExcavatorWorld(args.scene, args.track, asset_dir=args.asset_dir, asset_mass=60.0, push_sign=-1.0,
                           lateral_offset=args.lateral_offset, standoff=args.standoff, neighbour_dirs=neighbours)
    soil = SoilField.flat((args.trench[0] - 0.5, max(args.trench[1], args.dump_x) + 0.8), (-1.2, 1.2), 0.05, capacity_m3=args.capacity)
    heights, ground_points = _lidar_ground(args.scene, world.world_from_e, soil)
    soil = SoilField(soil.origin, soil.resolution, heights, capacity_m3=args.capacity)
    surface = float(np.median(heights))
    bucket_drop = 2 * BUCKET[0]  # the bucket hangs this far below the wrist

    # Scripted passes: (wrist x, wrist z, cutting?) waypoints.
    above = surface + bucket_drop + 0.9
    waypoints = [(args.trench[1], above, False)]
    for k in range(1, args.passes + 1):
        bottom = surface - k * args.pass_depth
        waypoints += [(args.trench[1], bottom + bucket_drop, False),  # down at the far end
                      (args.trench[0], bottom + bucket_drop, True),  # drag toward the machine
                      (args.trench[0], above, False),  # lift
                      (args.dump_x, above, False),  # over the spoil pile
                      ("dump", None, False),
                      (args.trench[1], above, False)]  # back
    world.reset(np.array(waypoints[0][:2]))
    wrist = np.array(waypoints[0][:2], dtype=float)
    every = int(round(1.0 / (args.fps * world.dt)))
    bucket = world.model.geom("bucket").id
    frames, step = {"asset": [], "parts": [], "neighbours": [], "heights": [], "load": []}, 0

    def record() -> None:
        frames["asset"].append(world.asset_pose())
        frames["parts"].append(world.part_poses())
        frames["neighbours"].append(world.neighbour_poses())
        frames["heights"].append(soil.heights.astype(np.float32).copy())
        frames["load"].append(soil.load_m3)

    def bucket_cut() -> None:
        centre = world.data.geom_xpos[bucket]
        rotation = world.data.geom_xmat[bucket].reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.asarray(BUCKET)
        world_corners = corners @ rotation.T + centre
        soil.cut(centre[:2], 0.0, (BUCKET[2], BUCKET[1]), float(world_corners[:, 2].min()))

    record()
    for target in waypoints[1:]:
        if target[0] == "dump":
            soil.dump((args.dump_x, 0.0), radius=0.7)
            continue
        goal = np.array(target[:2], dtype=float)
        steps = max(1, int(np.linalg.norm(goal - wrist) / args.speed / world.dt))
        start = wrist.copy()
        for i in range(1, steps + 1):
            world.step(start + (goal - start) * i / steps)
            if target[2]:
                bucket_cut()
            step += 1
            if step % every == 0:
                record()
        wrist = goal
    for _ in range(int(0.5 / world.dt)):
        world.step(wrist)
        step += 1
        if step % every == 0:
            record()

    asset_poses, part_poses = np.stack(frames["asset"]), np.stack(frames["parts"])
    neighbour_poses = np.stack(frames["neighbours"]) if world.neighbours else None
    dug = soil.heights < soil.initial - 0.01
    summary = {
        "track": world.token, "simulator": f"MuJoCo {mujoco.__version__}", "frames": len(asset_poses), "fps": args.fps,
        "passes": args.passes, "trench_depth_m": float((soil.initial - soil.heights).max()),
        "trench_area_m2": float(dug.sum() * soil.cell_area), "excavated_m3": float(-np.minimum(soil.heights - soil.initial, 0).sum() * soil.cell_area),
        "spoil_m3": float(np.maximum(soil.heights - soil.initial, 0).sum() * soil.cell_area),
        "volume_balance_m3": soil.volume_change() + soil.load_m3, "lidar_ground_points": ground_points,
        "barrier_displacement_m": float(np.linalg.norm(asset_poses[-1][:2, 3] - asset_poses[0][:2, 3])),
        "first_contact_frame": None, "max_joint_tracking_error_rad": world.tracking_error(),
    }
    payload = world.payload(asset_poses, part_poses, summary, neighbour_poses)
    payload["label"] = "MuJoCo excavator digging"
    payload["terrain"] = {"origin": list(soil.origin), "resolution": soil.resolution, "world_from_e": torch.from_numpy(world.world_from_e).float(),
                          "initial": torch.from_numpy(soil.initial.astype(np.float32)), "heights": torch.from_numpy(np.stack(frames["heights"]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
