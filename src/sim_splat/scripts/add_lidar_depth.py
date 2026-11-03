"""Add per-image sparse LiDAR depth to a converted nuScenes scene (``lidar_depth/``).

The reconstruction is only supervised by colour, and away from the ego path
its depth turns into a blur along the viewing rays: shape fitting from it
failed for most roadside objects, and depth-based localization is biased.
This script writes the supervision for a depth loss without reconverting the
scene (existing checkpoints stay valid):

* for every camera image, the LIDAR_TOP sweep nearest in time (same drive
  log, within ``--max-dt``) is moved to the world with its own ego pose and
  the scene's ``world_origin``, then projected with the image's camera;
* returns from the ego vehicle (``--ego-radius``) are dropped;
* the LiDAR sits above the cameras, so it also sees surfaces the camera does
  not (behind a pole, under a car).  Such a return lands among nearer ones in
  the image, so a point is kept only if it is not much farther than the
  nearest return in its neighbourhood (``--occlusion-window`` pixels);
* one return per pixel (the nearest).

Output: ``<scene>/lidar_depth/<image stem>.npz`` with ``uv`` (pixel centres,
float32) and ``depth`` (camera z, metres, float32).  ``InputDataset`` loads
them when the folder exists.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter


def _transform(record: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = record["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat((x, y, z, w)).as_matrix()
    matrix[:3, 3] = record["translation"]
    return matrix


def project(points_world: np.ndarray, c2w: np.ndarray, K: np.ndarray, width: int, height: int,
            window: int, slack: float) -> tuple[np.ndarray, np.ndarray]:
    """Sparse depth image of world points: (uv [N, 2] pixel centres, depth [N])."""

    camera = (points_world - c2w[:3, 3]) @ c2w[:3, :3]
    camera = camera[camera[:, 2] > 0.5]
    u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
    v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui, vi, z = u[inside].astype(np.int64), v[inside].astype(np.int64), camera[inside, 2]
    nearest = np.full((height, width), np.inf)
    np.minimum.at(nearest, (vi, ui), z)
    # Points occluded for the camera but seen by the higher LiDAR: farther than
    # the nearest return around them by more than a relative slack.
    neighbourhood = minimum_filter(nearest, size=window, mode="constant", cval=np.inf)
    rows, cols = np.nonzero(np.isfinite(nearest))
    depth = nearest[rows, cols]
    keep = depth <= neighbourhood[rows, cols] * (1.0 + slack) + 0.3
    uv = np.stack((cols[keep] + 0.5, rows[keep] + 0.5), axis=1).astype(np.float32)
    return uv, depth[keep].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True, help="converted scene directory")
    parser.add_argument("--nuscenes", type=Path, required=True, help="nuScenes root (metadata, samples/, sweeps/)")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--max-dt", type=float, default=0.03, help="s: LiDAR sweep to image time (20 Hz LiDAR)")
    parser.add_argument("--ego-radius", type=float, default=2.5)
    parser.add_argument("--occlusion-window", type=int, default=7)
    parser.add_argument("--occlusion-slack", type=float, default=0.05)
    args = parser.parse_args()

    meta = json.loads((args.scene_dir / "scene.json").read_text())["metadata"]
    origin = np.asarray(meta["world_origin"], dtype=np.float64)
    frames = json.loads((args.scene_dir / "cameras.json").read_text())["frames"]
    load = lambda name: {r["token"]: r for r in json.loads((args.nuscenes / args.version / f"{name}.json").read_text())}  # noqa: E731
    sample_data, ego_poses, calibrated = load("sample_data"), load("ego_pose"), load("calibrated_sensor")

    # LIDAR_TOP sweeps per drive log, sorted by time (only extracted files).
    sweeps: dict[str, list[tuple[int, dict]]] = {}
    for record in sample_data.values():
        name = Path(record["filename"]).name
        if "__LIDAR_TOP__" in name and (args.nuscenes / record["filename"]).is_file():
            sweeps.setdefault(name.split("__")[0], []).append((record["timestamp"], record))
    for items in sweeps.values():
        items.sort(key=lambda item: item[0])

    out = args.scene_dir / "lidar_depth"
    out.mkdir(exist_ok=True)
    counts, skipped, cache = [], 0, {}
    for frame in frames:
        name = Path(frame["image"]).stem
        log, stamp = name.split("__")[0], int(name.split("__")[-1])
        items = sweeps.get(log, [])
        times = [t for t, _ in items]
        i = bisect.bisect_left(times, stamp)
        candidates = [items[j] for j in (i - 1, i) if 0 <= j < len(items)]
        if not candidates:
            skipped += 1
            continue
        time, record = min(candidates, key=lambda item: abs(item[0] - stamp))
        if abs(time - stamp) * 1e-6 > args.max_dt:
            skipped += 1
            continue
        if record["token"] not in cache:
            points = np.fromfile(args.nuscenes / record["filename"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            points = points[np.linalg.norm(points[:, :2], axis=1) > args.ego_radius]
            to_world = _transform(ego_poses[record["ego_pose_token"]]) @ _transform(calibrated[record["calibrated_sensor_token"]])
            cache = {record["token"]: points @ to_world[:3, :3].T + to_world[:3, 3] - origin}  # sweeps repeat across 6 cameras
        uv, depth = project(cache[record["token"]], np.asarray(frame["c2w"], dtype=np.float64), np.asarray(frame["K"], dtype=np.float64),
                            frame["width"], frame["height"], args.occlusion_window, args.occlusion_slack)
        np.savez_compressed(out / f"{name}.npz", uv=uv, depth=depth)
        counts.append(len(depth))
    summary = {"images": len(frames), "with_depth": len(counts), "skipped": skipped,
               "points_per_image_median": float(np.median(counts)) if counts else 0.0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
