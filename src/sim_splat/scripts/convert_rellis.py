"""RELLIS-3D (off-road, camera + Ouster LiDAR + SLAM poses) -> SimSplat scene.

One sequence directory as distributed (``<seq>/pylon_camera_node/frameNNNNNN-<s>_<ms>.jpg``,
``<seq>/os1_cloud_node_kitti_bin/NNNNNN.bin``, ``<seq>/poses.txt``), plus the
calibration (camera intrinsics and distortion, camera -> LiDAR transform).

* **Cameras.** The camera and LiDAR frames are synchronised by index; each
  image's pose is ``lidar_pose[i] @ camera_to_lidar``.  Images are undistorted
  and downscaled (``--downscale``), so the scene is PINHOLE and rectified.
  The world is the first LiDAR frame, re-levelled so the ground plane fitted
  to the first scans is z = 0 with z up (the sky model and the excavator's
  ground assume that).
* **Seeds.** Every ``--seed-stride``-th scan, coloured from its synchronised
  image, voxel-downsampled.
* **Depth.** Each image gets the sparse depth of its own scan
  (``lidar_depth/``, same layout as ``add_lidar_depth.py``), for the depth loss.
* **Split.** Every ``--eval-interval``-th kept image is held out.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sim_splat.data.scene_format import export_scene
from sim_splat.scripts.add_lidar_depth import project
from sim_splat.types import Camera, DataparserOutputs


def _quat_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat((x, y, z, w)).as_matrix()


def _level(points: np.ndarray) -> np.ndarray:
    """World transform putting the dominant ground plane of ``points`` at z = 0, z up."""

    low = points[points[:, 2] < np.percentile(points[:, 2], 30)]
    rng = np.random.default_rng(0)
    best, best_inliers = None, -1
    for _ in range(300):
        sample = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        if normal[2] < 0.8:
            continue
        inliers = int((np.abs((low - sample[0]) @ normal) < 0.05).sum())
        if inliers > best_inliers:
            best, best_inliers = (normal, sample[0]), inliers
    normal, point = best
    axis = np.cross(normal, (0.0, 0.0, 1.0))
    angle = np.arccos(np.clip(normal[2], -1, 1))
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-9) * angle).as_matrix() if angle > 1e-6 else np.eye(3)
    level = np.eye(4)
    level[:3, :3] = rotation
    level[:3, 3] = -rotation @ point
    return level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True, help="RELLIS-3D sequence directory (e.g. .../00001)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="rellis")
    parser.add_argument("--frames", type=int, nargs=2, default=(0, 600), help="first, last (exclusive) frame index")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed-stride", type=int, default=5)
    parser.add_argument("--voxel", type=float, default=0.08)
    # Calibration (RELLIS-3D release): pylon camera intrinsics / distortion, and LiDAR -> camera.
    parser.add_argument("--intrinsics", type=float, nargs=4, default=(2813.643275, 2808.326079, 969.285772, 624.049972), metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--distortion", type=float, nargs=5, default=(-0.134313, -0.025905, 0.002181, 0.00084, 0.0))
    # transforms.yaml "os1_cloud_node-pylon_camera_node" maps camera coordinates into the LiDAR frame
    # (checked by projection: the other reading puts scan rings vertical and into the sky).
    parser.add_argument("--camera-to-lidar-q", type=float, nargs=4, default=(-0.50507811, 0.51206185, 0.49024953, -0.49228464), metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--camera-to-lidar-t", type=float, nargs=3, default=(-0.13165462, 0.03870398, -0.17253834))
    args = parser.parse_args()

    import cv2

    images = {int(re.match(r"frame(\d+)", path.name).group(1)): path for path in (args.sequence / "pylon_camera_node").glob("frame*.jpg")}
    scans = {int(path.stem): path for path in (args.sequence / "os1_cloud_node_kitti_bin").glob("*.bin")}
    poses_raw = np.loadtxt(args.sequence / "poses.txt").reshape(-1, 3, 4)
    poses = np.repeat(np.eye(4)[None], len(poses_raw), axis=0)
    poses[:, :3, :4] = poses_raw
    keep = [i for i in range(args.frames[0], min(args.frames[1], len(poses)), args.stride) if i in images and i in scans]
    if len(keep) < 3:
        raise ValueError(f"only {len(keep)} frames with image, scan and pose")

    camera_to_lidar = np.eye(4)
    camera_to_lidar[:3, :3] = _quat_matrix(*args.camera_to_lidar_q)
    camera_to_lidar[:3, 3] = args.camera_to_lidar_t
    load_scan = lambda i: np.fromfile(scans[i], dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)  # noqa: E731

    # Level the world on the first scans' ground.
    first = np.concatenate([load_scan(i) @ poses[i][:3, :3].T + poses[i][:3, 3] for i in keep[:5]])
    level = _level(first[np.linalg.norm(first[:, :2], axis=1) < 25])
    poses = level @ poses

    fx, fy, cx, cy = args.intrinsics
    K_raw = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1.0)))
    width_raw, height_raw = Image.open(images[keep[0]]).size
    K_new, _ = cv2.getOptimalNewCameraMatrix(K_raw, np.asarray(args.distortion), (width_raw, height_raw), 0.0)
    scale = 1.0 / args.downscale
    K = np.diag((scale, scale, 1.0)) @ K_new
    width, height = width_raw // args.downscale, height_raw // args.downscale
    maps = cv2.initUndistortRectifyMap(K_raw, np.asarray(args.distortion), None, K, (width, height), cv2.CV_32FC1)

    cache = args.output.parent / f".{args.output.name}_images"
    cache.mkdir(parents=True, exist_ok=True)
    cameras, image_paths, depth = [], [], {}
    xyz_parts, rgb_parts = [], []
    for index, i in enumerate(keep):
        c2w = poses[i] @ camera_to_lidar
        photo = cv2.remap(cv2.imread(str(images[i])), *maps, cv2.INTER_AREA if args.downscale > 1 else cv2.INTER_LINEAR)
        path = cache / f"{images[i].stem}.png"
        cv2.imwrite(str(path), photo)
        image_paths.append(path)
        cameras.append(Camera(c2w=torch.from_numpy(c2w.astype(np.float32)), K=torch.from_numpy(K.astype(np.float32)),
                              width=width, height=height, image_id=index, model="PINHOLE", is_rectified=True, sensor_id=0,
                              timestamp=float(re.search(r"-(\d+)_(\d+)", images[i].stem).group(1))
                              + float(re.search(r"-(\d+)_(\d+)", images[i].stem).group(2)) * 1e-3))
        world = load_scan(i) @ poses[i][:3, :3].T + poses[i][:3, 3]
        world = world[np.linalg.norm(load_scan(i)[:, :2], axis=1) > 1.5]  # the vehicle itself
        depth[path.stem] = project(world, c2w, K, width, height, window=7, slack=0.05)
        if index % args.seed_stride == 0:
            camera_points = (world - c2w[:3, 3]) @ c2w[:3, :3]
            front = camera_points[:, 2] > 0.5
            u = (K[0, 0] * camera_points[front, 0] / camera_points[front, 2] + K[0, 2]).astype(int)
            v = (K[1, 1] * camera_points[front, 1] / camera_points[front, 2] + K[1, 2]).astype(int)
            seen = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            xyz_parts.append(world[front][seen])
            rgb_parts.append(photo[v[seen], u[seen], ::-1] / 255.0)
    xyz, rgb = np.concatenate(xyz_parts), np.concatenate(rgb_parts)
    voxels = np.floor(xyz / args.voxel).astype(np.int64)
    _, unique = np.unique(voxels, axis=0, return_index=True)
    xyz, rgb = xyz[unique], rgb[unique]

    eval_indices = list(range(0, len(keep), args.eval_interval))
    eval_set = set(eval_indices)
    outputs = DataparserOutputs(
        image_paths=image_paths, cameras=cameras, eval_indices=eval_indices,
        train_indices=[i for i in range(len(keep)) if i not in eval_set],
        points_xyz=torch.from_numpy(xyz.astype(np.float32)), points_rgb=torch.from_numpy(rgb.astype(np.float32)),
        metadata={"source": "rellis3d", "sequence": args.sequence.name, "frames": [keep[0], keep[-1]], "stride": args.stride,
                  "world_from_first_lidar": level.tolist(), "dynamic_track_tokens": []},
    )
    export_scene(outputs, args.output, name=args.name)
    out = args.output / "lidar_depth"
    out.mkdir()
    for stem, (uv, z) in depth.items():
        np.savez_compressed(out / f"{stem}.npz", uv=uv, depth=z)
    print(json.dumps({"images": len(keep), "seed_points": int(len(xyz)), "size": [width, height],
                      "median_depth_points": float(np.median([len(z) for _, z in depth.values()]))}))


if __name__ == "__main__":
    main()
