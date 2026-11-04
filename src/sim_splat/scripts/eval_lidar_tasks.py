"""Downstream geometric tasks on a simulated depth sensor (rendered depth vs real LiDAR).

A simulator that renders depth is also a LiDAR / depth-camera simulator.  For
every held-out view, the rendered depth is sampled at exactly the pixels where
the real LiDAR has a return (same rays), so the "simulated scan" and the real
scan can be compared ray by ray.  The same simple perception stack runs on both:

1. **Ground segmentation**: a ground plane is fitted (RANSAC) to the *real*
   scan of the view; every point gets its height above it.
2. **Obstacle occupancy** (what a machine's safety layer uses): a point is an
   obstacle if it is 0.3-3 m above the ground; bird's-eye cells of
   ``--cell`` m holding at least 2 obstacle points are occupied.

Reported per renderer (cells within ``--range`` m):

* ``phantom_rate``: occupied in the simulated scan but not within one cell of
  any real occupied cell (false obstacles: the machine would stop for nothing);
* ``miss_rate``: real occupied cells with no simulated cell within one cell;
* ``occupancy_iou``;
* ``ground_height_error_m`` (median) and ``ground_over_10cm``: on rays whose real
  return is ground, how far the simulated surface is from the real ground (what
  excavation depth, object placement and the hidden-ground swap rely on);
* ``phantom_views``: share of views with at least one phantom cell.

Writes ``lidar_tasks.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def ground_plane(points: np.ndarray, iterations: int = 200, threshold: float = 0.1, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC plane (unit normal n with n_z > 0, offset d: n.x + d = 0) on the lower points."""

    low = points[points[:, 2] < np.percentile(points[:, 2], 50)]
    rng = np.random.default_rng(seed)
    best, best_count = (np.array((0.0, 0.0, 1.0)), -float(np.median(low[:, 2]))), -1
    for _ in range(iterations):
        sample = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < 0.9:
            continue
        offset = -float(normal @ sample[0])
        count = int((np.abs(low @ normal + offset) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    return best


def occupancy(points: np.ndarray, heights: np.ndarray, cell: float, low: float = 0.3, high: float = 3.0) -> set:
    obstacle = (heights > low) & (heights < high)
    keys, counts = np.unique(np.floor(points[obstacle, :2] / cell).astype(np.int64), axis=0, return_counts=True)
    return {tuple(key) for key, count in zip(keys, counts) if count >= 2}


def _near(cells: set, other: set) -> int:
    return sum(any((x + dx, y + dy) in other for dx in (-1, 0, 1) for dy in (-1, 0, 1)) for x, y in cells)


def _draw(path: Path, photo: np.ndarray, real_cells: set, panels: dict, c2w: np.ndarray, cell: float, size: int = 40) -> None:
    """Photo + one bird's-eye panel per renderer, in the camera's heading frame (up = forward):
    white = real obstacle cell, green = simulated and real, red = phantom, blue = missed."""

    from PIL import Image, ImageDraw

    forward = c2w[:3, 2][:2] / max(np.linalg.norm(c2w[:3, 2][:2]), 1e-9)
    left = np.array((-forward[1], forward[0]))
    scale = 9  # pixels per cell
    height = int(size / cell) * scale
    image = Image.fromarray((photo * 255).astype(np.uint8))
    image = image.resize((int(image.width * height / image.height), height))
    tiles = [image]

    def to_pixel(key):
        centre = (np.asarray(key) + 0.5) * cell - c2w[:2, 3]
        f, l = centre @ forward, centre @ left
        return int(height / 2 - l / cell * scale), int(height - f / cell * scale)

    for name, cells in panels.items():
        tile = Image.new("RGB", (height, height), (25, 25, 25))
        draw = ImageDraw.Draw(tile)
        for key in real_cells:
            x, y = to_pixel(key)
            near = any((key[0] + dx, key[1] + dy) in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            draw.rectangle((x, y, x + scale - 1, y + scale - 1), fill=(220, 220, 220) if near else (60, 120, 255))
        for key in cells:
            x, y = to_pixel(key)
            near = any((key[0] + dx, key[1] + dy) in real_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            draw.rectangle((x + 2, y + 2, x + scale - 3, y + scale - 3), fill=(60, 200, 80) if near else (240, 60, 50))
        draw.polygon([(height // 2, height - 2), (height // 2 - 6, height - 14), (height // 2 + 6, height - 14)], fill=(255, 220, 0))
        draw.rectangle((0, 0, 8 + 6 * len(name), 16), fill="black")
        draw.text((4, 3), name, fill="white")
        tiles.append(tile)
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles) + 4 * (len(tiles) - 1), height), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + 4
    canvas.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="scene whose held-out views (with lidar_depth) are scored")
    parser.add_argument("--renderer", action="append", required=True, metavar="NAME=SCENE,CONFIG,CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=float, default=0.5)
    parser.add_argument("--range", type=float, default=40.0, help="m from the camera")
    parser.add_argument("--figures", type=int, nargs="*", default=[], help="eval views (scene indices) to draw as BEV panels")
    args = parser.parse_args()

    from sim_splat.config import load_scene_config, load_train_config
    from sim_splat.data.dataparsers import parse_scene
    from sim_splat.data.input_dataset import InputDataset
    from sim_splat.engine.checkpoints import load_model_weights
    from sim_splat.models.sim_splat_model import SimSplatModel

    scene = parse_scene(load_scene_config(args.scene))
    dataset = InputDataset(scene)
    views = []
    for index in scene.eval_indices:
        lidar = dataset[index].lidar
        if lidar is not None and len(lidar) > 200:
            views.append((index, lidar.numpy()))

    def world_points(camera, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        K, c2w = camera.K.numpy().astype(np.float64), camera.c2w.numpy().astype(np.float64)
        rays = np.stack(((uv[:, 0] + 0.5 - K[0, 2]) / K[0, 0], (uv[:, 1] + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(uv))), axis=1)
        return (rays * depth[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]

    # Real scans: ground plane, heights and occupancy per view (renderer independent).
    real = {}
    for index, lidar in views:
        camera = scene.cameras[index]
        uv, z = lidar[:, :2].astype(np.float64), lidar[:, 2].astype(np.float64)
        keep = z < args.range
        points = world_points(camera, uv[keep], z[keep])
        normal, offset = ground_plane(points)
        heights = points @ normal + offset
        real[index] = {"keep": keep, "uv": uv[keep].astype(int), "points": points, "heights": heights, "plane": (normal, offset),
                       "cells": occupancy(points, heights, args.cell), "ground": np.abs(heights) < 0.15}

    results, drawn = {}, {index: {} for index in args.figures}
    for spec in args.renderer:
        name, paths = spec.split("=", 1)
        scene_path, config_path, checkpoint = (Path(item) for item in paths.split(","))
        own = parse_scene(load_scene_config(scene_path))
        by_name = {Path(path).name: camera for path, camera in zip(own.image_paths, own.cameras)}
        config = load_train_config(config_path)
        model = SimSplatModel(config.model, own, config.device)
        load_model_weights(checkpoint, model, config.device)
        model.eval()
        model.set_step(10**9)
        phantom = missed = sim_total = real_total = union = inter = phantom_views = 0
        ground_errors = []
        with torch.no_grad():
            for index, _ in views:
                camera = by_name[Path(scene.image_paths[index]).name]
                depth = model.get_outputs(camera)["depth"][..., 0].cpu().numpy().astype(np.float64)
                item = real[index]
                uv = item["uv"]
                simulated = world_points(scene.cameras[index], uv.astype(np.float64), depth[uv[:, 1], uv[:, 0]])
                normal, offset = item["plane"]
                heights = simulated @ normal + offset
                cells = occupancy(simulated, heights, args.cell)
                ghost = len(cells) - _near(cells, item["cells"])
                phantom += ghost
                phantom_views += ghost > 0
                missed += len(item["cells"]) - _near(item["cells"], cells)
                sim_total += len(cells)
                real_total += len(item["cells"])
                inter += len(cells & item["cells"])
                union += len(cells | item["cells"])
                ground_errors.append(np.abs(heights[item["ground"]] - item["heights"][item["ground"]]))
                if index in drawn:
                    drawn[index][name] = cells
        errors = np.concatenate(ground_errors)
        results[name] = {"phantom_rate": phantom / max(sim_total, 1), "miss_rate": missed / max(real_total, 1),
                         "occupancy_iou": inter / max(union, 1), "phantom_views": phantom_views / len(views),
                         "ground_height_error_m": float(np.median(errors)), "ground_over_10cm": float((errors > 0.1).mean()),
                         "simulated_cells": sim_total, "real_cells": real_total}
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    results["setup"] = {"views": len(views), "cell_m": args.cell, "range_m": args.range}
    for index, panels in drawn.items():
        if panels:
            _draw(args.output.parent / f"bev_{index:04d}.jpg", dataset[index].image.numpy(), real[index]["cells"], panels,
                  scene.cameras[index].c2w.numpy(), args.cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
