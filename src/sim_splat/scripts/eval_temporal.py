"""Temporal consistency and render speed of a reconstruction along a real camera trajectory.

A simulator replays sequences, so a renderer must not flicker.  Geometry and
poses are known, so no optical flow is needed: for consecutive frames t and
t+1 of one camera, every pixel of t+1 is lifted with the rendered depth of
t+1, moved into frame t, and the colour there is compared with the colour at
t+1 (backward warping).  Pixels that leave the image or are occluded (the
depth at the warped location disagrees by more than 5% + 0.1 m) are skipped.

The same warp is applied to the *real* photos (with the rendered depth, the
best geometry available).  Real consecutive frames differ too (exposure,
moving objects, warp error), so the render's residual is reported next to
the photos' own: a renderer is temporally consistent when its residual is
not above the real one.  Reported per renderer:

* ``warp_l1``: mean absolute RGB difference after warping (0-255), static
  pixels; ``warp_l1_objects``: inside annotated objects' masks;
* ``warp_lpips``: LPIPS between frame t+1 and warped frame t (tLPIPS-style);
* ``render_fps``: median over the trajectory (synchronised timing).

Writes ``temporal.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def warp_previous(previous: torch.Tensor, depth_next: torch.Tensor, depth_previous: torch.Tensor, camera_next, camera_previous):
    """Backward-warp image ``previous`` [H, W, C] into the next camera; returns (warped, valid [H, W])."""

    device = depth_next.device
    height, width = depth_next.shape
    v, u = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    K1, K0 = camera_next.K.to(device).float(), camera_previous.K.to(device).float()
    rays = torch.stack(((u + 0.5 - K1[0, 2]) / K1[0, 0], (v + 0.5 - K1[1, 2]) / K1[1, 1], torch.ones_like(depth_next)), -1)
    points = rays * depth_next[..., None]
    c2w1, c2w0 = camera_next.c2w.to(device).float(), camera_previous.c2w.to(device).float()
    world = points @ c2w1[:3, :3].T + c2w1[:3, 3]
    in0 = (world - c2w0[:3, 3]) @ c2w0[:3, :3]
    z0 = in0[..., 2].clamp_min(1e-3)
    u0 = K0[0, 0] * in0[..., 0] / z0 + K0[0, 2]
    v0 = K0[1, 1] * in0[..., 1] / z0 + K0[1, 2]
    grid = torch.stack((u0 / width * 2 - 1, v0 / height * 2 - 1), -1)[None]
    warped = torch.nn.functional.grid_sample(previous.permute(2, 0, 1)[None], grid, align_corners=False)[0].permute(1, 2, 0)
    sampled_depth = torch.nn.functional.grid_sample(depth_previous[None, None], grid, align_corners=False)[0, 0]
    valid = (in0[..., 2] > 0.1) & (u0 >= 0) & (u0 < width) & (v0 >= 0) & (v0 < height)
    valid &= (sampled_depth - z0).abs() < 0.05 * z0 + 0.1  # occluded in the previous frame
    return warped, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--renderer", action="append", required=True, metavar="NAME=SCENE,CONFIG,CHECKPOINT",
                        help="each with its own scene file; cameras are matched to the trajectory by image name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor", type=int, default=0, help="camera (sensor id) whose consecutive frames form the trajectory")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--start", type=int, default=0, help="first frame (position in the sensor's sequence)")
    args = parser.parse_args()

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    from sim_splat.config import load_scene_config, load_train_config
    from sim_splat.data.dataparsers import parse_scene
    from sim_splat.data.input_dataset import InputDataset
    from sim_splat.engine.checkpoints import load_model_weights
    from sim_splat.models.sim_splat_model import SimSplatModel

    scene = parse_scene(load_scene_config(args.scene))
    dataset = InputDataset(scene)
    sequence = sorted((i for i, camera in enumerate(scene.cameras) if camera.sensor_id == args.sensor), key=lambda i: scene.cameras[i].timestamp)
    sequence = sequence[args.start : args.start + args.frames]
    device = torch.device("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    results = {}
    real_done = False
    for spec in args.renderer:
        name, paths = spec.split("=", 1)
        scene_path, config_path, checkpoint = (Path(item) for item in paths.split(","))
        own_scene = parse_scene(load_scene_config(scene_path))
        by_name = {Path(path).name: camera for path, camera in zip(own_scene.image_paths, own_scene.cameras)}
        config = load_train_config(config_path)
        model = SimSplatModel(config.model, own_scene, config.device)
        load_model_weights(checkpoint, model, config.device)
        model.eval()
        model.set_step(10**9)
        frames, seconds = [], []
        with torch.no_grad():
            for index in sequence:
                camera = by_name[Path(scene.image_paths[index]).name]
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model.get_outputs(camera)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                batch = dataset[index]
                frames.append({"rgb": out["rgb"].clamp(0, 1), "depth": out["depth"][..., 0], "camera": camera,
                               "photo": batch.image.to(device), "static": batch.static_mask.to(device)[..., 0] > 0.5})
            stats = {"render": {"l1": [], "l1_objects": [], "lpips": []}, "real": {"l1": [], "l1_objects": [], "lpips": []}}
            for previous, current in zip(frames, frames[1:]):
                for source, key in (("render", "rgb"), ("real", "photo")):
                    if source == "real" and real_done:
                        continue
                    warped, valid = warp_previous(previous[key], current["depth"], previous["depth"], current["camera"], previous["camera"])
                    difference = (warped - current[key]).abs().mean(-1) * 255
                    static, objects = valid & current["static"], valid & ~current["static"]
                    if static.any():
                        stats[source]["l1"].append(float(difference[static].mean()))
                    if objects.sum() > 200:
                        stats[source]["l1_objects"].append(float(difference[objects].mean()))
                    masked = lambda image: torch.where(valid[..., None], image.clamp(0, 1), torch.zeros_like(image))  # noqa: E731
                    stats[source]["lpips"].append(float(lpips(masked(warped).permute(2, 0, 1)[None], masked(current[key]).permute(2, 0, 1)[None])))
        summarise = lambda values: float(np.mean(values)) if values else None  # noqa: E731
        results[name] = {"warp_l1": summarise(stats["render"]["l1"]), "warp_l1_objects": summarise(stats["render"]["l1_objects"]),
                         "warp_lpips": summarise(stats["render"]["lpips"]), "render_fps": 1.0 / float(np.median(seconds[1:])),
                         "gaussians": int(model.gauss_params["means"].shape[0]),
                         "resolution": [scene.cameras[sequence[0]].width, scene.cameras[sequence[0]].height]}
        if not real_done:
            results["real_photos"] = {"warp_l1": summarise(stats["real"]["l1"]), "warp_l1_objects": summarise(stats["real"]["l1_objects"]),
                                      "warp_lpips": summarise(stats["real"]["lpips"])}
            real_done = True
        del model
        torch.cuda.empty_cache()
        print(name, json.dumps(results[name]), flush=True)
    results["trajectory"] = {"sensor": args.sensor, "frames": len(sequence), "first_image": str(scene.image_paths[sequence[0]].name),
                             "gpu": torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
