"""Closed loop: camera perception drives a MuJoCo excavator pushing a look-alike asset.

Task: push the object ``--goal-distance`` m along the push direction and
leave it upright.  Each cycle:

1. **observe**: render the edited scene (inpainted background, the asset at
   its simulated pose, the excavator) from a fixed captured camera;
2. **perceive**: full-frame open-vocabulary detection, then 3D localization
   (rendered depth stands in for LiDAR).  The detection is associated with
   the previous estimate, starting from the object's mapped position;
3. **act**: put the bucket behind the object's estimated position and push
   by the remaining distance (at most ``--max-step`` m), then back off.

``--perception oracle`` uses the simulator's true pose instead, so the two
runs separate controller error from perception error.  Output:
``closed_loop.json`` (per-cycle truth, estimate and error),
``observations.mp4`` and the observation frames with detections drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.scripts.eval_perception import TiledDetector, localize
from sim_splat.scripts.render_edit import _to_uint8
from sim_splat.scripts.render_physics import _box_mesh_gaussians, _project_box
from sim_splat.scripts.simulate_excavator import BUCKET, ExcavatorWorld


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perception", choices=("camera", "oracle"), default="camera")
    parser.add_argument("--view", type=int, help="captured camera index to observe from (default: sees the object largest)")
    parser.add_argument("--goal-distance", type=float, default=1.2)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--max-step", type=float, default=0.8)
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--hit-height", type=float, default=0.35, help="low push: slide, don't topple")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--asset-mass", type=float, default=60.0)
    # "road barrier." boxes the whole row of barriers as one; the colour separates the unit.
    parser.add_argument("--prompt", default="an orange plastic barrier.")
    parser.add_argument("--surface-to-centre", action="store_true",
                        help="shift the depth estimate from the visible surface to the object centre (known size)")
    args = parser.parse_args()

    world = ExcavatorWorld(args.scene, args.track, asset_dir=args.asset_dir, asset_mass=args.asset_mass)
    scene = parse_scene(load_scene_config(args.scene))
    config = load_train_config(args.config)
    device = torch.device(config.device)
    model = SimSplatModel(config.model, scene, config.device)
    load_model_weights(args.background_checkpoint, model, config.device)
    model.eval()
    model.set_step(10**9)
    track = model.track_tokens.index(world.token)
    track_poses = model.object_to_world[:, track].detach().cpu().numpy().astype(np.float64)
    initial_inverse = np.linalg.inv(world.annotation)
    frames = track_poses.shape[0]
    everywhere = torch.ones((frames,), dtype=torch.bool)
    asset_index = model.add_actor("asset", torch.load(args.asset_dir / "gaussians.pt"),
                                  torch.from_numpy(world.asset_pose()).float().expand(frames, 4, 4).clone(), everywhere)
    part_indices = [model.add_actor(f"sim:{i}:{name}", _box_mesh_gaussians(list(world.model.geom_size[g]), tuple(world.model.geom_rgba[g][:3]),
                                                                          config.model.sh_degree),
                                    torch.eye(4).expand(frames, 4, 4).clone(), everywhere)
                    for i, (name, g) in enumerate(zip(world.PARTS, world.part_ids))]
    model.set_hidden_tracks([world.token])

    # Observation camera: the captured view that sees the object largest.
    if args.view is None:
        best = max((((b[2] - b[0]) * (b[3] - b[1]), i) for i in scene.train_indices
                    if (b := _project_box(track_poses[scene.cameras[i].image_id], world.asset_half, scene.cameras[i]))), default=None)
        args.view = best[1]
    camera = scene.cameras[args.view]
    frame = camera.image_id
    frame_from_sim = track_poses[frame] @ initial_inverse  # simulation world -> this frame's world (see render_physics)
    sim_from_frame = np.linalg.inv(frame_from_sim)
    e_from_world = np.linalg.inv(world.world_from_e)
    c2w, K = camera.c2w.numpy().astype(np.float64), camera.K.numpy().astype(np.float64)
    detector = TiledDetector("IDEA-Research/grounding-dino-tiny", args.prompt, 0.3) if args.perception == "camera" else None

    def render() -> tuple[Image.Image, np.ndarray]:
        model.object_to_world[:, asset_index] = torch.from_numpy(frame_from_sim @ world.asset_pose()).float().to(device)
        for index, pose in zip(part_indices, world.part_poses()):
            model.object_to_world[:, index] = torch.from_numpy(frame_from_sim @ pose).float().to(device)
        with torch.no_grad():
            out = model.get_outputs(camera)
        return Image.fromarray(_to_uint8(out["rgb"])), out["depth"][..., 0].cpu().numpy()

    def truth_x() -> float:
        return float((e_from_world @ world.asset_pose())[0, 3])

    estimate_world = (frame_from_sim @ world.asset_pose())[:3, 3]  # mapped prior (this frame's world)

    def perceive(image: Image.Image, depth: np.ndarray) -> tuple[float | None, dict]:
        nonlocal estimate_world
        if detector is None:
            return truth_x(), {}
        ground = float((frame_from_sim @ world.asset_pose())[2, 3] - world.asset_half[2])
        candidates = []
        for det in detector(image):
            point = localize(det["xyxy"], c2w, K, ground, depth).get("depth")
            if point is None:
                continue
            point = np.asarray(point)
            if args.surface_to_centre:
                ray = point - c2w[:3, 3]
                point = point + ray / np.linalg.norm(ray) * float(world.asset_half[0])
            candidates.append((float(np.linalg.norm(point[:2] - estimate_world[:2])), point, det))
        if not candidates:
            return None, {"detections": 0}
        distance, point, det = min(candidates, key=lambda item: item[0])
        if distance > 1.5:
            return None, {"detections": len(candidates), "nearest_m": distance}
        estimate_world = point
        sim_point = sim_from_frame @ np.append(point, 1.0)
        return float((e_from_world @ sim_point)[0]), {"box": det["xyxy"], "score": det["score"], "detections": len(candidates)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video: list[Image.Image] = []
    x0 = truth_x()
    goal = x0 + args.goal_distance
    hit_z = args.hit_height * world.height + BUCKET[0]
    park = np.array((x0 - world.reach - BUCKET[2] - 0.8, hit_z + 1.0))
    world.reset(park)
    wrist = park.copy()

    def move(target: np.ndarray, speed: float) -> None:
        nonlocal wrist
        distance = float(np.linalg.norm(target - wrist))
        steps = max(1, int(distance / speed / world.dt))
        every = max(1, int(1.0 / (12 * world.dt)))  # 12 fps observation video
        start = wrist.copy()
        for step in range(1, steps + 1):
            world.step(start + (target - start) * step / steps)
            if step % every == 0:
                video.append(render()[0])
        wrist = target.copy()
        for _ in range(int(0.4 / world.dt)):  # settle
            world.step(wrist)

    log = []
    for cycle in range(args.max_cycles + 1):
        image, depth = render()
        estimate, info = perceive(image, depth)
        true = truth_x()
        tilt = float(np.degrees(np.arccos(np.clip(world.asset_pose()[2, 2], -1, 1))))
        record = {"cycle": cycle, "true_x": true - x0, "estimate_x": None if estimate is None else estimate - x0,
                  "perception_error_m": None if estimate is None else abs(estimate - true), "tilt_deg": tilt, **info}
        log.append(record)
        drawn = image.copy()
        if "box" in info:
            ImageDraw.Draw(drawn).rectangle(info["box"], outline=(255, 0, 0), width=3)
        ImageDraw.Draw(drawn).text((8, 8), f"cycle {cycle}  true {true - x0:+.2f} m  est "
                                   f"{'—' if estimate is None else f'{estimate - x0:+.2f}'} m  goal {args.goal_distance:+.2f} m", fill=(255, 255, 0))
        drawn.save(args.output_dir / f"cycle_{cycle}.jpg", quality=90)
        print(json.dumps(record), flush=True)
        if estimate is None or abs(goal - estimate) < args.tolerance or cycle == args.max_cycles or tilt > 45:
            break
        push = float(np.clip(goal - estimate, -args.max_step, args.max_step))
        behind = estimate - world.reach - BUCKET[2] - 0.15
        move(np.array((behind, hit_z + 0.6)), 1.0)  # above and behind
        move(np.array((behind, hit_z)), 1.0)  # lower
        move(np.array((behind + 0.15 + push, hit_z)), args.speed)  # push (0.15 m to close the gap)
        move(np.array((behind + push - 0.3, hit_z + 0.6)), 1.0)  # back off and up, out of the camera's way

    final = truth_x() - x0
    summary = {"perception": args.perception, "view": args.view, "goal_m": args.goal_distance, "final_true_m": final,
               "final_error_m": abs(args.goal_distance - final), "success": abs(args.goal_distance - final) < args.tolerance and log[-1]["tilt_deg"] < 45,
               "cycles": len(log) - 1, "surface_to_centre": args.surface_to_centre,
               "median_perception_error_m": float(np.median([r["perception_error_m"] for r in log if r["perception_error_m"] is not None]))
               if any(r["perception_error_m"] is not None for r in log) else None}
    (args.output_dir / "closed_loop.json").write_text(json.dumps({"summary": summary, "cycles": log}, indent=1))
    print(json.dumps(summary, indent=1))
    if video:
        import cv2

        writer = cv2.VideoWriter(str(args.output_dir / "observations.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, video[0].size)
        for image in video:
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        writer.release()


if __name__ == "__main__":
    main()
