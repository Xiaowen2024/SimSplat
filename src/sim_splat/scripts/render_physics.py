"""Replay a physics simulation inside a trained scene along a smooth camera path.

Two kinds of simulation output are accepted:

* ``simulate_push.py`` (PyBullet): the pushed object is the actor's *own*
  reconstructed Gaussians, cloned into a new actor (``add_actor``) whose pose
  row is overwritten with the simulated pose every video frame;
* ``simulate_excavator.py`` (MuJoCo): the actor is *replaced* by a textured
  mesh asset, sampled into Gaussians and posed the same way.

Either way the original track is hidden in the edited render and the
simulator's own bodies (bucket, excavator links) are rendered as box assets.

Outputs: ``original/``, ``edited/``, ``background/``, ``real/`` frames,
``meta.json`` (per-frame projected boxes and hole statistics), ``push.mp4``
(original | edited, with zoomed crops) and ``collage.jpg`` (original / pushing
/ after).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sim_splat.assets import load_asset, mesh_to_gaussians, meshes_to_gaussians
from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.data.input_dataset import InputDataset
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.scripts.render_edit import _interpolate_camera_path, _labelled, _to_uint8
from sim_splat.types import Camera

_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _project_box(pose: np.ndarray, half: np.ndarray, camera: Camera) -> list[float] | None:
    """Pixel-space bounding box of an oriented 3D box, or None if not in front."""

    world = (_CORNERS * half) @ pose[:3, :3].T + pose[:3, 3]
    w2c = np.linalg.inv(camera.c2w.numpy().astype(np.float64))
    points = world @ w2c[:3, :3].T + w2c[:3, 3]
    if np.any(points[:, 2] <= 0.3):
        return None
    pixels = points @ camera.K.numpy().astype(np.float64).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    x0, y0 = np.clip(pixels.min(axis=0), 0, (camera.width, camera.height))
    x1, y1 = np.clip(pixels.max(axis=0), 0, (camera.width, camera.height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def _choose_views(outputs, pose: np.ndarray, half: np.ndarray, count: int) -> list[int]:
    """Captured frames of one sensor that see the target large and well inside the image."""

    best: dict[int, list[tuple[float, int]]] = {}
    for index, camera in enumerate(outputs.cameras):
        box = _project_box(pose, half, camera)
        if box is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (0.2 * camera.width < cx < 0.8 * camera.width and 0.15 * camera.height < cy < 0.85 * camera.height):
            continue
        best.setdefault(camera.sensor_id, []).append(((box[2] - box[0]) * (box[3] - box[1]), index))
    if not best:
        raise ValueError("No captured view sees the target well; choose --views explicitly")
    sensor = max(best, key=lambda key: max(best[key])[0])  # where the target appears largest
    frames = sorted(index for _, index in best[sensor])
    largest = max(best[sensor])[1]
    centre = frames.index(largest)
    start = max(0, min(centre - count // 2, len(frames) - count))
    return frames[start : start + count]


def _box_mesh_gaussians(half: list[float], colour: tuple[float, float, float], sh_degree: int) -> dict[str, torch.Tensor]:
    import trimesh

    mesh = trimesh.creation.box(extents=2.0 * np.asarray(half))
    mesh.visual.face_colors = np.array([*(np.asarray(colour) * 255), 255], dtype=np.uint8)
    return mesh_to_gaussians(mesh, 8000, sh_degree)


def _object_alpha(model: SimSplatModel, rows: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Silhouette (alpha) of the given Gaussian rows alone."""

    opacities = model.gauss_params["opacities"]
    backup = opacities.data.clone()
    opacities.data[~rows] = -30.0
    model.set_hidden_tracks([])
    alpha = model.get_outputs(camera)["alpha"][..., 0]
    opacities.data.copy_(backup)
    return alpha


def _crop(image: Image.Image, centre: tuple[float, float], size: tuple[int, int], out: tuple[int, int]) -> Image.Image:
    cx = int(np.clip(centre[0], size[0] / 2, image.width - size[0] / 2))
    cy = int(np.clip(centre[1], size[1] / 2, image.height - size[1] / 2))
    box = (cx - size[0] // 2, cy - size[1] // 2, cx + size[0] // 2, cy + size[1] // 2)
    return image.crop(box).resize(out, Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True, help="output of simulate_push.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="*", help="captured frame indices (same sensor) for the camera path")
    parser.add_argument("--num-views", type=int, default=4, help="captured views auto-selected for the path")
    parser.add_argument("--hold", type=int, default=24, help="extra still frames after the simulation ends")
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--asset-gaussians", type=int, default=60000)
    parser.add_argument("--asset-root", type=Path, help="all look-alike assets: those not simulated are shown at rest")
    parser.add_argument("--skip-depth", action="store_true", help="do not save per-frame depth (needed only by eval_perception)")
    args = parser.parse_args()

    scene = parse_scene(load_scene_config(args.scene))
    train_config = load_train_config(args.config)
    device = train_config.device
    model = SimSplatModel(train_config.model, scene, device)
    load_model_weights(args.checkpoint, model, device)
    model.eval()
    model.set_step(10**9)
    physics = torch.load(args.physics, map_location="cpu", weights_only=False)
    asset = physics.get("asset")
    poses = (asset["poses"] if asset else physics["actor_to_world"]).numpy().astype(np.float64)
    half = np.asarray(physics["box_half_extent"], dtype=np.float64)
    moved_half = np.asarray(asset["half_extent"], dtype=np.float64) if asset else half
    parts = physics.get("parts") or [{"name": "bucket", "half_extent": physics["pusher_half_extent"],
                                      "color": (0.95, 0.72, 0.08), "poses": physics["pusher_to_world"]}]
    label = physics.get("label", "sim bucket")
    initial = physics["initial_actor_to_world"].numpy().astype(np.float64)
    target = physics["track"]
    track = model.track_tokens.index(target)

    # The physics-driven actor: a sim asset, or a clone of the target's own Gaussians.
    frames = model.object_to_world.shape[0]
    if asset:
        moved = (torch.load(asset["gaussians"]) if "gaussians" in asset else
                 meshes_to_gaussians(load_asset(asset["path"], asset["height"]), args.asset_gaussians, train_config.model.sh_degree))
    else:
        own = model.point_track_indices == track
        moved = {name: parameter.detach()[own].clone() for name, parameter in model.gauss_params.items()}
    still = torch.from_numpy(initial).float().expand(frames, 4, 4).clone()
    everywhere = torch.ones((frames,), dtype=torch.bool)
    clone_index = model.add_actor(f"physics:{target}", moved, still, everywhere)
    part_indices = []
    for i, part in enumerate(parts):
        gaussians = _box_mesh_gaussians(part["half_extent"], tuple(part["color"]), train_config.model.sh_degree)
        part_indices.append(model.add_actor(f"sim:{i}:{part['name']}", gaussians, part["poses"][0].expand(frames, 4, 4).clone(), everywhere))
    sim_tokens = [f"sim:{i}:{part['name']}" for i, part in enumerate(parts)]
    # Other simulated bodies (look-alike neighbours pushed or hit in turn).
    neighbours = physics.get("neighbours") or []
    neighbour_indices = [model.add_actor(f"physics:nb:{item['token'][:8]}", torch.load(item["gaussians"]), still, everywhere)
                         for item in neighbours]
    physics_tokens = [f"physics:{target}", *(f"physics:nb:{item['token'][:8]}" for item in neighbours)]
    # Look-alike assets that are not simulated stay where they are (the background
    # was inpainted with all of them removed).
    from sim_splat.scripts.build_lookalike_asset import asset_poses as look_alike_poses

    simulated_tokens = {target, *(item["token"] for item in neighbours)}
    static_tokens = []
    if args.asset_root:
        for path in sorted(args.asset_root.glob("*/asset.json")):
            info = json.loads(path.read_text())
            if info["track"] in simulated_tokens or info["track"] not in model.track_tokens:
                continue
            own = model.object_to_world[:, model.track_tokens.index(info["track"])].detach().cpu().numpy()
            model.add_actor(f"static:{info['track'][:8]}", torch.load(path.parent / "gaussians.pt"),
                            torch.from_numpy(look_alike_poses(own, info)).float(), everywhere)
            static_tokens.append(info["track"])
    hidden_tracks = [target, *(item["token"] for item in neighbours), *static_tokens]

    # nuScenes annotations of a static object can drift between frames
    # (measured: 0.86 m in z for this barrier, since ego poses carry no real
    # elevation).  The reconstruction was fitted to each frame's own pose, so
    # physics is applied as a *relative* motion on top of that pose.
    track_poses = model.object_to_world[:, track].detach().cpu().numpy().astype(np.float64)
    initial_inverse = np.linalg.inv(initial)

    anchor = track_poses @ initial_inverse  # frame f's world <- simulation world
    if asset and "asset_dir" in asset:  # look-alike: anchor on the asset's per-frame pose (height-corrected)
        info = json.loads((Path(asset["asset_dir"]) / "asset.json").read_text())
        anchor = look_alike_poses(track_poses, info) @ np.linalg.inv(asset["initial_pose"].numpy().astype(np.float64))

    def to_frame(frame: int, pose: np.ndarray) -> np.ndarray:
        """Map a simulated world pose into frame ``frame``'s annotation world."""

        return anchor[frame] @ pose

    # Neighbours were snapped onto the simulator's ground plane: render their
    # simulated *motion* on top of their own photo-fitted pose instead.
    neighbour_photo_poses = []
    for item in neighbours:
        info = json.loads((Path(item["asset_dir"]) / "asset.json").read_text())
        own = model.object_to_world[:, model.track_tokens.index(item["token"])].detach().cpu().numpy()
        neighbour_photo_poses.append(look_alike_poses(own, info))

    def neighbour_pose(item: dict, photo_poses: np.ndarray, frame: int, step: int) -> np.ndarray:
        motion = item["poses"][step].numpy().astype(np.float64) @ np.linalg.inv(item["initial_pose"].numpy().astype(np.float64))
        return anchor[frame] @ motion @ np.linalg.inv(anchor[frame]) @ photo_poses[frame]

    def simulated(step: int, frame: int) -> np.ndarray:
        return to_frame(frame, poses[step])

    views = args.views or _choose_views(scene, initial, half, args.num_views)
    sim_frames = len(poses)
    total = sim_frames + args.hold
    dense = _interpolate_camera_path(scene.cameras, views, inserts=max(1, int(np.ceil(total / max(len(views) - 1, 1)))))
    path = [dense[int(round(item))] for item in np.linspace(0, len(dense) - 1, total)]

    dataset = InputDataset(scene)
    for name in ("original", "edited", "background", "real"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    width, height = scene.cameras[views[0]].width, scene.cameras[views[0]].height
    crop_size = (int(width / args.zoom), int(height / args.zoom))
    video_frames: list[Image.Image] = []
    with torch.no_grad():
        for k, (camera, source) in enumerate(path):
            step = min(k, sim_frames - 1)
            model.object_to_world[:, clone_index] = torch.from_numpy(simulated(step, source)).float().to(device)
            for part, index in zip(parts, part_indices):
                model.object_to_world[:, index] = torch.from_numpy(to_frame(source, part["poses"][step].numpy().astype(np.float64))).float().to(device)

            def pose_neighbours(at: int) -> None:
                for item, index, photo_poses in zip(neighbours, neighbour_indices, neighbour_photo_poses):
                    model.object_to_world[:, index] = torch.from_numpy(neighbour_pose(item, photo_poses, source, at)).float().to(device)

            renders = {}
            for name, hidden in (
                # With look-alike neighbours the background has all of them removed:
                # "original" is the simulated scene before anything moved.
                ("original", [f"physics:{target}", *sim_tokens, *hidden_tracks] if neighbours else [*physics_tokens, *sim_tokens]),
                ("edited", hidden_tracks),
                ("background", [*hidden_tracks, *physics_tokens, *sim_tokens]),
            ):
                pose_neighbours(0 if name == "original" else step)
                if name == "original" and neighbours:
                    model.object_to_world[:, clone_index] = torch.from_numpy(simulated(0, source)).float().to(device)
                    hidden = [*sim_tokens, *hidden_tracks]
                model.set_hidden_tracks(hidden)
                out = model.get_outputs(camera)
                model.object_to_world[:, clone_index] = torch.from_numpy(simulated(step, source)).float().to(device)
                renders[name] = out
                Image.fromarray(_to_uint8(out["rgb"])).save(args.output_dir / name / f"{k:04d}.png")
            is_real = camera is scene.cameras[source]
            if is_real:
                Image.fromarray(_to_uint8(dataset[source].image)).save(args.output_dir / "real" / f"{k:04d}.png")
            original_box = _project_box(track_poses[source], half, camera)
            current_box = _project_box(simulated(step, source), moved_half, camera)
            # Silhouettes of the object at its original and current pose: the
            # vacated area is old minus new; holes are measured behind the old one.
            original_rows = model.point_track_indices == track
            clone_rows = model.point_track_indices == clone_index
            now = _object_alpha(model, clone_rows, camera) > 0.5
            if asset:
                # A replaced object's own Gaussians are near-transparent (baked into the
                # static field): its original silhouette is the asset at its initial pose.
                model.object_to_world[:, clone_index] = torch.from_numpy(simulated(0, source)).float().to(device)
                before = _object_alpha(model, clone_rows, camera) > 0.5
                model.object_to_world[:, clone_index] = torch.from_numpy(simulated(step, source)).float().to(device)
            else:
                before = _object_alpha(model, original_rows, camera) > 0.5
            vacated = before & ~now

            def silhouette_box(mask: torch.Tensor) -> list[float] | None:
                ys, xs = torch.nonzero(mask, as_tuple=True)
                return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) > 20 else None

            moved_now = simulated(step, source)
            background_alpha = renders["background"]["alpha"][..., 0]
            hole = None
            if before.any():
                hole = {
                    "silhouette_pixels": int(before.sum()),
                    "vacated_fraction": float(vacated.sum() / before.sum()),
                    "mean_alpha": float(background_alpha[before].mean()),
                    "fraction_alpha_below_0.5": float((background_alpha[before] < 0.5).float().mean()),
                }
            records.append({"frame": k, "sim_step": step, "source_view": source, "is_real": is_real,
                            "original_box": original_box, "current_box": current_box, "background_alpha_in_original_box": hole,
                            # Ground truth for perception: tight 2D boxes from silhouettes, 3D centres in this frame's world.
                            "original_silhouette_box": silhouette_box(before), "current_silhouette_box": silhouette_box(now),
                            "original_centre": track_poses[source][:3, 3].tolist() if not asset else to_frame(source, poses[0])[:3, 3].tolist(),
                            "current_centre": moved_now[:3, 3].tolist(), "current_up": moved_now[:3, 2].tolist(),
                            "neighbour_centres": [neighbour_pose(item, photo, source, step)[:3, 3].tolist()
                                                  for item, photo in zip(neighbours, neighbour_photo_poses)],
                            "ground_z": float(to_frame(source, poses[0])[2, 3] - moved_half[2]),
                            "camera": {"c2w": camera.c2w.tolist(), "K": camera.K.tolist(), "width": camera.width, "height": camera.height}})
            for name in ("edited", "original") if not args.skip_depth else ():
                np.save(args.output_dir / name / f"{k:04d}_depth.npy", renders[name]["depth"][..., 0].half().cpu().numpy())

            # Video frame: original | edited on top, zoomed crops below.
            images = {name: Image.fromarray(_to_uint8(renders[name]["rgb"])) for name in ("original", "edited")}
            focus = original_box or [width / 2, height / 2, width / 2, height / 2]
            centre = ((focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2)
            canvas = Image.new("RGB", (2 * width, 2 * height))
            for column, name in enumerate(("original", "edited")):
                label = "original scene" if name == "original" else f"{label}  t={step / physics['fps']:.2f}s"
                canvas.paste(_labelled(np.asarray(images[name]), label), (column * width, 0))
                canvas.paste(_crop(images[name], centre, crop_size, (width, height)), (column * width, height))
            video_frames.append(canvas)

    import cv2

    writer = cv2.VideoWriter(str(args.output_dir / "push.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), physics["fps"], video_frames[0].size)
    for frame in video_frames:
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()

    # Collage: original / pushing / after, full frame (box outline) above a zoomed crop.
    contact = physics.get("first_contact_frame") or 0
    moments = (("original", 0, "original"), ("pushing", min(contact + physics["fps"] // 2, sim_frames - 1), "edited"),
               ("after", total - 1, "edited"))
    collage = Image.new("RGB", (3 * width, 2 * height))
    for column, (label, k, source) in enumerate(moments):
        image = Image.open(args.output_dir / source / f"{k:04d}.png").convert("RGB")
        record = records[k]
        outlined = image.copy()
        draw = ImageDraw.Draw(outlined)
        if record["original_box"]:
            draw.rectangle(record["original_box"], outline=(0, 200, 255), width=2)
        if label != "original" and record["current_box"]:
            draw.rectangle(record["current_box"], outline=(255, 80, 0), width=2)
        focus = record["original_box"] or [width / 2, height / 2, width / 2, height / 2]
        centre = ((focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2)
        collage.paste(_labelled(np.asarray(outlined), label), (column * width, 0))
        collage.paste(_crop(image, centre, crop_size, (width, height)), (column * width, height))
    collage.save(args.output_dir / "collage.jpg", quality=90)

    summary = {key: value for key, value in physics.items() if not isinstance(value, (torch.Tensor, dict, list))}
    if asset:
        summary["asset"] = {key: value for key, value in asset.items() if not isinstance(value, torch.Tensor)}
    summary["neighbours"] = [{key: value for key, value in item.items() if not isinstance(value, torch.Tensor)} for item in neighbours]
    summary.update({"views": views, "video_frames": total, "frames": records})
    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=1))


if __name__ == "__main__":
    main()
