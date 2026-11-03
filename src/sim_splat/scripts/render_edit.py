"""Render scene-graph edits: actor removal and simulator-asset replacement.

This reproduces the DrivingGaussian "dynamic object removal" demonstration on
a trained composite model: actors are hidden by track id (or by class:
``moving`` / ``all``) and the static Gaussian field fills the revealed
background.  ``--replace`` swaps an actor for a mesh asset that follows the
actor's per-frame pose — the hook a physics simulator will later drive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sim_splat.assets import fit_mesh_to_box, mesh_to_gaussians, procedural_vehicle_mesh
from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.datamanagers.full_image_datamanager import FullImageDataManager
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel, _psnr_from_mse


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def _labelled(image: np.ndarray, label: str) -> Image.Image:
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, 9 * len(label) + 12, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return result


def _resolve_tracks(selectors: list[str], tracks: list[dict], threshold: float) -> list[str]:
    tokens = [track["token"] for track in tracks]
    chosen: list[str] = []
    for selector in selectors:
        if selector == "all":
            chosen.extend(tokens)
        elif selector == "moving":
            chosen.extend(track["token"] for track in tracks if track["displacement"] > threshold)
        elif selector.startswith("category:"):
            prefix = selector.split(":", 1)[1]
            chosen.extend(track["token"] for track in tracks if track["category"].startswith(prefix))
        else:
            matches = [token for token in tokens if token.startswith(selector)]
            if len(matches) != 1:
                raise ValueError(f"Track selector {selector!r} matched {len(matches)} tracks")
            chosen.extend(matches)
    return sorted(set(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hide", action="append", default=[], help="track prefix, 'moving', 'all' or 'category:<prefix>'")
    parser.add_argument("--replace", action="append", default=[], help="track prefix to replace with a procedural sim vehicle")
    parser.add_argument("--asset-mesh", type=Path, help="optional mesh file used by --replace instead of the procedural vehicle")
    parser.add_argument("--asset-gaussians", type=int, default=60000)
    parser.add_argument("--moving-threshold", type=float, default=1.0, help="metres of keyframe displacement")
    parser.add_argument("--split", choices=("eval", "train", "all"), default="eval")
    parser.add_argument("--channel", type=int, help="only render this sensor id (e.g. 0 = first channel)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="also write an mp4 of the panels")
    parser.add_argument("--actor-file", action="append", type=Path, default=[], help="rigid actor from lift_masks_to_actor.py")
    args = parser.parse_args()

    scene_config = load_scene_config(args.scene)
    train_config = load_train_config(args.config)
    outputs = parse_scene(scene_config)
    manager = FullImageDataManager(outputs, seed=train_config.seed)
    device = train_config.device
    model = SimSplatModel(train_config.model, outputs, device)
    step = load_model_weights(args.checkpoint, model, device)
    model.eval()
    model.set_step(10**9)

    metadata = outputs.metadata
    tracks = [
        {"token": token, "category": category, "displacement": float(displacement), "box_size_xyz": size}
        for token, category, displacement, size in zip(
            metadata.get("dynamic_track_tokens", []),
            metadata.get("dynamic_track_categories", []),
            metadata.get("dynamic_track_displacement", []),
            metadata.get("dynamic_track_box_size_xyz", []),
        )
    ]
    frames = len(outputs.image_paths)
    for actor_path in args.actor_file:
        actor = torch.load(actor_path, map_location="cpu", weights_only=False)
        pose = torch.from_numpy(np.asarray(actor["object_to_world"], dtype=np.float32))
        model.add_actor(actor["name"], actor["gaussians"], pose.expand(frames, 4, 4).clone(), torch.ones((frames,), dtype=torch.bool))
        half = np.asarray(actor["half_extent_xyz"], dtype=np.float64)
        tracks.append({"token": actor["name"], "category": "lifted", "displacement": 0.0, "box_size_xyz": (2.0 * half).tolist()})

    hidden = _resolve_tracks(args.hide, tracks, args.moving_threshold)
    replaced = _resolve_tracks(args.replace, tracks, args.moving_threshold)
    for token in replaced:
        index = model.track_tokens.index(token)
        box_size = next(track["box_size_xyz"] for track in tracks if track["token"] == token)
        if box_size is None:
            raise ValueError(f"Track {token} has no box_size_xyz; cannot fit an asset to it")
        if args.asset_mesh is not None:
            import trimesh

            mesh = trimesh.load(args.asset_mesh, force="mesh")
        else:
            mesh = procedural_vehicle_mesh()
        mesh = fit_mesh_to_box(mesh, np.asarray(box_size, dtype=np.float64) / 2.0)
        gaussians = mesh_to_gaussians(mesh, args.asset_gaussians, train_config.model.sh_degree)
        model.add_actor(f"asset:{token}", gaussians, model.object_to_world[:, index].clone(), model.track_visible[:, index].clone())

    owners = model.point_track_indices
    report = {
        "checkpoint_step": step,
        "hidden_tracks": hidden,
        "replaced_tracks": replaced,
        "tracks": [
            {
                "token": track["token"],
                "category": track["category"],
                "displacement_m": round(track["displacement"], 2),
                "gaussians": int((owners == model.track_tokens.index(track["token"])).sum().item()),
            }
            for track in tracks
        ],
        "static_gaussians": int((owners < 0).sum().item()),
    }

    if args.split == "eval":
        indices = outputs.eval_indices
    elif args.split == "train":
        indices = outputs.train_indices
    else:
        indices = list(range(len(outputs.image_paths)))
    if args.channel is not None:
        indices = [index for index in indices if outputs.cameras[index].sensor_id == args.channel]
    indices = indices[:: max(args.stride, 1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Path] = []
    psnrs: list[float] = []
    edit_tokens = sorted(set(hidden) | set(replaced))
    asset_tokens = [f"asset:{token}" for token in replaced]
    with torch.no_grad():
        for count, index in enumerate(indices):
            batch = manager.dataset[index]
            target = batch.image.to(device)
            model.set_hidden_tracks(asset_tokens)
            full = model.get_outputs(batch.camera)["rgb"]
            psnrs.append(float(_psnr_from_mse((full - target).square().mean()).item()))
            images = [_labelled(_to_uint8(target), "ground truth"), _labelled(_to_uint8(full), f"reconstruction {psnrs[-1]:.1f} dB")]
            if edit_tokens:
                model.set_hidden_tracks(edit_tokens + asset_tokens)
                removed = model.get_outputs(batch.camera)["rgb"]
                images.append(_labelled(_to_uint8(removed), f"removed {len(edit_tokens)} actor(s)"))
            if replaced:
                model.set_hidden_tracks(edit_tokens)
                swapped = model.get_outputs(batch.camera)["rgb"]
                images.append(_labelled(_to_uint8(swapped), "replaced with sim asset"))
            width, height = images[0].size
            columns = 2
            rows = (len(images) + columns - 1) // columns
            canvas = Image.new("RGB", (width * columns, height * rows))
            for position, image in enumerate(images):
                canvas.paste(image, ((position % columns) * width, (position // columns) * height))
            path = args.output_dir / f"{count:04d}_{Path(batch.image_name).stem}.jpg"
            canvas.save(path, quality=92)
            panels.append(path)

    report["num_views"] = len(indices)
    report["reconstruction_psnr_mean"] = float(np.mean(psnrs)) if psnrs else None
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "tracks"}, indent=2))

    if args.video and panels:
        import cv2

        first = cv2.imread(str(panels[0]))
        writer = cv2.VideoWriter(str(args.output_dir / "edit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6, (first.shape[1], first.shape[0]))
        for path in panels:
            writer.write(cv2.imread(str(path)))
        writer.release()


if __name__ == "__main__":
    main()
