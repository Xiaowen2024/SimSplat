"""Fill the background behind a removed actor (2D inpainting distilled into 3D).

The ground and wall behind an object were never observed, so removing it
leaves semi-transparent, smeared background Gaussians.  This script:

1. renders the actor's silhouette in every training view that sees it and
   dilates it into a mask;
2. inpaints the *captured photo* inside the mask with LaMa (object erased
   with real texture and lighting around it);
3. fine-tunes only the static background Gaussians, alternating inpainted
   views (actor hidden) with ordinary training views (actor shown), so the
   rest of the scene is preserved.  Per-view inpaintings disagree slightly;
   the shared 3D Gaussians average them into one consistent fill.

With ``--asset-dir`` (a look-alike asset from ``build_lookalike_asset.py``)
the object is being *made interactive* rather than deleted.  Its appearance
was baked into the static field, and fine-tuning alone only recolours those
Gaussians: they stay on the object's surface and then cover the asset (a
green-asset test showed 27-48% of its pixels hidden).  So static Gaussians
inside the asset's volume (the union over frames of its per-frame pose) are
carved out first.  The volume alone is not enough: nuScenes cameras
disagree on height, so the static field spreads the object along viewing
rays.  So a static Gaussian is also carved when it projects inside the
asset's silhouette in front of its surface (within 1.5 m) in at least two
training views.  Masks come from the asset's silhouette.  In an inpainted view only
Gaussians *behind* the asset surface learn the fill: one in front would be
asked to show background there and to stay transparent in ordinary views
(where the asset renders the object), and it ends up as a half-transparent
veil over the asset.  In ordinary
views the asset renders the object, so the static field only has to learn
what lies around and behind it.

Writes a new checkpoint plus ``inpainted/`` (photo, mask, fill) for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.datamanagers.full_image_datamanager import FullImageDataManager
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.checkpoints import load_model_weights, save_checkpoint
from sim_splat.engine.optimizers import build_gaussian_optimizers
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.scripts.render_physics import _object_alpha, _project_box
from sim_splat.scripts.train import _scene_scale


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--track", required=True, help="unique prefix of the removed actor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--min-pixels", type=int, default=400, help="skip views where the actor covers fewer pixels")
    parser.add_argument("--dilate", type=int, default=9, help="mask dilation, pixels")
    parser.add_argument("--lr-scale", type=float, default=0.1, help="fraction of the training learning rates")
    parser.add_argument("--inpaint-fraction", type=float, default=0.5, help="share of steps on inpainted views")
    parser.add_argument("--asset-dir", type=Path, help="look-alike asset: carve its volume and use it as the object")
    parser.add_argument("--carve-margin", type=float, default=0.08, help="m, around the asset (not below it)")
    args = parser.parse_args()

    from simple_lama_inpainting import SimpleLama

    scene = parse_scene(load_scene_config(args.scene))
    train_config = load_train_config(args.config)
    device = torch.device(train_config.device)
    model = SimSplatModel(train_config.model, scene, train_config.device)
    step0 = load_model_weights(args.checkpoint, model, train_config.device)
    model.set_step(10**9)
    matches = [token for token in model.track_tokens if token.startswith(args.track)]
    if len(matches) != 1:
        raise ValueError(f"track prefix {args.track!r} matched {len(matches)} tracks")
    target = matches[0]
    track = model.track_tokens.index(target)
    manager = FullImageDataManager(scene, seed=train_config.seed)
    dataset = manager.dataset
    half = np.asarray(scene.metadata["dynamic_track_box_size_xyz"][track], dtype=np.float64) / 2.0
    hidden_object = [target]
    asset_index = None
    carved = 0
    if args.asset_dir:
        info = json.loads((args.asset_dir / "asset.json").read_text())
        asset_poses = model.object_to_world[:, track].detach().double() @ torch.tensor(info["local_correction"], device=device).double()
        asset_half = torch.tensor(info["size_xyz"], device=device).double() / 2.0
        static = model.point_track_indices < 0
        means = model.gauss_params["means"].detach().double()
        inside = torch.zeros_like(static)
        frames = torch.nonzero(model.track_visible[:, track]).flatten()
        for frame in frames[:: max(1, len(frames) // 60)]:
            pose = asset_poses[frame]
            local = (means - pose[:3, 3]) @ pose[:3, :3]
            inside |= ((local[:, :2].abs() <= asset_half[:2] + args.carve_margin).all(1)
                       & (local[:, 2] >= -asset_half[2] + 0.03) & (local[:, 2] <= asset_half[2] + args.carve_margin))
        keep = ~(inside & static)
        carved = int((~keep).sum())
        model.gauss_params = torch.nn.ParameterDict({name: torch.nn.Parameter(p.detach()[keep]) for name, p in model.gauss_params.items()})
        model.point_track_indices = model.point_track_indices[keep]
        print(f"carved {carved} static Gaussians inside the asset volume", flush=True)
        asset_index = model.add_actor("lookalike", torch.load(args.asset_dir / "gaussians.pt"), asset_poses.float(),
                                      torch.ones((asset_poses.shape[0],), dtype=torch.bool))
        # Occluders: static Gaussians in front of the asset inside its silhouette.
        votes = torch.zeros(len(model.point_track_indices), device=device)
        static = model.point_track_indices < 0
        model.eval()
        with torch.no_grad():
            for index in scene.train_indices:
                camera = scene.cameras[index]
                rows = model.point_track_indices == asset_index
                opacities = model.gauss_params["opacities"]
                backup = opacities.data.clone()
                opacities.data[~rows] = -30.0
                model.set_hidden_tracks([])
                out = model.get_outputs(camera)
                opacities.data.copy_(backup)
                silhouette, surface = out["alpha"][..., 0] > 0.5, out["depth"][..., 0]
                if silhouette.sum() < 50:
                    continue
                c2w, K = camera.c2w.to(device).float(), camera.K.to(device).float()
                in_camera = (model.gauss_params["means"].detach() - c2w[:3, 3]) @ c2w[:3, :3]
                z = in_camera[:, 2]
                u = (K[0, 0] * in_camera[:, 0] / z + K[0, 2]).round().long()
                v = (K[1, 1] * in_camera[:, 1] / z + K[1, 2]).round().long()
                ok = static & (z > 0.3) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
                hit = torch.zeros_like(ok)
                ui, vi = u[ok], v[ok]
                front = silhouette[vi, ui] & (z[ok] < surface[vi, ui] - 0.03) & (z[ok] > surface[vi, ui] - 1.5)
                hit[torch.nonzero(ok).flatten()[front]] = True
                votes += hit.float()
        occluders = votes >= 2
        keep = ~occluders
        model.gauss_params = torch.nn.ParameterDict({name: torch.nn.Parameter(p.detach()[keep]) for name, p in model.gauss_params.items()})
        model.point_track_indices = model.point_track_indices[keep]
        carved += int(occluders.sum())
        print(f"carved {int(occluders.sum())} static Gaussians occluding the asset", flush=True)
        hidden_object = [target, "lookalike"]

    # 1-2. Silhouette masks and LaMa fills on the captured photos.
    lama = SimpleLama()
    out = args.output_dir / "inpainted"
    out.mkdir(parents=True, exist_ok=True)
    views: list[tuple] = []  # (dataset index, filled image, mask, asset depth or None)
    model.eval()
    with torch.no_grad():
        for index in scene.train_indices:
            camera = scene.cameras[index]
            if not bool(model.track_visible[camera.image_id, track]):
                continue
            # Actor Gaussians can float far from the object along viewing rays;
            # keep only the silhouette inside the projected (enlarged) 3D box.
            pose = model.object_to_world[camera.image_id, asset_index if asset_index is not None else track].detach().cpu().numpy().astype(np.float64)
            box = _project_box(pose, (asset_half.cpu().numpy() if asset_index is not None else half) * 1.15, camera)
            if box is None:
                continue
            inside = np.zeros((camera.height, camera.width), dtype=bool)
            inside[int(box[1]) : int(np.ceil(box[3])), int(box[0]) : int(np.ceil(box[2]))] = True
            rows = model.point_track_indices == (asset_index if asset_index is not None else track)
            silhouette = (_object_alpha(model, rows, camera) > 0.3).cpu().numpy() & inside
            surface = None
            if asset_index is not None:
                backup = model.gauss_params["opacities"].data.clone()
                model.gauss_params["opacities"].data[~rows] = -30.0
                rendered = model.get_outputs(camera)
                model.gauss_params["opacities"].data.copy_(backup)
                surface = torch.where(rendered["alpha"][..., 0] > 0.5, rendered["depth"][..., 0], torch.full_like(rendered["depth"][..., 0], 1e6))
            if silhouette.sum() < args.min_pixels:
                continue
            mask = _dilate(silhouette, args.dilate) & _dilate(inside, args.dilate)
            photo = (dataset[index].image.numpy() * 255).round().astype(np.uint8)
            filled = np.asarray(lama(Image.fromarray(photo), Image.fromarray(mask.astype(np.uint8) * 255)))
            filled = filled[: photo.shape[0], : photo.shape[1]]  # LaMa pads to a multiple of 8
            views.append((index, torch.from_numpy(filled).float() / 255.0, torch.from_numpy(mask), surface))
            if len(views) <= 12:
                strip = np.concatenate((photo, np.repeat(mask[..., None] * 255, 3, axis=2).astype(np.uint8), filled), axis=1)
                Image.fromarray(strip).save(out / f"{index:05d}.jpg", quality=85)
    if not views:
        raise ValueError("No training view sees the actor; lower --min-pixels")
    print(f"inpainted {len(views)} views", flush=True)

    # 3. Fine-tune static Gaussians only (actor rows get zero gradient).
    model.train()
    optimizers = build_gaussian_optimizers(model, train_config.optimizer, scene_scale=_scene_scale(scene.cameras))
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * args.lr_scale
    static_rows = (model.point_track_indices < 0)[:, None]
    rng = np.random.default_rng(train_config.seed)
    log = []
    for step in range(args.steps):
        surface = None
        if rng.random() < args.inpaint_fraction:
            index, filled, mask, surface = views[rng.integers(len(views))]
            batch = dataset[index]
            batch.image = filled.to(device)
            # Supervise the filled region even where the static mask excluded the actor.
            batch.static_mask = torch.maximum(batch.static_mask, mask[..., None].float()).to(device)
            model.set_hidden_tracks(hidden_object)
        else:
            batch = manager.next_train(step0 + step)
            batch.image = batch.image.to(device)
            batch.static_mask = batch.static_mask.to(device)
            # With an asset, it (not the near-transparent track) renders the object.
            model.set_hidden_tracks([target] if asset_index is not None else [])
        outputs = model.get_outputs(batch.camera)
        loss = sum(model.get_loss_dict(outputs, batch).values())
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = static_rows[:, 0]
        if surface is not None:  # only Gaussians behind the asset may learn the fill
            camera = batch.camera
            c2w, K = camera.c2w.to(device).float(), camera.K.to(device).float()
            in_camera = (model.gauss_params["means"].detach() - c2w[:3, 3]) @ c2w[:3, :3]
            z = in_camera[:, 2].clamp_min(1e-3)
            u = (K[0, 0] * in_camera[:, 0] / z + K[0, 2]).round().long().clamp(0, camera.width - 1)
            v = (K[1, 1] * in_camera[:, 1] / z + K[1, 2]).round().long().clamp(0, camera.height - 1)
            trainable = trainable & ~(z < surface[v, u] - 0.03)
        for parameter in model.gauss_params.values():
            if parameter.grad is not None:
                parameter.grad.mul_(trainable.view(-1, *([1] * (parameter.grad.dim() - 1))))
        for optimizer in optimizers.values():
            optimizer.step()
        model.clamp_gaussian_parameters_()
        if step % 500 == 0:
            log.append({"step": step, "loss": float(loss)})
            print(f"step={step} loss={float(loss):.4f}", flush=True)

    model.set_hidden_tracks([])
    if asset_index is not None:  # drop the asset again: the checkpoint must match the scene's tracks
        keep = model.point_track_indices != asset_index
        model.gauss_params = torch.nn.ParameterDict({name: torch.nn.Parameter(p.detach()[keep]) for name, p in model.gauss_params.items()})
        model.point_track_indices = model.point_track_indices[keep]
        model.object_to_world = model.object_to_world[:, :asset_index]
        model.track_visible = model.track_visible[:, :asset_index]
        model.track_tokens.pop()
    save_checkpoint(args.output_dir / "final.pt", step0, model, optimizers)
    (args.output_dir / "inpaint.json").write_text(json.dumps(
        {"track": target, "views": [index for index, *_ in views], "steps": args.steps, "lr_scale": args.lr_scale,
         "dilate": args.dilate, "carved": carved, "asset_dir": str(args.asset_dir) if args.asset_dir else None, "log": log}, indent=1))


if __name__ == "__main__":
    main()
