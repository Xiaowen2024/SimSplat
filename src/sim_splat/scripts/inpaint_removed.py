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

    # 1-2. Silhouette masks and LaMa fills on the captured photos.
    lama = SimpleLama()
    out = args.output_dir / "inpainted"
    out.mkdir(parents=True, exist_ok=True)
    views: list[tuple[int, torch.Tensor, torch.Tensor]] = []  # (dataset index, filled image, mask)
    model.eval()
    with torch.no_grad():
        for index in scene.train_indices:
            camera = scene.cameras[index]
            if not bool(model.track_visible[camera.image_id, track]):
                continue
            # Actor Gaussians can float far from the object along viewing rays;
            # keep only the silhouette inside the projected (enlarged) 3D box.
            pose = model.object_to_world[camera.image_id, track].detach().cpu().numpy().astype(np.float64)
            box = _project_box(pose, half * 1.15, camera)
            if box is None:
                continue
            inside = np.zeros((camera.height, camera.width), dtype=bool)
            inside[int(box[1]) : int(np.ceil(box[3])), int(box[0]) : int(np.ceil(box[2]))] = True
            silhouette = (_object_alpha(model, model.point_track_indices == track, camera) > 0.3).cpu().numpy() & inside
            if silhouette.sum() < args.min_pixels:
                continue
            mask = _dilate(silhouette, args.dilate) & _dilate(inside, args.dilate)
            photo = (dataset[index].image.numpy() * 255).round().astype(np.uint8)
            filled = np.asarray(lama(Image.fromarray(photo), Image.fromarray(mask.astype(np.uint8) * 255)))
            filled = filled[: photo.shape[0], : photo.shape[1]]  # LaMa pads to a multiple of 8
            views.append((index, torch.from_numpy(filled).float() / 255.0, torch.from_numpy(mask)))
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
        if rng.random() < args.inpaint_fraction:
            index, filled, mask = views[rng.integers(len(views))]
            batch = dataset[index]
            batch.image = filled.to(device)
            # Supervise the filled region even where the static mask excluded the actor.
            batch.static_mask = torch.maximum(batch.static_mask, mask[..., None].float()).to(device)
            model.set_hidden_tracks([target])
        else:
            batch = manager.next_train(step0 + step)
            batch.image = batch.image.to(device)
            batch.static_mask = batch.static_mask.to(device)
            model.set_hidden_tracks([])
        outputs = model.get_outputs(batch.camera)
        loss = sum(model.get_loss_dict(outputs, batch).values())
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.gauss_params.values():
            if parameter.grad is not None:
                parameter.grad.mul_(static_rows.view(-1, *([1] * (parameter.grad.dim() - 1))))
        for optimizer in optimizers.values():
            optimizer.step()
        model.clamp_gaussian_parameters_()
        if step % 500 == 0:
            log.append({"step": step, "loss": float(loss)})
            print(f"step={step} loss={float(loss):.4f}", flush=True)

    model.set_hidden_tracks([])
    save_checkpoint(args.output_dir / "final.pt", step0, model, optimizers)
    (args.output_dir / "inpaint.json").write_text(json.dumps(
        {"track": target, "views": [index for index, _, _ in views], "steps": args.steps, "lr_scale": args.lr_scale,
         "dilate": args.dilate, "log": log}, indent=1))


if __name__ == "__main__":
    main()
