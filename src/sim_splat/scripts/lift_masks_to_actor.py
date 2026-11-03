"""Lift 2D object masks of a *static* scene into a rigid, removable actor.

nuScenes supplies 3D boxes per actor; a COLMAP capture (e.g. the ETH3D
trolley) does not.  This script recovers the same scene-graph split from 2D
masks only:

1. every Gaussian centre is projected into every view; it votes "object" in a
   view when it lands inside that view's object mask;
2. masked pixels are lifted with the rendered depth to visible *surface
   points*; their largest cluster defines the actor box (Gaussian centres
   would include faint near-camera floaters that also project into masks);
3. a view only counts for a Gaussian when the Gaussian is *visible* there
   (its depth matches the rendered expected depth); occluded background
   behind the object therefore never votes "object";
4. Gaussians visible in at least ``--min-views`` views and inside the mask in
   at least ``--vote`` of them are object candidates;
5. a gravity-aligned box is fitted to the surface cluster (up = mean camera up vector, heading = horizontal PCA);
6. voted Gaussians inside the box are written, in the box frame, to an actor file, and the
   remaining rows to a background-only checkpoint.

The actor file is a rigid body: ``render_edit.py --actor-file`` can render it
back at its original pose, at a simulator-provided pose, or replace it with a
mesh asset fitted to its box (which doubles as a collision proxy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel, _quat_multiply_wxyz, _rotation_to_quat_wxyz


def _find_mask(masks_dir: Path, image_path: Path) -> Path | None:
    matches = sorted(masks_dir.rglob(f"{image_path.stem}.png"))
    return matches[0] if matches else None


def _largest_cluster(points: np.ndarray, voxel: float) -> np.ndarray:
    """Boolean mask of the points in the largest 26-connected voxel component."""

    keys = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    grid = np.zeros(keys.max(axis=0) + 1, dtype=bool)
    grid[tuple(keys.T)] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if count == 0:
        return np.zeros((len(points),), dtype=bool)
    point_labels = labels[tuple(keys.T)]
    sizes = np.bincount(point_labels, minlength=count + 1)
    sizes[0] = 0
    return point_labels == int(np.argmax(sizes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object-masks", type=Path, required=True, help="PNG per image (matched by file stem)")
    parser.add_argument("--object-is-black", action="store_true", help="masks use the static-mask convention (object=0)")
    parser.add_argument("--name", required=True, help="actor name, e.g. trolley")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--vote", type=float, default=0.6)
    parser.add_argument("--cluster-voxel", type=float, default=0.0, help="0 = 1%% of the candidate extent")
    parser.add_argument("--depth-tolerance", type=float, default=0.05, help="relative depth slack for visibility")
    parser.add_argument("--up", default="camera", help="'camera' (orthogonal to camera x axes) or 'x,y,z'")
    parser.add_argument("--box-inflation", type=float, nargs=3, default=(0.05, 0.05, 0.05),
                        help="relative box growth (x y z) when assigning voted rows; grow x for objects cropped in the views")
    parser.add_argument("--min-surface-alpha", type=float, default=0.9)
    parser.add_argument("--pixel-stride", type=int, default=8, help="mask pixel subsampling for surface points")
    parser.add_argument("--reference-view", help="image stem whose mask is known to show the target (e.g. DSC_0704)")
    parser.add_argument("--consistency-scale", type=float, default=2.0, help="accept views within this x reference object radius")
    parser.add_argument("--max-mask-fraction", type=float, default=0.5, help="ignore near-full-frame masks")
    args = parser.parse_args()

    scene_config = load_scene_config(args.scene)
    train_config = load_train_config(args.config)
    outputs = parse_scene(scene_config)
    device = train_config.device
    model = SimSplatModel(train_config.model, outputs, device)
    load_model_weights(args.checkpoint, model, device)

    model.eval()
    model.set_step(10**9)
    static = model.point_track_indices < 0
    means = model.gauss_params["means"].detach()
    # Pass 1: per-view visibility, mask hits and lifted surface points.
    views: list[dict] = []
    for camera, image_path in zip(outputs.cameras, outputs.image_paths):
        mask_path = _find_mask(args.object_masks, image_path)
        if mask_path is None:
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        if args.object_is_black:
            mask = ~mask
        if mask.shape != (camera.height, camera.width):
            raise ValueError(f"{mask_path} is {mask.shape[::-1]}, expected {(camera.width, camera.height)}")
        if not mask.any():
            continue
        world_to_camera = torch.linalg.inv(camera.c2w.to(device))
        points = means @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = points[:, 2]
        pixels = points @ camera.K.to(device).T
        u = pixels[:, 0] / depth.clamp_min(1e-6)
        v = pixels[:, 1] / depth.clamp_min(1e-6)
        frame = (depth > 1e-3) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        rows = v.clamp(0, camera.height - 1).long()
        columns = u.clamp(0, camera.width - 1).long()
        with torch.no_grad():
            rendered = model.get_outputs(camera)
        rendered_depth = rendered["depth"][..., 0]
        rendered_alpha = rendered["alpha"][..., 0]
        # A view only votes for Gaussians that are the visible surface there.
        frame &= depth <= rendered_depth[rows, columns] * (1.0 + args.depth_tolerance) + 1e-3
        hit = frame & torch.from_numpy(mask).to(device)[rows, columns]
        # Masked pixels lifted with rendered depth: the visible object surface.
        mask_rows, mask_columns = np.nonzero(mask[:: args.pixel_stride, :: args.pixel_stride])
        pixel_v = torch.from_numpy(mask_rows * args.pixel_stride).to(device)
        pixel_u = torch.from_numpy(mask_columns * args.pixel_stride).to(device)
        # Expected depth is only a surface depth where the render is opaque;
        # at mask borders it blends object and floor and smears the box.
        opaque = rendered_alpha[pixel_v, pixel_u] >= args.min_surface_alpha
        pixel_v, pixel_u = pixel_v[opaque], pixel_u[opaque]
        if pixel_v.numel() == 0:
            continue
        z = rendered_depth[pixel_v, pixel_u]
        K = camera.K.to(device)
        camera_points = torch.stack(((pixel_u + 0.5 - K[0, 2]) / K[0, 0] * z, (pixel_v + 0.5 - K[1, 2]) / K[1, 1] * z, z), dim=-1)
        c2w = camera.c2w.to(device)
        surface = (camera_points @ c2w[:3, :3].T + c2w[:3, 3]).cpu().numpy()
        views.append({"name": image_path.stem, "fraction": float(mask.mean()), "frame": frame, "hit": hit,
                      "surface": surface, "centre": np.median(surface, axis=0)})
    if not views:
        raise FileNotFoundError(f"No non-empty masks in {args.object_masks} matched the scene's images")

    # Pass 2: keep only views whose mask lands on the *same* 3D object.  An
    # open-vocabulary detector run per image happily labels different objects
    # in different views (measured on ETH3D: 4-5 distinct objects plus
    # near-full-frame masks), so a reference view picks the target.
    usable = [view for view in views if view["fraction"] <= args.max_mask_fraction]
    if args.reference_view:
        reference = next((view for view in views if view["name"] == args.reference_view), None)
        if reference is None:
            raise ValueError(f"Reference view {args.reference_view} has no non-empty mask")
        anchor = reference["centre"]
        radius = args.consistency_scale * float(np.percentile(np.linalg.norm(reference["surface"] - anchor, axis=1), 90))
        usable = [view for view in usable if np.linalg.norm(view["centre"] - anchor) <= radius]
    else:
        anchor, radius = None, None
    if not usable:
        raise ValueError("No view passed the mask-consistency filter")
    in_view = torch.zeros((means.shape[0],), dtype=torch.int32, device=device)
    inside = torch.zeros_like(in_view)
    for view in usable:
        in_view += view["frame"].int()
        inside += view["hit"].int()
    used_views = len(usable)

    ratio = inside.float() / in_view.clamp_min(1).float()
    candidates = static & (in_view >= args.min_views) & (ratio >= args.vote)
    candidate_index = torch.where(candidates)[0].cpu().numpy()
    points = np.concatenate([view["surface"] for view in usable])
    if anchor is not None:
        points = points[np.linalg.norm(points - anchor, axis=1) <= radius]
    low, high = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
    voxel = args.cluster_voxel or 0.02 * float(np.linalg.norm(high - low))
    object_points = points[_largest_cluster(points, voxel)]

    # Gravity-aligned box: z = up, x = dominant horizontal direction.
    if args.up == "camera":
        # Hand-held captures have ~zero roll: every camera x axis is
        # horizontal, so up is the direction orthogonal to all of them.  The
        # mean camera up vector is biased by the usual downward pitch (this
        # tilted the measured ETH3D box), so it only fixes the sign here.
        right_axes = np.stack([camera.c2w[:3, 0].numpy() for camera in outputs.cameras])
        up = np.linalg.svd(right_axes, full_matrices=False)[2][-1]
        if up @ -np.mean([camera.c2w[:3, 1].numpy() for camera in outputs.cameras], axis=0) < 0:
            up = -up
    else:
        up = np.asarray([float(value) for value in args.up.split(",")])
    up = up / np.linalg.norm(up)
    centre = np.median(object_points, axis=0)
    offsets = object_points - centre
    horizontal = offsets - np.outer(offsets @ up, up)
    _, _, axes = np.linalg.svd(horizontal, full_matrices=False)
    x_axis = axes[0] - (axes[0] @ up) * up
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack((x_axis, np.cross(up, x_axis), up), axis=1)  # columns = box axes
    local = offsets @ rotation
    low, high = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
    centre = centre + rotation @ ((low + high) / 2.0)
    half_extent = (high - low) / 2.0
    # Every voted Gaussian (including faint ones) inside the box joins the
    # actor; voted floaters outside it stay in the background.
    candidate_local = (means[candidate_index].cpu().numpy() - centre) @ rotation
    in_box = np.all(np.abs(candidate_local) <= half_extent * (1.0 + np.asarray(args.box_inflation)), axis=1)
    object_index = candidate_index[in_box]
    object_to_world = np.eye(4, dtype=np.float32)
    object_to_world[:3, :3] = rotation
    object_to_world[:3, 3] = centre

    pose = torch.from_numpy(object_to_world).to(device)
    selected = torch.from_numpy(object_index).to(device)
    world_to_object = torch.linalg.inv(pose)
    gaussians = {name: parameter.detach()[selected].clone() for name, parameter in model.gauss_params.items()}
    gaussians["means"] = gaussians["means"] @ world_to_object[:3, :3].T + world_to_object[:3, 3]
    inverse_quat = _rotation_to_quat_wxyz(world_to_object[:3, :3])
    gaussians["quats"] = _quat_multiply_wxyz(inverse_quat.expand_as(gaussians["quats"]), torch.nn.functional.normalize(gaussians["quats"], dim=-1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_path = args.output_dir / f"{args.name}.actor.pt"
    torch.save(
        {
            "name": args.name,
            "gaussians": {name: value.cpu() for name, value in gaussians.items()},
            "object_to_world": object_to_world,
            "half_extent_xyz": half_extent.astype(np.float32),
            "source_checkpoint": str(args.checkpoint),
        },
        actor_path,
    )

    # Background-only checkpoint: identical model state minus the object rows.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    keep_rows = torch.ones((means.shape[0],), dtype=torch.bool)
    keep_rows[torch.from_numpy(object_index)] = False
    for key in list(state):
        if key.startswith("gauss_params.") or key == "point_track_indices":
            state[key] = state[key][keep_rows]
    checkpoint["optimizers"] = {}
    checkpoint["strategy_state"] = None
    background_path = args.output_dir / f"background_without_{args.name}.pt"
    torch.save(checkpoint, background_path)

    report = {
        "views_with_masks": len(views),
        "views_used": [view["name"] for view in usable],
        "candidates_after_vote": int(len(candidate_index)),
        "object_gaussians": int(len(object_index)),
        "background_gaussians": int(keep_rows.sum().item()),
        "box_centre": centre.tolist(),
        "box_half_extent": half_extent.tolist(),
        "actor_file": str(actor_path),
        "background_checkpoint": str(background_path),
    }
    (args.output_dir / f"{args.name}_lift_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
