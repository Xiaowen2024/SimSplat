"""Build a sim asset that looks like a reconstructed object: template mesh + photo texture.

Static street furniture (barriers, cones) never moves during capture, so the
reconstruction bakes it into the static field and its actor Gaussians stay
nearly transparent (scene-0061 barrier: 12 of 6011 actor Gaussians have
opacity > 0.5).  To make such an object interactive we rebuild it as a
conventional asset:

1. **Pose and size.** nuScenes ego poses have z = 0, so the shared world
   is not consistent in height and the *per-frame* annotation is the pose
   that is right relative to that frame's camera.  The asset is therefore
   ``annotation[frame] @ correction``, with one constant local correction
   fitted to the reconstruction.  Rendered depth is back-projected
   inside the projected box, the camera-facing side is found as the
   densest plane along the box's depth axis, and its points give top,
   bottom and ends.  Annotation boxes carry margin, are off by tens of cm
   along the depth axis, and their heights drift by up to ~0.9 m per frame.
2. **Shape.** A category template (``procedural_barrier_meshes``) sized to
   the annotation box; its convex parts double as physics colliders.
3. **Texture.** Surface samples are projected into every training photo
   that sees them, after a per-view vertical registration (the residual
   height error of per-frame annotations is several cm): the shift is the
   one under which the front face agrees with rendered depth while points
   just above its top edge do not.  Photo colours are mapped back through
   the per-sensor appearance transform, because the renderer applies it
   again.  Visibility is a depth test against the rendered depth
   and a facing test.  Each sample keeps its K best observations (frontal,
   close), and their robust mean is its colour.  Sides that were never
   photographed take the colour of the mirrored point (the template is
   symmetric front/back and left/right), then of the nearest observed sample.

Output directory: ``gaussians.pt`` (surfels, local frame), ``part_*.obj``
(collision/visual mesh), ``asset.json`` (world pose, size, coverage) and
``preview.jpg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sim_splat.assets import procedural_barrier_meshes, surfels_to_gaussians
from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.dataparsers import parse_scene
from sim_splat.data.input_dataset import InputDataset
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.scripts.render_edit import _to_uint8
from sim_splat.scripts.render_physics import _project_box


def _views(scene, poses: np.ndarray, half: np.ndarray, min_pixels: float) -> list[tuple[float, int]]:
    found = []
    for index in scene.train_indices:
        box = _project_box(poses[scene.cameras[index].image_id], half, scene.cameras[index])
        if box and (area := (box[2] - box[0]) * (box[3] - box[1])) >= min_pixels:
            found.append((area, index))
    return sorted(found, reverse=True)


def _fit_box(points: np.ndarray, cameras_local: np.ndarray, size: np.ndarray):
    """Refine a box (fixed yaw) from back-projected depth points in its per-frame local frame."""

    side = np.sign(np.median(cameras_local[:, 0]))
    near = points[(np.abs(points[:, 0]) < 1.2 * size[0]) & (np.abs(points[:, 1]) < 0.4 * size[1])
                  & (np.abs(points[:, 2]) < 0.3 * size[2])]
    histogram, edges = np.histogram(near[:, 0], bins=np.arange(-1.2 * size[0], 1.2 * size[0] + 0.02, 0.02))
    face_x = float(edges[np.argmax(histogram)] + 0.01)
    face = points[(np.abs(points[:, 0] - face_x) < 0.06) & (np.abs(points[:, 1]) < 0.8 * size[1])
                  & (np.abs(points[:, 2]) < 1.2 * size[2])]
    top = float(np.percentile(face[:, 2], 98))
    # The face plane also contains the kerb below the object: take the bottom
    # from the support surface just in front of the face instead.
    front = points[(side * (points[:, 0] - face_x) > 0.04) & (side * (points[:, 0] - face_x) < 0.3)
                   & (np.abs(points[:, 1]) < 0.4 * size[1]) & (points[:, 2] < top - 0.3)]
    levels, level_edges = np.histogram(front[:, 2], bins=np.arange(front[:, 2].min(), top, 0.02))
    bottom = float(level_edges[np.argmax(levels)] + 0.01)
    left, right = np.percentile(face[:, 1], [2, 98])
    depth = 0.75 * size[0]  # the back face is never observed: keep the category's proportion
    fitted = np.array((depth, right - left, top - bottom))
    # Face at mid-height of the tapered template sits at 0.36 * depth from the centre.
    centre = np.array((face_x - side * 0.36 * depth, (left + right) / 2, (top + bottom) / 2))
    correction = np.eye(4)
    correction[:3, 3] = centre
    fit = {"camera_side": float(side), "face_x": face_x, "top": top, "bottom": bottom, "face_bottom_p3": float(np.percentile(face[:, 2], 3)), "face_points": int(len(face)), "centre_offset_local": centre.round(3).tolist(),
           "annotation_size": size.round(3).tolist(), "fitted_size": fitted.round(3).tolist()}
    return correction, fitted, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="unedited reconstruction (depth for visibility)")
    parser.add_argument("--track", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surfels", type=int, default=120000)
    parser.add_argument("--min-pixels", type=float, default=600.0, help="projected box area for a view to be used")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    import trimesh
    from scipy.spatial import cKDTree

    scene = parse_scene(load_scene_config(args.scene))
    config = load_train_config(args.config)
    device = torch.device(config.device)
    model = SimSplatModel(config.model, scene, config.device)
    load_model_weights(args.checkpoint, model, config.device)
    model.eval()
    model.set_step(10**9)
    metadata = scene.metadata
    tokens = list(metadata["dynamic_track_tokens"])
    matches = [token for token in tokens if token.startswith(args.track)]
    if len(matches) != 1:
        raise ValueError(f"track prefix {args.track!r} matched {len(matches)} tracks")
    track = tokens.index(matches[0])
    visible = metadata["dynamic_track_visible"].numpy()
    annotation = metadata["dynamic_object_to_world"].numpy().astype(np.float64)[int(np.flatnonzero(visible[:, track])[0]), track]
    track_poses = model.object_to_world[:, track].detach().cpu().numpy().astype(np.float64)  # per-frame annotation
    size = np.asarray(metadata["dynamic_track_box_size_xyz"][track], dtype=np.float64)
    half = size / 2.0
    dataset = InputDataset(scene)
    views = _views(scene, track_poses, half, args.min_pixels)
    print(f"{len(views)} training views see the object", flush=True)

    # 1. Height from the reconstruction: top of the object in rendered depth.
    local_points, cameras_local = [], []
    with torch.no_grad():
        for _, index in views[:30]:
            camera = scene.cameras[index]
            to_local = np.linalg.inv(track_poses[camera.image_id])
            cameras_local.append(to_local[:3, :3] @ camera.c2w.numpy()[:3, 3] + to_local[:3, 3])
            box = _project_box(track_poses[camera.image_id], half, camera)
            depth = model.get_outputs(camera)["depth"][..., 0].cpu().numpy()
            v, u = np.mgrid[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            z = depth[v, u]
            rays = np.stack(((u + 0.5 - camera.K[0, 2].item()) / camera.K[0, 0].item(),
                             (v + 0.5 - camera.K[1, 2].item()) / camera.K[1, 1].item(), np.ones_like(z, dtype=np.float64)), -1)
            world = (rays * z[..., None]).reshape(-1, 3) @ camera.c2w.numpy()[:3, :3].T.astype(np.float64) + camera.c2w.numpy()[:3, 3]
            local_points.append(world @ to_local[:3, :3].T + to_local[:3, 3])
    local_points = np.concatenate(local_points)
    correction, size, fit = _fit_box(local_points, np.stack(cameras_local), size)
    pose = annotation @ correction  # first frame; other frames use track_poses[frame] @ correction
    half = size / 2.0
    print("fit:", json.dumps(fit), flush=True)

    # 2. Template mesh and surface samples (local frame).
    parts = procedural_barrier_meshes(size)
    mesh = trimesh.util.concatenate(parts)
    points, faces = trimesh.sample.sample_surface(mesh, args.surfels, seed=0)
    normals = mesh.face_normals[faces]
    radius = float(np.sqrt(mesh.area / args.surfels) * 0.9)
    local = torch.from_numpy(points).float().to(device)
    local_normals = torch.from_numpy(normals).float().to(device)

    # 3. Project into every view; keep each sample's K best observations.
    count, k = len(points), args.top_k
    best_weight = torch.zeros((count, k), device=device)
    best_colour = torch.zeros((count, k, 3), device=device)
    front = (local_normals[:, 0] * fit["camera_side"] > 0.5)
    rim = local[front & (local[:, 2] > local[:, 2].max() - 0.12)] + torch.tensor((0.0, 0.0, 0.08), device=device)
    face = local[front]

    def project(points_local, frame_pose, camera, depth):
        world = points_local @ frame_pose[:3, :3].T + frame_pose[:3, 3]
        c2w = camera.c2w.to(device).float()
        in_camera = (world - c2w[:3, 3]) @ c2w[:3, :3]
        z = in_camera[:, 2]
        K = camera.K.to(device).float()
        u = K[0, 0] * in_camera[:, 0] / z + K[0, 2]
        v = K[1, 1] * in_camera[:, 1] / z + K[1, 2]
        inside = (z > 0.5) & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
        ui, vi = u.clamp(0, camera.width - 1).long(), v.clamp(0, camera.height - 1).long()
        agrees = inside & ((depth[vi, ui] - z).abs() < 0.06 + 0.004 * z)
        return world, u, v, z, inside, agrees

    shifts = []
    with torch.no_grad():
        for _, index in views:
            camera = scene.cameras[index]
            depth = model.get_outputs(camera)["depth"][..., 0]
            base = torch.from_numpy(track_poses[camera.image_id] @ correction).float().to(device)
            best_score, best_dz = -1.0, 0.0
            for dz in np.arange(-0.2, 0.2001, 0.01):
                shifted = base.clone()
                shifted[:3, 3] += base[:3, 2] * float(dz)
                score = project(face, shifted, camera, depth)[5].float().mean() - project(rim, shifted, camera, depth)[5].float().mean()
                if score > best_score:
                    best_score, best_dz = float(score), float(dz)
            shifts.append(best_dz)
            frame_pose = base.clone()
            frame_pose[:3, 3] += base[:3, 2] * best_dz
            world, u, v, z, inside, agrees = project(local, frame_pose, camera, depth)
            world_normals = local_normals @ frame_pose[:3, :3].T
            c2w = camera.c2w.to(device).float()
            K = camera.K.to(device).float()
            facing = -(world_normals * (world - c2w[:3, 3])).sum(-1) / (world - c2w[:3, 3]).norm(dim=-1)
            seen = agrees & (facing > 0.2)
            if not seen.any():
                continue
            photo = dataset[index].image.to(device)
            if model.appearance is not None:  # back to the model's canonical colour space
                transform = model.appearance[camera.sensor_id].detach()
                photo = (photo - transform[:, 3]) @ torch.linalg.inv(transform[:, :3]).T
            grid = torch.stack((u / (camera.width - 1) * 2 - 1, v / (camera.height - 1) * 2 - 1), -1)[None, :, None]
            colour = torch.nn.functional.grid_sample(photo.permute(2, 0, 1)[None], grid, align_corners=True)[0, :, :, 0].T
            weight = torch.where(seen, facing * K[0, 0] / z, torch.zeros_like(z))
            merged_weight = torch.cat((best_weight, weight[:, None]), 1)
            merged_colour = torch.cat((best_colour, colour[:, None]), 1)
            best_weight, order = merged_weight.topk(k, dim=1)
            best_colour = torch.gather(merged_colour, 1, order[..., None].expand(-1, -1, 3))
    # Robust mean: drop observations far from the weighted mean (occluders, glare).
    valid = best_weight > 0
    mean = (best_colour * best_weight[..., None]).sum(1) / best_weight.sum(1, keepdim=True).clamp_min(1e-9)
    keep = valid & ((best_colour - mean[:, None]).norm(dim=-1) < 0.12)
    keep = torch.where(keep.any(1, keepdim=True), keep, valid)
    weight = best_weight * keep
    colours = ((best_colour * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
    observed = (weight.sum(1) > 0).cpu().numpy()
    fit["view_shift_dz_m"] = {"median": float(np.median(shifts)), "p10": float(np.percentile(shifts, 10)),
                              "p90": float(np.percentile(shifts, 90))}
    correction[2, 3] += float(np.median(shifts))  # the typical residual becomes part of the pose
    pose = annotation @ correction

    # 4. Unseen sides: mirrored sample, then nearest observed sample.
    source = np.full(count, "photo", dtype=object)
    tree = cKDTree(points[observed])
    observed_colours = colours[observed]
    for flip in ((-1, 1, 1), (1, -1, 1), (-1, -1, 1)):
        missing = ~observed & (source == "photo")
        if not missing.any():
            break
        distance, nearest = tree.query(points[missing] * np.array(flip))
        ok = distance < 2.5 * radius
        rows = np.flatnonzero(missing)[ok]
        colours[rows] = observed_colours[nearest[ok]]
        source[rows] = "mirror"
    rest = ~observed & (source == "photo")
    if rest.any():
        _, nearest = tree.query(points[rest])
        colours[rest] = observed_colours[nearest] * 0.85  # unseen (underside): slightly darker
        source[rest] = "nearest"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gaussians = surfels_to_gaussians(points, normals, np.clip(colours, 0, 1), radius, config.model.sh_degree)
    torch.save(gaussians, args.output_dir / "gaussians.pt")
    for i, part in enumerate(parts):
        part.export(args.output_dir / f"part_{i}.obj")
    coverage = {name: float(np.mean(source == name)) for name in ("photo", "mirror", "nearest")}
    info = {"track": matches[0], "category": metadata["dynamic_track_categories"][track], "size_xyz": size.tolist(),
            "fit": fit, "world_pose": pose.tolist(), "local_correction": correction.tolist(), "views": len(views), "surfels": count,
            "surfel_radius_m": radius, "coverage": coverage, "parts": [f"part_{i}.obj" for i in range(len(parts))]}
    (args.output_dir / "asset.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({key: value for key, value in info.items() if key != "world_pose"}, indent=1), flush=True)

    # Preview on the three largest views: photo | reconstruction | asset alone | asset in place of the object.
    frames = model.object_to_world.shape[0]
    asset_index = model.add_actor("lookalike", gaussians, torch.from_numpy(track_poses @ correction).float(),
                                  torch.ones((frames,), dtype=torch.bool))
    rows = []
    with torch.no_grad():
        for _, index in views[:3]:
            camera = scene.cameras[index]
            box = _project_box(track_poses[camera.image_id] @ correction, half, camera)
            pad = 0.6 * max(box[2] - box[0], box[3] - box[1])
            crop = (int(box[0] - pad), int(box[1] - pad / 2), int(box[2] + pad), int(box[3] + pad / 2))
            model.set_hidden_tracks(["lookalike"])
            recon = _to_uint8(model.get_outputs(camera)["rgb"])
            model.set_hidden_tracks([matches[0]])
            swapped = _to_uint8(model.get_outputs(camera)["rgb"])
            opacities = model.gauss_params["opacities"]
            backup = opacities.data.clone()
            opacities.data[model.point_track_indices != asset_index] = -30.0
            alone = _to_uint8(model.get_outputs(camera)["rgb"])
            opacities.data.copy_(backup)
            tiles = [Image.fromarray(image).crop(crop).resize((360, 220)) for image in
                     (_to_uint8(dataset[index].image), recon, alone, swapped)]
            row = Image.new("RGB", (4 * 360, 220))
            for column, tile in enumerate(tiles):
                row.paste(tile, (column * 360, 0))
            rows.append(row)
    sheet = Image.new("RGB", (4 * 360, 220 * len(rows)))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * 220))
    sheet.save(args.output_dir / "preview.jpg", quality=90)


if __name__ == "__main__":
    main()
