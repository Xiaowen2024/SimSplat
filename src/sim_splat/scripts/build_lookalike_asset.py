"""Build a sim asset that looks like a reconstructed object: template mesh + photo texture.

Static street furniture (barriers, cones) never moves during capture, so the
reconstruction bakes it into the static field and its actor Gaussians stay
nearly transparent (scene-0061 barrier: 12 of 6011 actor Gaussians have
opacity > 0.5).  To make such an object interactive we rebuild it as a
conventional asset:

1. **Pose and size.** nuScenes ego poses have z = 0, so the shared world
   is not consistent in height and the *per-frame* annotation is the pose
   that is right relative to that frame's camera.  The asset is therefore
   ``annotation[frame] @ correction`` plus a per-frame height shift
   (``asset_poses``).  Both come from the photos: SAM, prompted with the
   projected annotation box, segments the object in every view; its top
   and bottom edges (middle half of the columns) are intersected with the
   camera-facing face plane.  The median over views gives the height and
   centre; each view's offset from it is that frame's shift (up to ~0.3 m
   between frames).  Rendered depth is a poor guide for most objects: away
   from the ego path the reconstruction is a blur along viewing rays (an
   earlier depth-only fit made this barrier 25% too short).  It is the
   fallback when SAM is off.  Objects whose views disagree on height
   (occluded, or merged with a vehicle in front) are skipped: they stay in
   the reconstruction.  Length is the annotated one, since barriers in a
   row abut and no face separates them.
2. **Shape.** A category template (``procedural_barrier_meshes``) at the
   fitted size; its convex parts double as physics colliders.
3. **Texture.** Surface samples are projected into every training photo
   that sees them, at that view's height shift.  A sample is used if it
   lies inside the object's photo mask, faces the camera, has nothing
   clearly in front of it in rendered depth, and agrees with the static
   render (transient occluders like pedestrians are only in the photo).
   Photo colours are mapped back through the per-sensor appearance
   transform, because the renderer applies it again.  Each sample keeps
   its K best observations (frontal, close), and their robust mean is its
   colour.  Sides that were never photographed take the colour of the
   mirrored point (the template is symmetric front/back and left/right),
   then of the nearest observed sample.

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


def asset_poses(track_poses: np.ndarray, info: dict) -> np.ndarray:
    """Per-frame poses [frames, 4, 4] of a look-alike asset: the track's per-frame
    annotation, the fitted local correction, and the per-frame height correction."""

    poses = np.asarray(track_poses, dtype=np.float64) @ np.asarray(info["local_correction"], dtype=np.float64)
    shift = np.asarray(info.get("frame_shift_z") or np.zeros(len(poses)), dtype=np.float64)
    poses[:, :3, 3] += poses[:, :3, 2] * shift[:, None]
    return poses


def _views(scene, poses: np.ndarray, half: np.ndarray, min_pixels: float) -> list[tuple[float, int]]:
    found = []
    for index in scene.train_indices:
        box = _project_box(poses[scene.cameras[index].image_id], half, scene.cameras[index])
        if box and (area := (box[2] - box[0]) * (box[3] - box[1])) >= min_pixels:
            found.append((area, index))
    return sorted(found, reverse=True)


def _fit_box(points: np.ndarray, cameras_local: np.ndarray, size: np.ndarray):
    """Refine a box (fixed yaw) from back-projected depth points in its per-frame local frame.

    Barriers stand end to end in rows, and the annotation box carries margin
    in depth and height, so only the object's own footprint is used: the
    middle 80% of its length, and depth within the annotated box.  Length
    stays the annotated one (rows abut: a visible face cannot separate the
    units).  Bottom is the support surface just in front of the face; top is
    the highest surface inside the footprint; the face is the densest plane
    between them.
    """

    depth_a, length, height_a = size
    side = np.sign(np.median(cameras_local[:, 0]))
    own = points[(np.abs(points[:, 1]) < 0.4 * length) & (np.abs(points[:, 0]) < 0.6 * depth_a)
                 & (np.abs(points[:, 2]) < 1.2 * height_a)]
    # Support surface: the ground just outside the footprint on the camera side.
    ahead = points[(side * points[:, 0] > 0.5 * depth_a) & (side * points[:, 0] < 0.5 * depth_a + 0.6)
                   & (np.abs(points[:, 1]) < 0.4 * length) & (np.abs(points[:, 2]) < 1.2 * height_a)]
    levels, level_edges = np.histogram(ahead[:, 2], bins=np.arange(-1.2 * height_a, 1.2 * height_a + 0.02, 0.02))
    bottom = float(level_edges[np.argmax(levels)] + 0.01)
    body = own[(own[:, 2] > bottom + 0.1) & (own[:, 2] < bottom + 1.6)]
    top = float(np.percentile(body[:, 2], 98))
    middle = body[(body[:, 2] > bottom + 0.15) & (body[:, 2] < top - 0.1)]
    histogram, edges = np.histogram(middle[:, 0], bins=np.arange(-0.6 * depth_a, 0.6 * depth_a + 0.02, 0.02))
    face_x = float(edges[np.argmax(histogram)] + 0.01)
    depth = 0.75 * depth_a  # the back face is never observed: keep the category's proportion
    fitted = np.array((depth, 0.98 * length, top - bottom))
    # Face at mid-height of the tapered template sits at 0.36 * depth from the centre.
    centre = np.array((face_x - side * 0.36 * depth, 0.0, (top + bottom) / 2))
    correction = np.eye(4)
    correction[:3, 3] = centre
    fit = {"camera_side": float(side), "face_x": face_x, "top": top, "bottom": bottom, "face_points": int(len(middle)),
           "ground_points": int(len(ahead)), "centre_offset_local": centre.round(3).tolist(),
           "annotation_size": size.round(3).tolist(), "fitted_size": fitted.round(3).tolist()}
    return correction, fitted, fit


class PhotoSegmenter:
    """SAM prompted with the object's projected annotation box: its mask in a captured photo."""

    def __init__(self, model: str = "facebook/sam-vit-base", device: str = "cuda") -> None:
        from transformers import SamModel, SamProcessor

        self.model = SamModel.from_pretrained(model).to(device).eval()
        self.processor = SamProcessor.from_pretrained(model)
        self.device = device

    @torch.no_grad()
    def __call__(self, photo: np.ndarray, box) -> np.ndarray:
        inputs = self.processor(Image.fromarray(photo), input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                                   inputs["reshaped_input_sizes"].cpu())[0][0]
        return masks[int(outputs.iou_scores[0, 0].argmax())].numpy()


def _vertical_extent(mask: np.ndarray, box, camera, frame_pose: np.ndarray, depth_a: float):
    """Local-frame z of the object's top and bottom edge in one view, from its photo mask.

    Only the middle half of the projected box's columns is used (neighbours
    in a row and the object's own tapered ends do not count).  An edge that
    touches the search window (box enlarged by 30%) is not an edge but a
    merge with something above or below, and the view is skipped.  Image
    rows are turned into heights on the camera-facing face plane.
    """

    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    top_limit, bottom_limit = int(max(0, y0 - 0.3 * h)), int(min(camera.height, y1 + 0.3 * h))
    columns = np.arange(int(max(0, x0 + 0.25 * w)), int(min(camera.width, x1 - 0.25 * w)))
    if len(columns) < 3 or bottom_limit - top_limit < 6:
        return None
    window = mask[top_limit:bottom_limit, columns]
    filled = window.any(0)
    if filled.mean() < 0.6:
        return None
    rows = np.arange(top_limit, bottom_limit)
    tops = np.array([rows[np.argmax(col)] for col in window.T[filled]])
    bottoms = np.array([rows[len(col) - 1 - np.argmax(col[::-1])] for col in window.T[filled]])
    v_top, v_bottom = float(np.median(tops)), float(np.median(bottoms)) + 1.0
    if v_top <= top_limit + 1 or v_bottom >= bottom_limit - 1:
        return None
    to_local = np.linalg.inv(frame_pose)
    c2w = camera.c2w.numpy().astype(np.float64)
    K = camera.K.numpy().astype(np.float64)
    centre = to_local[:3, :3] @ c2w[:3, 3] + to_local[:3, 3]
    face_x = np.sign(centre[0]) * 0.25 * depth_a
    heights = []
    for v in (v_top, v_bottom):
        ray = to_local[:3, :3] @ (c2w[:3, :3] @ np.array((((x0 + x1) / 2 - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0)))
        if abs(ray[0]) < 1e-6:
            return None
        heights.append(centre[2] + ray[2] * (face_x - centre[0]) / ray[0])
    return heights[0], heights[1], float(np.sign(centre[0]))


def drop_actor(model, index: int) -> None:
    """Remove the most recently added actor (``add_actor``'s inverse)."""

    if index != len(model.track_tokens) - 1:
        raise ValueError("only the last actor can be dropped")
    keep = model.point_track_indices != index
    model.gauss_params = torch.nn.ParameterDict({name: torch.nn.Parameter(p.detach()[keep]) for name, p in model.gauss_params.items()})
    model.point_track_indices = model.point_track_indices[keep]
    model.object_to_world = model.object_to_world[:, :index]
    model.track_visible = model.track_visible[:, :index]
    model.track_tokens.pop()


def build_asset(model, scene, dataset, config, token: str, output_dir: Path, *, surfels: int, min_pixels: float,
                top_k: int, min_views: int = 1, fit_only: bool = False,
                occluder_threshold: float = 0.35, segmenter: PhotoSegmenter | None = None,
                max_height_spread: float = 0.15) -> dict:
    """Fit, texture and save one look-alike asset; returns its ``asset.json`` contents."""

    import trimesh
    from scipy.spatial import cKDTree

    device = torch.device(config.device)
    metadata = scene.metadata
    tokens = list(metadata["dynamic_track_tokens"])
    track = tokens.index(token)
    visible = metadata["dynamic_track_visible"].numpy()
    annotation = metadata["dynamic_object_to_world"].numpy().astype(np.float64)[int(np.flatnonzero(visible[:, track])[0]), track]
    track_poses = model.object_to_world[:, track].detach().cpu().numpy().astype(np.float64)  # per-frame annotation
    size = np.asarray(metadata["dynamic_track_box_size_xyz"][track], dtype=np.float64)
    half = size / 2.0
    views = _views(scene, track_poses, half, min_pixels)
    print(f"{token[:8]}: {len(views)} training views see the object", flush=True)
    if len(views) < min_views:
        raise ValueError(f"only {len(views)} views of at least {min_pixels:.0f} px")
    # 1. Pose and size.  With a segmenter: the object's top and bottom edges in
    # the photos (robust where the reconstruction is a blur of depth, which is
    # most objects that are not close to the road).  Otherwise, or with too
    # few clean views, from rendered depth.
    masks: dict[int, np.ndarray] = {}
    view_shift: dict[int, float] = {}
    correction = fit = None
    if segmenter is not None:
        extents = {}
        for _, index in views:
            camera = scene.cameras[index]
            box = _project_box(track_poses[camera.image_id], half, camera)
            photo = (dataset[index].image.numpy() * 255).round().astype(np.uint8)
            masks[index] = segmenter(photo, box)
            extent = _vertical_extent(masks[index], box, camera, track_poses[camera.image_id], size[0])
            if extent is not None:
                extents[index] = extent
        if len(extents) >= 3:
            tops, bottoms, sides = (np.array(values) for values in zip(*extents.values()))
            top, bottom = float(np.median(tops)), float(np.median(bottoms))
            height = top - bottom
            heights_all = tops - bottoms
            spread = float(np.percentile(heights_all, 75) - np.percentile(heights_all, 25))
            if spread > max_height_spread:
                # Masks disagree between views: the object is occluded or merged with
                # something else (e.g. a vehicle parked in front).  A hard case: leave it
                # in the reconstruction rather than build a wrong asset.
                raise ValueError(f"photo masks disagree on height (IQR {spread:.2f} m): occluded or merged")
            if 0.5 * size[2] <= height <= 1.3 * size[2]:
                for index, (t, b, _) in extents.items():
                    view_shift[index] = (t + b) / 2 - (top + bottom) / 2
                heights = tops - bottoms
                depth = 0.75 * size[0]
                side = float(np.sign(np.median(sides)))
                correction = np.eye(4)
                correction[2, 3] = (top + bottom) / 2
                fit = {"method": "photo_masks", "camera_side": side, "top": top, "bottom": bottom, "mask_views": len(extents),
                       # Spread between views (interquartile range).  Top and bottom move
                       # together when the per-frame pose is off vertically (shift); the
                       # height only varies when the masks themselves disagree.
                       "height_spread_m": float(np.percentile(heights, 75) - np.percentile(heights, 25)),
                       "shift_spread_m": float(np.percentile(list(view_shift.values()), 75) - np.percentile(list(view_shift.values()), 25)),
                       "centre_offset_local": correction[:3, 3].round(3).tolist(), "annotation_size": size.round(3).tolist(),
                       "fitted_size": [round(depth, 3), round(0.98 * size[1], 3), round(height, 3)]}
                size = np.array((depth, 0.98 * size[1], height))
    if correction is None:
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
        correction, size, fit = _fit_box(np.concatenate(local_points), np.stack(cameras_local), size)
        fit["method"] = "rendered_depth"
    # Per-frame vertical correction: nuScenes per-frame poses drift in height
    # (ego z = 0); the photos measure by how much.  Median over a frame's
    # cameras, interpolated over frames without a measurement.
    frame_shift = np.zeros(track_poses.shape[0])
    if view_shift:
        by_frame: dict[int, list[float]] = {}
        for index, shift in view_shift.items():
            by_frame.setdefault(scene.cameras[index].image_id, []).append(shift)
        known = np.array(sorted(by_frame))
        frame_shift = np.interp(np.arange(track_poses.shape[0]), known, [float(np.median(by_frame[f])) for f in known])
    pose = annotation @ correction  # first frame; other frames: asset_poses()
    half = size / 2.0
    print("fit:", json.dumps(fit), flush=True)
    if fit_only:
        return {"track": token, "category": metadata["dynamic_track_categories"][track], "size_xyz": size.tolist(), "fit": fit,
                "views": len(views), "coverage": {"photo": 1.0}}

    # 2. Template mesh and surface samples (local frame).
    parts = procedural_barrier_meshes(size)
    mesh = trimesh.util.concatenate(parts)
    points, faces = trimesh.sample.sample_surface(mesh, surfels, seed=0)
    normals = mesh.face_normals[faces]
    radius = float(np.sqrt(mesh.area / surfels) * 0.9)
    local = torch.from_numpy(points).float().to(device)
    local_normals = torch.from_numpy(normals).float().to(device)

    # 3. Project into every view; keep each sample's K best observations.
    count, k = len(points), top_k
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
            rendered = model.get_outputs(camera)
            depth = rendered["depth"][..., 0]
            base = torch.from_numpy(track_poses[camera.image_id] @ correction).float().to(device)
            if index in view_shift:  # measured in this photo
                best_dz = float(np.clip(view_shift[index], -0.3, 0.3))
            else:  # register against rendered depth
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
            if index in masks:
                # Inside the object's photo mask: no background, no occluders.  Rendered
                # depth around most objects is a blur along the viewing rays, so it only
                # vetoes a sample when something is clearly in front (> 0.5 m).
                mask = torch.from_numpy(masks[index]).to(device)
                ui, vi = u.clamp(0, camera.width - 1).long(), v.clamp(0, camera.height - 1).long()
                seen = inside & mask[vi, ui] & (depth[vi, ui] > z - 0.5) & (facing > 0.2)
            else:
                seen = agrees & (facing > 0.2)
            if not seen.any():
                continue
            photo = dataset[index].image.to(device)
            grid = torch.stack((u / (camera.width - 1) * 2 - 1, v / (camera.height - 1) * 2 - 1), -1)[None, :, None]
            # Transient occluders (pedestrians, passing cars) are in the photo but
            # not in the static reconstruction: drop samples where the two disagree.
            sample = lambda image: torch.nn.functional.grid_sample(image.permute(2, 0, 1)[None], grid, align_corners=True)[0, :, :, 0].T
            seen = seen & ((sample(photo) - sample(rendered["rgb"].clamp(0, 1))).norm(dim=-1) < occluder_threshold)
            if not seen.any():
                continue
            if model.appearance is not None:  # back to the model's canonical colour space
                transform = model.appearance[camera.sensor_id].detach()
                photo = (photo - transform[:, 3]) @ torch.linalg.inv(transform[:, :3]).T
            colour = sample(photo)
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
    if fit["method"] == "rendered_depth":
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

    output_dir.mkdir(parents=True, exist_ok=True)
    gaussians = surfels_to_gaussians(points, normals, np.clip(colours, 0, 1), radius, config.model.sh_degree)
    torch.save(gaussians, output_dir / "gaussians.pt")
    for i, part in enumerate(parts):
        part.export(output_dir / f"part_{i}.obj")
    coverage = {name: float(np.mean(source == name)) for name in ("photo", "mirror", "nearest")}
    info = {"track": token, "category": metadata["dynamic_track_categories"][track], "size_xyz": size.tolist(),
            "fit": fit, "world_pose": pose.tolist(), "local_correction": correction.tolist(), "frame_shift_z": frame_shift.round(4).tolist(),
            "views": len(views), "surfels": count,
            "surfel_radius_m": radius, "coverage": coverage, "parts": [f"part_{i}.obj" for i in range(len(parts))]}
    (output_dir / "asset.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({key: value for key, value in info.items() if key != "world_pose"}, indent=1), flush=True)

    # Preview on the three largest views: photo | reconstruction | asset alone | asset in place of the object.
    frames = model.object_to_world.shape[0]
    info_poses = asset_poses(track_poses, info)
    asset_index = model.add_actor("lookalike", gaussians, torch.from_numpy(info_poses).float(),
                                  torch.ones((frames,), dtype=torch.bool))
    rows = []
    try:
        with torch.no_grad():
            for _, index in views[:3]:
                camera = scene.cameras[index]
                box = _project_box(info_poses[camera.image_id], half, camera)
                if box is None:
                    continue
                pad = 0.6 * max(box[2] - box[0], box[3] - box[1])
                crop = (int(box[0] - pad), int(box[1] - pad / 2), int(box[2] + pad), int(box[3] + pad / 2))
                model.set_hidden_tracks(["lookalike"])
                recon = _to_uint8(model.get_outputs(camera)["rgb"])
                model.set_hidden_tracks([token])
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
        sheet.save(output_dir / "preview.jpg", quality=90)
    finally:
        model.set_hidden_tracks([])
        drop_actor(model, asset_index)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="unedited reconstruction (depth for visibility)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--track", help="one track (unique token prefix); assets go straight into --output-dir")
    target.add_argument("--category", help="every track of this category; one sub-directory per track")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surfels", type=int, default=120000)
    parser.add_argument("--min-pixels", type=float, default=600.0, help="projected box area for a view to be used")
    parser.add_argument("--min-views", type=int, default=8, help="--category: skip objects seen in fewer views")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--fit-only", action="store_true", help="--category: fit boxes only, write fits.json")
    parser.add_argument("--no-segmenter", action="store_true", help="fit from rendered depth only (no SAM photo masks)")
    args = parser.parse_args()

    scene = parse_scene(load_scene_config(args.scene))
    config = load_train_config(args.config)
    model = SimSplatModel(config.model, scene, config.device)
    load_model_weights(args.checkpoint, model, config.device)
    model.eval()
    model.set_step(10**9)
    dataset = InputDataset(scene)
    tokens = list(scene.metadata["dynamic_track_tokens"])
    options = {"surfels": args.surfels, "min_pixels": args.min_pixels, "top_k": args.top_k,
               "segmenter": None if args.no_segmenter else PhotoSegmenter(device=config.device)}
    if args.track:
        matches = [token for token in tokens if token.startswith(args.track)]
        if len(matches) != 1:
            raise ValueError(f"track prefix {args.track!r} matched {len(matches)} tracks")
        build_asset(model, scene, dataset, config, matches[0], args.output_dir, **options)
        return
    categories = scene.metadata["dynamic_track_categories"]
    summary = []
    for token, category in zip(tokens, categories):
        if category != args.category:
            continue
        try:
            info = build_asset(model, scene, dataset, config, token, args.output_dir / token[:8], min_views=args.min_views,
                               fit_only=args.fit_only, **options)
            summary.append({"track": token, "status": "built", "views": info["views"], "fitted_size": info["size_xyz"],
                            "annotation_size": info["fit"]["annotation_size"], "coverage": info["coverage"], "fit": info["fit"]})
        except Exception as error:  # noqa: BLE001 - record and continue with the next object
            summary.append({"track": token, "status": "skipped", "reason": str(error)})
            print(f"{token[:8]}: skipped ({error})", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ("fits.json" if args.fit_only else "summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"built {sum(item['status'] == 'built' for item in summary)} of {len(summary)} {args.category} assets", flush=True)


if __name__ == "__main__":
    main()
