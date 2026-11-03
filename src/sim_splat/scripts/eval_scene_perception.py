"""Scene-level perception check: every look-alike asset at once, real vs reconstruction vs swap.

``eval_perception.py`` checks one object.  This script checks a whole
scene in which many objects (e.g. every barrier) were rebuilt as sim
assets (``build_lookalike_asset.py --category``) and removed from the
background (``inpaint_removed.py --asset-root``).  On sampled camera views
it runs the same detector on three images:

* **real**: the captured photo;
* **reconstruction**: the unedited 3DGS render;
* **swap**: inpainted background + all assets, i.e. what a simulator built
  on this scene would show;
* optionally more unedited renderers (``--renderer``, e.g. vanilla 3DGS or
  ablations), scored like the reconstruction: does a better-fidelity
  renderer also agree better with what the detector does on real photos?

Ground truth comes from the assets.  Each asset is rendered alone, and the
pixels where it is the nearest surface in the swap render give its
*visible* (modal) box.  An object is scored when it has at least
``--min-pixels`` visible pixels and at least half of its silhouette is
visible; otherwise it is an *ignore* region, and detections on it are not
false positives.  Annotated objects of the same category that have no
asset (too few views to build one) are also ignore regions.  The same
boxes are used for all three images: the assets were fitted to the photos,
so they mark where the real objects are.

Per (view, object) and per image it records:
* detection: IoU of the best one-to-one match (greedy, by IoU);
* localization: a matched box is turned into a 3D centre from depth
  (median depth in the inner box, moved by half the object's depth along
  the ray).  Depth is rendered: the reconstruction's for the photo and
  for the reconstruction (a LiDAR stand-in), the swap's for the swap;
* appearance: crop LPIPS of reconstruction / swap against the photo.

The summary has recall, precision, IoU and localization error per image
type, broken down by distance and visibility.  It also reports *paired*
outcomes per (view, object): a "sim gap" is a detection on the photo that
is missing on the swap, the case that would make a perception test in
simulation disagree with reality.  Per-object tables find the objects
behind the gap.  "Flip rate" is temporal stability: the objects are static,
so a detection toggling between consecutive frames of a camera is flicker.
Render speed is reported per renderer.  "Merged" counts objects whose best-covering detection
also covers a neighbour: a row of barriers boxed as one.  Writes ``scene_perception.json``, ``objects.csv`` and
``failures/`` (photo | reconstruction | swap crops, green = ground truth,
red = detections).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sim_splat.scripts.eval_perception import TiledDetector, _iou

SOURCES = ("real", "reconstruction", "swap")
DISTANCE_BINS = ((0, 10), (10, 20), (20, 35), (35, 1e9))


def _match(detections: list[dict], boxes: list[list[float]]) -> list[tuple[int | None, float]]:
    """Greedy one-to-one matching by IoU: (detection index or None, IoU) per box."""

    pairs = sorted(((_iou(d["xyxy"], b), i, j) for i, d in enumerate(detections) for j, b in enumerate(boxes)), reverse=True)
    used_d, out = set(), [(None, 0.0)] * len(boxes)
    for iou, i, j in pairs:
        if iou <= 0 or i in used_d or out[j][0] is not None:
            continue
        used_d.add(i)
        out[j] = (i, iou)
    return out


def _ioa(inner, outer) -> float:
    """Fraction of ``inner`` inside ``outer``."""

    x0, y0, x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / area


def _centre_from_depth(box, depth: np.ndarray, c2w: np.ndarray, K: np.ndarray, half_depth: float) -> np.ndarray | None:
    w, h = box[2] - box[0], box[3] - box[1]
    inner = depth[int(box[1] + 0.25 * h) : int(box[3] - 0.25 * h) + 1, int(box[0] + 0.25 * w) : int(box[2] - 0.25 * w) + 1]
    if not inner.size:
        return None
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    direction = c2w[:3, :3] @ np.array(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0))
    surface = c2w[:3, 3] + direction * float(np.median(inner))
    ray = direction / np.linalg.norm(direction)
    return surface + ray * half_depth  # visible surface -> centre


def _rate(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _summary(rows: list[dict], sources=SOURCES) -> dict:
    out = {"objects": len(rows)}
    for source in sources:
        hits = [r[f"{source}_iou"] >= 0.5 for r in rows]
        out[f"{source}_recall@0.5"] = _rate(hits)
        out[f"{source}_recall@0.3"] = _rate(r[f"{source}_iou"] >= 0.3 for r in rows)
        # Merged: the best detection on this object also covers a neighbour (row boxed as one).
        out[f"{source}_merged"] = _rate(r.get(f"{source}_merged", False) for r in rows)
        errors = [r[f"{source}_error_m"] for r in rows if r.get(f"{source}_error_m") is not None]
        out[f"{source}_median_error_m"] = float(np.median(errors)) if errors else None
        ious = [r[f"{source}_iou"] for r in rows if r[f"{source}_iou"] >= 0.5]
        out[f"{source}_mean_iou"] = _rate(ious)
    for source in sources[1:]:
        out[f"{source}_crop_psnr"] = _rate(r[f"{source}_psnr"] for r in rows if f"{source}_psnr" in r)
        out[f"{source}_crop_lpips"] = _rate(r[f"{source}_lpips"] for r in rows if f"{source}_lpips" in r)
        both = [(r["real_iou"] >= 0.5, r[f"{source}_iou"] >= 0.5) for r in rows]
        out[f"{source}_agreement_with_real"] = _rate(a == b for a, b in both)
        out[f"{source}_sim_gap"] = _rate(a and not b for a, b in both)  # real detected, this one missed
        out[f"{source}_sim_extra"] = _rate(b and not a for a, b in both)
    # Temporal stability: the objects are static, so a detection that appears and
    # disappears between consecutive frames of one camera is flicker (detector
    # noise on real photos; detector noise plus rendering instability on renders).
    series: dict[tuple, list[dict]] = {}
    for r in rows:
        series.setdefault((r["sensor"], r["track"]), []).append(r)
    for source in sources:
        flips = []
        for items in series.values():
            items = sorted(items, key=lambda r: r["frame"])
            for a, b in zip(items, items[1:]):
                if b["frame"] - a["frame"] <= 2:
                    flips.append((a[f"{source}_iou"] >= 0.3) != (b[f"{source}_iou"] >= 0.3))
        out[f"{source}_flip_rate"] = _rate(flips)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="unedited reconstruction")
    parser.add_argument("--background-checkpoint", type=Path, required=True, help="inpainted background (inpaint_removed --asset-root)")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="a plastic road barrier.")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--min-pixels", type=float, default=400.0)
    parser.add_argument("--min-visible-fraction", type=float, default=0.5)
    parser.add_argument("--occlusion-margin", type=float, default=0.3, help="m: nearer surfaces than this occlude")
    parser.add_argument("--renderer", action="append", default=[], metavar="NAME=SCENE,CONFIG,CHECKPOINT",
                        help="another unedited reconstruction to score like 'reconstruction' (e.g. vanilla 3DGS, ablations); "
                             "cameras are matched by image name")
    args = parser.parse_args()

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    from sim_splat.config import load_scene_config, load_train_config
    from sim_splat.data.dataparsers import parse_scene
    from sim_splat.data.input_dataset import InputDataset
    from sim_splat.engine.checkpoints import load_model_weights
    from sim_splat.models.sim_splat_model import SimSplatModel
    from sim_splat.scripts.build_lookalike_asset import asset_poses
    from sim_splat.scripts.render_edit import _to_uint8
    from sim_splat.scripts.render_physics import _project_box

    scene = parse_scene(load_scene_config(args.scene))
    config = load_train_config(args.config)
    device = torch.device(config.device)
    models = {}
    for name, checkpoint in (("reconstruction", args.checkpoint), ("swap", args.background_checkpoint)):
        model = SimSplatModel(config.model, scene, config.device)
        load_model_weights(checkpoint, model, config.device)
        model.eval()
        model.set_step(10**9)
        models[name] = model
    recon, swap = models["reconstruction"], models["swap"]
    # Other renderers of the same captures (their own scene files; cameras matched by image name).
    renderers = {}
    for spec in args.renderer:
        name, paths = spec.split("=", 1)
        scene_path, config_path, checkpoint = (Path(item) for item in paths.split(","))
        other_scene = parse_scene(load_scene_config(scene_path))
        other_config = load_train_config(config_path)
        model = SimSplatModel(other_config.model, other_scene, other_config.device)
        load_model_weights(checkpoint, model, other_config.device)
        model.eval()
        model.set_step(10**9)
        by_name = {Path(path).name: camera for path, camera in zip(other_scene.image_paths, other_scene.cameras)}
        renderers[name] = (model, by_name)
    sources = ("real", "reconstruction", *renderers, "swap")
    render_seconds = {source: [] for source in sources[1:]}

    def timed(source, model, camera):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = model.get_outputs(camera)
        torch.cuda.synchronize(device)
        render_seconds[source].append(time.perf_counter() - start)
        return out

    infos = [json.loads(path.read_text()) for path in sorted(args.asset_root.glob("*/asset.json"))]
    category = infos[0]["category"]
    tokens = recon.track_tokens
    frames = recon.object_to_world.shape[0]
    assets = []  # dicts: token, track, actor index, poses [frames, 4, 4], half size
    for info in infos:
        track = tokens.index(info["track"])
        poses = asset_poses(recon.object_to_world[:, track].detach().cpu().numpy(), info)
        gaussians = torch.load(args.asset_root / info["track"][:8] / "gaussians.pt")
        actor = swap.add_actor(f"lookalike:{info['track'][:8]}", gaussians, torch.from_numpy(poses).float(), torch.ones((frames,), dtype=torch.bool))
        assets.append({"token": info["track"], "track": track, "actor": actor, "poses": poses, "half": np.asarray(info["size_xyz"]) / 2.0})
    swap.set_hidden_tracks([asset["token"] for asset in assets])
    built = {asset["token"] for asset in assets}
    categories = scene.metadata["dynamic_track_categories"]
    unbuilt = [index for index, token in enumerate(tokens) if categories[index] == category and token not in built]
    sizes = scene.metadata["dynamic_track_box_size_xyz"]
    track_poses = recon.object_to_world.detach().cpu().numpy().astype(np.float64)

    def render_alone(rows: torch.Tensor, camera):
        opacities = swap.gauss_params["opacities"]
        backup = opacities.data.clone()
        opacities.data[~rows] = -30.0
        hidden = swap.hidden_track_ids
        swap.hidden_track_ids = set()
        try:
            out = swap.get_outputs(camera)
        finally:
            opacities.data.copy_(backup)
            swap.hidden_track_ids = hidden
        return out["alpha"][..., 0], out["depth"][..., 0]

    # Views with at least one scorable object, spread over the sequence.
    candidates = []
    for index in sorted(scene.train_indices + scene.eval_indices):
        camera = scene.cameras[index]
        if any(_project_box(asset["poses"][camera.image_id], asset["half"], camera) for asset in assets):
            candidates.append(index)
    step = max(1, len(candidates) // args.max_views)
    views = candidates[::step][: args.max_views]
    print(f"{len(assets)} assets, {len(unbuilt)} unbuilt {category}; {len(views)} of {len(candidates)} candidate views", flush=True)

    detector = TiledDetector(args.dino_model, args.prompt, args.threshold)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    dataset = InputDataset(scene)
    eval_set = set(scene.eval_indices)
    rows: list[dict] = []
    false_positives = {source: 0 for source in sources}
    detections_total = {source: 0 for source in sources}
    failures = []
    with torch.no_grad():
        for index in views:
            camera = scene.cameras[index]
            c2w, K = camera.c2w.numpy().astype(np.float64), camera.K.numpy().astype(np.float64)
            out_recon, out_swap = timed("reconstruction", recon, camera), timed("swap", swap, camera)
            scene_depth = out_swap["depth"][..., 0]
            gts, ignores = [], []
            for asset in assets:
                pose = asset["poses"][camera.image_id]
                if not _project_box(pose, asset["half"], camera):
                    continue
                alpha, depth = render_alone(swap.point_track_indices == asset["actor"], camera)
                silhouette = alpha > 0.5
                if int(silhouette.sum()) < 20:
                    continue
                # Rendered depth is alpha-blended: a faint floater in front lowers it
                # without hiding the object, so only a clear margin counts as occlusion.
                visible = silhouette & (depth <= scene_depth + args.occlusion_margin)
                ys, xs = torch.nonzero(visible, as_tuple=True)
                fraction = float(visible.sum()) / float(silhouette.sum())
                if len(xs) < args.min_pixels or fraction < args.min_visible_fraction:
                    sy, sx = torch.nonzero(silhouette, as_tuple=True)
                    ignores.append([float(sx.min()), float(sy.min()), float(sx.max() + 1), float(sy.max() + 1)])
                    continue
                gts.append({"asset": asset, "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                            "pixels": int(len(xs)), "visible_fraction": fraction,
                            "distance_m": float(np.linalg.norm(pose[:3, 3] - c2w[:3, 3])), "centre": pose[:3, 3]})
            for track in unbuilt:
                if bool(recon.track_visible[camera.image_id, track]):
                    box = _project_box(track_poses[camera.image_id, track], np.asarray(sizes[track]) / 2.0, camera)
                    if box:
                        ignores.append(list(box))
            if not gts:
                continue
            photo = dataset[index].image.to(device)
            images = {"real": photo, "reconstruction": out_recon["rgb"].clamp(0, 1)}
            depths = {"real": out_recon["depth"][..., 0].cpu().numpy(), "reconstruction": out_recon["depth"][..., 0].cpu().numpy()}
            image_name = Path(scene.image_paths[index]).name
            for name, (model, by_name) in renderers.items():
                out = timed(name, model, by_name[image_name])
                images[name], depths[name] = out["rgb"].clamp(0, 1), out["depth"][..., 0].cpu().numpy()
            images["swap"], depths["swap"] = out_swap["rgb"].clamp(0, 1), scene_depth.cpu().numpy()
            view_rows = [{"view": index, "frame": camera.image_id, "split": "eval" if index in eval_set else "train", "sensor": camera.sensor_id,
                          "track": g["asset"]["token"][:8], "pixels": g["pixels"], "visible_fraction": round(g["visible_fraction"], 3),
                          "distance_m": round(g["distance_m"], 2)} for g in gts]
            found = {}
            for source, tensor in images.items():
                image = Image.fromarray(_to_uint8(tensor))
                detections = detector(image)
                found[source] = (image, detections)
                matches = _match(detections, [g["box"] for g in gts])
                matched = {i for i, iou in matches if i is not None and iou >= 0.5}
                detections_total[source] += len(detections)
                for i, det in enumerate(detections):
                    if i in matched:
                        continue
                    if any(_iou(det["xyxy"], g["box"]) >= 0.3 for g in gts) or any(_ioa(det["xyxy"], box) >= 0.5 for box in ignores):
                        continue  # partial match or an ignore region: neither true nor false
                    false_positives[source] += 1
                for row, g, (i, iou) in zip(view_rows, gts, matches):
                    row[f"{source}_iou"] = round(iou, 3)
                    covering = max(detections, key=lambda d: _ioa(g["box"], d["xyxy"]), default=None)
                    neighbours = [other["box"] for other in gts if other is not g] + ignores
                    row[f"{source}_merged"] = bool(covering is not None and _ioa(g["box"], covering["xyxy"]) > 0.7
                                                   and any(_ioa(box, covering["xyxy"]) > 0.5 for box in neighbours))
                    row[f"{source}_error_m"] = None
                    if i is not None and iou >= 0.5:
                        centre = _centre_from_depth(detections[i]["xyxy"], depths[source], c2w, K, float(g["asset"]["half"][0]))
                        if centre is not None:
                            row[f"{source}_error_m"] = round(float(np.linalg.norm(centre[:2] - g["centre"][:2])), 3)
            for row, g in zip(view_rows, gts):
                box = g["box"]
                pad_w, pad_h = 0.2 * (box[2] - box[0]), 0.2 * (box[3] - box[1])
                x0, y0 = int(max(0, box[0] - pad_w)), int(max(0, box[1] - pad_h))
                x1, y1 = int(min(camera.width, box[2] + pad_w)), int(min(camera.height, box[3] + pad_h))
                resize = lambda t: torch.nn.functional.interpolate(t.permute(2, 0, 1)[None], size=(128, 128), mode="bilinear")
                for source in sources[1:]:
                    crop, reference = images[source][y0:y1, x0:x1], photo[y0:y1, x0:x1]
                    row[f"{source}_lpips"] = round(float(lpips(resize(crop), resize(reference))), 4)
                    row[f"{source}_psnr"] = round(float(-10 * torch.log10(((crop - reference) ** 2).mean().clamp_min(1e-10))), 3)
                if row["real_iou"] >= 0.5 and row["swap_iou"] < 0.5 and len(failures) < 80:
                    failures.append((row, box, {source: found[source] for source in SOURCES}))  # photo | reconstruction | swap
            rows.extend(view_rows)
            print(f"view {index}: {len(gts)} objects, {len(ignores)} ignored", flush=True)

    summary = {"all": _summary(rows, sources)}
    for split in ("train", "eval"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            summary[f"split_{split}"] = _summary(subset, sources)
    for low, high in DISTANCE_BINS:
        subset = [r for r in rows if low <= r["distance_m"] < high]
        if subset:
            summary[f"distance_{low}-{high if high < 1e9 else 'inf'}m"] = _summary(subset, sources)
    for name, test in (("visible<0.8", lambda r: r["visible_fraction"] < 0.8), ("visible>=0.8", lambda r: r["visible_fraction"] >= 0.8)):
        subset = [r for r in rows if test(r)]
        if subset:
            summary[name] = _summary(subset, sources)
    for source in sources:
        true = sum(r[f"{source}_iou"] >= 0.5 for r in rows)
        summary["all"][f"{source}_precision"] = true / max(true + false_positives[source], 1)
        summary["all"][f"{source}_detections"] = detections_total[source]
    for source, seconds in render_seconds.items():
        if len(seconds) > 1:  # the first render pays for warm-up
            summary["all"][f"{source}_render_fps"] = 1.0 / float(np.median(seconds[1:]))
    objects = {}
    for row in rows:
        objects.setdefault(row["track"], []).append(row)
    per_object = []
    for track, subset in sorted(objects.items()):
        item = {"track": track, "views": len(subset), "median_distance_m": float(np.median([r["distance_m"] for r in subset])),
                **{key: value for key, value in _summary(subset, sources).items() if key != "objects"}}
        per_object.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scene_perception.json").write_text(json.dumps(
        {"prompt": args.prompt, "category": category, "assets": len(assets), "unbuilt": len(unbuilt), "views": len(views),
         "summary": summary, "objects": per_object, "rows": rows}, indent=1))
    with open(args.output_dir / "objects.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_object[0].keys()))
        writer.writeheader()
        writer.writerows(per_object)
    folder = args.output_dir / "failures"
    folder.mkdir(exist_ok=True)
    for row, box, images in failures:
        pad = 0.8 * max(box[2] - box[0], box[3] - box[1])
        crop = (int(box[0] - pad), int(box[1] - pad / 2), int(box[2] + pad), int(box[3] + pad / 2))
        tiles = []
        for source in SOURCES:
            image, detections = images[source]
            image = image.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            for det in detections:
                draw.rectangle(det["xyxy"], outline=(255, 0, 0), width=2)
            tile = image.crop(crop)
            tile = tile.resize((360, max(1, int(360 * tile.height / max(tile.width, 1)))))
            ImageDraw.Draw(tile).text((4, 4), f"{source} IoU {row[f'{source}_iou']:.2f}", fill=(255, 255, 0))
            tiles.append(tile)
        sheet = Image.new("RGB", (360 * 3, max(t.height for t in tiles)))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (360 * i, 0))
        sheet.save(folder / f"gap_{row['track']}_{row['view']:05d}.jpg", quality=88)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
