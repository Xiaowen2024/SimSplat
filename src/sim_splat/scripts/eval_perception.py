"""Perception check for edited scenes: full-frame detection and 3D localization.

Unlike ``eval_edit_detection.py`` (which crops around the known box), the
detector here sees the whole image, so it must find the object, separate it
from its neighbours, and box it tightly.  Two localizers then turn each
matched detection into a 3D position:

* **mono**: the bottom-centre of the 2D box intersected with the ground
  plane under the object's original pose (what a camera-only stack does);
* **depth**: the median rendered depth inside the box, back-projected at
  the box centre (a stand-in for LiDAR / stereo).

Ground truth comes from the simulator: tight 2D boxes from rendered
silhouettes, and the object's 3D centre.

Modes:

``static``: every training view that sees the object, at rest (t = 0), on
  the real photo, the reconstruction, and the swapped render (inpainted
  background + look-alike asset).  This tests whether perception treats
  the asset like the real object *before* anything moves.  It also reports
  crop PSNR / LPIPS against the photo.

``sequence``: a ``render_physics.py`` output (original vs edited per video
  frame, real photo where the path hits a captured view).  It reports
  results per phase, plus ghost detections at the vacated spot.

Writes ``perception.json`` and a ``failures/`` gallery (misses, worst
localization) for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


class TiledDetector:
    """Grounding DINO over the full frame plus overlapping 2x2 tiles, merged by NMS."""

    def __init__(self, model: str, prompt: str, threshold: float) -> None:
        from transformers import pipeline

        self.pipe = pipeline(task="zero-shot-object-detection", model=model, device=0)
        self.prompt, self.threshold = prompt, threshold

    def __call__(self, image: Image.Image) -> list[dict]:
        import torchvision

        width, height = image.size
        windows = [(0, 0, width, height)]
        tw, th = int(width * 0.6), int(height * 0.6)
        for x in (0, width - tw):
            for y in (0, height - th):
                windows.append((x, y, x + tw, y + th))
        boxes, scores = [], []
        for x0, y0, x1, y1 in windows:
            crop = image.crop((x0, y0, x1, y1))
            scale = 1000 / crop.width
            found = self.pipe(crop.resize((1000, int(crop.height * scale))), candidate_labels=[self.prompt], threshold=self.threshold)
            for item in found:
                b = item["box"]
                box = [b["xmin"] / scale + x0, b["ymin"] / scale + y0, b["xmax"] / scale + x0, b["ymax"] / scale + y0]
                if (box[2] - box[0]) > 0.9 * width:  # whole-row / whole-image boxes
                    continue
                boxes.append(box)
                scores.append(item["score"])
        if not boxes:
            return []
        keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
        return [{"xyxy": boxes[i], "score": float(scores[i])} for i in keep]


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def _best(detections: list[dict], box) -> tuple[dict | None, float]:
    if box is None or not detections:
        return None, 0.0
    scored = [(_iou(d["xyxy"], box), d) for d in detections]
    iou, det = max(scored, key=lambda item: item[0])
    return det, iou


def _ray(c2w: np.ndarray, K: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    direction = c2w[:3, :3] @ np.array(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0))
    return c2w[:3, 3], direction


def localize(box, c2w: np.ndarray, K: np.ndarray, ground_z: float, depth: np.ndarray | None) -> dict:
    out = {}
    origin, direction = _ray(c2w, K, (box[0] + box[2]) / 2, box[3])
    if direction[2] < -1e-6:
        out["mono"] = (origin + direction * (ground_z - origin[2]) / direction[2]).tolist()
    if depth is not None:
        w, h = box[2] - box[0], box[3] - box[1]
        inner = depth[int(box[1] + 0.25 * h) : int(box[3] - 0.25 * h) + 1, int(box[0] + 0.25 * w) : int(box[2] - 0.25 * w) + 1]
        if inner.size:
            origin, direction = _ray(c2w, K, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            out["depth"] = (origin + direction * float(np.median(inner))).tolist()
    return out


def _score(det, iou, box, centre, c2w, K, ground_z, depth) -> dict:
    row = {"gt_box": box, "iou": iou, "detected@0.5": iou >= 0.5, "detected@0.3": iou >= 0.3,
           "det_box": det["xyxy"] if det else None, "det_score": det["score"] if det else None}
    if det is not None and iou >= 0.3:
        for name, point in localize(det["xyxy"], c2w, K, ground_z, depth).items():
            row[f"error_{name}_m"] = float(np.linalg.norm(np.asarray(point)[:2] - np.asarray(centre)[:2]))
    return row


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {"frames": len(rows), "recall@0.5": float(np.mean([r["detected@0.5"] for r in rows])),
           "recall@0.3": float(np.mean([r["detected@0.3"] for r in rows]))}
    ious = [r["iou"] for r in rows if r["detected@0.3"]]
    out["mean_iou_matched"] = float(np.mean(ious)) if ious else None
    for name in ("mono", "depth"):
        errors = [r[f"error_{name}_m"] for r in rows if f"error_{name}_m" in r]
        if errors:
            out[f"median_error_{name}_m"] = float(np.median(errors))
            out[f"p90_error_{name}_m"] = float(np.percentile(errors, 90))
    for key in ("crop_psnr", "crop_lpips"):
        values = [r[key] for r in rows if key in r]
        if values:
            out[key] = float(np.mean(values))
    return out


def _draw(image: Image.Image, gt, det, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if gt:
        draw.rectangle(gt, outline=(0, 255, 0), width=2)
    if det:
        draw.rectangle(det, outline=(255, 0, 0), width=2)
    draw.text((4, 4), label, fill=(255, 255, 0))
    return image


def run_sequence(args, detector: TiledDetector) -> dict:
    meta = json.loads((args.renders / "meta.json").read_text())
    frames = meta["frames"]
    sim_frames = max(item["sim_step"] for item in frames) + 1
    contact = meta.get("first_contact_frame") or 0
    rows: dict[str, list[dict]] = {}
    failures = []
    for record in frames:
        k = record["frame"]
        phase = "after" if k >= sim_frames - args.after_frames else ("before" if record["sim_step"] < contact else "pushing")
        cam = record["camera"]
        c2w, K = np.asarray(cam["c2w"]), np.asarray(cam["K"])
        for source in ("real", "original", "edited"):
            path = args.renders / source / f"{k:04d}.png"
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGB")
            moved = source == "edited"
            box = record["current_silhouette_box"] if moved else record["original_silhouette_box"]
            centre = record["current_centre"] if moved else record["original_centre"]
            depth_path = args.renders / ("edited" if moved else "original") / f"{k:04d}_depth.npy"
            depth = np.load(depth_path).astype(np.float32) if depth_path.is_file() else None
            detections = detector(image)
            det, iou = _best(detections, box)
            row = {"frame": k, "phase": phase, **_score(det, iou, box, centre, c2w, K, record["ground_z"], depth)}
            if moved and record["original_silhouette_box"] and box and _iou(record["original_silhouette_box"], box) < 0.2:
                # The object has left: any detection on its old spot is a ghost.
                row["ghost"] = _best(detections, record["original_silhouette_box"])[1] >= 0.3
            rows.setdefault(f"{source}/{phase}", []).append(row)
            if box and (iou < 0.5 or row.get("error_depth_m", 0) > 1.0):
                failures.append((source, k, image, box, det["xyxy"] if det else None, iou))
    summary = {key: _summarise(value) for key, value in sorted(rows.items())}
    ghosts = [r["ghost"] for key, value in rows.items() if key.startswith("edited") for r in value if "ghost" in r]
    summary["ghost_rate_when_vacated"] = float(np.mean(ghosts)) if ghosts else None
    _save_failures(args.output_dir, failures)
    return {"mode": "sequence", "renders": str(args.renders), "summary": summary, "rows": rows}


def run_static(args, detector: TiledDetector) -> dict:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    from sim_splat.config import load_scene_config, load_train_config
    from sim_splat.data.dataparsers import parse_scene
    from sim_splat.data.input_dataset import InputDataset
    from sim_splat.engine.checkpoints import load_model_weights
    from sim_splat.models.sim_splat_model import SimSplatModel
    from sim_splat.scripts.build_lookalike_asset import _views
    from sim_splat.scripts.render_edit import _to_uint8
    from sim_splat.scripts.render_physics import _object_alpha

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
    target = next(token for token in recon.track_tokens if token.startswith(args.track))
    track = recon.track_tokens.index(target)
    info = json.loads((args.asset_dir / "asset.json").read_text())
    correction = np.asarray(info["local_correction"])
    half = np.asarray(info["size_xyz"]) / 2.0
    track_poses = recon.object_to_world[:, track].detach().cpu().numpy().astype(np.float64)
    asset_poses = track_poses @ correction
    asset_index = swap.add_actor("lookalike", torch.load(args.asset_dir / "gaussians.pt"), torch.from_numpy(asset_poses).float(),
                                 torch.ones((asset_poses.shape[0],), dtype=torch.bool))
    swap.set_hidden_tracks([target])
    asset_rows = swap.point_track_indices == asset_index
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    dataset = InputDataset(scene)
    rows: dict[str, list[dict]] = {}
    failures = []
    with torch.no_grad():
        for _, index in _views(scene, asset_poses, half, args.min_pixels):
            camera = scene.cameras[index]
            silhouette = _object_alpha(swap, asset_rows, camera) > 0.5
            swap.set_hidden_tracks([target])
            ys, xs = torch.nonzero(silhouette, as_tuple=True)
            if len(xs) < 50:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            pose = asset_poses[camera.image_id]
            c2w, K = camera.c2w.numpy().astype(np.float64), camera.K.numpy().astype(np.float64)
            ground_z = float(pose[2, 3] - half[2])
            photo = dataset[index].image.to(device)
            out_recon, out_swap = recon.get_outputs(camera), swap.get_outputs(camera)
            images = {"real": photo, "reconstruction": out_recon["rgb"].clamp(0, 1), "swap": out_swap["rgb"].clamp(0, 1)}
            depths = {"real": out_recon["depth"], "reconstruction": out_recon["depth"], "swap": out_swap["depth"]}
            pad_w, pad_h = 0.2 * (box[2] - box[0]), 0.2 * (box[3] - box[1])
            x0, y0 = int(max(0, box[0] - pad_w)), int(max(0, box[1] - pad_h))
            x1, y1 = int(min(camera.width, box[2] + pad_w)), int(min(camera.height, box[3] + pad_h))
            for source, tensor in images.items():
                image = Image.fromarray(_to_uint8(tensor))
                det, iou = _best(detector(image), box)
                row = {"view": index, **_score(det, iou, box, pose[:3, 3], c2w, K, ground_z, depths[source][..., 0].cpu().numpy())}
                if source != "real":
                    a, b = tensor[y0:y1, x0:x1], photo[y0:y1, x0:x1]
                    row["crop_psnr"] = float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))
                    resize = lambda t: torch.nn.functional.interpolate(t.permute(2, 0, 1)[None], size=(128, 128), mode="bilinear")
                    row["crop_lpips"] = float(lpips(resize(a), resize(b)))
                rows.setdefault(source, []).append(row)
                if iou < 0.5:
                    failures.append((source, index, image, box, det["xyxy"] if det else None, iou))
    summary = {key: _summarise(value) for key, value in rows.items()}
    _save_failures(args.output_dir, failures)
    return {"mode": "static", "asset": str(args.asset_dir), "summary": summary, "rows": rows}


def _save_failures(output_dir: Path, failures: list) -> None:
    folder = output_dir / "failures"
    folder.mkdir(parents=True, exist_ok=True)
    for source, k, image, box, det, iou in failures[:60]:
        pad = 0.8 * max(box[2] - box[0], box[3] - box[1])
        crop = (int(box[0] - pad), int(box[1] - pad / 2), int(box[2] + pad), int(box[3] + pad / 2))
        tile = _draw(image, box, det, f"{source} {k} IoU={iou:.2f} (green GT, red det)").crop(crop)
        tile.resize((480, int(480 * tile.height / max(tile.width, 1)))).save(folder / f"{source}_{k:04d}.jpg", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("static", "sequence"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="road barrier.")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--renders", type=Path, help="sequence: render_physics.py output")
    parser.add_argument("--after-frames", type=int, default=12)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="static: unedited reconstruction")
    parser.add_argument("--background-checkpoint", type=Path, help="static: inpainted background for the swap")
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--track")
    parser.add_argument("--min-pixels", type=float, default=600.0)
    args = parser.parse_args()

    detector = TiledDetector(args.dino_model, args.prompt, args.threshold)
    result = run_sequence(args, detector) if args.mode == "sequence" else run_static(args, detector)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "perception.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
