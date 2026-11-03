"""Perception check for an edited scene: does a detector still see the object?

Runs on the output of ``render_physics.py``.  For every video frame it runs
off-the-shelf detectors on

* ``real``: captured photos (only frames where the path camera is a real view),
* ``original``: the unedited reconstruction,
* ``edited``: the reconstruction with the object pushed by the simulator,

and matches detections to the target's projected 3D box.  Reported:

* **recall** of the target at its *current* location, per phase
  (before contact / pushing / after);
* **ghosts**: detections at the original location in the *background-only*
  render (object removed), i.e. whether the spot still looks like the object
  (baked shadows, smudges) -- this does not depend on where the object went;
* **holes**: background opacity behind the object's original silhouette, and
  how much of that silhouette the moved object has vacated (``meta.json``).

COCO detectors (YOLO) have no barrier or traffic-cone class, so the target is
detected with an open-vocabulary model (Grounding DINO) and a text prompt.
YOLO is still run to report any COCO object appearing at the vacated spot.

Detection runs on an upscaled crop around each box: on full frames Grounding
DINO boxes a whole *row* of barriers as one object, so single instances
never match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def _matches(detections: list[dict], box: list[float] | None, threshold: float) -> bool:
    """IoU match, or a same-sized detection centred inside the box.

    A projected 3D box is looser than the visible silhouette, which caps IoU
    for small, thin objects even when the detection is correct.
    """

    if box is None:
        return False
    area = (box[2] - box[0]) * (box[3] - box[1])
    for item in detections:
        x0, y0, x1, y1 = item["xyxy"]
        centred = box[0] <= (x0 + x1) / 2 <= box[2] and box[1] <= (y0 + y1) / 2 <= box[3]
        ratio = (x1 - x0) * (y1 - y0) / max(area, 1e-9)
        if _iou(item["xyxy"], box) >= threshold or (centred and 0.3 <= ratio <= 2.5):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", type=Path, required=True, help="render_physics.py output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="an orange and white plastic barrier.")
    parser.add_argument("--edited-prompt", help="prompt for the edited object when it was replaced by a different asset")
    parser.add_argument("--crop-pad", type=float, default=1.0, help="crop half-size = box size * (0.5 + pad)")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--after-frames", type=int, default=12, help="last N simulated frames count as 'after'")
    args = parser.parse_args()

    from transformers import pipeline
    from ultralytics import YOLO

    meta = json.loads((args.renders / "meta.json").read_text())
    frames = meta["frames"]
    sim_frames = max(item["sim_step"] for item in frames) + 1
    contact = meta.get("first_contact_frame") or 0
    dino = pipeline(task="zero-shot-object-detection", model=args.dino_model, device=0)
    yolo = YOLO(args.yolo_model)

    def detect_target(image: Image.Image, box: list[float] | None, prompt: str) -> list[dict]:
        if box is None:
            return []
        size = max(box[2] - box[0], box[3] - box[1]) * (1.0 + 2.0 * args.crop_pad)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        left, top = cx - size / 2, cy - size / 2
        crop = image.crop((int(left), int(top), int(left + size), int(top + size))).resize((512, 512))
        scale = 512 / size
        found = dino(crop, candidate_labels=[prompt], threshold=args.dino_threshold)
        return [{"score": float(item["score"]),
                 "xyxy": [item["box"]["xmin"] / scale + left, item["box"]["ymin"] / scale + top,
                          item["box"]["xmax"] / scale + left, item["box"]["ymax"] / scale + top]} for item in found]

    def detect_coco(image: Image.Image) -> list[dict]:
        result = yolo.predict(np.asarray(image), conf=0.25, verbose=False)[0]
        return [{"label": result.names[int(box.cls.item())], "score": float(box.conf.item()), "xyxy": [float(v) for v in box.xyxy[0].tolist()]}
                for box in result.boxes]

    def phase(record: dict) -> str:
        if record["frame"] >= sim_frames - args.after_frames:
            return "after"
        return "before" if record["sim_step"] < contact else "pushing"

    rows = []
    for record in frames:
        k = record["frame"]
        row = {"frame": k, "phase": phase(record), "is_real": record["is_real"]}
        for kind in ("real", "original", "edited"):
            path = args.renders / kind / f"{k:04d}.png"
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGB")
            box = record["original_box"] if kind != "edited" else record["current_box"]
            target = detect_target(image, box, args.edited_prompt if kind == "edited" and args.edited_prompt else args.prompt)
            row[kind] = {"target_detected": _matches(target, box, args.iou), "detections": target}
        background = Image.open(args.renders / "background" / f"{k:04d}.png").convert("RGB")
        ghosts = detect_target(background, record["original_box"], args.prompt)
        row["background"] = {"ghost_at_original": _matches(ghosts, record["original_box"], args.iou), "detections": ghosts,
                             "coco_at_original": [item["label"] for item in detect_coco(background)
                                                  if record["original_box"] and _iou(item["xyxy"], record["original_box"]) >= args.iou]}
        row["hole"] = record["background_alpha_in_original_box"]
        rows.append(row)

    def rate(kind: str, which: str, rows_subset: list[dict]) -> dict:
        valid = [row for row in rows_subset if kind in row]
        hits = sum(row[kind]["target_detected"] for row in valid)
        return {"frames": len(valid), "detected": hits, "recall": round(hits / len(valid), 3) if valid else None}

    summary: dict = {"prompt": args.prompt, "edited_prompt": args.edited_prompt, "iou_threshold": args.iou, "phases": {}}
    for name in ("before", "pushing", "after"):
        subset = [row for row in rows if row["phase"] == name]
        summary["phases"][name] = {kind: rate(kind, name, subset) for kind in ("real", "original", "edited")}
    after = [row for row in rows if row["phase"] == "after"]
    holes = [row["hole"] for row in rows if row["hole"]]
    both = [row for row in rows if "original" in row and row["original"]["target_detected"]]
    summary["edited_detected_when_original_detected"] = (
        round(sum(row["edited"]["target_detected"] for row in both) / len(both), 3) if both else None)
    summary["vacated_location"] = {
        "frames": len(rows),
        "ghost_rate_background_only": round(sum(row["background"]["ghost_at_original"] for row in rows) / len(rows), 3),
        "coco_objects_at_original": sorted({label for row in rows for label in row["background"]["coco_at_original"]}),
        "background_mean_alpha_behind_object": round(float(np.mean([h["mean_alpha"] for h in holes])), 3) if holes else None,
        "background_fraction_alpha_below_0.5": round(float(np.mean([h["fraction_alpha_below_0.5"] for h in holes])), 3) if holes else None,
        "vacated_fraction_after": round(float(np.mean([row["hole"]["vacated_fraction"] for row in after if row["hole"]])), 3)
        if any(row["hole"] for row in after) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**summary, "frames": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
