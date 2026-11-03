"""Generate conservative static masks from Grounding DINO detection boxes.

This is a deliberately simple *box-mask baseline*: it answers whether removing
chair-like image regions changes the reconstruction. It does not claim precise
instance segmentation; use a SAM refinement stage before treating masks as
production-quality actor boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _image_paths(images_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [path for path in images_dir.rglob("*") if path.suffix.lower() in extensions]
    if not paths:
        raise ValueError(f"No images found in {images_dir}")
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create static masks from Grounding DINO boxes.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", default="chair,church pew,bench")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--dilate-px", type=int, default=8)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.dilate_px < 0:
        raise ValueError("--dilate-px must be non-negative")
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one text prompt")

    try:
        import torch
        from transformers import pipeline
    except ImportError as error:
        raise ImportError(
            "Install optional masking dependencies first: "
            "pip install -e '.[masking]'"
        ) from error
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    detector = pipeline(
        task="zero-shot-object-detection",
        model=args.model,
        device=0 if args.device == "cuda" else -1,
    )
    image_paths = _image_paths(args.images)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "kind": "grounding_dino_box_mask_baseline",
        "model": args.model,
        "labels": labels,
        "threshold": args.threshold,
        "dilate_px": args.dilate_px,
        "images": {},
    }

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(args.images)
        output_path = (args.output / relative_path).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        detections = detector(image, candidate_labels=labels, threshold=args.threshold)

        # White means supervise static background; black means ignore candidate actor.
        mask = Image.new("L", image.size, color=255)
        draw = ImageDraw.Draw(mask)
        records: list[dict[str, object]] = []
        for detection in detections:
            box = detection["box"]
            coordinates = [int(round(box[key])) for key in ("xmin", "ymin", "xmax", "ymax")]
            draw.rectangle(coordinates, fill=0)
            records.append(
                {
                    "label": str(detection["label"]),
                    "score": float(detection["score"]),
                    "xyxy": coordinates,
                }
            )
        if args.dilate_px:
            # MaxFilter dilates white; invert to dilate the black ignore region.
            kernel = 2 * args.dilate_px + 1
            inverted = Image.fromarray(255 - np.asarray(mask, dtype=np.uint8))
            mask = Image.fromarray(255 - np.asarray(inverted.filter(ImageFilter.MaxFilter(kernel)), dtype=np.uint8))
        mask.save(output_path)
        manifest["images"][str(relative_path)] = records
        print(f"[{index}/{len(image_paths)}] {relative_path}: {len(records)} candidate boxes")

    with (args.output / "dino_box_mask_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
