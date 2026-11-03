"""Refine Grounding-DINO boxes into instance masks with SAM (box prompt).

Box masks are adequate for excluding a region from supervision, but lifting an
object into 3D needs its silhouette: a box also covers the wall and floor
around the object, and multi-view voting cannot carve away background that is
behind the object from every viewpoint.  This takes the highest-scoring box
per image from ``generate_dino_box_masks.py``'s manifest and writes SAM masks
in the static-mask convention (white = background, black = object).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--box-manifest", type=Path, required=True, help="dino_box_mask_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    args = parser.parse_args()

    import torch
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(args.model).to(device).eval()
    processor = SamProcessor.from_pretrained(args.model)
    manifest = json.loads(args.box_manifest.read_text())
    report: dict[str, object] = {"model": args.model, "boxes": args.box_manifest.name, "images": {}}
    for relative, records in sorted(manifest["images"].items()):
        image_path = args.images / relative
        output_path = (args.output / relative).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        mask = np.full((image.height, image.width), 255, dtype=np.uint8)
        best = max(records, key=lambda record: record["score"], default=None)
        if best is not None and best["score"] >= args.min_score:
            inputs = processor(image, input_boxes=[[best["xyxy"]]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            silhouette = masks[0][0, 0].numpy().astype(bool)
            mask[silhouette] = 0
            report["images"][relative] = {"box": best["xyxy"], "score": best["score"], "object_fraction": float(silhouette.mean())}
        else:
            report["images"][relative] = {"box": None}
        Image.fromarray(mask).save(output_path)
        print(f"{relative}: {report['images'][relative]}")
    (args.output / "sam_mask_manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
