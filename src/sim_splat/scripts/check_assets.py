"""Sanity checks for a batch of look-alike assets (``build_lookalike_asset.py --category``).

Flags fits that are physically implausible against the annotation box and
the category template, and stacks every ``preview.jpg`` into one contact
sheet for review:

* ``short`` / ``tall``: fitted height outside [0.6, 1.25] x annotated height
  (the face plane caught the kerb, a neighbour's top, or a wall);
* ``long``: fitted length > 1.25 x annotated length (neighbours in a row
  merged into one face);
* ``face_behind``: the fitted camera-facing plane lies more than half the
  annotated depth *behind* the box centre (a surface behind the object);
* ``unstable``: photo-mask fits whose measured height varies by more than
  15 cm between views (interquartile range): occluded, or merged with a
  neighbour.  A common vertical shift of top and bottom is not a flag:
  that is per-frame pose drift, corrected by ``frame_shift_z``;
* ``unseen``: under 10% of the surface was textured from photos.

Writes ``checks.json`` and ``contact_sheet.jpg`` next to the assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(info: dict) -> list[str]:
    fit = info["fit"]
    annotated, fitted = fit["annotation_size"], fit["fitted_size"]
    flags = []
    if fitted[2] < 0.6 * annotated[2]:
        flags.append("short")
    if fitted[2] > 1.25 * annotated[2]:
        flags.append("tall")
    if fitted[1] > 1.25 * annotated[1]:
        flags.append("long")
    if "face_x" in fit and fit["camera_side"] * fit["face_x"] < -0.5 * annotated[0]:
        flags.append("face_behind")
    if fit.get("height_spread_m", 0.0) > 0.15:
        flags.append("unstable")
    if info["coverage"]["photo"] < 0.10:
        flags.append("unseen")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset_root", type=Path)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    results = []
    for path in sorted(args.asset_root.glob("*/asset.json")):
        info = json.loads(path.read_text())
        results.append({"track": info["track"][:8], "views": info["views"], "flags": check(info),
                        "annotated_size": info["fit"]["annotation_size"], "fitted_size": info["fit"]["fitted_size"],
                        "photo_coverage": round(info["coverage"]["photo"], 3)})
    (args.asset_root / "checks.json").write_text(json.dumps(results, indent=1))
    rows = []
    for item in results:
        preview = args.asset_root / item["track"] / "preview.jpg"
        if not preview.is_file():
            continue
        image = Image.open(preview).convert("RGB")
        row = image.crop((0, 0, image.width, 220)).resize((image.width // 2, 110))
        label = f"{item['track']}  {'/'.join(item['flags']) or 'ok'}  fit {[round(v, 2) for v in item['fitted_size']]}"
        ImageDraw.Draw(row).text((4, 4), label, fill=(255, 0, 0) if item["flags"] else (0, 255, 0))
        rows.append(row)
    if rows:
        columns = 2
        height = -(-len(rows) // columns)
        sheet = Image.new("RGB", (rows[0].width * columns, rows[0].height * height))
        for i, row in enumerate(rows):
            sheet.paste(row, ((i % columns) * row.width, (i // columns) * row.height))
        sheet.save(args.asset_root / "contact_sheet.jpg", quality=85)
    flagged = [item for item in results if item["flags"]]
    print(f"{len(results)} assets, {len(flagged)} flagged")
    for item in flagged:
        print(item["track"], item["flags"], "annotated", item["annotated_size"], "fitted", item["fitted_size"])


if __name__ == "__main__":
    main()
