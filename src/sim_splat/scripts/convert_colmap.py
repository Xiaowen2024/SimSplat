"""Convert a COLMAP reconstruction into a SimSplat scene directory.

Uses the COLMAP scene YAML (``dataparser: colmap``) for paths, masks, point
budget and world normalization, then writes the generic format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_splat.config import load_scene_config
from sim_splat.data.dataparsers.colmap_dataparser import ColmapDataParser
from sim_splat.data.scene_format import export_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="scene YAML with dataparser: colmap")
    parser.add_argument("--output", type=Path, required=True, help="new scene directory (must not exist)")
    parser.add_argument("--link-images", action="store_true", help="symlink instead of copying RGB files")
    args = parser.parse_args()
    config = load_scene_config(args.scene)
    if config.dataparser != "colmap":
        raise ValueError(f"{args.scene} is not a COLMAP scene config")
    outputs = ColmapDataParser(config).get_dataparser_outputs()
    print(f"wrote {export_scene(outputs, args.output, name=config.name, link_images=args.link_images)}")


if __name__ == "__main__":
    main()
