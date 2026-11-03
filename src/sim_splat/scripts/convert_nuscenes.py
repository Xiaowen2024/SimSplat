"""Convert one nuScenes scene into a SimSplat scene directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_splat.converters.nuscenes import NuScenesConverter, load_conversion_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/convert/<name>.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new scene directory (must not exist)")
    parser.add_argument("--link-images", action="store_true", help="symlink instead of copying RGB files")
    args = parser.parse_args()
    scene_dir = NuScenesConverter(load_conversion_config(args.config)).convert(args.output, link_images=args.link_images)
    print(f"wrote {scene_dir}")


if __name__ == "__main__":
    main()
