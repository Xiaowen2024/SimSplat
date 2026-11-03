from __future__ import annotations

from sim_splat.config import SceneConfig
from sim_splat.data.dataparsers.base_dataparser import DataParser
from sim_splat.data.dataparsers.colmap_dataparser import ColmapDataParser
from sim_splat.data.dataparsers.scene_dataparser import SceneDataParser
from sim_splat.types import DataparserOutputs

# "scene" is the generic format every converter writes.  "colmap" is kept so a
# raw COLMAP reconstruction can still be trained without a conversion step.
DATAPARSERS: dict[str, type[DataParser]] = {"scene": SceneDataParser, "colmap": ColmapDataParser}


def parse_scene(config: SceneConfig) -> DataparserOutputs:
    if config.dataparser not in DATAPARSERS:
        raise ValueError(f"Unknown dataparser {config.dataparser!r}; choose from {sorted(DATAPARSERS)}")
    return DATAPARSERS[config.dataparser](config).get_dataparser_outputs()


__all__ = ["DATAPARSERS", "ColmapDataParser", "DataParser", "SceneDataParser", "parse_scene"]
