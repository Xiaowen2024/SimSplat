"""The generic parser: reads a SimSplat scene directory (README.md, "Scene format").

New data sources should not add a parser here; they add a converter under
``sim_splat.converters`` that writes this format.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sim_splat.config import SceneConfig
from sim_splat.data.dataparsers.base_dataparser import DataParser
from sim_splat.data.scene_format import load_scene
from sim_splat.types import DataparserOutputs


@dataclass
class SceneDataParser(DataParser):
    config: SceneConfig

    def get_dataparser_outputs(self) -> DataparserOutputs:
        outputs = load_scene(self.config.data)
        count = len(outputs.image_paths)
        if self.config.train_frame_indices is not None:
            # Explicit subset for sanity checks (e.g. a single-frame overfit).
            requested = list(self.config.train_frame_indices)
            if not requested or any(index < 0 or index >= count for index in requested):
                raise ValueError(f"train_frame_indices must be valid frame indices, got {requested}")
            held_out = [index for index in range(count) if index not in set(requested)]
            return replace(outputs, train_indices=requested, eval_indices=held_out)
        if not outputs.eval_indices:
            # Scenes exported without a split fall back to every Nth frame.
            eval_indices = list(range(0, count, self.config.eval_interval))
            train_indices = [index for index in outputs.train_indices if index not in set(eval_indices)]
            return replace(outputs, train_indices=train_indices, eval_indices=eval_indices)
        return outputs
