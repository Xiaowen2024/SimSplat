"""Aggregate evaluation and qualitative rendering for a saved SimSplat checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.datamanagers.full_image_datamanager import FullImageDataManager
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.checkpoints import load_model_weights
from sim_splat.models.sim_splat_model import SimSplatModel


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SimSplat checkpoint over every held-out view.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--hide-track", action="append", default=[], help="nuScenes instance token to omit before rendering")
    parser.add_argument("--split", choices=("eval", "train"), default="eval", help="Which fixed split to render and score")
    args = parser.parse_args()

    scene_config = load_scene_config(args.scene)
    train_config = load_train_config(args.config)
    if args.hide_track:
        train_config.model.hidden_track_ids = tuple(args.hide_track)
    outputs = parse_scene(scene_config)
    manager = FullImageDataManager(outputs, seed=train_config.seed)
    model = SimSplatModel(train_config.model, outputs, train_config.device)
    checkpoint_step = load_model_weights(args.checkpoint, model, train_config.device)
    model.eval()
    device = torch.device(train_config.device)
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[dict[str, float]] = []
    with torch.no_grad():
        indices = outputs.eval_indices if args.split == "eval" else outputs.train_indices
        for index, dataset_index in enumerate(indices):
            batch = manager.dataset[dataset_index]
            batch.image = batch.image.to(device)
            batch.static_mask = batch.static_mask.to(device)
            prediction = model.get_outputs(batch.camera)["rgb"]
            metrics = model.get_metrics_dict({"rgb": prediction}, batch)
            measurements.append(metrics)
            if args.render_dir:
                _save_rgb(args.render_dir / f"{index:03d}_{batch.image_name}", prediction)
                _save_rgb(args.render_dir / f"{index:03d}_{Path(batch.image_name).stem}_target.png", batch.image)
                _save_rgb(args.render_dir / f"{index:03d}_{Path(batch.image_name).stem}_static_mask.png", batch.static_mask.expand_as(batch.image))

    aggregate = {key: float(np.mean([record[key] for record in measurements])) for key in measurements[0]}
    result = {"checkpoint_step": checkpoint_step, "num_eval_views": len(measurements), "mean": aggregate, "per_view": measurements}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
