from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sim_splat.config import load_scene_config, load_train_config
from sim_splat.data.datamanagers.full_image_datamanager import FullImageDataManager
from sim_splat.data.dataparsers import parse_scene
from sim_splat.engine.optimizers import build_extra_optimizers, build_gaussian_optimizers
from sim_splat.engine.trainer import Trainer
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.pipelines.full_image_pipeline import FullImagePipeline


def _scene_scale(cameras: list) -> float:
    """Robust camera-trajectory extent used by adaptive refinement.

    A sparse SfM point cloud can contain distant outliers.  Refinement
    thresholds describe projected size near the observed scene, so they should
    be normalized by where cameras actually travelled, not the point-cloud
    bounding box.
    """

    positions = torch.stack([camera.c2w[:3, 3] for camera in cameras]).to(dtype=torch.float32)
    center = positions.mean(dim=0)
    return max(torch.linalg.vector_norm(positions - center, dim=-1).max().item(), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a standalone SimSplat model.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", help="override the config's experiment_name (output directory)")
    args = parser.parse_args()

    scene_config = load_scene_config(args.scene)
    train_config = load_train_config(args.config)
    if args.experiment_name:
        train_config.experiment_name = args.experiment_name
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    outputs = parse_scene(scene_config)
    datamanager = FullImageDataManager(outputs, seed=train_config.seed)
    model = SimSplatModel(train_config.model, outputs, train_config.device)
    pipeline = FullImagePipeline(datamanager, model, train_config.device)
    scene_scale = _scene_scale(outputs.cameras)
    optimizers = build_gaussian_optimizers(model, train_config.optimizer, scene_scale=scene_scale)
    extra_optimizers = build_extra_optimizers(model, train_config.optimizer)
    tracks = outputs.metadata.get("dynamic_track_tokens", [])
    owners = outputs.metadata.get("dynamic_point_track_indices")
    print(
        f"images={len(outputs.image_paths)} train={len(outputs.train_indices)} eval={len(outputs.eval_indices)} "
        f"seeds={len(outputs.points_xyz)} actor_seeds={int((owners >= 0).sum()) if owners is not None else 0} "
        f"tracks={len(tracks)} scene_scale={scene_scale:.2f}",
        flush=True,
    )

    strategy = None
    if train_config.densify:
        from sim_splat.strategies.budgeted_default import BudgetedDefaultStrategy

        densification = train_config.densification
        strategy = BudgetedDefaultStrategy(
            max_gaussians=densification.max_gaussians,
            max_new_per_refine=densification.max_new_per_refine,
            prune_opa=densification.prune_opacity,
            grow_grad2d=densification.grow_grad2d,
            refine_start_iter=densification.start_step,
            refine_stop_iter=densification.stop_step,
            refine_every=densification.every,
            reset_every=densification.reset_every,
            revised_opacity=densification.revised_opacity,
            verbose=True,
            on_duplicate=model.duplicate_track_owners,
            on_split=model.split_track_owners,
            on_prune=model.prune_track_owners,
        )

    Trainer(
        config=train_config,
        pipeline=pipeline,
        optimizers=optimizers,
        strategy=strategy,
        scene_scale=scene_scale,
        extra_optimizers=extra_optimizers,
    ).train()


if __name__ == "__main__":
    main()
