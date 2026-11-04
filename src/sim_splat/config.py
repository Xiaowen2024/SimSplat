from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class SceneConfig:
    """Which scene to train on.  Dataset-specific options live in converters.

    ``dataparser: scene`` (default) reads a SimSplat scene directory written
    by a converter; ``colmap`` reads a raw COLMAP reconstruction directly.
    """

    name: str
    data: Path  # scene directory (scene) or dataset root (colmap)
    dataparser: Literal["scene", "colmap"] = "scene"
    eval_interval: int = 8  # used only when the source defines no split
    # Optional deterministic training subset for geometric sanity checks.
    # Indices refer to the parser's frame order, not filenames.
    train_frame_indices: tuple[int, ...] | None = None
    # --- COLMAP only ------------------------------------------------------
    images_path: str = "images"
    colmap_path: str = "sparse/0"
    masks_path: str | None = None
    max_init_points: int = 100_000
    # Center and scale world coordinates using the registered camera trajectory.
    # RGB pixels and intrinsics are intentionally untouched.
    normalize_world_space: bool = False
    # A scene may contain several physical cameras; their real models are read
    # per camera from calibration, never assumed globally here.
    calibration_policy: Literal["preserve", "rectify"] = "preserve"
    unsupported_model_policy: Literal["error", "rectify"] = "error"


@dataclass
class ModelConfig:
    sh_degree: int = 3
    init_opacity: float = 0.1
    init_scale_multiplier: float = 1.0
    init_knn_k: int = 4
    ssim_lambda: float = 0.2
    static_mask_loss_weight: float = 1.0
    use_scale_regularization: bool = False
    max_gauss_ratio: float = 10.0
    # Track IDs to omit at render time.  This is the editing handle produced
    # by the DrivingGaussian-style object/background decomposition.
    hidden_track_ids: tuple[str, ...] = ()
    # Add one SH band every this many steps (standard 3DGS schedule).
    sh_degree_interval: int = 1000
    # Learned equirectangular sky composited through Gaussian transmittance.
    sky: bool = False
    sky_resolution: tuple[int, int] = (128, 256)
    sky_initial_rgb: float = 0.7
    # Per-sensor 3x4 colour transform for multi-camera exposure differences.
    appearance_affine: bool = False
    # Sparse LiDAR depth supervision (scene/lidar_depth/, add_lidar_depth.py):
    # relative L1 between rendered expected depth and LiDAR at static pixels.
    depth_loss_weight: float = 0.0
    depth_loss_max_m: float = 80.0
    # Per-image SE(3) pose correction learned with the scene (training views only):
    # absorbs pose / sync / calibration error that otherwise blurs the reconstruction.
    pose_refine: bool = False


@dataclass
class GaussianOptimizerConfig:
    means_lr: float = 1.6e-4
    scales_lr: float = 5.0e-3
    quats_lr: float = 1.0e-3
    opacities_lr: float = 5.0e-2
    features_dc_lr: float = 2.5e-3
    features_rest_lr: float = 1.25e-4
    eps: float = 1.0e-15
    # Exponential decay prevents a few early, high-gradient frames from
    # permanently inflating Gaussians in sparse or dynamic sequences.
    final_lr_ratio: float = 0.1
    # Original 3DGS decays only the position learning rate.  Decaying colour, opacity and
    # scale too (False, the historical default here) freezes them late and underfits.
    decay_means_only: bool = False
    sky_lr: float = 1.0e-2
    appearance_lr: float = 1.0e-3
    pose_lr: float = 5.0e-4


@dataclass
class DensificationConfig:
    """Budget and schedule for adaptive Gaussian refinement.

    A raw DefaultStrategy can add every high-gradient primitive at the same
    refinement event, which is particularly dangerous for large SfM scenes.
    This configuration makes the capacity constraint explicit and reproducible.
    """

    max_gaussians: int = 150_000
    max_new_per_refine: int = 2_500
    start_step: int = 500
    stop_step: int = 6_000
    every: int = 100
    reset_every: int = 3_000
    grow_grad2d: float = 2.0e-4
    prune_opacity: float = 0.005
    revised_opacity: bool = True


@dataclass
class TrainConfig:
    experiment_name: str
    max_steps: int = 30_000
    seed: int = 42
    device: str = "cuda"
    output_dir: Path = Path("outputs")
    eval_every: int = 1_000
    save_every: int = 5_000
    densify: bool = True
    log_every: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: GaussianOptimizerConfig = field(default_factory=GaussianOptimizerConfig)
    densification: DensificationConfig = field(default_factory=DensificationConfig)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_scene_config(path: Path) -> SceneConfig:
    values = _load_yaml(path)
    values["data"] = Path(values["data"])
    for key in ("train_frame_indices",):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return SceneConfig(**values)


def load_train_config(path: Path) -> TrainConfig:
    values = _load_yaml(path)
    values["output_dir"] = Path(values.get("output_dir", "outputs"))
    model_values = dict(values.get("model", {}))
    for key in ("hidden_track_ids", "sky_resolution"):
        if key in model_values:
            model_values[key] = tuple(model_values[key])
    values["model"] = ModelConfig(**model_values)
    values["optimizer"] = GaussianOptimizerConfig(**values.get("optimizer", {}))
    values["densification"] = DensificationConfig(**values.get("densification", {}))
    return TrainConfig(**values)
