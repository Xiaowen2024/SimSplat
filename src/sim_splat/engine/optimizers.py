from __future__ import annotations

import torch

from sim_splat.config import GaussianOptimizerConfig
from sim_splat.models.sim_splat_model import SimSplatModel


def build_gaussian_optimizers(
    model: SimSplatModel, config: GaussianOptimizerConfig, *, scene_scale: float = 1.0
) -> dict[str, torch.optim.Optimizer]:
    """Create one Adam optimizer per mutable Gaussian tensor.

    gsplat's DefaultStrategy requires this one-key/one-parameter convention so
    that it can extend or prune Adam state alongside a Gaussian tensor.
    """

    if scene_scale <= 0:
        raise ValueError(f"scene_scale must be positive, got {scene_scale}")
    learning_rates = {
        # Translation lives in world units, unlike every other Gaussian
        # attribute. Match the standard 3DGS convention by scaling only its LR
        # by a robust camera-trajectory extent.
        "means": config.means_lr * scene_scale,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
        "opacities": config.opacities_lr,
        "features_dc": config.features_dc_lr,
        "features_rest": config.features_rest_lr,
    }
    if config.eps <= 0 or any(value <= 0 for value in learning_rates.values()):
        raise ValueError("Adam epsilon and every Gaussian learning rate must be positive")
    if not 0 < config.final_lr_ratio <= 1:
        raise ValueError("final_lr_ratio must lie in (0, 1]")
    missing = set(learning_rates).difference(model.gauss_params)
    if missing:
        raise KeyError(f"Model is missing Gaussian parameters required by optimizer: {sorted(missing)}")

    return {
        name: torch.optim.Adam(
            [{"params": [model.gauss_params[name]], "lr": learning_rate, "initial_lr": learning_rate, "name": name}],
            eps=config.eps,
        )
        for name, learning_rate in learning_rates.items()
    }


def build_extra_optimizers(model: SimSplatModel, config: GaussianOptimizerConfig) -> dict[str, torch.optim.Optimizer]:
    """Adam for non-Gaussian parameters (sky map, per-sensor appearance).

    They are kept out of the Gaussian optimizer dict because gsplat
    strategies require params and optimizers to share exactly the same keys.
    """

    learning_rates = {"sky": config.sky_lr, "appearance": config.appearance_lr, "pose": config.pose_lr}
    return {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": learning_rates[name], "initial_lr": learning_rates[name], "name": name}],
            eps=config.eps,
        )
        for name, parameter in model.extra_parameters().items()
    }
