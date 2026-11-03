from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    strategy_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "strategy_state": strategy_state,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    device: str,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    checkpoint_optimizers = checkpoint["optimizers"]
    if set(checkpoint_optimizers) != set(optimizers):
        raise ValueError("Checkpoint optimizer keys do not match current model")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint_optimizers[name])
    return int(checkpoint["step"])


def load_model_weights(path: Path, model: nn.Module, device: str) -> int:
    """Load a (possibly densified) checkpoint into a freshly built SimSplatModel.

    Adaptive refinement changes the row count of every Gaussian tensor and of
    the per-row actor ownership buffer, so both are resized before a strict
    ``load_state_dict``.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    if "gauss_params.means" not in state:
        raise KeyError(f"{path} does not contain gauss_params.means")
    names = tuple(model.gauss_params.keys())
    model.gauss_params = nn.ParameterDict(
        {name: nn.Parameter(state[f"gauss_params.{name}"].detach().clone()) for name in names}
    ).to(device)
    if "point_track_indices" in state:
        model.point_track_indices = state["point_track_indices"].detach().clone().to(device)
    model.load_state_dict(state)
    return int(checkpoint["step"])
