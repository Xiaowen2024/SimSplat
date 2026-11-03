from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from sim_splat.types import Batch, Camera


class Model(nn.Module, ABC):
    """Common model interface used by a pipeline and trainer."""

    @abstractmethod
    def get_outputs(self, camera: Camera) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_loss_dict(self, outputs: dict[str, Any], batch: Batch) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_metrics_dict(self, outputs: dict[str, Any], batch: Batch) -> dict[str, float]:
        raise NotImplementedError
