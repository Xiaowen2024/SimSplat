from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Pipeline(ABC):
    """Standard interface consumed by the trainer."""

    @abstractmethod
    def get_train_loss_dict(self, step: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_metrics(self, step: int) -> dict[str, float]:
        raise NotImplementedError
