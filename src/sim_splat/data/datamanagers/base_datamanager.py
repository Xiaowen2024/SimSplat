from __future__ import annotations

from abc import ABC, abstractmethod

from sim_splat.types import Batch


class DataManager(ABC):
    """Owns per-step sampling policy and data residency, not scene parsing."""

    @abstractmethod
    def next_train(self, step: int) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Batch:
        raise NotImplementedError
