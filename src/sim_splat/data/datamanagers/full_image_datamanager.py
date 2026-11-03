from __future__ import annotations

import random

from sim_splat.data.input_dataset import InputDataset
from sim_splat.data.datamanagers.base_datamanager import DataManager
from sim_splat.types import Batch, DataparserOutputs


class FullImageDataManager(DataManager):
    """Samples whole images; it deliberately knows nothing about Gaussian rendering."""

    def __init__(self, outputs: DataparserOutputs, seed: int = 42) -> None:
        self.dataset = InputDataset(outputs)
        self._train_indices = list(outputs.train_indices)
        self._eval_indices = list(outputs.eval_indices)
        if not self._train_indices or not self._eval_indices:
            raise ValueError("Both train and eval splits must contain at least one image")
        self._rng = random.Random(seed)
        self._train_queue: list[int] = []

    def _next_train_index(self) -> int:
        if not self._train_queue:
            self._train_queue = self._train_indices.copy()
            self._rng.shuffle(self._train_queue)
        return self._train_queue.pop()

    def next_train(self, step: int) -> Batch:
        del step
        return self.dataset[self._next_train_index()]

    def next_eval(self, step: int) -> Batch:
        return self.dataset[self._eval_indices[step % len(self._eval_indices)]]

    def fixed_eval_batches(self) -> list[Batch]:
        return [self.dataset[index] for index in self._eval_indices]
