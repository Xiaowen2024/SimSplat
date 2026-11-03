from __future__ import annotations

import torch

from sim_splat.data.datamanagers.full_image_datamanager import FullImageDataManager
from sim_splat.models.base_model import Model
from sim_splat.pipelines.base_pipeline import Pipeline


class FullImagePipeline(Pipeline):
    """Framework bridge: manager → model outputs → losses/metrics."""

    def __init__(self, datamanager: FullImageDataManager, model: Model, device: str) -> None:
        self.datamanager = datamanager
        self.model = model
        self.device = torch.device(device)

    def _to_device(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(self.device, non_blocking=True)

    def get_train_loss_dict(self, step: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
        batch = self.datamanager.next_train(step)
        batch.image = self._to_device(batch.image)
        batch.static_mask = self._to_device(batch.static_mask)
        outputs = self.model.get_outputs(batch.camera)
        return outputs, self.model.get_loss_dict(outputs, batch), self.model.get_metrics_dict(outputs, batch)

    @torch.no_grad()
    def get_eval_metrics(self, step: int) -> dict[str, float]:
        self.model.eval()
        batch = self.datamanager.next_eval(step)
        batch.image = self._to_device(batch.image)
        batch.static_mask = self._to_device(batch.static_mask)
        outputs = self.model.get_outputs(batch.camera)
        metrics = self.model.get_metrics_dict(outputs, batch)
        self.model.train()
        return metrics
