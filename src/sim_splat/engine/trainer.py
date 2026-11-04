from __future__ import annotations

from typing import Any

import time

import torch

from sim_splat.config import TrainConfig
from sim_splat.engine.checkpoints import save_checkpoint
from sim_splat.engine.jsonl_logger import JsonlLogger
from sim_splat.pipelines.full_image_pipeline import FullImagePipeline


class Trainer:
    """Optimization loop with optional gsplat densification/pruning strategy."""

    def __init__(
        self,
        config: TrainConfig,
        pipeline: FullImagePipeline,
        optimizers: dict[str, torch.optim.Optimizer],
        strategy: Any | None = None,
        scene_scale: float = 1.0,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.optimizers = optimizers
        # Sky/appearance optimizers: stepped every iteration, never touched
        # by densification, and not learning-rate decayed.
        self.extra_optimizers = extra_optimizers or {}
        self.strategy = strategy
        self.output_dir = config.output_dir / config.experiment_name
        # One metrics file per run; appending silently mixed earlier runs.
        metrics_path = self.output_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        self.logger = JsonlLogger(metrics_path)
        self.strategy_state: dict[str, Any] | None = None
        if self.strategy is not None:
            self.strategy.check_sanity(self.pipeline.model.gauss_params, self.optimizers)
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    def _all_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {**self.optimizers, **{f"extra/{name}": value for name, value in self.extra_optimizers.items()}}

    def _zero_grad(self) -> None:
        for optimizer in self._all_optimizers().values():
            optimizer.zero_grad(set_to_none=True)

    def _step(self) -> None:
        for optimizer in self._all_optimizers().values():
            optimizer.step()
        self.pipeline.model.clamp_gaussian_parameters_()

    def _decay_learning_rates(self, step: int) -> None:
        """Exponential schedule shared by the independently mutable tensors."""

        progress = min(step / max(self.config.max_steps - 1, 1), 1.0)
        multiplier = self.config.optimizer.final_lr_ratio**progress
        for name, optimizer in self.optimizers.items():
            if self.config.optimizer.decay_means_only and name != "means":
                continue
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier

    def train(self) -> None:
        self.pipeline.model.train()
        started = time.time()
        for step in range(self.config.max_steps):
            self._decay_learning_rates(step)
            if hasattr(self.pipeline.model, "set_step"):
                self.pipeline.model.set_step(step)
            outputs, loss_dict, metric_dict = self.pipeline.get_train_loss_dict(step)
            raster_info = outputs.get("raster_info")
            if self.strategy is not None:
                if raster_info is None:
                    raise KeyError("Densification requires outputs['raster_info']")
                self.strategy.step_pre_backward(
                    params=self.pipeline.model.gauss_params,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=raster_info,
                )

            loss = sum(loss_dict.values())
            self._zero_grad()
            loss.backward()
            self._step()
            if self.strategy is not None:
                self.strategy.step_post_backward(
                    params=self.pipeline.model.gauss_params,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=raster_info,
                    packed=True,
                )
                # Refinement may create new parameter rows after the regular
                # optimizer step, so enforce finite parameter bounds again.
                self.pipeline.model.clamp_gaussian_parameters_()

            scalars = {**{key: value.detach().item() for key, value in loss_dict.items()}, **metric_dict}
            self.logger.log(step, scalars)
            if step % self.config.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"step={step} loss={loss.item():.4f} psnr={metric_dict['psnr']:.2f} "
                    f"gaussians={int(metric_dict['gaussian_count'])} elapsed={elapsed:.0f}s",
                    flush=True,
                )
            if step and step % self.config.save_every == 0:
                save_checkpoint(
                    self.output_dir / "checkpoints" / f"step_{step:08d}.pt",
                    step,
                    self.pipeline.model,
                    self._all_optimizers(),
                    self.strategy_state,
                )
            if step and step % self.config.eval_every == 0:
                eval_metrics = self.pipeline.get_eval_metrics(step)
                self.logger.log(step, {f"eval/{key}": value for key, value in eval_metrics.items()})
                print(f"eval step={step} psnr={eval_metrics['psnr']:.2f}", flush=True)

        final_step = self.config.max_steps
        save_checkpoint(
            self.output_dir / "checkpoints" / "final.pt",
            final_step,
            self.pipeline.model,
            self._all_optimizers(),
            self.strategy_state,
        )
        self.logger.log(final_step, {f"eval/{key}": value for key, value in self.pipeline.get_eval_metrics(final_step).items()})
