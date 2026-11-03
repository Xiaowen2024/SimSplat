"""A capacity-bounded variant of gsplat's adaptive densification policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _topk_mask(scores: torch.Tensor, candidates: torch.Tensor, budget: int) -> torch.Tensor:
    """Keep at most ``budget`` candidate entries with the largest score."""

    selected = torch.zeros_like(candidates)
    indices = torch.where(candidates)[0]
    if budget <= 0 or indices.numel() == 0:
        return selected
    if indices.numel() <= budget:
        selected[indices] = True
        return selected
    ranked = torch.topk(scores[indices], k=budget, sorted=False).indices
    selected[indices[ranked]] = True
    return selected


class BudgetedDefaultStrategy:
    """Wrap gsplat refinement with a hard maximum primitive count.

    At every refinement event, candidate splits/duplicates are ranked by the
    accumulated screen-space mean gradient.  Only the strongest candidates
    fitting the remaining capacity can be added.  This is deliberately a
    strategy-level policy: it changes scene capacity without contaminating the
    renderer or model with a dataset-specific magic number.
    """

    def __init__(
        self,
        *,
        max_gaussians: int,
        max_new_per_refine: int,
        on_duplicate: Callable[[torch.Tensor], None] | None = None,
        on_split: Callable[[torch.Tensor], None] | None = None,
        on_prune: Callable[[torch.Tensor], None] | None = None,
        **default_kwargs: Any,
    ) -> None:
        if max_gaussians < 1:
            raise ValueError(f"max_gaussians must be positive, got {max_gaussians}")
        if max_new_per_refine < 1:
            raise ValueError(f"max_new_per_refine must be positive, got {max_new_per_refine}")
        try:
            from gsplat.strategy import DefaultStrategy
            from gsplat.strategy.ops import duplicate, split
        except ImportError as error:  # pragma: no cover - requires CUDA gsplat environment
            raise ImportError("BudgetedDefaultStrategy requires gsplat") from error
        self._default = DefaultStrategy(**default_kwargs)
        self._duplicate = duplicate
        self._split = split
        self.max_gaussians = max_gaussians
        self.max_new_per_refine = max_new_per_refine
        # Non-optimised per-Gaussian metadata (e.g. object track ownership)
        # must undergo exactly the same row operations as ParameterDict.
        self.on_duplicate = on_duplicate
        self.on_split = on_split
        self.on_prune = on_prune

    def check_sanity(self, *args: Any, **kwargs: Any) -> None:
        self._default.check_sanity(*args, **kwargs)

    def initialize_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._default.initialize_state(*args, **kwargs)

    def step_pre_backward(self, *args: Any, **kwargs: Any) -> None:
        self._default.step_pre_backward(*args, **kwargs)

    @torch.no_grad()
    def _grow_gs(self, params: Any, optimizers: Any, state: dict[str, Any], step: int) -> tuple[int, int]:
        count = state["count"]
        gradients = state["grad2d"] / count.clamp_min(1)
        remaining = self.max_gaussians - len(params["means"])
        if remaining <= 0:
            return 0, 0

        is_high_gradient = gradients > self._default.grow_grad2d
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self._default.grow_scale3d * state["scene_scale"]
        duplicate_candidates = is_high_gradient & is_small
        split_candidates = is_high_gradient & ~is_small
        if step < self._default.refine_scale2d_stop_iter:
            split_candidates |= state["radii"] > self._default.grow_scale2d
        # One primitive is added by either operation.  Resolve overlap toward
        # duplication, the less disruptive operation for a small Gaussian.
        split_candidates &= ~duplicate_candidates

        all_candidates = duplicate_candidates | split_candidates
        keep = _topk_mask(gradients, all_candidates, min(remaining, self.max_new_per_refine))
        duplicate_mask = duplicate_candidates & keep
        split_mask = split_candidates & keep
        num_duplicate = int(duplicate_mask.sum().item())
        num_split = int(split_mask.sum().item())
        if num_duplicate:
            if self.on_duplicate is not None:
                self.on_duplicate(duplicate_mask)
            self._duplicate(params=params, optimizers=optimizers, state=state, mask=duplicate_mask)
        if num_split:
            # Duplicates appended above must not become eligible to split in
            # this same event; split's mask uses the updated parameter length.
            split_mask = torch.cat((split_mask, torch.zeros(num_duplicate, dtype=torch.bool, device=split_mask.device)))
            if self.on_split is not None:
                # The appended duplicate rows are false in this padded mask;
                # passing the full, current-length vector preserves alignment
                # with ownership after the preceding duplicate operation.
                self.on_split(split_mask)
            self._split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=split_mask,
                revised_opacity=self._default.revised_opacity,
            )
        return num_duplicate, num_split

    def step_post_backward(self, *, params: Any, optimizers: Any, state: dict[str, Any], step: int, info: dict[str, Any], packed: bool) -> None:
        """Same state/prune/reset semantics as gsplat DefaultStrategy."""

        policy = self._default
        if step >= policy.refine_stop_iter:
            return
        policy._update_state(params, state, info, packed=packed)
        if step > policy.refine_start_iter and step % policy.refine_every == 0 and step % policy.reset_every >= policy.pause_refine_after_reset:
            n_duplicate, n_split = self._grow_gs(params, optimizers, state, step)
            # Mirror DefaultStrategy._prune_gs so side metadata undergoes the
            # same stable filtering operation as every optimised tensor.
            is_prune = torch.sigmoid(params["opacities"].flatten()) < policy.prune_opa
            if step > policy.reset_every:
                is_too_big = torch.exp(params["scales"]).max(dim=-1).values > policy.prune_scale3d * state["scene_scale"]
                if step < policy.refine_scale2d_stop_iter:
                    is_too_big |= state["radii"] > policy.prune_scale2d
                is_prune |= is_too_big
            n_prune = int(is_prune.sum().item())
            if n_prune:
                if self.on_prune is not None:
                    self.on_prune(is_prune)
                from gsplat.strategy.ops import remove

                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
            if policy.verbose:
                print(f"Step {step}: {n_duplicate} duplicated, {n_split} split, {n_prune} pruned; {len(params['means'])} Gaussian(s).")
            state["grad2d"].zero_()
            state["count"].zero_()
            if policy.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
        if step % policy.reset_every == 0:
            # Kept as an imported gsplat operation through DefaultStrategy's module.
            from gsplat.strategy.ops import reset_opa

            reset_opa(params=params, optimizers=optimizers, state=state, value=policy.prune_opa * 2.0)
