from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sim_splat.types import Batch, DataparserOutputs


class InputDataset:
    """Generic full-image dataset; scene interpretation stays in the parser."""

    def __init__(self, outputs: DataparserOutputs, cache_images: bool = True) -> None:
        self.outputs = outputs
        # Decoded uint8 RGB per index; JPEG/PNG decoding otherwise dominates
        # a multi-camera training step on the CPU.
        self.cache_images = cache_images
        self._rgb_cache: dict[int, torch.Tensor] = {}
        if len(outputs.mask_paths) not in (0, len(outputs.image_paths)):
            raise ValueError("mask_paths must be empty or align one-to-one with image_paths")

    def __len__(self) -> int:
        return len(self.outputs.image_paths)

    @staticmethod
    def _load_rgb_uint8(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8))

    def _load_rgb(self, index: int) -> torch.Tensor:
        cached = self._rgb_cache.get(index)
        if cached is None:
            cached = self._load_rgb_uint8(self.outputs.image_paths[index])
            if self.cache_images:
                self._rgb_cache[index] = cached
        return cached.to(torch.float32) / 255.0

    @staticmethod
    def _load_mask(path: Path | None, height: int, width: int) -> torch.Tensor:
        if path is None:
            return torch.ones((height, width, 1), dtype=torch.float32)
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if mask.shape != (height, width):
            raise ValueError(f"Mask {path} has {mask.shape}; expected {(height, width)}")
        return torch.from_numpy((mask > 0.5).astype(np.float32))[..., None]

    def __getitem__(self, index: int) -> Batch:
        path = self.outputs.image_paths[index]
        camera = self.outputs.cameras[index]
        image = self._load_rgb(index)
        height, width = image.shape[:2]
        if (height, width) != (camera.height, camera.width):
            raise ValueError(
                f"Image {path} is {width}x{height}; camera expects {camera.width}x{camera.height}"
            )
        mask_path = self.outputs.mask_paths[index] if self.outputs.mask_paths else None
        return Batch(image=image, static_mask=self._load_mask(mask_path, height, width), camera=camera, image_name=path.name)
