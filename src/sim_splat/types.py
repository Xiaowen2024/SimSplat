from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Camera:
    """One calibrated camera expressed in the project's c2w convention.

    ``model`` and ``distortion_params`` preserve source calibration. They must
    not be silently discarded because a scene happens to use rectified views.
    """

    c2w: torch.Tensor  # [4, 4], camera -> world extrinsics
    K: torch.Tensor  # [3, 3], intrinsics at this image's pixel grid
    width: int
    height: int
    image_id: int
    model: str = "PINHOLE"  # Original model, read per camera from calibration.
    distortion_params: torch.Tensor | None = None  # Model-native lens coefficients.
    is_rectified: bool = False  # True only when RGB pixels were explicitly rectified.
    sensor_id: int = 0  # Physical camera index; selects per-sensor appearance.
    timestamp: float = 0.0  # Seconds; used for actor pose interpolation and reporting.


@dataclass(frozen=True)
class DataparserOutputs:
    """Immutable metadata produced once by a scene parser."""

    image_paths: list[Path]
    cameras: list[Camera]
    train_indices: list[int]
    eval_indices: list[int]
    points_xyz: torch.Tensor  # [N, 3] sparse point coordinates in world space
    points_rgb: torch.Tensor  # [N, 3] sparse point colors
    mask_paths: list[Path | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """A full-image batch consumed by a Gaussian-splatting model."""

    image: torch.Tensor
    static_mask: torch.Tensor
    camera: Camera
    image_name: str
    # Optional sparse LiDAR depth [N, 3]: pixel-centre u, v and camera-z depth (metres).
    lidar: torch.Tensor | None = None
