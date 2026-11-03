from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sim_splat.config import SceneConfig
from sim_splat.data.dataparsers.base_dataparser import DataParser
from sim_splat.types import Camera, DataparserOutputs


# Number of non-distortion parameters at the front of COLMAP's camera.params.
# The remaining values are retained verbatim in Camera.distortion_params.
_INTRINSIC_PARAM_COUNTS: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 3,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 4,
    "FULL_OPENCV": 4,
    "FOV": 4,
    "SIMPLE_RADIAL_FISHEYE": 3,
    "RADIAL_FISHEYE": 3,
    "THIN_PRISM_FISHEYE": 4,
}


def _as_dict(map_like: Any) -> dict[int, Any]:
    """Turn a pycolmap map view into a normal integer-keyed dictionary."""

    return {int(key): value for key, value in map_like.items()}


def _camera_model_name(camera: Any) -> str:
    """Handle the small pycolmap naming differences across releases."""

    model = getattr(camera, "model_name", None)
    if callable(model):
        model = model()
    if model is None:
        model = getattr(camera, "model", None)
    if model is None:
        raise ValueError("COLMAP camera has no model/model_name")
    return str(getattr(model, "name", model)).replace("CameraModelId.", "")


def _matrix(value: Any, name: str) -> np.ndarray:
    """Extract a finite 3x3 or 4x4 matrix from a pycolmap object."""

    if callable(value):
        value = value()
    matrix = getattr(value, "matrix", value)
    if callable(matrix):
        matrix = matrix()
    result = np.asarray(matrix, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _distortion_params(camera: Any, model: str) -> torch.Tensor | None:
    """Preserve model-native lens coefficients without treating them as pinhole."""

    try:
        intrinsic_count = _INTRINSIC_PARAM_COUNTS[model]
    except KeyError as error:
        raise ValueError(
            f"Unsupported COLMAP camera model {model!r}. Add its parameter layout "
            "or explicitly run a calibrated rectification preprocessor."
        ) from error

    params = np.asarray(camera.params, dtype=np.float32)
    if params.ndim != 1 or len(params) < intrinsic_count:
        raise ValueError(f"Camera model {model} has malformed params: shape={params.shape}")
    coefficients = params[intrinsic_count:]
    return torch.from_numpy(coefficients.copy()) if len(coefficients) else None


def _mask_path(masks_dir: Path, image_name: str) -> Path:
    """Find a mask with the same relative stem as a COLMAP image."""

    relative_name = Path(image_name)
    candidates = [
        masks_dir / relative_name,
        masks_dir / relative_name.with_suffix(".png"),
        masks_dir / relative_name.with_suffix(".jpg"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No mask for {image_name}; tried: {attempted}")


@dataclass
class ColmapDataParser(DataParser):
    """Read a calibrated COLMAP reconstruction into SimSplat scene metadata.

    Raw distorted RGBs remain paired with their original per-camera model and
    distortion coefficients. This parser never labels a distorted image as
    pinhole; explicit pixel rectification belongs in a separate preprocessor.
    """

    config: SceneConfig

    def get_dataparser_outputs(self) -> DataparserOutputs:
        if self.config.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.config.eval_interval}")
        if self.config.calibration_policy == "rectify":
            raise NotImplementedError(
                "Rectification requires materializing a new RGB/K pair. Run the "
                "calibrated rectification preprocessor first, then parse its output."
            )

        scene_root = self.config.data.expanduser()
        images_dir = scene_root / self.config.images_path
        colmap_dir = scene_root / self.config.colmap_path
        masks_dir = scene_root / self.config.masks_path if self.config.masks_path else None
        for label, path in (("images", images_dir), ("COLMAP", colmap_dir), ("masks", masks_dir)):
            if path is not None and not path.is_dir():
                raise FileNotFoundError(f"{label} directory does not exist: {path}")

        try:
            import pycolmap
        except ImportError as error:
            raise ImportError(
                "ColmapDataParser requires pycolmap. Install project dependencies "
                "with `pip install -e .`."
            ) from error

        reconstruction = pycolmap.Reconstruction(str(colmap_dir))
        cameras_by_id = _as_dict(reconstruction.cameras)
        images_by_id = _as_dict(reconstruction.images)
        registered_ids = [int(image_id) for image_id in reconstruction.reg_image_ids()]
        if not registered_ids:
            raise ValueError(f"No registered images in COLMAP reconstruction: {colmap_dir}")

        # Filename ordering makes split assignment reproducible across pycolmap versions.
        registered_ids.sort(key=lambda image_id: str(images_by_id[image_id].name))

        image_paths: list[Path] = []
        cameras: list[Camera] = []
        mask_paths: list[Path | None] = []
        model_counts: Counter[str] = Counter()

        for image_id in registered_ids:
            colmap_image = images_by_id[image_id]
            image_name = str(colmap_image.name)
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"COLMAP image {image_name!r} is missing from {images_dir}")

            camera_id = int(colmap_image.camera_id)
            if camera_id not in cameras_by_id:
                raise ValueError(f"Image {image_name} references missing camera ID {camera_id}")
            colmap_camera = cameras_by_id[camera_id]
            model = _camera_model_name(colmap_camera)
            distortion = _distortion_params(colmap_camera, model)

            K = _matrix(getattr(colmap_camera, "calibration_matrix"), "camera calibration matrix")
            if K.shape != (3, 3):
                raise ValueError(f"Camera {camera_id} produced K with shape {K.shape}, expected (3, 3)")

            w2c = _matrix(getattr(colmap_image, "cam_from_world"), "COLMAP world-to-camera pose")
            if w2c.shape == (3, 4):
                w2c_h = np.eye(4, dtype=np.float32)
                w2c_h[:3, :4] = w2c
                w2c = w2c_h
            if w2c.shape != (4, 4):
                raise ValueError(f"Image {image_name} produced pose shape {w2c.shape}, expected (3, 4) or (4, 4)")
            c2w = np.linalg.inv(w2c).astype(np.float32)

            with Image.open(image_path) as rgb:
                image_width, image_height = rgb.size
            calibration_width = int(colmap_camera.width)
            calibration_height = int(colmap_camera.height)
            if (image_width, image_height) != (calibration_width, calibration_height):
                raise ValueError(
                    f"RGB/calibration resolution mismatch for {image_path}: RGB is "
                    f"{image_width}x{image_height}, calibration is "
                    f"{calibration_width}x{calibration_height}. Use calibration that "
                    "matches this image scale or explicitly generate a resized calibration."
                )

            if masks_dir is not None:
                mask_path = _mask_path(masks_dir, image_name)
                with Image.open(mask_path) as mask:
                    if mask.size != (image_width, image_height):
                        raise ValueError(
                            f"Mask/RGB resolution mismatch for {mask_path}: mask is "
                            f"{mask.width}x{mask.height}, RGB is "
                            f"{image_width}x{image_height}"
                        )
                mask_paths.append(mask_path)

            image_paths.append(image_path)
            cameras.append(
                Camera(
                    c2w=torch.from_numpy(c2w),
                    K=torch.from_numpy(K),
                    width=image_width,
                    height=image_height,
                    image_id=image_id,
                    model=model,
                    distortion_params=distortion,
                    is_rectified=False,
                )
            )
            model_counts[model] += 1

        points_by_id = _as_dict(reconstruction.points3D)
        if not points_by_id:
            raise ValueError(f"No sparse 3D points in COLMAP reconstruction: {colmap_dir}")
        point_ids = sorted(points_by_id)
        points_xyz = torch.from_numpy(
            np.asarray([points_by_id[point_id].xyz for point_id in point_ids], dtype=np.float32)
        )
        points_rgb = torch.from_numpy(
            np.asarray([points_by_id[point_id].color for point_id in point_ids], dtype=np.float32) / 255.0
        )
        raw_point_count = len(points_xyz)
        if self.config.max_init_points > 0 and raw_point_count > self.config.max_init_points:
            # Initialization need not retain every SfM point: deterministic
            # subsampling keeps memory bounded while preserving broad coverage.
            generator = torch.Generator(device="cpu").manual_seed(42)
            selected = torch.randperm(raw_point_count, generator=generator)[: self.config.max_init_points]
            points_xyz, points_rgb = points_xyz[selected], points_rgb[selected]

        world_normalization: dict[str, Any] = {"enabled": False}
        if self.config.normalize_world_space:
            # Use the camera trajectory, rather than sparse-point bounds: SfM
            # clouds may contain remote outliers but the reconstructed views
            # define the observed scene's meaningful scale.
            positions = torch.stack([camera.c2w[:3, 3] for camera in cameras]).to(dtype=torch.float32)
            center = positions.mean(dim=0)
            scale = torch.linalg.vector_norm(positions - center, dim=-1).max().clamp_min(1e-6)
            normalized_cameras: list[Camera] = []
            for camera in cameras:
                c2w = camera.c2w.clone()
                c2w[:3, 3] = (c2w[:3, 3] - center) / scale
                normalized_cameras.append(
                    Camera(
                        c2w=c2w,
                        K=camera.K,
                        width=camera.width,
                        height=camera.height,
                        image_id=camera.image_id,
                        model=camera.model,
                        distortion_params=camera.distortion_params,
                        is_rectified=camera.is_rectified,
                    )
                )
            cameras = normalized_cameras
            points_xyz = (points_xyz - center) / scale
            world_normalization = {
                "enabled": True,
                "center": center.tolist(),
                "scale": float(scale.item()),
            }

        eval_indices = list(range(0, len(image_paths), self.config.eval_interval))
        eval_set = set(eval_indices)
        train_indices = [index for index in range(len(image_paths)) if index not in eval_set]
        if not train_indices or not eval_indices:
            raise ValueError(
                f"Invalid eval_interval={self.config.eval_interval} for {len(image_paths)} registered images"
            )

        return DataparserOutputs(
            image_paths=image_paths,
            cameras=cameras,
            train_indices=train_indices,
            eval_indices=eval_indices,
            points_xyz=points_xyz,
            points_rgb=points_rgb,
            mask_paths=mask_paths,
            metadata={
                "source": "colmap",
                "colmap_dir": str(colmap_dir),
                "images_dir": str(images_dir),
                "num_registered_images": len(image_paths),
                "num_sparse_points_raw": raw_point_count,
                "num_sparse_points_initialized": len(points_xyz),
                "camera_models": dict(sorted(model_counts.items())),
                "calibration_policy": self.config.calibration_policy,
                "is_rectified": False,
                "world_normalization": world_normalization,
            },
        )
