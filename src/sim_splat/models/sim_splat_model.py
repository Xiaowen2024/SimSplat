from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from sim_splat.config import ModelConfig
from sim_splat.models.base_model import Model
from sim_splat.types import Batch, Camera, DataparserOutputs


_SH_C0 = 0.28209479177387814


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to the degree-zero SH coefficient."""

    return (rgb - 0.5) / _SH_C0


def _initial_log_scales(points_xyz: torch.Tensor, k: int, multiplier: float) -> torch.Tensor:
    """Initialize isotropic Gaussian scale from local sparse-point spacing.

    The nearest-neighbor query deliberately runs through a CPU KD-tree rather
    than materializing an N×N torch.cdist matrix on the GPU.
    """

    point_count = points_xyz.shape[0]
    if point_count < 2:
        raise ValueError("At least two sparse COLMAP points are required to initialize Gaussians")
    if k < 2:
        raise ValueError(f"init_knn_k must be at least 2, got {k}")
    if multiplier <= 0:
        raise ValueError(f"init_scale_multiplier must be positive, got {multiplier}")

    effective_k = min(k, point_count)
    points_np = points_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    distances, _ = cKDTree(points_np).query(points_np, k=effective_k, workers=-1)
    if effective_k == 2:
        distances = distances[:, None]
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float32)
    rms_distance = np.sqrt(np.mean(neighbor_distances**2, axis=1))
    # Duplicate COLMAP points can yield zero distance; keep scale positive in log space.
    scalar_scales = torch.from_numpy(np.clip(rms_distance * multiplier, 1e-6, None))
    return torch.log(scalar_scales).unsqueeze(-1).repeat(1, 3)


def _padded_coefficients(coefficients: torch.Tensor, size: int) -> torch.Tensor:
    """Return a zero-padded, single-camera coefficient tensor for gsplat."""

    if coefficients.numel() > size:
        raise ValueError(f"Expected at most {size} distortion coefficients, got {coefficients.numel()}")
    padded = torch.zeros((1, size), dtype=coefficients.dtype, device=coefficients.device)
    padded[0, : coefficients.numel()] = coefficients
    return padded


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    weights = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    weights = weights / weights.sum()
    return (weights[:, None] * weights[None, :]).expand(channels, 1, size, size).contiguous()


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM (11x11 Gaussian window, sigma 1.5) for HWC RGB in [0, 1].

    This is the window used by 3DGS, Splatfacto and pytorch-msssim. The former
    11x11 *box* window was a measured training failure: on full-resolution
    Gaussian renders Adam drove even an SSIM-only objective away from the
    target (1-SSIM rose from 0.39 to 0.70) although its autograd was exact.
    Check any cheaper window with a single-image overfit before adopting it.
    """

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(
            "SSIM expects aligned HWC RGB tensors; got "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    height, width, _ = prediction.shape
    if height < 1 or width < 1:
        raise ValueError("SSIM requires non-empty images")
    # Keep the reference window for normal images while supporting tiny tests.
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    window = _gaussian_window(window_size, 1.5, 3, prediction.device, prediction.dtype)
    x = prediction.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)

    def blur(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value, window, groups=3)

    mean_x, mean_y = blur(x), blur(y)
    variance_x = blur(x * x) - mean_x.square()
    variance_y = blur(y * y) - mean_y.square()
    covariance = blur(x * y) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return ssim.mean()


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 normalized by supervised pixels, not by the whole image area."""

    denominator = mask.sum() * prediction.shape[-1]
    if denominator.detach().item() <= 0:
        raise ValueError("static_mask contains no supervised pixels")
    return ((prediction - target).abs() * mask).sum() / denominator


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum() * prediction.shape[-1]
    if denominator.detach().item() <= 0:
        raise ValueError("static_mask contains no supervised pixels")
    return ((prediction - target).square() * mask).sum() / denominator


def _psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def _rotation_to_quat_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert proper rotation matrices [..., 3, 3] to wxyz quaternions [..., 4]."""

    m = rotation
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    # Four algebraically equivalent candidates; pick the numerically largest.
    candidates = torch.stack(
        (
            torch.stack((1 + m00 + m11 + m22, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1),
            torch.stack((m[..., 2, 1] - m[..., 1, 2], 1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), -1),
            torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1]), -1),
            torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22), -1),
        ),
        dim=-2,
    )
    diagonal = torch.stack((m00 + m11 + m22, m00, m11, m22), dim=-1)
    best = torch.argmax(diagonal, dim=-1)
    chosen = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    quaternion = F.normalize(chosen, dim=-1)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def _quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product, broadcasting a [4] world pose over local quaternions."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack((lw * rw - lx * rx - ly * ry - lz * rz,
                        lw * rx + lx * rw + ly * rz - lz * ry,
                        lw * ry - lx * rz + ly * rw + lz * rx,
                        lw * rz + lx * ry - ly * rx + lz * rw), dim=-1)


class SkyEnvironmentMap(nn.Module):
    """Learned equirectangular radiance at infinity, indexed by world ray direction.

    LiDAR never returns from the sky, so a pure Gaussian scene renders it as
    the (black) background.  On nuScenes the overcast sky covers roughly a
    third of each frame, which alone bounded the old single-frame PSNR near 8.
    The map is composited only through the Gaussian transmittance.
    """

    def __init__(self, height: int, width: int, initial_rgb: float) -> None:
        super().__init__()
        logit = math.log(initial_rgb / (1.0 - initial_rgb))
        self.texture = nn.Parameter(torch.full((1, 3, height, width), logit, dtype=torch.float32))

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Map unit world directions [H, W, 3] (z up) to RGB [H, W, 3]."""

        longitude = torch.atan2(directions[..., 1], directions[..., 0]) / math.pi
        latitude = torch.asin(directions[..., 2].clamp(-1.0, 1.0)) / (math.pi / 2.0)
        grid = torch.stack((longitude, -latitude), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(self.texture, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return torch.sigmoid(sampled[0].permute(1, 2, 0))


class SimSplatModel(Model):
    """Standalone 3DGS scene graph backed by gsplat.

    One vectorized ParameterDict holds every Gaussian.  ``point_track_indices``
    assigns each row either to the static world field (-1) or to one tracked
    actor whose rows live in that actor's canonical box frame.  Rendering a
    frame transforms actor rows by their per-frame object-to-world pose; hiding
    an actor zeroes its opacity *without* changing row indices, so gsplat's
    packed ``gaussian_ids`` (and therefore densification statistics) always
    refer to the same rows as the optimizer state.
    """

    def __init__(self, config: ModelConfig, scene: DataparserOutputs, device: str) -> None:
        super().__init__()
        self.config = config
        self.scene = scene
        self.device = torch.device(device)
        self.track_tokens: list[str] = list(scene.metadata.get("dynamic_track_tokens", ()))
        track_indices = scene.metadata.get("dynamic_point_track_indices")
        transforms = scene.metadata.get("dynamic_object_to_world")
        visible = scene.metadata.get("dynamic_track_visible")
        point_count = scene.points_xyz.shape[0]
        if track_indices is None:
            track_indices = torch.full((point_count,), -1, dtype=torch.long)
        if transforms is None:
            transforms = torch.empty((len(scene.image_paths), 0, 4, 4), dtype=torch.float32)
        if visible is None:
            visible = torch.empty((len(scene.image_paths), 0), dtype=torch.bool)
        if track_indices.shape != (point_count,):
            raise ValueError("dynamic_point_track_indices must align with points_xyz")
        if transforms.shape[:2] != (len(scene.image_paths), len(self.track_tokens)) or transforms.shape[-2:] != (4, 4):
            raise ValueError("dynamic_object_to_world must be [frames, tracks, 4, 4]")
        self.register_buffer("point_track_indices", track_indices.to(dtype=torch.long, device=self.device))
        self.register_buffer("object_to_world", transforms.to(dtype=torch.float32, device=self.device))
        self.register_buffer("track_visible", visible.to(dtype=torch.bool, device=self.device))
        self.hidden_track_ids: set[str] = set()
        self.set_hidden_tracks(config.hidden_track_ids)
        self.active_sh_degree = config.sh_degree
        self.populate_gaussians()

        # Non-Gaussian parameters are kept outside gauss_params because gsplat
        # strategies require a one-to-one params/optimizers key match.
        self.sky = SkyEnvironmentMap(*config.sky_resolution, config.sky_initial_rgb) if config.sky else None
        sensor_count = max((camera.sensor_id for camera in scene.cameras), default=0) + 1
        if config.appearance_affine:
            identity = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1)
            self.appearance = nn.Parameter(identity.repeat(sensor_count, 1, 1))
        else:
            self.appearance = None
        self.to(self.device)

    # ------------------------------------------------------------------
    # Scene-graph editing
    # ------------------------------------------------------------------
    def set_hidden_tracks(self, tokens: Any) -> None:
        """Hide actors by full track token or unique token prefix."""

        resolved: set[str] = set()
        for token in tokens:
            matches = [candidate for candidate in self.track_tokens if candidate.startswith(token)]
            if len(matches) != 1:
                raise ValueError(f"Track id {token!r} matched {len(matches)} tracks")
            resolved.add(matches[0])
        self.hidden_track_ids = resolved

    def _hidden_track_mask(self) -> torch.Tensor:
        mask = torch.zeros((len(self.track_tokens),), dtype=torch.bool, device=self.device)
        for index, token in enumerate(self.track_tokens):
            if token in self.hidden_track_ids:
                mask[index] = True
        return mask

    @torch.no_grad()
    def add_actor(
        self,
        token: str,
        gaussians: dict[str, torch.Tensor],
        object_to_world: torch.Tensor,
        visible: torch.Tensor,
    ) -> int:
        """Append an externally authored actor (e.g. a simulator asset).

        ``gaussians`` uses this model's parameter names in the actor's local
        frame.  ``object_to_world`` [frames, 4, 4] and ``visible`` [frames] are
        the per-frame poses; a physics engine can overwrite them every tick via
        ``object_to_world[:, track_index]`` without touching any Gaussian.
        """

        if token in self.track_tokens:
            raise ValueError(f"Actor {token!r} already exists")
        frames = self.object_to_world.shape[0]
        if object_to_world.shape != (frames, 4, 4) or visible.shape != (frames,):
            raise ValueError("Actor poses must be [frames, 4, 4] with a [frames] visibility vector")
        track_index = len(self.track_tokens)
        self.track_tokens.append(token)
        self.object_to_world = torch.cat((self.object_to_world, object_to_world.to(self.object_to_world)[:, None]), dim=1)
        self.track_visible = torch.cat((self.track_visible, visible.to(self.track_visible)[:, None]), dim=1)
        count = gaussians["means"].shape[0]
        self.gauss_params = nn.ParameterDict(
            {
                name: nn.Parameter(torch.cat((parameter.detach(), gaussians[name].to(parameter))))
                for name, parameter in self.gauss_params.items()
            }
        )
        owners = torch.full((count,), track_index, dtype=torch.long, device=self.device)
        self.point_track_indices = torch.cat((self.point_track_indices, owners))
        return track_index

    def _composed_gaussians(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pose actor rows into the world for one frame.

        Returns world means, world quaternions, a per-row visibility mask, and
        per-row local->world rotations (None for a static-only scene) used to
        evaluate view-dependent colour in each actor's own frame.
        """
        means = self.gauss_params["means"]
        quats = self.gauss_params["quats"]
        owners = self.point_track_indices
        dynamic = owners >= 0
        if not self.track_tokens or not bool(dynamic.any()):
            return means, quats, torch.ones_like(owners, dtype=torch.bool), None
        if not 0 <= frame_index < self.object_to_world.shape[0]:
            raise IndexError(f"Unknown frame index {frame_index}")
        transforms = self.object_to_world[frame_index].to(dtype=means.dtype)
        active_tracks = self.track_visible[frame_index] & ~self._hidden_track_mask()
        owner_rows = owners[dynamic]
        rotations = transforms[owner_rows, :3, :3]
        world_means = means.clone()
        world_means[dynamic] = torch.einsum("nij,nj->ni", rotations, means[dynamic]) + transforms[owner_rows, :3, 3]
        track_quats = _rotation_to_quat_wxyz(transforms[:, :3, :3])
        world_quats = quats.clone()
        world_quats[dynamic] = _quat_multiply_wxyz(track_quats[owner_rows], F.normalize(quats[dynamic], dim=-1))
        visible = torch.ones_like(owners, dtype=torch.bool)
        visible[dynamic] = active_tracks[owner_rows]
        local_to_world = torch.eye(3, device=means.device, dtype=means.dtype).expand(means.shape[0], 3, 3).clone()
        local_to_world[dynamic] = rotations
        return world_means, world_quats, visible, local_to_world

    @torch.no_grad()
    def duplicate_track_owners(self, mask: torch.Tensor) -> None:
        """Apply gsplat's duplicate row operation to immutable track metadata."""

        if mask.shape != self.point_track_indices.shape:
            raise ValueError("Duplicate mask must align with per-Gaussian ownership")
        self.point_track_indices = torch.cat((self.point_track_indices, self.point_track_indices[mask].clone()))

    @torch.no_grad()
    def split_track_owners(self, mask: torch.Tensor) -> None:
        """Apply gsplat's split row operation to immutable track metadata.

        gsplat's split removes each selected row and appends two children, so
        ownership is filtered and the selected owners are appended twice.
        """

        if mask.shape != self.point_track_indices.shape:
            raise ValueError("Split mask must align with per-Gaussian ownership")
        selected = self.point_track_indices[mask]
        self.point_track_indices = torch.cat((self.point_track_indices[~mask], selected, selected))

    @torch.no_grad()
    def prune_track_owners(self, mask: torch.Tensor) -> None:
        """Apply gsplat's stable prune operation to immutable track metadata."""

        if mask.shape != self.point_track_indices.shape:
            raise ValueError("Prune mask must align with per-Gaussian ownership")
        self.point_track_indices = self.point_track_indices[~mask].clone()

    def populate_gaussians(self) -> None:
        """Create the vectorized, trainable Gaussian state from seed points."""

        points_xyz = self.scene.points_xyz.detach().to(dtype=torch.float32, device="cpu")
        points_rgb = self.scene.points_rgb.detach().to(dtype=torch.float32, device="cpu")
        if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
            raise ValueError(f"points_xyz must have shape [N, 3], got {tuple(points_xyz.shape)}")
        if points_rgb.shape != points_xyz.shape:
            raise ValueError(
                f"points_rgb must match points_xyz shape {tuple(points_xyz.shape)}, got {tuple(points_rgb.shape)}"
            )
        if not torch.isfinite(points_xyz).all() or not torch.isfinite(points_rgb).all():
            raise ValueError("Seed points and colors must be finite")
        if torch.any(points_rgb < 0) or torch.any(points_rgb > 1):
            raise ValueError("points_rgb must be normalized to [0, 1]")
        if not 0.0 < self.config.init_opacity < 1.0:
            raise ValueError(f"init_opacity must lie in (0, 1), got {self.config.init_opacity}")

        if self.config.sh_degree < 0:
            raise ValueError(f"sh_degree must be non-negative, got {self.config.sh_degree}")
        point_count = points_xyz.shape[0]
        log_scales = _initial_log_scales(
            points_xyz,
            k=self.config.init_knn_k,
            multiplier=self.config.init_scale_multiplier,
        )
        # A data-adaptive ceiling is more portable than a hard metric value:
        # LiDAR and SfM scenes differ wildly in world coordinate scale. The
        # 95th percentile ignores isolated sparse-point outliers.
        initial_scale_ceiling = torch.quantile(torch.exp(log_scales[:, 0]), 0.95) * self.config.max_gauss_ratio
        max_log_scale = torch.log(initial_scale_ceiling.clamp_min(1e-6))
        # Clamp *before the first rasterization*: one distant sparse point can
        # otherwise cover the whole first image and corrupt its optimizer state.
        log_scales = log_scales.clamp(max=max_log_scale)
        self.register_buffer("max_log_scale", max_log_scale.to(self.device))
        sh_bases = (self.config.sh_degree + 1) ** 2

        quats = torch.zeros((point_count, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        opacity_logits = torch.full(
            (point_count, 1),
            fill_value=math.log(self.config.init_opacity / (1.0 - self.config.init_opacity)),
            dtype=torch.float32,
        )
        # Colour is always stored as SH (degree 0 = DC term); colours are
        # evaluated by this model, never by gsplat, so actors can use their
        # own local view directions.
        features_dc = _rgb_to_sh(points_rgb)
        features_rest = torch.zeros((point_count, sh_bases - 1, 3), dtype=torch.float32)

        self.gauss_params = nn.ParameterDict(
            {
                "means": nn.Parameter(points_xyz),
                "scales": nn.Parameter(log_scales),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(opacity_logits),
                "features_dc": nn.Parameter(features_dc),
                "features_rest": nn.Parameter(features_rest),
            }
        ).to(self.device)

    def extra_parameters(self) -> dict[str, nn.Parameter]:
        """Trainable non-Gaussian parameters (sky map, per-sensor colour)."""

        extras: dict[str, nn.Parameter] = {}
        if self.sky is not None:
            extras["sky"] = self.sky.texture
        if self.appearance is not None:
            extras["appearance"] = self.appearance
        return extras

    def set_step(self, step: int) -> None:
        """Standard 3DGS schedule: one more SH band every ``sh_degree_interval`` steps."""

        self.active_sh_degree = min(self.config.sh_degree, step // max(self.config.sh_degree_interval, 1))

    @torch.no_grad()
    def clamp_gaussian_parameters_(self) -> None:
        """Keep a few unstable splats from swallowing an entire sparse scene."""

        self.gauss_params["scales"].clamp_(min=math.log(1e-6), max=float(self.max_log_scale.item()))
        # Splitting may use a logit-space opacity transform.  In float32,
        # sigmoid(logit) can round to exactly one and a subsequent logit would
        # become +inf.  Keep alpha finite while still allowing opaque surfaces.
        self.gauss_params["opacities"].nan_to_num_(nan=0.0, posinf=6.0, neginf=-10.0).clamp_(min=-10.0, max=6.0)

    def _rasterizer_camera_kwargs(self, camera: Camera, dtype: torch.dtype) -> dict[str, Any]:
        """Map retained COLMAP calibration to gsplat's native camera arguments."""

        if camera.is_rectified:
            return {"camera_model": "pinhole"}

        model = camera.model.upper()
        coefficients = camera.distortion_params
        if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
            if coefficients is not None and coefficients.numel() != 0:
                raise ValueError(f"{model} must not carry distortion coefficients")
            return {"camera_model": "pinhole"}
        if coefficients is None:
            raise ValueError(f"Distorted camera {model} is missing distortion_params")
        coefficients = coefficients.to(device=self.device, dtype=dtype).flatten()

        if model == "SIMPLE_RADIAL":
            return {"camera_model": "pinhole", "radial_coeffs": _padded_coefficients(coefficients, 6)}
        if model == "RADIAL":
            return {"camera_model": "pinhole", "radial_coeffs": _padded_coefficients(coefficients, 6)}
        if model == "OPENCV":
            if coefficients.numel() != 4:
                raise ValueError(f"OPENCV expects [k1, k2, p1, p2], got {coefficients.numel()} values")
            return {
                "camera_model": "pinhole",
                "radial_coeffs": _padded_coefficients(coefficients[:2], 6),
                "tangential_coeffs": coefficients[2:].reshape(1, 2),
            }
        if model == "FULL_OPENCV":
            if coefficients.numel() != 8:
                raise ValueError(
                    "FULL_OPENCV expects [k1, k2, p1, p2, k3, k4, k5, k6], "
                    f"got {coefficients.numel()} values"
                )
            radial = torch.cat((coefficients[:2], coefficients[4:]))
            return {
                "camera_model": "pinhole",
                "radial_coeffs": radial.reshape(1, 6),
                "tangential_coeffs": coefficients[2:4].reshape(1, 2),
            }
        if model in {"SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"}:
            return {"camera_model": "fisheye", "radial_coeffs": _padded_coefficients(coefficients, 4)}

        raise NotImplementedError(
            f"COLMAP model {model} has no native gsplat adapter. Use an explicitly "
            "rectified RGB/K pair instead of rendering its distorted pixels as pinhole."
        )

    def _gaussian_colors(self, means: torch.Tensor, local_to_world: torch.Tensor | None, camera_center: torch.Tensor) -> torch.Tensor:
        """Evaluate SH colour with each row's view direction in its own frame."""

        from gsplat import spherical_harmonics

        coefficients = torch.cat((self.gauss_params["features_dc"].unsqueeze(1), self.gauss_params["features_rest"]), dim=1)
        degree = min(self.active_sh_degree, self.config.sh_degree)
        if degree == 0:
            return torch.clamp_min(coefficients[:, 0] * _SH_C0 + 0.5, 0.0)
        directions = means - camera_center
        if local_to_world is not None:
            # Rotate world view directions into each actor's canonical frame.
            directions = torch.einsum("nji,nj->ni", local_to_world, directions)
        bases = (degree + 1) ** 2
        return torch.clamp_min(spherical_harmonics(degree, directions, coefficients[:, :bases]) + 0.5, 0.0)

    def _camera_ray_directions(self, camera: Camera, c2w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(camera.height, device=self.device, dtype=c2w.dtype) + 0.5,
            torch.arange(camera.width, device=self.device, dtype=c2w.dtype) + 0.5,
            indexing="ij",
        )
        camera_dirs = torch.stack(((xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)), dim=-1)
        return F.normalize(camera_dirs @ c2w[:3, :3].T, dim=-1)

    def get_outputs(self, camera: Camera) -> dict[str, Any]:
        """Differentiably rasterize this model from one calibrated camera view."""

        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise ImportError("get_outputs requires the gsplat package") from error

        if camera.width <= 0 or camera.height <= 0:
            raise ValueError(f"Camera resolution must be positive, got {camera.width}x{camera.height}")
        if camera.c2w.shape != (4, 4):
            raise ValueError(f"camera.c2w must have shape (4, 4), got {tuple(camera.c2w.shape)}")
        if camera.K.shape != (3, 3):
            raise ValueError(f"camera.K must have shape (3, 3), got {tuple(camera.K.shape)}")

        means, quats, visible, local_to_world = self._composed_gaussians(camera.image_id)
        dtype = means.dtype
        c2w = camera.c2w.to(device=self.device, dtype=dtype)
        K = camera.K.to(device=self.device, dtype=dtype)
        colors = self._gaussian_colors(means, local_to_world, c2w[:3, 3])
        # Hidden or not-yet-visible actors keep their rows (and therefore
        # their gaussian_ids) but contribute nothing and receive no gradient.
        opacities = torch.sigmoid(self.gauss_params["opacities"]).squeeze(-1) * visible.to(dtype)

        render_colors, render_alphas, raster_info = rasterization(
            means=means,
            quats=quats,
            # The optimizer owns unconstrained log-scales and opacity logits;
            # gsplat takes physical scales and alpha values.
            scales=torch.exp(self.gauss_params["scales"]),
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=camera.width,
            height=camera.height,
            sh_degree=None,
            packed=True,
            render_mode="RGB+ED",
            **self._rasterizer_camera_kwargs(camera, dtype),
        )
        expected_color_shape = (1, camera.height, camera.width, 4)
        expected_alpha_shape = (1, camera.height, camera.width, 1)
        if render_colors.shape != expected_color_shape or render_alphas.shape != expected_alpha_shape:
            raise RuntimeError(
                "Unexpected gsplat render shape: "
                f"colors={tuple(render_colors.shape)}, alpha={tuple(render_alphas.shape)}, "
                f"expected {expected_color_shape} and {expected_alpha_shape}"
            )
        rgb = render_colors[0, ..., :3]
        alpha = render_alphas[0]
        sky_rgb = None
        if self.sky is not None:
            sky_rgb = self.sky(self._camera_ray_directions(camera, c2w, K))
            rgb = rgb + (1.0 - alpha) * sky_rgb
        if self.appearance is not None:
            if camera.sensor_id >= self.appearance.shape[0]:
                raise IndexError(f"No appearance transform for sensor {camera.sensor_id}")
            transform = self.appearance[camera.sensor_id]
            rgb = rgb @ transform[:, :3].T + transform[:, 3]
        return {
            "rgb": rgb.clamp(0.0, 1.0),
            "depth": render_colors[0, ..., 3:4],
            "alpha": alpha,
            "sky_rgb": sky_rgb,
            "raster_info": raster_info,
        }

    @staticmethod
    def _validate_supervision(outputs: dict[str, Any], batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "rgb" not in outputs:
            raise KeyError("Renderer outputs must contain an 'rgb' image")
        prediction = outputs["rgb"]
        target = batch.image
        mask = batch.static_mask
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("outputs['rgb'] must be a torch.Tensor")
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
            raise ValueError(
                "Rendered RGB and target must be aligned HWC RGB tensors; got "
                f"rendered={tuple(prediction.shape)}, target={tuple(target.shape)}"
            )
        if mask.shape != (*target.shape[:2], 1):
            raise ValueError(f"static_mask must have shape {(*target.shape[:2], 1)}, got {tuple(mask.shape)}")
        if prediction.device != target.device or prediction.device != mask.device:
            raise ValueError("Rendered RGB, target RGB, and static_mask must share one device")
        if torch.any(mask < 0) or torch.any(mask > 1):
            raise ValueError("static_mask must lie in [0, 1]")
        return prediction, target, mask

    def get_loss_dict(self, outputs: dict[str, Any], batch: Batch) -> dict[str, torch.Tensor]:
        """Return differentiable static-background reconstruction loss terms.

        Both terms are mask-normalized. An all-white mask exactly recovers the
        standard full-image L1 + SSIM objective; masked pixels contribute no
        photometric gradient. Masked SSIM currently zeroes ignored pixels in
        both images, which is a documented first-order approximation.
        """

        prediction, target, mask = self._validate_supervision(outputs, batch)
        if not 0.0 <= self.config.ssim_lambda <= 1.0:
            raise ValueError(f"ssim_lambda must lie in [0, 1], got {self.config.ssim_lambda}")
        if self.config.static_mask_loss_weight <= 0:
            raise ValueError("static_mask_loss_weight must be positive")

        static_l1 = _masked_l1(prediction, target, mask)
        masked_prediction = prediction * mask
        masked_target = target * mask
        static_ssim_loss = 1.0 - _ssim(masked_prediction, masked_target)
        weight = self.config.static_mask_loss_weight
        losses = {
            "loss_l1": weight * (1.0 - self.config.ssim_lambda) * static_l1,
            "loss_ssim": weight * self.config.ssim_lambda * static_ssim_loss,
        }
        if self.config.use_scale_regularization:
            excess = torch.relu(self.gauss_params["scales"] - self.max_log_scale)
            # A small soft penalty gives useful gradients before the hard cap
            # in Trainer clamps a post-step outlier.
            losses["loss_scale_reg"] = 0.01 * excess.square().mean()
        return losses

    def get_metrics_dict(self, outputs: dict[str, Any], batch: Batch) -> dict[str, float]:
        """Report full-image and simulation-static reconstruction quality."""

        prediction, target, mask = self._validate_supervision(outputs, batch)
        with torch.no_grad():
            full_mse = (prediction - target).square().mean()
            static_mse = _masked_mse(prediction, target, mask)
            masked_ssim = _ssim(prediction * mask, target * mask)
            return {
                "psnr": _psnr_from_mse(full_mse).item(),
                "static_psnr": _psnr_from_mse(static_mse).item(),
                "static_ssim": masked_ssim.item(),
                "static_pixel_fraction": mask.mean().item(),
                "gaussian_count": float(self.gauss_params["means"].shape[0]),
            }
