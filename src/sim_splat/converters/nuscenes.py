"""Convert a nuScenes scene into a SimSplat scene directory.

    python -m sim_splat.scripts.convert_nuscenes --config configs/convert/<x>.yaml --output <scene_dir>

This is a *converter*, not a parser: it runs once, and training reads the
generic scene directory (README.md, "Scene format").  It consumes the dataset's
supplied sensor calibration and ego poses.  Gaussians are seeded from keyframe LiDAR sweeps:
returns inside an annotated actor box are stored in that actor's canonical box
frame, everything else forms the static world field.  This is the
DrivingGaussian/Street-Gaussians style background/actor decomposition that
makes actors removable or replaceable at render time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import yaml

from sim_splat.data.scene_format import export_scene
from sim_splat.types import Camera, DataparserOutputs


@dataclass
class NuScenesConversionConfig:
    """Everything nuScenes-specific; none of it reaches the training config."""

    name: str
    data: Path  # nuScenes root containing <version>/, samples/, sweeps/
    scene_name: str
    camera_channels: tuple[str, ...] = ("CAM_FRONT",)
    nuscenes_version: str = "v1.0-mini"
    # Include 12 Hz non-keyframe images; actor poses are interpolated from
    # the 2 Hz keyframe annotations.
    use_sweeps: bool = False
    sweep_stride: int = 1
    # Explicit integer downscale; K is scaled with the cached RGB.
    image_downscale: int = 1
    # Hold out every Nth keyframe image (sweeps only ever add training views).
    eval_interval: int = 8
    # LiDAR seed preprocessing.
    max_init_points: int = 800_000
    lidar_voxel_size: float = 0.1
    object_voxel_size: float = 0.05
    ego_radius: float = 2.5
    lidar_seed_frame_indices: tuple[int, ...] | None = None
    # Annotated categories (prefix match) that become rigid actor tracks.
    dynamic_categories: tuple[str, ...] = ("vehicle.", "human.pedestrian.")
    box_margin: float = 0.15
    object_random_seeds: int = 300
    track_time_tolerance: float = 0.25
    # Bottom-row fraction showing the ego body, per channel (static mask).
    ego_mask_bottom_fraction: dict[str, float] = field(default_factory=dict)
    # Optional external static masks under <data>/<masks_path>.
    masks_path: str | None = None


def load_conversion_config(path: Path) -> NuScenesConversionConfig:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values["data"] = Path(values["data"])
    for key in ("camera_channels", "dynamic_categories", "lidar_seed_frame_indices"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return NuScenesConversionConfig(**values)


def _records(root: Path, version: str, name: str) -> dict[str, dict]:
    with (root / version / f"{name}.json").open(encoding="utf-8") as handle:
        return {record["token"]: record for record in json.load(handle)}


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Encountered zero-length nuScenes quaternion")
    return quaternion / norm


def _rotation(quaternion: list[float] | np.ndarray) -> np.ndarray:
    """Return a 3x3 matrix from nuScenes' scalar-first [w, x, y, z] quaternion."""

    w, x, y, z = _normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(rotation: list[float] | np.ndarray, translation: list[float] | np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation(rotation)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _record_transform(record: dict) -> np.ndarray:
    return _transform(record["rotation"], record["translation"])


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0, q1 = _normalize_quaternion(q0), _normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return _normalize_quaternion(q0 + alpha * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)


@dataclass
class ActorTrack:
    """Keyframe annotations of one instance, sorted by time (seconds)."""

    token: str
    category: str
    times: np.ndarray  # [K]
    translations: np.ndarray  # [K, 3]
    rotations: np.ndarray  # [K, 4] wxyz
    size_wlh: np.ndarray  # [3]; nuScenes stores width, length, height

    @property
    def half_extent_xyz(self) -> np.ndarray:
        """Half box extent in the box's local axes.

        nuScenes box frame: x forward along the *length*, y left along the
        *width*, z up.  ``size`` is [w, l, h], so local x pairs with l and
        local y with w.  (Comparing local x with w was a former bug: a
        4.5 m x 1.9 m car became a 1.9 m long, 4.5 m wide slab.)
        """

        width, length, height = self.size_wlh
        return np.array([length, width, height], dtype=np.float64) / 2.0

    def pose_at(self, time: float, tolerance: float) -> np.ndarray | None:
        """Object-to-world pose at ``time`` or None outside the annotated span."""

        if time < self.times[0] - tolerance or time > self.times[-1] + tolerance:
            return None
        if len(self.times) == 1 or time <= self.times[0]:
            return _transform(self.rotations[0], self.translations[0])
        if time >= self.times[-1]:
            return _transform(self.rotations[-1], self.translations[-1])
        upper = int(np.searchsorted(self.times, time, side="right"))
        lower = upper - 1
        alpha = (time - self.times[lower]) / max(self.times[upper] - self.times[lower], 1e-9)
        translation = (1 - alpha) * self.translations[lower] + alpha * self.translations[upper]
        return _transform(_slerp(self.rotations[lower], self.rotations[upper], alpha), translation)

    @property
    def displacement(self) -> float:
        return float(np.linalg.norm(self.translations[-1, :2] - self.translations[0, :2]))


def _voxel_downsample(points: np.ndarray, voxel: float, groups: np.ndarray | None = None) -> np.ndarray:
    """Indices of one representative point per occupied voxel (per group)."""

    if voxel <= 0 or len(points) == 0:
        return np.arange(len(points))
    keys = np.floor(points / voxel).astype(np.int64)
    if groups is not None:
        keys = np.concatenate((groups[:, None].astype(np.int64), keys), axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


@dataclass
class _ImageRecord:
    sample_data: dict
    channel_index: int
    time: float


@dataclass
class NuScenesConverter:
    """Multi-camera nuScenes reader with LiDAR seeds and actor tracks."""

    config: NuScenesConversionConfig

    def convert(self, output: Path, link_images: bool = False) -> Path:
        return export_scene(self.build(), output, name=self.config.name, link_images=link_images)

    def _image_path(self, root: Path, filename: str) -> tuple[Path, int, int]:
        """Return the RGB path at the configured scale, caching resized copies."""

        source = root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing extracted camera frame {source}")
        scale = self.config.image_downscale
        if scale < 1:
            raise ValueError(f"image_downscale must be a positive integer, got {scale}")
        if scale == 1:
            with Image.open(source) as image:
                return source, *image.size
        cached = (root / "sim_splat_cache" / f"x{scale}" / filename).with_suffix(".png")
        if not cached.is_file():
            cached.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                width, height = image.size
                if width % scale or height % scale:
                    raise ValueError(f"{source} size {image.size} is not divisible by downscale {scale}")
                image.convert("RGB").resize((width // scale, height // scale), Image.Resampling.BOX).save(cached)
        with Image.open(cached) as image:
            return cached, *image.size

    def _ego_mask(self, root: Path, channel: str, width: int, height: int) -> Path:
        """One cached static mask per channel: 0 on the ego body, 1 elsewhere."""

        fraction = self.config.ego_mask_bottom_fraction.get(channel, 0.0)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"ego mask fraction for {channel} must lie in [0, 1), got {fraction}")
        path = root / "sim_splat_cache" / "ego_masks" / f"{channel}_{width}x{height}_{fraction:.3f}.png"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.full((height, width), 255, dtype=np.uint8)
            mask[height - int(round(fraction * height)) :] = 0
            Image.fromarray(mask).save(path)
        return path

    def build(self) -> DataparserOutputs:
        config = self.config
        if not config.scene_name:
            raise ValueError("NuScenesDataParser requires scene_name in the scene YAML")
        root = config.data.expanduser()
        version = config.nuscenes_version
        if not (root / version).is_dir():
            raise FileNotFoundError(f"Expected nuScenes metadata under {root / version}")
        masks_dir = root / config.masks_path if config.masks_path else None
        if masks_dir is not None and not masks_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")
        channels = tuple(config.camera_channels)

        load = lambda name: _records(root, version, name)  # noqa: E731
        scenes, samples, sample_data = load("scene"), load("sample"), load("sample_data")
        annotations, instances, categories = load("sample_annotation"), load("instance"), load("category")
        calibrated, sensors, ego_poses = load("calibrated_sensor"), load("sensor"), load("ego_pose")
        selected = next((scene for scene in scenes.values() if scene["name"] == config.scene_name), None)
        if selected is None:
            available = ", ".join(sorted(scene["name"] for scene in scenes.values()))
            raise ValueError(f"Unknown nuScenes scene {config.scene_name!r}; available: {available}")

        def channel_of(record: dict) -> str:
            return sensors[calibrated[record["calibrated_sensor_token"]]["sensor_token"]]["channel"]

        def sensor_to_world(record: dict) -> np.ndarray:
            return _record_transform(ego_poses[record["ego_pose_token"]]) @ _record_transform(
                calibrated[record["calibrated_sensor_token"]]
            )

        keyframes: dict[str, dict[str, dict]] = {}
        for record in sample_data.values():
            if record["is_key_frame"]:
                keyframes.setdefault(record["sample_token"], {})[channel_of(record)] = record
        ordered_samples: list[dict] = []
        token = selected["first_sample_token"]
        while token:
            sample = samples[token]
            missing = [name for name in (*channels, "LIDAR_TOP") if name not in keyframes.get(token, {})]
            if missing:
                raise ValueError(f"Sample {token} lacks {missing}")
            ordered_samples.append(sample)
            token = sample["next"]
        if len(ordered_samples) < 3:
            raise ValueError("Need at least three keyframes for a train/eval split")
        sample_tokens = {sample["token"] for sample in ordered_samples}

        # ---- Camera images (keyframes, optionally 12 Hz sweeps) ----------
        image_records: list[_ImageRecord] = []
        for channel_index, channel in enumerate(channels):
            record = keyframes[ordered_samples[0]["token"]][channel]
            last_time = keyframes[ordered_samples[-1]["token"]][channel]["timestamp"]
            since_keyframe = 0
            while record is not None and record["timestamp"] <= last_time:
                since_keyframe = 0 if record["is_key_frame"] else since_keyframe + 1
                keep = record["is_key_frame"] or (
                    config.use_sweeps and since_keyframe % max(config.sweep_stride, 1) == 0
                )
                if keep:
                    image_records.append(_ImageRecord(record, channel_index, record["timestamp"] * 1e-6))
                record = sample_data[record["next"]] if record["next"] else None
        image_records.sort(key=lambda item: (item.time, item.channel_index))

        # ---- Actor tracks from keyframe annotations -----------------------
        raw_tracks: dict[str, list[tuple[float, dict]]] = {}
        for annotation in annotations.values():
            if annotation["sample_token"] not in sample_tokens:
                continue
            category = categories[instances[annotation["instance_token"]]["category_token"]]["name"]
            if not any(category.startswith(prefix) for prefix in config.dynamic_categories):
                continue
            time = samples[annotation["sample_token"]]["timestamp"] * 1e-6
            raw_tracks.setdefault(annotation["instance_token"], []).append((time, annotation))
        tracks: list[ActorTrack] = []
        for instance_token, items in sorted(raw_tracks.items(), key=lambda item: min(time for time, _ in item[1])):
            items.sort(key=lambda item: item[0])
            category = categories[instances[instance_token]["category_token"]]["name"]
            tracks.append(
                ActorTrack(
                    token=instance_token,
                    category=category,
                    times=np.array([time for time, _ in items]),
                    translations=np.array([annotation["translation"] for _, annotation in items], dtype=np.float64),
                    rotations=np.array([annotation["rotation"] for _, annotation in items], dtype=np.float64),
                    size_wlh=np.asarray(items[0][1]["size"], dtype=np.float64),
                )
            )

        # Recentre the world on the camera trajectory.  float32 was measured to
        # be adequate at nuScenes' ~2 km map offsets, but a local origin keeps
        # exported assets and simulator poses in convenient coordinates.
        camera_to_world = [sensor_to_world(item.sample_data) for item in image_records]
        origin = np.mean([pose[:3, 3] for pose in camera_to_world], axis=0)

        # ---- Cameras -----------------------------------------------------
        scale = config.image_downscale
        unknown = set(config.ego_mask_bottom_fraction) - set(channels)
        if unknown:
            raise ValueError(f"ego_mask_bottom_fraction names unselected channels {sorted(unknown)}")
        if masks_dir is not None and config.ego_mask_bottom_fraction:
            raise ValueError("Combine ego masks into masks_path offline; both sources are not merged here")
        cameras: list[Camera] = []
        image_paths: list[Path] = []
        mask_paths: list[Path | None] = []
        for index, (item, c2w) in enumerate(zip(image_records, camera_to_world)):
            path, width, height = self._image_path(root, item.sample_data["filename"])
            K = np.asarray(calibrated[item.sample_data["calibrated_sensor_token"]]["camera_intrinsic"], dtype=np.float64)
            if K.shape != (3, 3):
                raise ValueError(f"Malformed camera intrinsic matrix: {K.shape}")
            K = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ K
            local_c2w = c2w.copy()
            local_c2w[:3, 3] -= origin
            image_paths.append(path)
            cameras.append(
                Camera(
                    c2w=torch.from_numpy(local_c2w.astype(np.float32)),
                    K=torch.from_numpy(K.astype(np.float32)),
                    width=width,
                    height=height,
                    image_id=index,
                    model="PINHOLE",
                    is_rectified=True,
                    sensor_id=item.channel_index,
                    timestamp=item.time,
                )
            )
            if config.ego_mask_bottom_fraction:
                mask_paths.append(self._ego_mask(root, channels[item.channel_index], width, height))
            elif masks_dir is not None:
                relative = Path(item.sample_data["filename"])
                if relative.parts and relative.parts[0] in ("samples", "sweeps"):
                    relative = Path(*relative.parts[1:])
                mask_path = (masks_dir / relative).with_suffix(".png")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing nuScenes static mask {mask_path}")
                with Image.open(mask_path) as mask:
                    if mask.size != (width, height):
                        raise ValueError(f"Mask {mask_path} is {mask.size}; expected {(width, height)}")
                mask_paths.append(mask_path)

        transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (len(cameras), len(tracks), 4, 4)).copy()
        visible_tracks = np.zeros((len(cameras), len(tracks)), dtype=np.bool_)
        for frame_index, camera in enumerate(cameras):
            for track_index, track in enumerate(tracks):
                pose = track.pose_at(camera.timestamp, config.track_time_tolerance)
                if pose is not None:
                    pose[:3, 3] -= origin
                    transforms[frame_index, track_index] = pose
                    visible_tracks[frame_index, track_index] = True

        # ---- LiDAR seeds -------------------------------------------------
        seed_xyz, seed_rgb, seed_owner = self._lidar_seeds(
            root, ordered_samples, keyframes, channels, tracks, sensor_to_world, origin, calibrated
        )
        if len(seed_xyz) == 0:
            raise ValueError("No LiDAR points projected into the selected cameras")

        # Hold out every Nth *keyframe* image.  Sweeps only ever add training
        # views, so keyframe-only and sweep runs are scored on identical images.
        keyframe_images = [index for index, item in enumerate(image_records) if item.sample_data["is_key_frame"]]
        eval_indices = keyframe_images[:: config.eval_interval]
        eval_set = set(eval_indices)
        train_indices = [index for index in range(len(image_paths)) if index not in eval_set]
        return DataparserOutputs(
            image_paths=image_paths,
            cameras=cameras,
            train_indices=train_indices,
            eval_indices=eval_indices,
            points_xyz=torch.from_numpy(seed_xyz.astype(np.float32)),
            points_rgb=torch.from_numpy(seed_rgb.astype(np.float32)),
            mask_paths=mask_paths,
            metadata={
                "source": "nuscenes",
                "scene_name": config.scene_name,
                "camera_channels": list(channels),
                "num_keyframes": len(ordered_samples),
                "num_lidar_seed_points": int(len(seed_xyz)),
                "world_origin": origin.tolist(),
                "dynamic_track_tokens": [track.token for track in tracks],
                "dynamic_track_categories": [track.category for track in tracks],
                "dynamic_track_displacement": [track.displacement for track in tracks],
                # Full box extent along the box's local x (length), y (width), z.
                "dynamic_track_box_size_xyz": [(2.0 * track.half_extent_xyz).tolist() for track in tracks],
                "dynamic_point_track_indices": torch.from_numpy(seed_owner),
                "dynamic_object_to_world": torch.from_numpy(transforms),
                "dynamic_track_visible": torch.from_numpy(visible_tracks),
            },
        )

    def _lidar_seeds(
        self,
        root: Path,
        ordered_samples: list[dict],
        keyframes: dict[str, dict[str, dict]],
        channels: tuple[str, ...],
        tracks: list[ActorTrack],
        sensor_to_world,
        origin: np.ndarray,
        calibrated: dict[str, dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Colour keyframe LiDAR returns from synchronized cameras and assign owners."""

        config = self.config
        rng = np.random.default_rng(42)
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        owner_parts: list[np.ndarray] = []
        seeded_tracks: set[int] = set()
        for sample_index, sample in enumerate(ordered_samples):
            if config.lidar_seed_frame_indices is not None and sample_index not in config.lidar_seed_frame_indices:
                continue
            records = keyframes[sample["token"]]
            lidar_record = records["LIDAR_TOP"]
            lidar_path = root / lidar_record["filename"]
            if not lidar_path.is_file():
                raise FileNotFoundError(f"Missing extracted LiDAR frame {lidar_path}")
            lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            # Returns from the ego vehicle's own roof/body are not scene content.
            lidar = lidar[np.linalg.norm(lidar[:, :2], axis=1) > config.ego_radius]
            lidar_to_world = sensor_to_world(lidar_record)
            world = lidar @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]

            # Actor ownership at the LiDAR timestamp.
            time = lidar_record["timestamp"] * 1e-6
            owner = np.full((len(world),), -1, dtype=np.int64)
            local = world.copy()
            best = np.full((len(world),), np.inf)
            poses: dict[int, np.ndarray] = {}
            for track_index, track in enumerate(tracks):
                pose = track.pose_at(time, 1e-3)
                if pose is None:
                    continue
                poses[track_index] = pose
                in_box = (world - pose[:3, 3]) @ pose[:3, :3]
                half = track.half_extent_xyz
                margin = config.box_margin
                inside = (
                    (np.abs(in_box[:, 0]) <= half[0] + margin)
                    & (np.abs(in_box[:, 1]) <= half[1] + margin)
                    # Do not extend below the box floor: that is road surface.
                    & (in_box[:, 2] >= -half[2])
                    & (in_box[:, 2] <= half[2] + margin)
                )
                distance = np.linalg.norm(in_box, axis=1)
                take = inside & (distance < best)
                owner[take] = track_index
                local[take] = in_box[take]
                best[take] = distance[take]

            # Random actor seeds (in local box coordinates) for actors still
            # unseeded; coloured from whichever synchronized camera sees them.
            random_local: list[np.ndarray] = []
            random_owner: list[np.ndarray] = []
            for track_index, pose in poses.items():
                if track_index in seeded_tracks or config.object_random_seeds <= 0:
                    continue
                half = tracks[track_index].half_extent_xyz
                points = rng.uniform(-0.8, 0.8, size=(config.object_random_seeds, 3)) * half
                random_local.append(points)
                random_owner.append(np.full((len(points),), track_index, dtype=np.int64))
            if random_local:
                extra_local = np.concatenate(random_local)
                extra_owner = np.concatenate(random_owner)
                extra_world = np.stack([poses[int(o)][:3, :3] @ p + poses[int(o)][:3, 3] for p, o in zip(extra_local, extra_owner)])
                world = np.concatenate((world, extra_world))
                local = np.concatenate((local, extra_local))
                owner = np.concatenate((owner, extra_owner))

            colours = np.full((len(world), 3), np.nan)
            for channel in channels:
                record = records[channel]
                with Image.open(root / record["filename"]) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                height, width = rgb.shape[:2]
                world_to_camera = np.linalg.inv(sensor_to_world(record))
                camera_xyz = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                K = np.asarray(calibrated[record["calibrated_sensor_token"]]["camera_intrinsic"], dtype=np.float64)
                depth = camera_xyz[:, 2]
                in_front = depth > 0.5
                pixels = np.zeros((len(world), 2))
                pixels[in_front] = (camera_xyz[in_front] @ K.T)[:, :2] / depth[in_front, None]
                hit = (
                    in_front
                    & np.isnan(colours[:, 0])
                    & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
                )
                columns = pixels[hit, 0].astype(np.int64)
                rows = pixels[hit, 1].astype(np.int64)
                colours[hit] = rgb[rows, columns]
            seen = ~np.isnan(colours[:, 0])
            for track_index in set(owner[seen & (owner >= 0)].tolist()):
                seeded_tracks.add(int(track_index))
            stored = np.where((owner >= 0)[:, None], local, world - origin)
            xyz_parts.append(stored[seen])
            rgb_parts.append(colours[seen])
            owner_parts.append(owner[seen])

        if not xyz_parts:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
        xyz = np.concatenate(xyz_parts)
        rgb = np.concatenate(rgb_parts)
        owner = np.concatenate(owner_parts)

        static = np.where(owner < 0)[0]
        static = static[_voxel_downsample(xyz[static], config.lidar_voxel_size)]
        if len(static) > config.max_init_points:
            static = np.sort(rng.choice(static, size=config.max_init_points, replace=False))
        actors = np.where(owner >= 0)[0]
        actors = actors[_voxel_downsample(xyz[actors], config.object_voxel_size, groups=owner[actors])]
        keep = np.concatenate((static, actors))
        return xyz[keep], np.clip(rgb[keep], 0.0, 1.0), owner[keep]
