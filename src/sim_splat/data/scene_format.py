"""SimSplat scene directory: the single on-disk interchange format.

Every data source (COLMAP, nuScenes, a fleet log, a simulator capture) is
converted *once* into this layout; the training workflow only ever reads it.
See README.md ("Scene format") for the specification.

    <scene>/
    ├─ scene.json      format version, name, source, free-form metadata
    ├─ cameras.json    one record per image: path, K, c2w, size, split, ...
    ├─ points.ply      seed points: xyz, rgb, owning track (-1 = static)
    ├─ tracks.json     optional rigid actors and their per-frame poses
    ├─ images/         RGB files referenced by cameras.json
    └─ masks/          optional static masks (white = supervise)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim_splat.types import Camera, DataparserOutputs

FORMAT_NAME = "sim_splat_scene"
FORMAT_VERSION = 1

# Metadata keys the model and editing tools consume (filled from tracks.json).
TRACK_KEYS = (
    "dynamic_track_tokens",
    "dynamic_track_categories",
    "dynamic_track_displacement",
    "dynamic_track_box_size_xyz",
    "dynamic_point_track_indices",
    "dynamic_object_to_world",
    "dynamic_track_visible",
)

_PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("track", "<i4")]
)
_PLY_TYPES = {"<f4": "float", "|u1": "uchar", "<i4": "int"}


# ---------------------------------------------------------------------------
# points.ply
# ---------------------------------------------------------------------------
def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, track: np.ndarray) -> None:
    """Binary little-endian PLY with xyz (float), rgb (uchar), track (int)."""

    count = len(xyz)
    if rgb.shape != (count, 3) or xyz.shape != (count, 3) or track.shape != (count,):
        raise ValueError("points.ply needs xyz [N,3], rgb [N,3] and track [N]")
    data = np.empty(count, dtype=_PLY_DTYPE)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    colours = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    data["red"], data["green"], data["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    data["track"] = track
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property {_PLY_TYPES[_PLY_DTYPE[name].str]} {name}" for name in _PLY_DTYPE.names]
    header.append("end_header")
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(data.tobytes())


def read_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``write_points_ply`` output; a missing ``track`` property means all static."""

    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
            if not line and len(header) > 64:
                raise ValueError(f"{path} has no PLY end_header")
        if header[0] != "ply" or "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path} must be a binary little-endian PLY")
        count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
        names = {"float": "<f4", "uchar": "u1", "int": "<i4", "double": "<f8"}
        fields = [(line.split()[2], names[line.split()[1]]) for line in header if line.startswith("property")]
        data = np.frombuffer(handle.read(count * np.dtype(fields).itemsize), dtype=np.dtype(fields), count=count)
    xyz = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32)
    rgb = np.stack((data["red"], data["green"], data["blue"]), axis=1).astype(np.float32) / 255.0
    track = data["track"].astype(np.int64) if "track" in data.dtype.names else np.full((count,), -1, dtype=np.int64)
    return xyz, rgb, track


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _place(source: Path, directory: Path, scene_dir: Path, placed: dict[Path, str], link: bool) -> str:
    """Copy (or symlink) one file into the scene; returns its scene-relative path."""

    source = source.resolve()
    if source in placed:
        return placed[source]
    relative = f"{directory.name}/{source.name}"
    target = scene_dir / relative
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Two inputs map to {relative}; give them unique file names")
    if link:
        target.symlink_to(source)
    else:
        shutil.copy2(source, target)
    placed[source] = relative
    return relative


def export_scene(outputs: DataparserOutputs, scene_dir: Path, name: str, link_images: bool = False) -> Path:
    """Write any parser's outputs as a scene directory (the converter back end)."""

    scene_dir.mkdir(parents=True, exist_ok=False)
    (scene_dir / "images").mkdir()
    if outputs.mask_paths:
        (scene_dir / "masks").mkdir()
    eval_set = set(outputs.eval_indices)
    train_set = set(outputs.train_indices)
    placed: dict[Path, str] = {}
    frames: list[dict[str, Any]] = []
    for index, (camera, image_path) in enumerate(zip(outputs.cameras, outputs.image_paths)):
        record: dict[str, Any] = {
            "image": _place(image_path, scene_dir / "images", scene_dir, placed, link_images),
            "width": camera.width,
            "height": camera.height,
            "K": camera.K.tolist(),
            "c2w": camera.c2w.tolist(),
            "camera_model": camera.model,
            "distortion": None if camera.distortion_params is None else camera.distortion_params.flatten().tolist(),
            "is_rectified": camera.is_rectified,
            "sensor": camera.sensor_id,
            "timestamp": camera.timestamp,
            "split": "eval" if index in eval_set else ("train" if index in train_set else "unused"),
        }
        if outputs.mask_paths and outputs.mask_paths[index] is not None:
            record["mask"] = _place(outputs.mask_paths[index], scene_dir / "masks", scene_dir, placed, link_images)
        frames.append(record)
    (scene_dir / "cameras.json").write_text(json.dumps({"frames": frames}, indent=1))

    metadata = outputs.metadata
    track_owner = metadata.get("dynamic_point_track_indices")
    owner = np.full((len(outputs.points_xyz),), -1, dtype=np.int64) if track_owner is None else track_owner.numpy()
    write_points_ply(scene_dir / "points.ply", outputs.points_xyz.numpy(), outputs.points_rgb.numpy(), owner)

    tokens = list(metadata.get("dynamic_track_tokens", []))
    if tokens:
        transforms = metadata["dynamic_object_to_world"].numpy()
        visible = metadata["dynamic_track_visible"].numpy()
        categories = metadata.get("dynamic_track_categories", [""] * len(tokens))
        sizes = metadata.get("dynamic_track_box_size_xyz", [None] * len(tokens))
        tracks = [
            {
                "id": token,
                "category": categories[index],
                "box_size_xyz": None if sizes[index] is None else [float(v) for v in sizes[index]],
                "poses": {str(frame): transforms[frame, index].tolist() for frame in np.flatnonzero(visible[:, index])},
            }
            for index, token in enumerate(tokens)
        ]
        (scene_dir / "tracks.json").write_text(json.dumps({"tracks": tracks}))

    extra = {key: value for key, value in metadata.items() if key not in TRACK_KEYS}
    serializable = json.loads(json.dumps(extra, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    scene = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "name": name, "metadata": serializable}
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    return scene_dir


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_scene(scene_dir: Path) -> DataparserOutputs:
    """Read a scene directory into the framework's parser contract."""

    scene_dir = scene_dir.expanduser()
    scene = json.loads((scene_dir / "scene.json").read_text())
    if scene.get("format") != FORMAT_NAME:
        raise ValueError(f"{scene_dir} is not a {FORMAT_NAME} directory")
    if scene.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported scene format version {scene.get('version')} (expected {FORMAT_VERSION})")
    frames = json.loads((scene_dir / "cameras.json").read_text())["frames"]
    if not frames:
        raise ValueError(f"{scene_dir}/cameras.json has no frames")

    cameras: list[Camera] = []
    image_paths: list[Path] = []
    mask_paths: list[Path | None] = []
    train_indices: list[int] = []
    eval_indices: list[int] = []
    has_masks = any("mask" in frame for frame in frames)
    for index, frame in enumerate(frames):
        image_path = scene_dir / frame["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"cameras.json references missing image {image_path}")
        c2w = torch.tensor(frame["c2w"], dtype=torch.float32)
        K = torch.tensor(frame["K"], dtype=torch.float32)
        if c2w.shape != (4, 4) or K.shape != (3, 3):
            raise ValueError(f"Frame {index}: c2w must be 4x4 and K 3x3")
        distortion = frame.get("distortion")
        cameras.append(
            Camera(
                c2w=c2w,
                K=K,
                width=int(frame["width"]),
                height=int(frame["height"]),
                image_id=index,
                model=frame.get("camera_model", "PINHOLE"),
                distortion_params=None if distortion is None else torch.tensor(distortion, dtype=torch.float32),
                is_rectified=bool(frame.get("is_rectified", False)),
                sensor_id=int(frame.get("sensor", 0)),
                timestamp=float(frame.get("timestamp", 0.0)),
            )
        )
        image_paths.append(image_path)
        if has_masks:
            mask_paths.append(scene_dir / frame["mask"] if "mask" in frame else None)
        split = frame.get("split", "train")
        if split == "train":
            train_indices.append(index)
        elif split == "eval":
            eval_indices.append(index)
        elif split != "unused":
            raise ValueError(f"Frame {index}: split must be train, eval or unused, got {split!r}")

    xyz, rgb, owner = read_points_ply(scene_dir / "points.ply")
    metadata: dict[str, Any] = dict(scene.get("metadata", {}))
    metadata.update({"source": metadata.get("source", "scene"), "scene_dir": str(scene_dir), "scene_name": scene["name"]})

    tracks_path = scene_dir / "tracks.json"
    tracks = json.loads(tracks_path.read_text())["tracks"] if tracks_path.is_file() else []
    if owner.max(initial=-1) >= len(tracks):
        raise ValueError("points.ply references a track index that tracks.json does not define")
    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (len(frames), len(tracks), 4, 4)).copy()
    visible = np.zeros((len(frames), len(tracks)), dtype=bool)
    displacement: list[float] = []
    for track_index, track in enumerate(tracks):
        frames_with_pose = sorted(int(frame) for frame in track["poses"])
        for frame in frames_with_pose:
            transforms[frame, track_index] = np.asarray(track["poses"][str(frame)], dtype=np.float32)
            visible[frame, track_index] = True
        if frames_with_pose:
            first, last = transforms[frames_with_pose[0], track_index], transforms[frames_with_pose[-1], track_index]
            displacement.append(float(np.linalg.norm(last[:2, 3] - first[:2, 3])))
        else:
            displacement.append(0.0)
    metadata.update(
        {
            "dynamic_track_tokens": [track["id"] for track in tracks],
            "dynamic_track_categories": [track.get("category", "") for track in tracks],
            "dynamic_track_displacement": displacement,
            "dynamic_track_box_size_xyz": [track.get("box_size_xyz") for track in tracks],
            "dynamic_point_track_indices": torch.from_numpy(owner),
            "dynamic_object_to_world": torch.from_numpy(transforms),
            "dynamic_track_visible": torch.from_numpy(visible),
        }
    )
    return DataparserOutputs(
        image_paths=image_paths,
        cameras=cameras,
        train_indices=train_indices,
        eval_indices=eval_indices,
        points_xyz=torch.from_numpy(xyz),
        points_rgb=torch.from_numpy(rgb),
        mask_paths=mask_paths,
        metadata=metadata,
    )
