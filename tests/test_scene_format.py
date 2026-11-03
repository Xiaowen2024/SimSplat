from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sim_splat.config import SceneConfig
from sim_splat.data.dataparsers import parse_scene
from sim_splat.data.scene_format import export_scene, read_points_ply, write_points_ply
from sim_splat.types import Camera, DataparserOutputs


def _outputs(tmp_path: Path) -> DataparserOutputs:
    paths, masks, cameras = [], [], []
    mask = tmp_path / "ego.png"
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(mask)
    for index in range(3):
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(path)
        paths.append(path)
        masks.append(mask)  # one mask file shared by several frames
        c2w = torch.eye(4)
        c2w[0, 3] = float(index)
        cameras.append(Camera(c2w=c2w, K=torch.tensor([[4.0, 0, 4], [0, 4.0, 3], [0, 0, 1]]), width=8, height=6,
                              image_id=index, sensor_id=index % 2, timestamp=0.5 * index, is_rectified=True))
    transforms = torch.eye(4).repeat(3, 1, 1, 1)
    transforms[2, 0, 0, 3] = 5.0
    return DataparserOutputs(
        image_paths=paths,
        cameras=cameras,
        train_indices=[1, 2],
        eval_indices=[0],
        points_xyz=torch.tensor([[0.0, 0, 1], [1.0, 0, 2], [0.0, 0.5, 0]]),
        points_rgb=torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
        mask_paths=masks,
        metadata={
            "source": "test",
            "world_origin": [1.0, 2.0, 3.0],
            "dynamic_track_tokens": ["car"],
            "dynamic_track_categories": ["vehicle.car"],
            "dynamic_track_box_size_xyz": [[4.5, 1.9, 1.5]],
            "dynamic_point_track_indices": torch.tensor([-1, -1, 0]),
            "dynamic_object_to_world": transforms,
            "dynamic_track_visible": torch.tensor([[False], [True], [True]]),
        },
    )


def test_points_ply_round_trip(tmp_path: Path) -> None:
    xyz = np.array([[0.5, -1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.5], [0.2, 0.4, 0.6]])
    write_points_ply(tmp_path / "p.ply", xyz, rgb, np.array([-1, 3]))
    read_xyz, read_rgb, track = read_points_ply(tmp_path / "p.ply")
    np.testing.assert_array_equal(read_xyz, xyz)
    np.testing.assert_allclose(read_rgb, rgb, atol=1 / 255)
    assert track.tolist() == [-1, 3]


def test_export_then_parse_preserves_scene(tmp_path: Path) -> None:
    source = _outputs(tmp_path)
    scene_dir = export_scene(source, tmp_path / "scene", name="unit")
    parsed = parse_scene(SceneConfig(name="unit", data=scene_dir))

    assert parsed.train_indices == [1, 2] and parsed.eval_indices == [0]
    assert len(list((scene_dir / "masks").iterdir())) == 1  # shared mask copied once
    for before, after in zip(source.cameras, parsed.cameras):
        torch.testing.assert_close(after.c2w, before.c2w)
        torch.testing.assert_close(after.K, before.K)
        assert (after.sensor_id, after.timestamp, after.is_rectified) == (before.sensor_id, before.timestamp, True)
    torch.testing.assert_close(parsed.points_xyz, source.points_xyz)
    metadata = parsed.metadata
    assert metadata["world_origin"] == [1.0, 2.0, 3.0]
    assert metadata["dynamic_track_tokens"] == ["car"]
    assert metadata["dynamic_track_box_size_xyz"] == [[4.5, 1.9, 1.5]]
    assert metadata["dynamic_track_visible"].tolist() == [[False], [True], [True]]
    assert metadata["dynamic_point_track_indices"].tolist() == [-1, -1, 0]
    torch.testing.assert_close(metadata["dynamic_object_to_world"], source.metadata["dynamic_object_to_world"])
    assert metadata["dynamic_track_displacement"] == [pytest.approx(5.0)]


def test_train_frame_indices_override_split(tmp_path: Path) -> None:
    scene_dir = export_scene(_outputs(tmp_path), tmp_path / "scene", name="unit")
    parsed = parse_scene(SceneConfig(name="unit", data=scene_dir, train_frame_indices=(1,)))
    assert parsed.train_indices == [1] and parsed.eval_indices == [0, 2]
