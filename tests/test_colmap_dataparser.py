from __future__ import annotations

import sys
import types

import numpy as np
import torch
from PIL import Image

from sim_splat.config import SceneConfig
from sim_splat.data.dataparsers.colmap_dataparser import ColmapDataParser


class _FakeCamera:
    def __init__(self, model_name: str, params: list[float]) -> None:
        self.model_name = model_name
        self.params = np.asarray(params, dtype=np.float64)
        self.width = 4
        self.height = 3

    def calibration_matrix(self) -> np.ndarray:
        return np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])


class _FakePose:
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


class _FakeImage:
    def __init__(self, name: str, camera_id: int) -> None:
        self.name = name
        self.camera_id = camera_id
        self.cam_from_world = _FakePose()


class _FakePoint:
    def __init__(self, xyz: list[float], color: list[int]) -> None:
        self.xyz = np.asarray(xyz)
        self.color = np.asarray(color)


class _FakeReconstruction:
    def __init__(self, _: str) -> None:
        self.cameras = {
            1: _FakeCamera("PINHOLE", [2.0, 2.0, 2.0, 1.5]),
            2: _FakeCamera("OPENCV", [2.0, 2.0, 2.0, 1.5, 0.1, -0.2, 0.01, 0.02]),
        }
        self.images = {10: _FakeImage("frame_a.png", 1), 11: _FakeImage("frame_b.png", 2)}
        self.points3D = {4: _FakePoint([1.0, 2.0, 3.0], [255, 128, 0])}

    def reg_image_ids(self) -> list[int]:
        return [11, 10]


def test_parser_preserves_calibration_and_builds_scene(tmp_path, monkeypatch) -> None:
    images = tmp_path / "images"
    sparse = tmp_path / "sparse"
    images.mkdir()
    sparse.mkdir()
    Image.new("RGB", (4, 3)).save(images / "frame_a.png")
    Image.new("RGB", (4, 3)).save(images / "frame_b.png")
    monkeypatch.setitem(sys.modules, "pycolmap", types.SimpleNamespace(Reconstruction=_FakeReconstruction))

    outputs = ColmapDataParser(
        SceneConfig(name="fixture", data=tmp_path, images_path="images", colmap_path="sparse", eval_interval=2)
    ).get_dataparser_outputs()

    assert [path.name for path in outputs.image_paths] == ["frame_a.png", "frame_b.png"]
    assert outputs.train_indices == [1]
    assert outputs.eval_indices == [0]
    assert outputs.cameras[0].c2w[0, 3].item() == -1.0
    assert outputs.cameras[0].distortion_params is None
    assert outputs.cameras[1].model == "OPENCV"
    torch.testing.assert_close(
        outputs.cameras[1].distortion_params,
        torch.tensor([0.1, -0.2, 0.01, 0.02]),
    )
    assert outputs.points_xyz.shape == (1, 3)
    torch.testing.assert_close(outputs.points_rgb, torch.tensor([[1.0, 128.0 / 255.0, 0.0]]))
