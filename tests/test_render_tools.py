from __future__ import annotations

import numpy as np
import torch

from sim_splat.scripts.render_edit import _interpolate_camera_path
from sim_splat.scripts.render_physics import _project_box
from sim_splat.types import Camera


def _camera(x: float, yaw_deg: float, sensor: int = 0) -> Camera:
    c2w = torch.eye(4)
    angle = np.radians(yaw_deg)
    c2w[:3, :3] = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
    c2w[0, 3] = x
    K = torch.tensor([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    return Camera(c2w=c2w, K=K, width=100, height=80, image_id=0, sensor_id=sensor, is_rectified=True)


def test_camera_path_interpolates_pose_and_keeps_endpoints() -> None:
    cameras = [_camera(0.0, 0.0), _camera(2.0, 20.0)]
    path = _interpolate_camera_path(cameras, [0, 1], inserts=3)
    assert len(path) == 5
    assert path[0][0] is cameras[0] and path[-1][0] is cameras[1]
    middle = path[2][0]
    assert abs(float(middle.c2w[0, 3]) - 1.0) < 1e-5
    yaw = np.degrees(np.arctan2(float(middle.c2w[0, 2]), float(middle.c2w[2, 2])))
    assert abs(yaw - 10.0) < 1e-3


def test_camera_path_never_blends_different_sensors() -> None:
    cameras = [_camera(0.0, 0.0, sensor=0), _camera(2.0, 0.0, sensor=1)]
    assert len(_interpolate_camera_path(cameras, [0, 1], inserts=3)) == 2


def test_project_box_in_front_and_behind() -> None:
    camera = _camera(0.0, 0.0)
    pose = np.eye(4)
    pose[2, 3] = 5.0  # 5 m in front of an OpenCV camera looking along +z
    box = _project_box(pose, np.array([0.5, 0.5, 0.5]), camera)
    assert box is not None and box[0] < 50 < box[2] and box[1] < 40 < box[3]
    pose[2, 3] = -5.0
    assert _project_box(pose, np.array([0.5, 0.5, 0.5]), camera) is None
