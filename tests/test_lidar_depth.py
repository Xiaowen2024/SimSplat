import numpy as np
import pytest

pytest.importorskip("scipy")
from sim_splat.scripts.add_lidar_depth import project


def _camera():
    c2w = np.eye(4)  # OpenCV: looking down +z
    K = np.array(((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)))
    return c2w, K


def test_project_keeps_nearest_return_per_pixel():
    c2w, K = _camera()
    points = np.array(((0.0, 0.0, 5.0), (0.0, 0.0, 9.0), (0.5, 0.0, 5.0)))
    uv, depth = project(points, c2w, K, 100, 100, window=1, slack=0.05)
    assert sorted(depth.tolist()) == [5.0, 5.0]
    assert np.allclose(uv[np.argmin(uv[:, 0])], (50.5, 50.5))


def test_project_drops_returns_hidden_behind_a_nearer_surface():
    c2w, K = _camera()
    # A wall at 5 m next to (one pixel away from) a return at 20 m: the LiDAR,
    # mounted higher, saw past the wall's edge; the camera cannot.
    points = np.array(((0.0, 0.0, 5.0), (0.2, 0.0, 20.0), (0.0, 1.0, 20.0)))
    uv, depth = project(points, c2w, K, 100, 100, window=7, slack=0.05)
    assert 5.0 in depth.tolist()
    assert 20.0 not in depth[np.abs(uv[:, 1] - 50.5) < 3].tolist()  # hidden one near the wall dropped
    assert (np.abs(uv[:, 1] - 55.5) < 1).any()  # the far return away from the wall stays


def test_project_ignores_points_behind_or_outside():
    c2w, K = _camera()
    points = np.array(((0.0, 0.0, -3.0), (100.0, 0.0, 5.0)))
    uv, depth = project(points, c2w, K, 100, 100, window=3, slack=0.05)
    assert len(depth) == 0
