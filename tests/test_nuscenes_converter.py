from __future__ import annotations

import numpy as np
import pytest

from sim_splat.converters.nuscenes import ActorTrack, _voxel_downsample


def _track() -> ActorTrack:
    yaw90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    return ActorTrack(
        token="t",
        category="vehicle.car",
        times=np.array([0.0, 0.5]),
        translations=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], yaw90]),
        size_wlh=np.array([1.9, 4.5, 1.5]),
    )


def test_box_half_extent_uses_length_along_local_x() -> None:
    np.testing.assert_allclose(_track().half_extent_xyz, [2.25, 0.95, 0.75])


def test_pose_interpolates_translation_and_rotation() -> None:
    pose = _track().pose_at(0.25, tolerance=0.0)
    np.testing.assert_allclose(pose[:3, 3], [5.0, 0.0, 0.0])
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    assert yaw == pytest.approx(np.pi / 4)


def test_pose_is_none_outside_annotated_span() -> None:
    assert _track().pose_at(1.0, tolerance=0.25) is None
    assert _track().pose_at(0.7, tolerance=0.25) is not None


def test_voxel_downsample_respects_groups() -> None:
    points = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    assert len(_voxel_downsample(points, 0.1)) == 1
    assert len(_voxel_downsample(points, 0.1, groups=np.array([0, 0, 1]))) == 2
