import numpy as np
import pytest

pytest.importorskip("trimesh")
from sim_splat.scripts.simulate_excavator import BOOM, PIVOT, STICK, _ik


@pytest.mark.parametrize("wrist", [(2.8, 1.5), (3.5, 1.0), (4.2, 1.2), (3.0, 0.4)])
def test_ik_reaches_wrist_with_vertical_bucket(wrist):
    boom, stick, bucket = _ik(np.array(wrist))
    # Hinges about +y: a link at angle a points along (cos a, -sin a) in (x, z).
    elbow = PIVOT + BOOM * np.array((np.cos(boom), -np.sin(boom)))
    tip = elbow + STICK * np.array((np.cos(boom + stick), -np.sin(boom + stick)))
    assert np.allclose(tip, wrist, atol=1e-6)
    assert elbow[1] > max(PIVOT[1], wrist[1])  # elbow-up excavator pose
    assert np.isclose(boom + stick + bucket, np.pi / 2)  # bucket hangs straight down


def test_asset_poses_applies_correction_and_frame_shift():
    from sim_splat.scripts.build_lookalike_asset import asset_poses

    track = np.repeat(np.eye(4)[None], 3, axis=0)
    track[1, :3, 3] = (1.0, 2.0, 0.0)
    correction = np.eye(4)
    correction[2, 3] = 0.1
    poses = asset_poses(track, {"local_correction": correction.tolist(), "frame_shift_z": [0.0, 0.2, -0.1]})
    assert np.allclose(poses[0, :3, 3], (0.0, 0.0, 0.1))
    assert np.allclose(poses[1, :3, 3], (1.0, 2.0, 0.3))
    assert np.allclose(poses[2, :3, 3], (0.0, 0.0, 0.0))
    # Without a per-frame shift the correction alone applies.
    assert np.allclose(asset_poses(track, {"local_correction": correction.tolist()})[2, :3, 3], (0.0, 0.0, 0.1))
