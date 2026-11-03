import torch

from sim_splat.scripts.eval_temporal import warp_previous
from sim_splat.types import Camera


def _camera(x: float = 0.0) -> Camera:
    c2w = torch.eye(4)
    c2w[0, 3] = x
    K = torch.tensor(((20.0, 0.0, 16.0), (0.0, 20.0, 12.0), (0.0, 0.0, 1.0)))
    return Camera(c2w=c2w, K=K, width=32, height=24, image_id=0)


def test_warp_with_identical_poses_is_the_identity():
    image = torch.rand(24, 32, 3)
    depth = torch.full((24, 32), 5.0)
    warped, valid = warp_previous(image, depth, depth, _camera(), _camera())
    assert valid.float().mean() > 0.9
    assert torch.allclose(warped[valid], image[valid], atol=1e-5)


def test_warp_follows_a_sideways_camera_move():
    # A fronto-parallel wall at 5 m; moving the camera 0.25 m right shifts it 1 px left.
    image = torch.zeros(24, 32, 3)
    image[:, 10] = 1.0
    depth = torch.full((24, 32), 5.0)
    warped, valid = warp_previous(image, depth, depth, _camera(0.25), _camera(0.0))
    assert warped[12, 9, 0] > 0.9 and valid[12, 9]
