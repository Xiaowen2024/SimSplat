import numpy as np

from sim_splat.scripts.eval_lidar_tasks import _near, ground_plane, occupancy


def test_ground_plane_and_obstacle_occupancy():
    rng = np.random.default_rng(0)
    ground = np.column_stack((rng.uniform(0, 20, 2000), rng.uniform(-5, 5, 2000), 0.5 + rng.normal(0, 0.01, 2000)))
    box = np.column_stack((rng.uniform(10, 10.4, 50), rng.uniform(0, 0.4, 50), rng.uniform(0.9, 1.5, 50)))
    points = np.concatenate((ground, box))
    normal, offset = ground_plane(points)
    heights = points @ normal + offset
    assert np.abs(heights[:2000]).max() < 0.06
    cells = occupancy(points, heights, 0.5)
    assert cells and all(19 <= x <= 20 and 0 <= y <= 0 for x, y in cells)  # only the box
    assert _near({(5, 5)}, {(6, 6)}) == 1 and _near({(5, 5)}, {(7, 5)}) == 0
