import numpy as np

from sim_splat.terrain import SoilField


def _field(**kwargs):
    return SoilField.flat((0.0, 4.0), (-1.0, 1.0), 0.05, **kwargs)


def test_cut_removes_soil_into_the_bucket_and_conserves_volume():
    soil = _field(capacity_m3=10.0)
    taken = soil.cut(centre=(2.0, 0.0), yaw=0.0, half_xy=(0.25, 0.4), bottom_z=-0.3)
    assert taken > 0
    assert np.isclose(soil.load_m3, taken)
    assert np.isclose(soil.volume_change(), -taken)
    assert np.isclose(soil.height_at(2.0, 0.0), -0.3)
    assert np.isclose(soil.height_at(3.5, 0.0), 0.0)  # untouched outside the footprint


def test_a_full_bucket_stops_cutting():
    soil = _field(capacity_m3=0.05)
    soil.cut(centre=(2.0, 0.0), yaw=0.0, half_xy=(0.25, 0.4), bottom_z=-0.5)
    assert np.isclose(soil.load_m3, 0.05)
    assert soil.cut(centre=(1.0, 0.0), yaw=0.0, half_xy=(0.25, 0.4), bottom_z=-0.5) == 0.0


def test_dump_and_settle_conserve_volume_and_limit_slopes():
    soil = _field(capacity_m3=10.0, repose_deg=35.0)
    taken = soil.cut(centre=(1.0, 0.0), yaw=0.0, half_xy=(0.3, 0.3), bottom_z=-0.4)
    spilled = soil.dump(centre=(3.0, 0.0), radius=0.3)
    assert np.isclose(spilled, taken)
    assert soil.load_m3 == 0.0
    assert abs(soil.volume_change()) < 1e-9
    assert soil.height_at(3.0, 0.0) > 0.05  # a mound
    slope = np.abs(np.diff(soil.heights[:, 50:], axis=1)).max() / soil.resolution
    assert slope <= np.tan(np.radians(35.0)) * 1.3  # mound sides relaxed (iterative, approximately)


def test_soil_surfels_have_fixed_size_and_show_only_the_excavation():
    from sim_splat.terrain import soil_surfels

    soil = _field(capacity_m3=10.0)
    before = soil_surfels(soil.heights, soil.initial, soil.origin, soil.resolution)
    soil.cut(centre=(2.0, 0.0), yaw=0.0, half_xy=(0.3, 0.3), bottom_z=-0.4)
    points, normals, colours, visible = soil_surfels(soil.heights, soil.initial, soil.origin, soil.resolution)
    assert len(points) == len(before[0])  # fixed slots: updated in place
    assert not before[3].any()  # nothing changed, nothing drawn
    shown = points[visible]
    assert len(shown) > 0
    assert np.all(np.abs(shown[:, 0] - 2.0) < 0.5) and np.all(np.abs(shown[:, 1]) < 0.5)
    assert shown[:, 2].min() < -0.35  # trench floor
    walls = visible & (np.abs(normals[:, 2]) < 1e-6)
    assert walls.any()  # vertical faces of the cut are drawn


def test_cutting_force_reduces_to_rankine_passive_pressure():
    from sim_splat.terrain import SoilParams, cutting_force

    soil = SoilParams("smooth sand", 16000.0, 0.0, 30.0, 0.0, 0.0)
    out = cutting_force(0.4, 0.9, soil, rake_deg=90.0)
    kp = np.tan(np.radians(45 + 15)) ** 2
    assert np.isclose(out["total"], 0.5 * 16000.0 * 0.4**2 * kp * 0.9, rtol=1e-3)
    assert np.isclose(out["rho_deg"], 45 - 15, atol=0.2)
    assert np.isclose(out["horizontal"], out["total"]) and abs(out["vertical_up"]) < 1e-6 * out["total"]


def test_cutting_force_grows_with_depth_and_soil_strength():
    from sim_splat.terrain import SOILS, cutting_force

    for soil in SOILS.values():
        shallow, deep = cutting_force(0.1, 0.9, soil)["horizontal"], cutting_force(0.3, 0.9, soil)["horizontal"]
        assert deep > shallow > 0
    assert cutting_force(0.2, 0.9, SOILS["clay"])["horizontal"] > cutting_force(0.2, 0.9, SOILS["loam"])["horizontal"] \
        > cutting_force(0.2, 0.9, SOILS["sand"])["horizontal"]
    assert cutting_force(0.0, 0.9, SOILS["clay"])["total"] == 0.0


def test_bearing_capacity_matches_prandtl_for_clay():
    from sim_splat.terrain import SoilParams, bearing_capacity

    clay = SoilParams("undrained clay", 18000.0, 40e3, 0.0, 0.0, 0.0)
    assert np.isclose(bearing_capacity(clay, 0.9, 0.4), (2 + np.pi) * 40e3 * 0.36, rtol=1e-6)  # N_c = 5.14, N_gamma = 0
