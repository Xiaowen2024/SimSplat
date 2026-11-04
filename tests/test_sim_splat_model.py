from __future__ import annotations

from pathlib import Path
import sys
import types

import torch
import pytest

from sim_splat.config import ModelConfig
from sim_splat.config import GaussianOptimizerConfig
from sim_splat.engine.optimizers import build_gaussian_optimizers
from sim_splat.models.sim_splat_model import SimSplatModel
from sim_splat.types import Batch, Camera, DataparserOutputs


def _scene() -> DataparserOutputs:
    return DataparserOutputs(
        image_paths=[Path("frame.png")],
        cameras=[],
        train_indices=[0],
        eval_indices=[0],
        points_xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        points_rgb=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]),
    )


def test_populate_gaussians_registers_vectorized_trainable_state() -> None:
    model = SimSplatModel(ModelConfig(sh_degree=2, init_opacity=0.1), _scene(), device="cpu")

    assert set(model.gauss_params) == {
        "means",
        "scales",
        "quats",
        "opacities",
        "features_dc",
        "features_rest",
    }
    assert model.gauss_params["means"].shape == (4, 3)
    assert model.gauss_params["scales"].shape == (4, 3)
    assert model.gauss_params["quats"].shape == (4, 4)
    assert model.gauss_params["opacities"].shape == (4, 1)
    assert model.gauss_params["features_dc"].shape == (4, 3)
    assert model.gauss_params["features_rest"].shape == (4, 8, 3)
    torch.testing.assert_close(model.gauss_params["means"], _scene().points_xyz)
    torch.testing.assert_close(model.gauss_params["quats"][:, 0], torch.ones(4))
    assert all(parameter.requires_grad for parameter in model.gauss_params.values())


def test_get_outputs_calls_calibrated_gsplat_renderer(monkeypatch) -> None:
    received: dict[str, object] = {}

    def fake_rasterization(**kwargs):
        received.update(kwargs)
        return (
            torch.zeros((1, 3, 4, 4)),
            torch.zeros((1, 3, 4, 1)),
            {"means2d": torch.zeros((4, 2))},
        )

    gsplat_module = types.ModuleType("gsplat")
    rendering_module = types.ModuleType("gsplat.rendering")
    rendering_module.rasterization = fake_rasterization
    gsplat_module.rendering = rendering_module
    gsplat_module.spherical_harmonics = lambda degree, dirs, coeffs: coeffs[:, 0] * 0.0
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_module)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering_module)

    model = SimSplatModel(ModelConfig(sh_degree=2), _scene(), device="cpu")
    camera = Camera(
        c2w=torch.eye(4),
        K=torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]),
        width=4,
        height=3,
        image_id=1,
        model="OPENCV",
        distortion_params=torch.tensor([0.1, -0.2, 0.01, 0.02]),
    )

    outputs = model.get_outputs(camera)

    assert outputs["rgb"].shape == (3, 4, 3)
    assert outputs["depth"].shape == (3, 4, 1)
    assert outputs["alpha"].shape == (3, 4, 1)
    assert received["camera_model"] == "pinhole"
    torch.testing.assert_close(received["radial_coeffs"], torch.tensor([[0.1, -0.2, 0, 0, 0, 0]]))
    torch.testing.assert_close(received["tangential_coeffs"], torch.tensor([[0.01, 0.02]]))
    # Colours are evaluated by the model (per-actor local view directions),
    # so gsplat receives plain RGB.
    assert received["colors"].shape == (4, 3)
    assert received["sh_degree"] is None
    assert received["opacities"].shape == (4,)


def test_static_loss_and_metrics_exclude_masked_dynamic_pixel() -> None:
    model = SimSplatModel(ModelConfig(sh_degree=0, ssim_lambda=0.2), _scene(), device="cpu")
    target = torch.zeros((3, 4, 3))
    prediction = target.clone()
    prediction[0, 0] = 1.0  # Simulated moving-object error.
    prediction.requires_grad_()
    mask = torch.ones((3, 4, 1))
    mask[0, 0] = 0.0
    batch = Batch(
        image=target,
        static_mask=mask,
        camera=Camera(c2w=torch.eye(4), K=torch.eye(3), width=4, height=3, image_id=1),
        image_name="frame.png",
    )

    losses = model.get_loss_dict({"rgb": prediction}, batch)
    sum(losses.values()).backward()
    metrics = model.get_metrics_dict({"rgb": prediction}, batch)

    assert losses["loss_l1"].item() == pytest.approx(0.0)
    assert losses["loss_ssim"].item() == pytest.approx(0.0)
    assert prediction.grad[0, 0].abs().sum().item() == pytest.approx(0.0)
    assert metrics["static_psnr"] > metrics["psnr"]
    assert metrics["static_pixel_fraction"] == pytest.approx(11.0 / 12.0)
    assert metrics["gaussian_count"] == 4.0


def test_gaussian_optimizers_have_one_parameter_per_strategy_key() -> None:
    model = SimSplatModel(ModelConfig(), _scene(), device="cpu")
    optimizers = build_gaussian_optimizers(model, GaussianOptimizerConfig())

    assert set(optimizers) == set(model.gauss_params)
    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(1.6e-4)
    for name, optimizer in optimizers.items():
        assert optimizer.param_groups[0]["params"] == [model.gauss_params[name]]


def test_ssim_is_one_for_identical_images_and_lower_otherwise() -> None:
    from sim_splat.models.sim_splat_model import _ssim

    torch.manual_seed(0)
    image = torch.rand((32, 40, 3))
    assert _ssim(image, image).item() == pytest.approx(1.0, abs=1e-5)
    assert _ssim((image * 0.5).clamp(0, 1), image).item() < 0.99


def test_rotation_to_quaternion_round_trip() -> None:
    from sim_splat.models.sim_splat_model import _rotation_to_quat_wxyz

    torch.manual_seed(0)
    quats = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
    quats = torch.where(quats[:, :1] < 0, -quats, quats)
    w, x, y, z = quats.unbind(-1)
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )
    torch.testing.assert_close(_rotation_to_quat_wxyz(rotation), quats, atol=1e-5, rtol=1e-5)


def _actor_scene() -> DataparserOutputs:
    transforms = torch.eye(4).repeat(2, 1, 1, 1)  # [frames=2, tracks=1, 4, 4]
    transforms[1, 0, 0, 3] = 10.0  # actor moves 10 m along x in frame 1
    return DataparserOutputs(
        image_paths=[Path("a.png"), Path("b.png")],
        cameras=[],
        train_indices=[0],
        eval_indices=[1],
        points_xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        points_rgb=torch.full((4, 3), 0.5),
        metadata={
            "dynamic_track_tokens": ["bus-token"],
            "dynamic_point_track_indices": torch.tensor([-1, -1, 0, 0]),
            "dynamic_object_to_world": transforms,
            "dynamic_track_visible": torch.tensor([[True], [True]]),
        },
    )


def test_hidden_actor_keeps_rows_and_zeroes_opacity() -> None:
    model = SimSplatModel(ModelConfig(sh_degree=0), _actor_scene(), device="cpu")
    means, _, visible, _ = model._composed_gaussians(1)
    assert means.shape[0] == 4  # row indices are never subset
    torch.testing.assert_close(means[2], torch.tensor([10.0, 2.0, 0.0]))
    assert visible.tolist() == [True, True, True, True]
    model.set_hidden_tracks(["bus"])
    _, _, visible, _ = model._composed_gaussians(1)
    assert visible.tolist() == [True, True, False, False]


def test_split_track_owners_matches_gsplat_row_layout() -> None:
    # gsplat.strategy.ops.split builds cat([p[rest], p[sel], p[sel]]).
    model = SimSplatModel(ModelConfig(sh_degree=0), _actor_scene(), device="cpu")
    model.split_track_owners(torch.tensor([False, False, True, False]))
    assert model.point_track_indices.tolist() == [-1, -1, 0, 0, 0]
    model.duplicate_track_owners(torch.tensor([True, False, False, False, False]))
    assert model.point_track_indices.tolist() == [-1, -1, 0, 0, 0, -1]


def test_add_actor_appends_rows_and_pose_track() -> None:
    model = SimSplatModel(ModelConfig(sh_degree=1), _actor_scene(), device="cpu")
    gaussians = {
        "means": torch.zeros((2, 3)),
        "scales": torch.zeros((2, 3)),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * 2),
        "opacities": torch.zeros((2, 1)),
        "features_dc": torch.zeros((2, 3)),
        "features_rest": torch.zeros((2, 3, 3)),
    }
    index = model.add_actor("asset:car", gaussians, model.object_to_world[:, 0].clone(), torch.tensor([True, True]))
    assert index == 1
    assert model.gauss_params["means"].shape[0] == 6
    assert model.point_track_indices.tolist()[-2:] == [1, 1]
    means, _, _, _ = model._composed_gaussians(1)
    torch.testing.assert_close(means[-1], torch.tensor([10.0, 0.0, 0.0]))


def test_pose_refinement_applies_only_in_training_mode():
    scene = _scene()
    model = SimSplatModel(ModelConfig(sh_degree=0, pose_refine=True), scene, device="cpu")
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    camera = Camera(c2w=c2w, K=torch.eye(3), width=4, height=4, image_id=0)
    assert "pose" in model.extra_parameters()
    assert torch.allclose(model.refined_c2w(camera), camera.c2w)  # starts at zero correction
    with torch.no_grad():
        model.pose_delta[camera.image_id] = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.0, 0.0])
    model.train()
    refined = model.refined_c2w(camera)
    assert not torch.allclose(refined, camera.c2w)
    assert torch.allclose(refined[:3, :3].T @ refined[:3, :3], torch.eye(3), atol=1e-5)  # still a rotation
    assert torch.allclose(refined[:3, 3], camera.c2w[:3, 3] + 0.2 * camera.c2w[:3, 0], atol=1e-5)
    model.eval()
    assert torch.allclose(model.refined_c2w(camera), camera.c2w)
    model.refine_captured_views = True  # scoring training views
    assert not torch.allclose(model.refined_c2w(camera), camera.c2w)
