"""Convert simulator meshes into Gaussian actors that the scene graph can pose.

A simulator asset (URDF/OBJ/GLB) is a triangle mesh with a rigid pose that a
physics engine updates every tick.  Turning its surface into flat, opaque
Gaussians lets the *same* gsplat pass render the neural background and the
asset, so occlusion in both directions is resolved by the rasterizer's depth
ordering rather than by an approximate 2D depth composite.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sim_splat.models.sim_splat_model import _rgb_to_sh, _rotation_to_quat_wxyz


def procedural_vehicle_mesh(color: tuple[float, float, float] = (0.95, 0.45, 0.1)):
    """A deliberately synthetic 'sim car': body box + cabin box + four wheels.

    Unit-free; ``fit_mesh_to_box`` rescales it to a track's box.  Local axes
    follow nuScenes boxes: x forward, y left, z up, centred at the origin.
    """

    import trimesh

    body = trimesh.creation.box(extents=(1.0, 1.0, 0.45))
    body.apply_translation((0.0, 0.0, -0.1))
    cabin = trimesh.creation.box(extents=(0.55, 0.9, 0.35))
    cabin.apply_translation((-0.05, 0.0, 0.3))
    parts = [(body, color), (cabin, (0.2, 0.25, 0.3))]
    for x in (-0.32, 0.32):
        for y in (-0.5, 0.5):
            wheel = trimesh.creation.cylinder(radius=0.17, height=0.12, sections=24)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
            wheel.apply_translation((x, y, -0.33))
            parts.append((wheel, (0.05, 0.05, 0.05)))
    for mesh, rgb in parts:
        mesh.visual.face_colors = np.array([*(np.asarray(rgb) * 255), 255], dtype=np.uint8)
    return trimesh.util.concatenate([mesh for mesh, _ in parts])


def fit_mesh_to_box(mesh, half_extent_xyz: np.ndarray):
    """Centre ``mesh`` and scale it anisotropically to a box's extents."""

    mesh = mesh.copy()
    low, high = mesh.bounds
    mesh.apply_translation(-(low + high) / 2.0)
    extent = np.maximum(high - low, 1e-6)
    mesh.vertices = mesh.vertices * (2.0 * np.asarray(half_extent_xyz) / extent)
    return mesh


def mesh_to_gaussians(
    mesh,
    count: int,
    sh_degree: int,
    light_direction: tuple[float, float, float] = (0.3, 0.2, 1.0),
    opacity: float = 0.99,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Sample flat surfels on the mesh surface in SimSplat parameter layout."""

    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[faces]
    colors = mesh.visual.face_colors[faces, :3].astype(np.float64) / 255.0
    # Bake simple Lambert shading so geometry reads in the render; a later
    # version would use the simulator's material and the scene's lighting.
    light = np.asarray(light_direction, dtype=np.float64)
    light /= np.linalg.norm(light)
    colors = np.clip(colors * (0.55 + 0.45 * np.clip(normals @ light, 0.0, 1.0))[:, None], 0.0, 1.0)

    # Disc radius from surface area per sample; thin along the normal.
    radius = math.sqrt(mesh.area / count) * 0.9
    scales = np.tile(np.log([radius, radius, radius * 0.05]), (count, 1))
    # Rotation whose local z axis is the face normal.
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    tangent = np.cross(helper, normals)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(normals, tangent)
    rotation = torch.from_numpy(np.stack((tangent, bitangent, normals), axis=-1)).float()

    bases = (sh_degree + 1) ** 2
    return {
        "means": torch.from_numpy(points).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": _rotation_to_quat_wxyz(rotation),
        "opacities": torch.full((count, 1), math.log(opacity / (1.0 - opacity))),
        "features_dc": _rgb_to_sh(torch.from_numpy(colors).float()),
        "features_rest": torch.zeros((count, bases - 1, 3)),
    }
