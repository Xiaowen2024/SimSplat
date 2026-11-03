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


def procedural_barrier_meshes(size_xyz) -> list:
    """Plastic water-filled road barrier fitted to an annotation box.

    Local axes follow the track box: x = depth (thin), y = length, z = up,
    centred on the box.  A slightly tapered body plus a top lip, as two
    convex parts so a physics engine can use them directly as colliders.
    Proportions are from the scene-0061 barriers (orange/white units on the
    kerb): the annotation box includes some margin, so the body is narrower.
    """

    import trimesh

    depth, length, height = (float(v) for v in size_xyz)
    body_height = 0.9 * height
    bottom, top, lip = 0.78 * depth, 0.66 * depth, 0.72 * depth
    half_length = 0.485 * length
    z0 = -height / 2

    def prism(profile_xz, y_half):
        vertices = [(x, y, z) for y in (-y_half, y_half) for x, z in profile_xz]
        return trimesh.Trimesh(vertices).convex_hull

    body = prism([(-bottom / 2, z0), (bottom / 2, z0), (top / 2, z0 + body_height), (-top / 2, z0 + body_height)], half_length)
    cap = prism([(-lip / 2, z0 + body_height - 0.04), (lip / 2, z0 + body_height - 0.04), (lip / 2, z0 + height), (-lip / 2, z0 + height)],
                half_length + 0.01)
    return [body, cap]


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


def load_asset(path, height: float):
    """Load a GLB/OBJ asset as a list of meshes, upright (z up), centred, ``height`` m tall.

    glTF is y-up; the result follows the scene-graph convention (x forward,
    y left, z up) with the bounding-box centre at the origin, so the actor's
    pose is the pose of its box centre.  Sub-meshes are kept separate so each
    keeps its own texture.
    """

    import trimesh

    loaded = trimesh.load(path, force="scene")
    meshes = [mesh for mesh in loaded.dump() if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    if str(path).lower().endswith((".glb", ".gltf")):
        y_up_to_z_up = trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0))
        for mesh in meshes:
            mesh.apply_transform(y_up_to_z_up)
    low = np.min([mesh.bounds[0] for mesh in meshes], axis=0)
    high = np.max([mesh.bounds[1] for mesh in meshes], axis=0)
    scale = height / max(high[2] - low[2], 1e-9)
    for mesh in meshes:
        mesh.apply_translation(-(low + high) / 2.0)
        mesh.apply_scale(scale)
    return meshes


def _surface_colours(mesh, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] at surface samples: texture lookup, else vertex/face colours."""

    import trimesh

    visual = mesh.visual
    if getattr(visual, "kind", None) == "texture" and visual.uv is not None:
        material = visual.material
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        factor = np.asarray(getattr(material, "baseColorFactor", None) if getattr(material, "baseColorFactor", None) is not None
                            else (255, 255, 255, 255), dtype=np.float64)[:3]
        factor = factor / 255.0 if factor.max() > 1.0 else factor
        if image is None:
            return np.tile(factor, (len(points), 1))
        barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[faces], points)
        uv = np.einsum("nk,nkj->nj", barycentric, np.asarray(visual.uv)[mesh.faces[faces]])
        texture = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        height, width = texture.shape[:2]
        column = (np.mod(uv[:, 0], 1.0) * (width - 1)).round().astype(int)
        row = ((1.0 - np.mod(uv[:, 1], 1.0)) * (height - 1)).round().astype(int)
        return texture[row, column] * factor
    return np.asarray(visual.face_colors)[faces, :3].astype(np.float64) / 255.0


def meshes_to_gaussians(meshes, count: int, sh_degree: int, **kwargs) -> dict[str, torch.Tensor]:
    """``mesh_to_gaussians`` over several meshes, samples split by surface area."""

    areas = np.array([mesh.area for mesh in meshes])
    counts = np.maximum(1, np.round(count * areas / areas.sum()).astype(int))
    parts = [mesh_to_gaussians(mesh, int(n), sh_degree, seed=i, **kwargs) for i, (mesh, n) in enumerate(zip(meshes, counts))]
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}


def mesh_to_gaussians(
    mesh,
    count: int,
    sh_degree: int,
    light_direction: tuple[float, float, float] = (0.3, 0.2, 1.0),
    opacity: float = 0.99,
    seed: int = 0,
    ambient: float = 0.55,
) -> dict[str, torch.Tensor]:
    """Sample flat surfels on the mesh surface in SimSplat parameter layout."""

    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[faces]
    colors = _surface_colours(mesh, points, faces)
    # Bake simple Lambert shading so geometry reads in the render; a later
    # version would use the simulator's material and the scene's lighting.
    light = np.asarray(light_direction, dtype=np.float64)
    light /= np.linalg.norm(light)
    colors = np.clip(colors * (ambient + (1.0 - ambient) * np.clip(normals @ light, 0.0, 1.0))[:, None], 0.0, 1.0)

    # Disc radius from surface area per sample; thin along the normal.
    return surfels_to_gaussians(points, normals, colors, math.sqrt(mesh.area / count) * 0.9, sh_degree, opacity)


def surfels_to_gaussians(
    points: np.ndarray, normals: np.ndarray, colors: np.ndarray, radius: float, sh_degree: int, opacity: float = 0.99
) -> dict[str, torch.Tensor]:
    """Flat, opaque discs (z axis = surface normal) in SimSplat parameter layout."""

    count = len(points)
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
