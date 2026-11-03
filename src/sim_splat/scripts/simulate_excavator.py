"""Replace a reconstructed actor with a sim asset and let a MuJoCo excavator push it.

The target track (e.g. a barrier) is removed from the scene and a textured
mesh asset (GLB/OBJ) is placed upright at its location, on the ground under
its annotation box.  An excavator -- static undercarriage and cab, plus a
boom / stick / bucket arm on three hinge joints driven by position servos --
sweeps its bucket through the asset along a scripted path (planar two-link
IK).  The asset is a free body with its convex hull as collider; other
actors within ``--collider-radius`` are static boxes.

Output (``torch.save``), replayed by ``render_physics.py``:

* ``asset``: path, height and ``poses`` [K, 4, 4] of the asset (box-centre
  frame of ``load_asset``) in the world;
* ``parts``: excavator geoms, each with ``half_extent``, ``color`` and
  ``poses`` [K, 4, 4];
* ``initial_actor_to_world``: the removed track's annotation pose, used to
  map the simulation onto each frame's own annotation (see render_physics).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim_splat.assets import load_asset
from sim_splat.config import load_scene_config
from sim_splat.data.dataparsers import parse_scene

YELLOW, DARK, GREY = (0.93, 0.68, 0.1), (0.3, 0.3, 0.3), (0.55, 0.65, 0.72)
BOOM, STICK, BUCKET = 2.6, 1.9, (0.3, 0.45, 0.2)  # link lengths; bucket box half extents
PIVOT = np.array((0.5, 1.3))  # boom pivot (x, z) in the excavator frame


def _yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _ik(wrist: np.ndarray) -> tuple[float, float, float]:
    """Boom, stick, bucket angles (about +y; positive pitches +x down) that put
    the stick tip at ``wrist`` (x, z) with the bucket hanging vertically."""

    u = np.array((wrist[0] - PIVOT[0], PIVOT[1] - wrist[1]))  # x right, y down
    reach = np.clip(np.linalg.norm(u), abs(BOOM - STICK) + 1e-3, BOOM + STICK - 1e-3)
    elbow = float(np.arccos(np.clip((reach**2 - BOOM**2 - STICK**2) / (2 * BOOM * STICK), -1.0, 1.0)))
    boom = float(np.arctan2(u[1], u[0]) - np.arctan2(STICK * np.sin(elbow), BOOM + STICK * np.cos(elbow)))
    return boom, elbow, np.pi / 2 - boom - elbow


def _box(name: str, pos, half, rgb, extra: str = "") -> str:
    return (f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" size="{half[0]} {half[1]} {half[2]}" '
            f'rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" {extra}/>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--track", required=True, help="unique prefix of the track to replace")
    parser.add_argument("--asset", type=Path, required=True, help="GLB/OBJ mesh asset")
    parser.add_argument("--asset-height", type=float, default=0.95, help="m")
    parser.add_argument("--asset-mass", type=float, default=15.0, help="kg (traffic drum with ballast)")
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--push-sign", type=float, default=1.0, help="+1 / -1: push along the track box's +x or -x")
    parser.add_argument("--standoff", type=float, default=3.6, help="excavator base to asset, m")
    parser.add_argument("--hit-height", type=float, default=0.6, help="bucket centre height / asset height")
    parser.add_argument("--travel", type=float, default=1.4, help="bucket sweep length, m")
    parser.add_argument("--speed", type=float, default=1.0, help="bucket sweep speed, m/s")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--collider-radius", type=float, default=4.0)
    args = parser.parse_args()

    import mujoco
    import trimesh

    metadata = parse_scene(load_scene_config(args.scene)).metadata
    tokens: list[str] = list(metadata["dynamic_track_tokens"])
    matches = [token for token in tokens if token.startswith(args.track)]
    if len(matches) != 1:
        raise ValueError(f"track prefix {args.track!r} matched {len(matches)} tracks")
    track = tokens.index(matches[0])
    transforms = metadata["dynamic_object_to_world"].numpy().astype(np.float64)
    visible = metadata["dynamic_track_visible"].numpy()
    annotation = transforms[int(np.flatnonzero(visible[:, track])[0]), track]
    half = np.asarray(metadata["dynamic_track_box_size_xyz"][track], dtype=np.float64) / 2.0
    ground_z = float(annotation[2, 3] - half[2])

    # Excavator frame E: x = push direction, z up, origin on the ground under
    # the excavator; the asset stands at (standoff, 0).
    yaw = _yaw(annotation[:3, :3])
    direction = np.array((np.cos(yaw), np.sin(yaw), 0.0)) * np.sign(args.push_sign)
    world_from_e = np.eye(4)
    world_from_e[:3, :3] = np.stack((direction, np.cross((0, 0, 1), direction), (0, 0, 1)), axis=1)
    world_from_e[:3, 3] = (*(annotation[:2, 3] - direction[:2] * args.standoff), ground_z)
    e_from_world = np.linalg.inv(world_from_e)

    meshes = load_asset(args.asset, args.asset_height)
    hull = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes]).convex_hull
    if len(hull.vertices) > 400:
        hull = trimesh.Trimesh(hull.vertices[np.random.default_rng(0).choice(len(hull.vertices), 400, replace=False)]).convex_hull
    asset_half = (hull.bounds[1] - hull.bounds[0]) / 2.0

    statics = []
    for other, token in enumerate(tokens):
        frames = np.flatnonzero(visible[:, other])
        if other == track or not len(frames):
            continue
        pose = transforms[frames[0], other]
        if np.linalg.norm(pose[:2, 3] - annotation[:2, 3]) > args.collider_radius:
            continue
        local = e_from_world @ pose
        size = np.asarray(metadata["dynamic_track_box_size_xyz"][other], dtype=np.float64) / 2.0
        statics.append(f'<geom type="box" pos="{local[0, 3]} {local[1, 3]} {local[2, 3]}" size="{size[0]} {size[1]} {size[2]}" '
                       f'euler="0 0 {_yaw(local[:3, :3])}" contype="1" conaffinity="1" group="3"/>')

    # contype/conaffinity: arm (2) touches only the asset (1|2); world statics (1) touch only the asset.
    start = np.array((args.standoff - asset_half[0] - BUCKET[2] - 0.6, args.hit_height * args.asset_height + BUCKET[0] + 1.0))
    q0 = _ik(start)
    arm = "contype=\"2\" conaffinity=\"2\""
    xml = f"""
<mujoco model="excavator_push">
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <asset><mesh name="asset" vertex="{' '.join(map(str, hull.vertices.ravel()))}"/></asset>
  <worldbody>
    <geom name="ground" type="plane" size="50 50 0.1" contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
    {''.join(statics)}
    {_box("tracks", (0, 0, 0.3), (1.2, 0.9, 0.3), DARK, 'contype="0" conaffinity="0"')}
    {_box("cab", (-0.2, 0, 1.0), (0.7, 0.7, 0.4), YELLOW, 'contype="0" conaffinity="0"')}
    {_box("cabin", (0.0, 0.35, 1.7), (0.45, 0.32, 0.3), GREY, 'contype="0" conaffinity="0"')}
    <body name="boom" pos="{PIVOT[0]} 0 {PIVOT[1]}">
      <joint name="boom" type="hinge" axis="0 1 0" damping="20000" ref="0"/>
      {_box("boom", (BOOM / 2, 0, 0), (BOOM / 2, 0.15, 0.18), YELLOW, arm)}
      <body name="stick" pos="{BOOM} 0 0">
        <joint name="stick" type="hinge" axis="0 1 0" damping="10000"/>
        {_box("stick", (STICK / 2, 0, 0), (STICK / 2, 0.11, 0.13), YELLOW, arm)}
        <body name="bucket" pos="{STICK} 0 0">
          <joint name="bucket" type="hinge" axis="0 1 0" damping="5000"/>
          {_box("bucket", (BUCKET[0], 0, 0), BUCKET, DARK, arm)}
        </body>
      </body>
    </body>
    <body name="asset" pos="{args.standoff} 0 {asset_half[2] + 0.002}">
      <freejoint/>
      <geom name="asset" type="mesh" mesh="asset" mass="{args.asset_mass}" friction="{args.friction} 0.005 0.0001"
            contype="1" conaffinity="3"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="boom" kp="400000" forcerange="-1e6 1e6"/>
    <position joint="stick" kp="200000" forcerange="-1e6 1e6"/>
    <position joint="bucket" kp="100000" forcerange="-1e6 1e6"/>
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints = [model.joint(name).qposadr[0] for name in ("boom", "stick", "bucket")]
    data.qpos[joints] = q0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)

    # Bucket path (stick tip): lower behind the asset, sweep through it, lift.
    hit = np.array((start[0] + 0.2, args.hit_height * args.asset_height + BUCKET[0]))
    sweep = args.travel / args.speed

    def wrist(t: float) -> np.ndarray:
        if t < 1.0:
            s = 0.5 - 0.5 * np.cos(np.pi * t)
            return start + (hit - start) * s
        if t < 1.0 + sweep:
            return hit + np.array((args.speed * (t - 1.0), 0.0))
        s = min(1.0, t - 1.0 - sweep)
        return hit + np.array((args.travel, 0.9 * (0.5 - 0.5 * np.cos(np.pi * s))))

    asset_body = model.body("asset").id
    asset_geom, bucket_geom = model.geom("asset").id, model.geom("bucket").id
    part_names = ["tracks", "cab", "cabin", "boom", "stick", "bucket"]
    part_ids = [model.geom(name).id for name in part_names]

    def pose(position, rotation) -> np.ndarray:
        local = np.eye(4)
        local[:3, :3] = np.asarray(rotation).reshape(3, 3)
        local[:3, 3] = position
        return world_from_e @ local

    steps = int(round(args.seconds / model.opt.timestep))
    every = int(round(1.0 / (args.fps * model.opt.timestep)))
    asset_poses, part_poses, contacts, tracking = [], [], [], []
    for step in range(steps + 1):
        if step:
            data.ctrl[:] = _ik(wrist(step * model.opt.timestep))
            mujoco.mj_step(model, data)
        if step % every == 0:
            asset_poses.append(pose(data.xpos[asset_body], data.xmat[asset_body]))
            part_poses.append([pose(data.geom_xpos[g], data.geom_xmat[g]) for g in part_ids])
            touching = any({c.geom1, c.geom2} == {asset_geom, bucket_geom} for c in data.contact[: data.ncon])
            contacts.append(touching)
            tracking.append(float(np.abs(data.qpos[joints] - data.ctrl).max()))

    asset_poses = np.stack(asset_poses)
    part_poses = np.stack(part_poses)  # [K, parts, 4, 4]
    tilt = float(np.degrees(np.arccos(np.clip(asset_poses[-1][2, 2], -1.0, 1.0))))
    summary = {
        "track": matches[0],
        "simulator": f"MuJoCo {mujoco.__version__}",
        "frames": len(asset_poses),
        "fps": args.fps,
        "asset_path": str(args.asset),
        "displacement_m": float(np.linalg.norm(asset_poses[-1][:2, 3] - asset_poses[0][:2, 3])),
        "final_tilt_deg": tilt,
        "max_tilt_deg": float(np.degrees(np.arccos(np.clip(asset_poses[:, 2, 2], -1.0, 1.0))).max()),
        "toppled": tilt > 45.0,
        "first_contact_frame": int(np.argmax(contacts)) if any(contacts) else None,
        "max_joint_tracking_error_rad": max(tracking),
        "static_colliders": len(statics),
        "ground_z": ground_z,
        "mass_kg": args.asset_mass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **summary,
            "label": "MuJoCo excavator",
            "box_half_extent": half.tolist(),
            "initial_actor_to_world": torch.from_numpy(annotation).float(),
            "asset": {"path": str(args.asset), "height": args.asset_height, "half_extent": asset_half.tolist(),
                      "poses": torch.from_numpy(asset_poses).float()},
            "parts": [
                {"name": name, "half_extent": [float(v) for v in model.geom_size[g]],
                 "color": [float(v) for v in model.geom_rgba[g][:3]], "poses": torch.from_numpy(part_poses[:, i]).float()}
                for i, (name, g) in enumerate(zip(part_names, part_ids))
            ],
        },
        args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
