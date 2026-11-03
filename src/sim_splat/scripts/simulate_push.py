"""Simulate a kinematic 'excavator bucket' pushing a reconstructed actor (PyBullet).

The target actor becomes a dynamic rigid box with the actor's own 3D box
size.  Its initial orientation keeps only the box yaw (box z = world z), and
the ground plane sits at the box bottom.  Other actors within
``--collider-radius`` are static box colliders.  A kinematic box (the bucket)
is driven along the target's local x axis, at a configurable height.

The output stores full 6-DoF poses sampled at the video frame rate:

* ``actor_to_world`` [K, 4, 4]: pose to apply to the actor's *own* Gaussians
  (its local annotation frame), ready for ``model.object_to_world[:, track]``.
* ``pusher_to_world`` [K, 4, 4] and ``pusher_half_extent``: so the bucket can
  be rendered as a simulator asset.

Physics never touches the renderer: ``render_physics.py`` replays these poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim_splat.config import load_scene_config
from sim_splat.data.dataparsers import parse_scene


def _yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _matrix(position, quaternion_xyzw, bullet) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(bullet.getMatrixFromQuaternion(quaternion_xyzw)).reshape(3, 3)
    result[:3, 3] = position
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--track", required=True, help="unique target-track prefix")
    parser.add_argument("--output", type=Path, required=True, help="physics .pt file")
    parser.add_argument("--mass", type=float, default=60.0, help="kg (water-filled plastic barrier)")
    parser.add_argument("--friction", type=float, default=0.8, help="actor surface friction")
    parser.add_argument("--ground-friction", type=float, default=1.0,
                        help="Bullet multiplies the two bodies' coefficients: effective mu = friction * ground_friction")
    parser.add_argument("--push-sign", type=float, default=1.0, help="+1 / -1: push along the box's +x or -x")
    parser.add_argument("--push-axis", choices=("x", "y"), default="x", help="box axis to push along")
    parser.add_argument("--hit-height", type=float, default=0.75, help="bucket centre height as a fraction of box height")
    parser.add_argument("--speed", type=float, default=1.2, help="bucket speed, m/s")
    parser.add_argument("--travel", type=float, default=1.8, help="bucket travel distance, m")
    parser.add_argument("--seconds", type=float, default=4.0, help="simulated duration")
    parser.add_argument("--fps", type=int, default=24, help="pose sampling rate (= video frame rate)")
    parser.add_argument("--collider-radius", type=float, default=4.0)
    args = parser.parse_args()

    import pybullet as bullet

    outputs = parse_scene(load_scene_config(args.scene))
    metadata = outputs.metadata
    tokens: list[str] = list(metadata["dynamic_track_tokens"])
    matches = [token for token in tokens if token.startswith(args.track)]
    if len(matches) != 1:
        raise ValueError(f"track prefix {args.track!r} matched {len(matches)} tracks")
    track = tokens.index(matches[0])
    transforms = metadata["dynamic_object_to_world"].numpy().astype(np.float64)
    visible = metadata["dynamic_track_visible"].numpy()
    frame = int(np.flatnonzero(visible[:, track])[0])
    annotation = transforms[frame, track]  # world_from_actor_local
    half = np.asarray(metadata["dynamic_track_box_size_xyz"][track], dtype=np.float64) / 2.0

    # Physics box frame: same centre, yaw only (z = world z).
    yaw = _yaw(annotation[:3, :3])
    box0 = np.eye(4)
    box0[:3, :3] = np.array(((np.cos(yaw), -np.sin(yaw), 0), (np.sin(yaw), np.cos(yaw), 0), (0, 0, 1)))
    box0[:3, 3] = annotation[:3, 3]
    # actor_local -> physics box frame (constant): applied after every step.
    box_from_actor = np.linalg.inv(box0) @ annotation
    ground_z = float(annotation[2, 3] - half[2])

    client = bullet.connect(bullet.DIRECT)
    bullet.setGravity(0, 0, -9.81, physicsClientId=client)
    hz = 240
    bullet.setTimeStep(1.0 / hz, physicsClientId=client)
    plane = bullet.createMultiBody(0, bullet.createCollisionShape(bullet.GEOM_PLANE, physicsClientId=client),
                                   basePosition=(0, 0, ground_z), physicsClientId=client)
    bullet.changeDynamics(plane, -1, lateralFriction=args.ground_friction, physicsClientId=client)
    quat0 = bullet.getQuaternionFromEuler((0, 0, yaw))
    body = bullet.createMultiBody(
        args.mass, bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=half.tolist(), physicsClientId=client),
        basePosition=box0[:3, 3].tolist(), baseOrientation=quat0, physicsClientId=client,
    )
    bullet.changeDynamics(body, -1, lateralFriction=args.friction, physicsClientId=client)

    # Neighbouring actors (e.g. the next barrier in the row) are static colliders.
    colliders: list[str] = []
    for other, token in enumerate(tokens):
        frames = np.flatnonzero(visible[:, other])
        if other == track or not len(frames):
            continue
        pose = transforms[frames[0], other]
        if np.linalg.norm(pose[:2, 3] - annotation[:2, 3]) > args.collider_radius:
            continue
        other_half = np.asarray(metadata["dynamic_track_box_size_xyz"][other], dtype=np.float64) / 2.0
        bullet.createMultiBody(0, bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=other_half.tolist(), physicsClientId=client),
                               basePosition=pose[:3, 3].tolist(), baseOrientation=bullet.getQuaternionFromEuler((0, 0, _yaw(pose[:3, :3]))),
                               physicsClientId=client)
        colliders.append(token)

    # Kinematic bucket: thin plate facing the target, moved by pose resets.
    axis = 0 if args.push_axis == "x" else 1
    direction = box0[:3, axis] * np.sign(args.push_sign)
    lateral = 1 - axis
    pusher_half = np.zeros(3)
    pusher_half[axis] = 0.15
    pusher_half[lateral] = max(half[lateral] * 0.8, 0.3)
    height = ground_z + args.hit_height * 2.0 * half[2]
    # The plate must not overhang the box top: a tipping box's back-top edge
    # first *rises*, and an overhanging bucket pins it flat (measured: 0 deg
    # tilt with a 0.6 m plate, 94 deg with a 0.2 m plate at the same height).
    pusher_half[2] = max(0.03, min(0.1, ground_z + 2.0 * half[2] - height - 0.02))
    start = box0[:3, 3] - direction * (half[axis] + pusher_half[axis] + 0.25)
    start[2] = height
    pusher = bullet.createMultiBody(0, bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=pusher_half.tolist(), physicsClientId=client),
                                    basePosition=start.tolist(), baseOrientation=quat0, physicsClientId=client)
    pusher_rotation = box0[:3, :3]

    steps = int(round(args.seconds * hz))
    every = max(1, hz // args.fps)
    actor_to_world, pusher_to_world, contact_frames = [], [], []
    for step in range(steps + 1):
        moved = min(step / hz * args.speed, args.travel)
        position = start + direction * moved
        bullet.resetBasePositionAndOrientation(pusher, position.tolist(), quat0, physicsClientId=client)
        if step:
            bullet.stepSimulation(physicsClientId=client)
        if step % every == 0:
            box_pose = _matrix(*bullet.getBasePositionAndOrientation(body, physicsClientId=client), bullet)
            actor_to_world.append(box_pose @ box_from_actor)
            pusher_pose = np.eye(4)
            pusher_pose[:3, :3] = pusher_rotation
            pusher_pose[:3, 3] = position
            pusher_to_world.append(pusher_pose)
            contact_frames.append(bool(bullet.getContactPoints(body, pusher, physicsClientId=client)))
    bullet.disconnect(client)

    actor_to_world = np.stack(actor_to_world)
    final = actor_to_world[-1] @ np.linalg.inv(box_from_actor)
    tilt = float(np.degrees(np.arccos(np.clip(final[2, 2], -1.0, 1.0))))
    summary = {
        "track": matches[0],
        "frames": len(actor_to_world),
        "fps": args.fps,
        "displacement_m": float(np.linalg.norm(final[:2, 3] - box0[:2, 3])),
        "final_tilt_deg": tilt,
        "toppled": tilt > 45.0,
        "first_contact_frame": int(np.argmax(contact_frames)) if any(contact_frames) else None,
        "static_colliders": len(colliders),
        "ground_z": ground_z,
        "mass_kg": args.mass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **summary,
            "actor_to_world": torch.from_numpy(actor_to_world).float(),
            "pusher_to_world": torch.from_numpy(np.stack(pusher_to_world)).float(),
            "pusher_half_extent": pusher_half.tolist(),
            "box_half_extent": half.tolist(),
            "initial_actor_to_world": torch.from_numpy(annotation).float(),
        },
        args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
