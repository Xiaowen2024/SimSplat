"""Replace a reconstructed actor with a sim asset and let a MuJoCo excavator push it.

The target track (e.g. a barrier) is removed from the scene and replaced by
either a textured mesh asset (``--asset``, GLB/OBJ, placed upright on the
ground under the annotation box) or a look-alike asset built from the
scene's own photos (``--asset-dir``, from ``build_lookalike_asset.py``,
placed at its fitted pose with its convex parts as colliders).  An excavator -- static undercarriage and cab, plus a
boom / stick / bucket arm on three hinge joints driven by position servos --
sweeps its bucket through the asset along a scripted path (planar two-link
IK).  The asset is a free body with its convex hull as collider; other
actors within ``--collider-radius`` are static boxes (shrunk by 10 cm, and
dropped where they overlap a simulated object).  With ``--asset-root``
every other look-alike asset nearby is a free body too, so a push can run
down a row of barriers (``--push-axis y``).  Those neighbours are snapped
onto the ground plane: their relative heights carry the per-frame height
error and would otherwise start sunk in or floating.  A 1 s settle
resolves the remaining contact overlaps before the arm moves.

Output (``torch.save``), replayed by ``render_physics.py``:

* ``asset``: path, height and ``poses`` [K, 4, 4] of the asset (box-centre
  frame of ``load_asset``) in the world;
* ``parts``: excavator geoms, each with ``half_extent``, ``color`` and
  ``poses`` [K, 4, 4];
* ``initial_actor_to_world``: the removed track's annotation pose, used to
  map the simulation onto each frame's own annotation (see render_physics);
* ``neighbours``: per simulated neighbour, its asset and poses; renderers
  apply its simulated motion to its own photo-fitted pose.
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
from sim_splat.scripts.build_lookalike_asset import asset_poses

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


def _obb_overlap(a: tuple, b: tuple) -> bool:
    """2D oriented boxes (centre xy, yaw, half xy): separating-axis test."""

    def corners(box):
        (cx, cy), yaw, (hx, hy) = box
        axes = np.array(((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw))))
        return np.array((cx, cy)) + np.array([sx * hx * axes[0] + sy * hy * axes[1] for sx in (-1, 1) for sy in (-1, 1)])

    ca, cb = corners(a), corners(b)
    for yaw in (a[1], b[1]):
        for axis in ((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw))):
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


class ExcavatorWorld:
    """MuJoCo world: the asset in place of a track, neighbours as static boxes, and the excavator.

    Frame E has x along the push direction and z up, with its origin on the
    ground under the excavator; the asset starts at (standoff, 0).  The arm
    is commanded by its stick-tip ("wrist") position in E (x, z), through IK
    and position servos.  ``reset(start)`` sets the arm at a wrist position
    without dynamics.
    """

    PARTS = ("tracks", "cab", "cabin", "boom", "stick", "bucket")

    def __init__(self, scene: Path, track: str, *, asset: Path | None = None, asset_dir: Path | None = None,
                 asset_height: float = 0.95, asset_mass: float = 15.0, friction: float = 0.8, push_sign: float = 1.0, push_axis: str = "x",
                 lateral_offset: float = 0.0, standoff: float = 3.6, collider_radius: float = 4.0, neighbour_dirs: list[Path] | tuple = ()) -> None:
        import mujoco
        import trimesh

        self.mujoco = mujoco
        metadata = parse_scene(load_scene_config(scene)).metadata
        tokens: list[str] = list(metadata["dynamic_track_tokens"])
        matches = [token for token in tokens if token.startswith(track)]
        if len(matches) != 1:
            raise ValueError(f"track prefix {track!r} matched {len(matches)} tracks")
        self.token = matches[0]
        index = tokens.index(self.token)
        transforms = metadata["dynamic_object_to_world"].numpy().astype(np.float64)
        visible = metadata["dynamic_track_visible"].numpy()
        annotation = transforms[int(np.flatnonzero(visible[:, index])[0]), index]
        self.annotation = annotation
        self.track_half = np.asarray(metadata["dynamic_track_box_size_xyz"][index], dtype=np.float64) / 2.0
        yaw = _yaw(annotation[:3, :3])

        # Asset collider parts (local frame, centred) and its first-frame world pose.
        first = int(np.flatnonzero(visible[:, index])[0])
        neighbours = []
        if asset_dir:
            info = json.loads((asset_dir / "asset.json").read_text())
            hulls = [trimesh.load(asset_dir / name, force="mesh").convex_hull for name in info["parts"]]
            target_poses = asset_poses(transforms[:, index], info)
            asset_pose = target_poses[first]
            self.asset_record = {"gaussians": str(asset_dir / "gaussians.pt"), "height": float(info["size_xyz"][2]),
                                 "asset_dir": str(asset_dir), "initial_pose": torch.from_numpy(asset_pose).float()}
            # Other look-alike assets as free bodies.  Per-frame poses drift, so each
            # is placed *relative to the target* in a frame where both are annotated.
            for path in neighbour_dirs:
                other_info = json.loads((Path(path) / "asset.json").read_text())
                other = tokens.index(other_info["track"])
                if other == index:
                    continue
                both = np.flatnonzero(visible[:, index] & visible[:, other])
                if not len(both):
                    continue
                frame = int(both[np.argmin(np.abs(both - first))])
                other_poses = asset_poses(transforms[:, other], other_info)
                relative = np.linalg.inv(target_poses[frame]) @ other_poses[frame]
                pose = asset_pose @ relative
                if np.linalg.norm(pose[:2, 3] - asset_pose[:2, 3]) > collider_radius:
                    continue
                neighbours.append({"token": other_info["track"], "track": other, "pose": pose, "dir": str(path),
                                   "hulls": [trimesh.load(Path(path) / name, force="mesh").convex_hull for name in other_info["parts"]],
                                   "half": np.asarray(other_info["size_xyz"]) / 2.0})
        else:
            meshes = load_asset(asset, asset_height)
            hull = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes]).convex_hull
            if len(hull.vertices) > 400:
                hull = trimesh.Trimesh(hull.vertices[np.random.default_rng(0).choice(len(hull.vertices), 400, replace=False)]).convex_hull
            hulls = [hull]
            asset_pose = np.eye(4)
            asset_pose[:3, :3] = np.array(((np.cos(yaw), -np.sin(yaw), 0), (np.sin(yaw), np.cos(yaw), 0), (0, 0, 1)))
            asset_pose[:3, 3] = (*annotation[:2, 3], annotation[2, 3] - self.track_half[2] + (hull.bounds[1, 2] - hull.bounds[0, 2]) / 2)
            self.asset_record = {"path": str(asset), "height": asset_height}
        low = np.min([h.bounds[0] for h in hulls], axis=0)
        high = np.max([h.bounds[1] for h in hulls], axis=0)
        self.asset_half = (high - low) / 2.0
        self.ground_z = float(asset_pose[2, 3] - self.asset_half[2])
        self.asset_mass = asset_mass

        heading = yaw + (np.pi / 2 if push_axis == "y" else 0.0)  # "y": along a row of barriers
        direction = np.array((np.cos(heading), np.sin(heading), 0.0)) * np.sign(push_sign)
        world_from_e = np.eye(4)
        world_from_e[:3, :3] = np.stack((direction, np.cross((0, 0, 1), direction), (0, 0, 1)), axis=1)
        side = np.cross((0, 0, 1), direction)[:2]  # +y of frame E
        # A lateral offset aims the bucket at e.g. the joint between two barriers.
        world_from_e[:3, 3] = (*(asset_pose[:2, 3] - direction[:2] * standoff + side * lateral_offset), self.ground_z)
        self.world_from_e = world_from_e
        e_from_world = np.linalg.inv(world_from_e)
        asset_in_e = e_from_world @ asset_pose
        asset_quat = np.zeros(4)
        mujoco.mju_mat2Quat(asset_quat, asset_in_e[:3, :3].ravel())
        self.height = 2.0 * self.asset_half[2]
        self.reach = float(np.abs(asset_in_e[0, :3]) @ self.asset_half)  # asset extent along the push direction
        self.standoff = standoff

        # Footprints of the simulated objects in E: (centre xy, yaw, half xy, bottom z).
        def lowest(pose: np.ndarray, parts) -> float:  # lowest collider vertex (objects may lean slightly)
            return float(min((h.vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() for h in parts))

        # Objects on one pavement, but their relative heights carry the per-frame
        # height error (+-15 cm): on the ground plane they would start sunk in or
        # floating.  Physics uses a clean world (every object resting on the plane);
        # renderers apply only the simulated *motion* to each object's own
        # photo-fitted pose, so this snap never shows.
        footprints = [((asset_in_e[0, 3], asset_in_e[1, 3]), _yaw(asset_in_e[:3, :3]), tuple(self.asset_half[:2]))]
        for item in neighbours:
            local = e_from_world @ item["pose"]
            local[2, 3] -= lowest(local, item["hulls"])
            item["pose"] = world_from_e @ local
            low = np.min([h.bounds[0] for h in item["hulls"]], axis=0)
            high = np.max([h.bounds[1] for h in item["hulls"]], axis=0)
            footprints.append(((local[0, 3], local[1, 3]), _yaw(local[:3, :3]), tuple((high - low)[:2] / 2.0)))

        statics = []
        self.dropped_statics = 0
        simulated = {index} | {item["track"] for item in neighbours}
        for other in range(len(tokens)):
            frames = np.flatnonzero(visible[:, other])
            if other in simulated or not len(frames):
                continue
            pose = transforms[frames[0], other]
            if np.linalg.norm(pose[:2, 3] - annotation[:2, 3]) > collider_radius:
                continue
            local = e_from_world @ pose
            # Annotation boxes carry margin: shrink them, and drop any that still
            # overlaps a simulated object (it would push it at t = 0).
            size = np.maximum(np.asarray(metadata["dynamic_track_box_size_xyz"][other], dtype=np.float64) / 2.0 - 0.1, 0.05)
            box = ((local[0, 3], local[1, 3]), _yaw(local[:3, :3]), tuple(size[:2]))
            if any(_obb_overlap(box, footprint) for footprint in footprints):
                self.dropped_statics += 1
                continue
            statics.append(f'<geom type="box" pos="{local[0, 3]} {local[1, 3]} {local[2, 3]}" size="{size[0]} {size[1]} {size[2]}" '
                           f'euler="0 0 {_yaw(local[:3, :3])}" contype="1" conaffinity="1" group="3"/>')
        self.static_colliders = len(statics)

        # contype/conaffinity: arm (2) touches only the asset (1|2); world statics (1) touch only the asset.
        arm = 'contype="2" conaffinity="2"'
        xml = f"""
<mujoco model="excavator_push">
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <asset>{''.join(f'<mesh name="asset{i}" vertex="{" ".join(map(str, h.vertices.ravel()))}"/>' for i, h in enumerate(hulls))}{''.join(f'<mesh name="nb{n}_{i}" vertex="{" ".join(map(str, h.vertices.ravel()))}"/>' for n, item in enumerate(neighbours) for i, h in enumerate(item["hulls"]))}</asset>
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
    <body name="asset" pos="{asset_in_e[0, 3]} {asset_in_e[1, 3]} {asset_in_e[2, 3] + 0.002}" quat="{' '.join(map(str, asset_quat))}">
      <freejoint/>
      {''.join(f'<geom name="asset{i}" type="mesh" mesh="asset{i}" mass="{asset_mass / len(hulls)}" '
               f'friction="{friction} 0.005 0.0001" contype="1" conaffinity="3"/>' for i in range(len(hulls)))}
    </body>
    {''.join(self._free_body(f"nb{n}", item, e_from_world, asset_mass, friction) for n, item in enumerate(neighbours))}
  </worldbody>
  <actuator>
    <position joint="boom" kp="400000" forcerange="-1e6 1e6"/>
    <position joint="stick" kp="200000" forcerange="-1e6 1e6"/>
    <position joint="bucket" kp="100000" forcerange="-1e6 1e6"/>
  </actuator>
</mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)  # body poses valid before the first step
        self.joints = [self.model.joint(name).qposadr[0] for name in ("boom", "stick", "bucket")]
        self.asset_body = self.model.body("asset").id
        self.asset_geoms = {self.model.geom(f"asset{i}").id for i in range(len(hulls))}
        self.bucket_geom = self.model.geom("bucket").id
        self.part_ids = [self.model.geom(name).id for name in self.PARTS]
        self.neighbours = neighbours
        self.neighbour_bodies = [self.model.body(f"nb{n}").id for n in range(len(neighbours))]
        if neighbours:
            self._settle()

    def _settle(self, seconds: float = 1.0) -> None:
        """Let abutting objects resolve their small initial overlaps (a few cm)
        with the arm raised out of reach, then start from rest."""

        raised = np.array((0.5, 4.0))
        self.reset(raised)
        for _ in range(int(seconds / self.dt)):
            self.step(raised)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _free_body(name: str, item: dict, e_from_world: np.ndarray, mass: float, friction: float) -> str:
        import mujoco

        local = e_from_world @ item["pose"]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, local[:3, :3].ravel())
        geoms = "".join(f'<geom type="mesh" mesh="{name}_{i}" mass="{mass / len(item["hulls"])}" friction="{friction} 0.005 0.0001" '
                        f'contype="1" conaffinity="3"/>' for i in range(len(item["hulls"])))
        return (f'<body name="{name}" pos="{local[0, 3]} {local[1, 3]} {local[2, 3] + 0.002}" quat="{" ".join(map(str, quat))}">'
                f'<freejoint/>{geoms}</body>')

    def neighbour_poses(self) -> list[np.ndarray]:
        return [self._pose(self.data.xpos[b], self.data.xmat[b]) for b in self.neighbour_bodies]

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, wrist: np.ndarray) -> None:
        q = _ik(wrist)
        self.data.qpos[self.joints] = q
        self.data.ctrl[:] = q
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, wrist: np.ndarray) -> None:
        self.data.ctrl[:] = _ik(wrist)
        self.mujoco.mj_step(self.model, self.data)

    def _pose(self, position, rotation) -> np.ndarray:
        local = np.eye(4)
        local[:3, :3] = np.asarray(rotation).reshape(3, 3)
        local[:3, 3] = position
        return self.world_from_e @ local

    def asset_pose(self) -> np.ndarray:
        return self._pose(self.data.xpos[self.asset_body], self.data.xmat[self.asset_body])

    def part_poses(self) -> list[np.ndarray]:
        return [self._pose(self.data.geom_xpos[g], self.data.geom_xmat[g]) for g in self.part_ids]

    def touching(self) -> bool:
        return any(self.bucket_geom in (c.geom1, c.geom2) and ({c.geom1, c.geom2} - {self.bucket_geom}) <= self.asset_geoms
                   for c in self.data.contact[: self.data.ncon])

    def tracking_error(self) -> float:
        return float(np.abs(self.data.qpos[self.joints] - self.data.ctrl).max())

    def payload(self, asset_poses: np.ndarray, part_poses: np.ndarray, summary: dict,
                neighbour_poses: np.ndarray | None = None) -> dict:
        """What ``render_physics.py`` replays: asset and excavator-part poses per video frame."""

        return {
            "neighbours": [
                {"token": item["token"], "gaussians": str(Path(item["dir"]) / "gaussians.pt"), "half_extent": item["half"].tolist(),
                 "asset_dir": item["dir"], "initial_pose": torch.from_numpy(neighbour_poses[0, n]).float(),
                 "poses": torch.from_numpy(neighbour_poses[:, n]).float()}
                for n, item in enumerate(self.neighbours)
            ] if neighbour_poses is not None and len(self.neighbours) else [],
            **summary,
            "label": "MuJoCo excavator",
            "box_half_extent": self.track_half.tolist(),
            "initial_actor_to_world": torch.from_numpy(self.annotation).float(),
            "asset": {**self.asset_record, "half_extent": self.asset_half.tolist(), "poses": torch.from_numpy(asset_poses).float()},
            "parts": [
                {"name": name, "half_extent": [float(v) for v in self.model.geom_size[g]],
                 "color": [float(v) for v in self.model.geom_rgba[g][:3]], "poses": torch.from_numpy(part_poses[:, i]).float()}
                for i, (name, g) in enumerate(zip(self.PARTS, self.part_ids))
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--track", required=True, help="unique prefix of the track to replace")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--asset", type=Path, help="GLB/OBJ mesh asset")
    source.add_argument("--asset-dir", type=Path, help="look-alike asset directory (build_lookalike_asset.py)")
    parser.add_argument("--asset-height", type=float, default=0.95, help="m")
    parser.add_argument("--asset-mass", type=float, default=15.0, help="kg (traffic drum with ballast)")
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--push-sign", type=float, default=1.0, help="+1 / -1: push along the track box's +axis or -axis")
    parser.add_argument("--push-axis", choices=("x", "y"), default="x", help="box axis to push along (y: along a row)")
    parser.add_argument("--lateral-offset", type=float, default=0.0, help="m: shift the excavator sideways (hit a joint)")
    parser.add_argument("--standoff", type=float, default=3.6, help="excavator base to asset, m")
    parser.add_argument("--hit-height", type=float, default=0.6, help="bucket centre height / asset height")
    parser.add_argument("--travel", type=float, default=1.4, help="bucket sweep length, m")
    parser.add_argument("--speed", type=float, default=1.0, help="bucket sweep speed, m/s")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--collider-radius", type=float, default=4.0)
    parser.add_argument("--asset-root", type=Path, help="other look-alike assets (one per sub-directory) become free bodies too")
    args = parser.parse_args()

    import mujoco

    world = ExcavatorWorld(args.scene, args.track, asset=args.asset, asset_dir=args.asset_dir, asset_height=args.asset_height,
                           asset_mass=args.asset_mass, friction=args.friction, push_sign=args.push_sign, push_axis=args.push_axis,
                           lateral_offset=args.lateral_offset,
                           standoff=args.standoff, collider_radius=args.collider_radius,
                           neighbour_dirs=sorted(path.parent for path in args.asset_root.glob("*/asset.json")) if args.asset_root else ())
    start = np.array((args.standoff - world.reach - BUCKET[2] - 0.6, args.hit_height * world.height + BUCKET[0] + 1.0))
    world.reset(start)

    # Bucket path (stick tip): lower behind the asset, sweep through it, lift.
    hit = np.array((start[0] + 0.2, args.hit_height * world.height + BUCKET[0]))
    sweep = args.travel / args.speed

    def wrist(t: float) -> np.ndarray:
        if t < 1.0:
            s = 0.5 - 0.5 * np.cos(np.pi * t)
            return start + (hit - start) * s
        if t < 1.0 + sweep:
            return hit + np.array((args.speed * (t - 1.0), 0.0))
        s = min(1.0, t - 1.0 - sweep)
        return hit + np.array((args.travel, 0.9 * (0.5 - 0.5 * np.cos(np.pi * s))))

    steps = int(round(args.seconds / world.dt))
    every = int(round(1.0 / (args.fps * world.dt)))
    asset_poses, part_poses, contacts, tracking, neighbour_poses = [], [], [], [], []
    for step in range(steps + 1):
        if step:
            world.step(wrist(step * world.dt))
        if step % every == 0:
            asset_poses.append(world.asset_pose())
            part_poses.append(world.part_poses())
            neighbour_poses.append(world.neighbour_poses())
            contacts.append(world.touching())
            tracking.append(world.tracking_error())

    asset_poses = np.stack(asset_poses)
    part_poses = np.stack(part_poses)  # [K, parts, 4, 4]
    neighbour_poses = np.stack(neighbour_poses) if world.neighbours else None  # [K, neighbours, 4, 4]
    moved = [float(np.linalg.norm(neighbour_poses[-1, n, :2, 3] - neighbour_poses[0, n, :2, 3])) for n in range(len(world.neighbours))] \
        if world.neighbours else []
    tilt = float(np.degrees(np.arccos(np.clip(asset_poses[-1][2, 2], -1.0, 1.0))))
    summary = {
        "track": world.token,
        "simulator": f"MuJoCo {mujoco.__version__}",
        "frames": len(asset_poses),
        "fps": args.fps,
        "asset": str(args.asset_dir or args.asset),
        "displacement_m": float(np.linalg.norm(asset_poses[-1][:2, 3] - asset_poses[0][:2, 3])),
        "final_tilt_deg": tilt,
        "max_tilt_deg": float(np.degrees(np.arccos(np.clip(asset_poses[:, 2, 2], -1.0, 1.0))).max()),
        "toppled": tilt > 45.0,
        "first_contact_frame": int(np.argmax(contacts)) if any(contacts) else None,
        "max_joint_tracking_error_rad": max(tracking),
        "static_colliders": world.static_colliders,
        "static_colliders_dropped_overlap": world.dropped_statics,
        "neighbour_bodies": len(world.neighbours),
        "neighbours_moved_over_5cm": int(sum(m > 0.05 for m in moved)),
        "max_neighbour_displacement_m": max(moved) if moved else 0.0,
        "ground_z": world.ground_z,
        "mass_kg": args.asset_mass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(world.payload(asset_poses, part_poses, summary, neighbour_poses), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
