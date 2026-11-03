"""Deformable ground for excavation: a kinematic height-field soil model.

The reconstruction's ground is a static surface; an excavator changes it.
``SoilField`` is the usual lightweight model for this in machine simulation:

* the terrain is a height field (cell heights over a regular x-y grid);
* **cut**: wherever the bucket's cutting box overlaps the soil, the column
  above the bucket's bottom is removed and its volume is added to the load
  (up to the bucket's capacity);
* **dump**: the load is spilled as a smooth mound of the same volume centred
  under the bucket;
* **settle**: slopes steeper than the angle of repose relax (sand-pile
  relaxation), so walls of a trench and the sides of a mound stay plausible.

Volume is conserved: removed = load + deposited.  The field is in frame E
of the excavator world (x forward, z up), heights in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SoilField:
    origin: tuple[float, float]  # x, y of cell (0, 0)'s centre
    resolution: float
    heights: np.ndarray  # [rows (y), cols (x)]
    capacity_m3: float = 0.35
    repose_deg: float = 40.0
    load_m3: float = 0.0
    initial: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=np.float64).copy()
        self.initial = self.heights.copy()

    @classmethod
    def flat(cls, x_range: tuple[float, float], y_range: tuple[float, float], resolution: float, height: float = 0.0,
             **kwargs) -> "SoilField":
        cols = int(round((x_range[1] - x_range[0]) / resolution)) + 1
        rows = int(round((y_range[1] - y_range[0]) / resolution)) + 1
        return cls((x_range[0], y_range[0]), resolution, np.full((rows, cols), height), **kwargs)

    @property
    def cell_area(self) -> float:
        return self.resolution**2

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.heights.shape
        xs = self.origin[0] + self.resolution * np.arange(cols)
        ys = self.origin[1] + self.resolution * np.arange(rows)
        return np.meshgrid(xs, ys)

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear height (outside the grid: the nearest edge)."""

        rows, cols = self.heights.shape
        fx = np.clip((np.asarray(x) - self.origin[0]) / self.resolution, 0, cols - 1)
        fy = np.clip((np.asarray(y) - self.origin[1]) / self.resolution, 0, rows - 1)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        x1, y1 = np.minimum(x0 + 1, cols - 1), np.minimum(y0 + 1, rows - 1)
        tx, ty = fx - x0, fy - y0
        h = self.heights
        return (h[y0, x0] * (1 - tx) * (1 - ty) + h[y0, x1] * tx * (1 - ty) + h[y1, x0] * (1 - tx) * ty + h[y1, x1] * tx * ty)

    def _inside(self, centre, yaw: float, half_xy) -> np.ndarray:
        gx, gy = self.grid()
        dx, dy = gx - centre[0], gy - centre[1]
        lx = dx * np.cos(yaw) + dy * np.sin(yaw)
        ly = -dx * np.sin(yaw) + dy * np.cos(yaw)
        return (np.abs(lx) <= half_xy[0]) & (np.abs(ly) <= half_xy[1])

    def cut(self, centre, yaw: float, half_xy, bottom_z: float) -> float:
        """Remove soil above ``bottom_z`` under the bucket footprint; returns the volume taken."""

        room = self.capacity_m3 - self.load_m3
        if room <= 0:
            return 0.0
        inside = self._inside(centre, yaw, half_xy)
        excess = np.where(inside, np.maximum(self.heights - bottom_z, 0.0), 0.0)
        volume = float(excess.sum() * self.cell_area)
        if volume <= 0:
            return 0.0
        take = min(1.0, room / volume)  # a full bucket stops cutting
        self.heights -= excess * take
        self.load_m3 += volume * take
        return volume * take

    def dump(self, centre, radius: float = 0.6) -> float:
        """Spill the whole load as a smooth mound (same volume) around ``centre``."""

        if self.load_m3 <= 0:
            return 0.0
        gx, gy = self.grid()
        weight = np.exp(-((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2) / (2 * (radius / 2) ** 2))
        total = float(weight.sum() * self.cell_area)
        if total <= 0:
            return 0.0
        spilled = self.load_m3
        self.heights += weight * (spilled / total)
        self.load_m3 = 0.0
        self.settle()
        return spilled

    def settle(self, iterations: int = 2000, tolerance: float = 1e-4) -> None:
        """Relax slopes steeper than the angle of repose until converged (mass-conserving)."""

        limit = np.tan(np.radians(self.repose_deg)) * self.resolution
        h = self.heights
        for _ in range(iterations):
            moved = False
            for axis in (0, 1):
                difference = np.diff(h, axis=axis)
                excess = np.where(np.abs(difference) > limit + tolerance, (np.abs(difference) - limit) / 2.0 * np.sign(difference), 0.0)
                if not np.any(excess):
                    continue
                moved = True
                transfer = 0.5 * excess  # under-relaxed for stability
                if axis == 0:
                    h[:-1, :] += transfer
                    h[1:, :] -= transfer
                else:
                    h[:, :-1] += transfer
                    h[:, 1:] -= transfer
            if not moved:
                break

    def volume_change(self) -> float:
        """Net soil volume vs. the initial field (should equal minus the load)."""

        return float((self.heights - self.initial).sum() * self.cell_area)

    def changed(self, tolerance: float = 0.01) -> np.ndarray:
        return np.abs(self.heights - self.initial) > tolerance


SOIL_RGB = np.array((0.58, 0.45, 0.32))  # moist sub-base soil; never photographed, so authored
CRUST_RGB = np.array((0.36, 0.36, 0.35))  # cut edge of the pavement layer


def soil_surfels(heights: np.ndarray, initial: np.ndarray, origin: tuple[float, float], resolution: float, *,
                 wall_slots: int = 6, crust_m: float = 0.08, changed_m: float = 0.01, seed: int = 0):
    """Fixed-size surfel set for the excavated / deposited part of a height field.

    Returns ``points``, ``normals``, ``colours`` [N, 3] and ``visible`` [N] in the
    field's frame, with N = cells x (1 + ``wall_slots``) whatever the state, so
    a renderer can update them in place frame by frame without flicker.

    * top: one disc per cell at its current height, normal from the height
      gradient, Lambert-shaded;
    * walls: up to ``wall_slots`` discs stacked over the drop to the lowest
      4-neighbour, facing it, darker with depth (a cheap ambient-occlusion
      term).  The top ``crust_m`` of a cut is the pavement layer.

    Only cells that changed (or border a change) are visible: everywhere else
    the reconstruction itself is the ground.
    """

    rows, cols = heights.shape
    gx, gy = np.meshgrid(origin[0] + resolution * np.arange(cols), origin[1] + resolution * np.arange(rows))
    dz_dy, dz_dx = np.gradient(heights, resolution)
    normals_top = np.stack((-dz_dx, -dz_dy, np.ones_like(heights)), axis=-1)
    normals_top /= np.linalg.norm(normals_top, axis=-1, keepdims=True)
    changed = np.abs(heights - initial) > changed_m
    active = changed.copy()
    active[1:] |= changed[:-1]
    active[:-1] |= changed[1:]
    active[:, 1:] |= changed[:, :-1]
    active[:, :-1] |= changed[:, 1:]
    noise = np.random.default_rng(seed).uniform(0.85, 1.15, size=(rows, cols, 1))
    sun = np.array((0.3, 0.2, 0.93))
    sun /= np.linalg.norm(sun)
    shade = 0.7 + 0.3 * np.clip(normals_top @ sun, 0.0, 1.0)
    depth_below = np.clip(initial - heights, 0.0, None)
    occlusion = 1.0 - 0.45 * np.clip(depth_below / 0.6, 0.0, 1.0)
    top_colour = np.clip(SOIL_RGB * noise * (shade * occlusion)[..., None], 0, 1)
    points = [np.stack((gx, gy, heights), axis=-1).reshape(-1, 3)]
    normals = [normals_top.reshape(-1, 3)]
    colours = [top_colour.reshape(-1, 3)]
    visible = [changed.reshape(-1)]

    # Walls: drop to the lowest 4-neighbour, and its direction.
    padded = np.pad(heights, 1, mode="edge")
    neighbours = np.stack((padded[1:-1, 2:], padded[1:-1, :-2], padded[2:, 1:-1], padded[:-2, 1:-1]))  # +x, -x, +y, -y
    directions = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
    lowest = neighbours.argmin(axis=0)
    drop = heights - neighbours.min(axis=0)
    facing = directions[lowest]
    wall_normals = np.concatenate((facing, np.zeros((rows, cols, 1))), axis=-1)
    for slot in range(wall_slots):
        fraction = (slot + 0.5) / wall_slots
        z = heights - fraction * drop
        offset = facing * resolution * 0.5
        points.append(np.stack((gx + offset[..., 0], gy + offset[..., 1], z), axis=-1).reshape(-1, 3))
        normals.append(wall_normals.reshape(-1, 3))
        below_surface = initial - z
        crust = below_surface < crust_m
        darkness = 1.0 - 0.5 * np.clip(below_surface / 0.6, 0.0, 1.0)
        colour = np.where(crust[..., None], CRUST_RGB, SOIL_RGB * noise[..., 0:1]) * (0.85 * darkness)[..., None]
        colours.append(np.clip(colour, 0, 1).reshape(-1, 3))
        # Only real walls: slopes at the angle of repose drop less than a cell per cell.
        visible.append((active & (drop > resolution) & (slot < np.ceil(drop / resolution * 2))).reshape(-1))
    return (np.concatenate(points), np.concatenate(normals), np.concatenate(colours), np.concatenate(visible))
