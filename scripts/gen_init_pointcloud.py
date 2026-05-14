#!/usr/bin/env python3
"""
Generate a synthetic initialization point cloud for splatfacto.

Produces a PLY file with points sampled on the surface of a target object
at the world origin. Nerfstudio reads this via "ply_file_path" in transforms.json
and uses it to initialize 3D Gaussian positions instead of random initialization.

Shape is selected by --target (looked up in drone_recon.targets.TARGETS):
  fire hydrant   →  hydrant  (cylinder body + dome + nozzles)
  car / default  →  cylinder (parameterized radius/height)
  traffic cone   →  cylinder (small, orange)

Usage:
  python3 scripts/gen_init_pointcloud.py ~/recon_output
  python3 scripts/gen_init_pointcloud.py ~/recon_output --target car
"""

import sys
import json
import struct
import math
import random
import argparse
import numpy as np
from pathlib import Path

# Allow the script to import the shared targets table whether it's run from
# the source tree or the installed share/ directory.
_THIS = Path(__file__).resolve()
_PKG_PARENTS = [
    _THIS.parent.parent,                                  # source: src/drone_recon
    Path.home() / 'ros2_ws/src',                          # source root fallback
    Path('/home/mgallai/ros2_ws/install/drone_recon/lib/python3.12/site-packages'),
]
for _p in _PKG_PARENTS:
    if (_p / 'drone_recon' / 'targets.py').exists():
        sys.path.insert(0, str(_p))
        break
try:
    from drone_recon import targets as _targets
except ImportError:
    _targets = None


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a binary PLY file with XYZ (+ optional RGB) points."""
    n = len(points)
    has_color = colors is not None

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines += ["end_header", ""]
    header = "\n".join(header_lines).encode()

    with open(path, "wb") as f:
        f.write(header)
        for i in range(n):
            x, y, z = float(points[i, 0]), float(points[i, 1]), float(points[i, 2])
            f.write(struct.pack("<fff", x, y, z))
            if has_color:
                r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
                f.write(struct.pack("<BBB", r, g, b))


def hydrant_points(n_points: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points on a simplified fire hydrant shape at the world origin.

    The hydrant model in the scene sits at (0, 0, 0) with its base on the ground.
    Approximate geometry:
      - Main body: cylinder  r=0.12m, z=0.00..0.45m
      - Dome cap:  hemisphere r=0.12m, center z=0.45m
      - Lateral nozzles: two small cylinders at z=0.30, offset ±Y by 0.12m
    """
    rng = random.Random(seed)

    pts = []
    cols = []

    def red():
        return (200 + rng.randint(0, 40), rng.randint(0, 30), rng.randint(0, 30))

    # Main body cylinder (lateral surface)
    n_body = int(n_points * 0.55)
    R_body, z_lo, z_hi = 0.12, 0.0, 0.45
    for _ in range(n_body):
        theta = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(z_lo, z_hi)
        pts.append([R_body * math.cos(theta), R_body * math.sin(theta), z])
        cols.append(red())

    # Dome cap (hemisphere on top of body)
    n_dome = int(n_points * 0.20)
    for _ in range(n_dome):
        # Rejection sample upper hemisphere
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            z = rng.uniform(0, 1)
            if x * x + y * y + z * z <= 1.0:
                break
        r = math.sqrt(x * x + y * y + z * z)
        pts.append([R_body * x / r, R_body * y / r, 0.45 + R_body * z / r])
        cols.append(red())

    # Top nut (small cylinder on dome)
    n_nut = int(n_points * 0.05)
    R_nut = 0.04
    for _ in range(n_nut):
        theta = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(0.57, 0.63)
        pts.append([R_nut * math.cos(theta), R_nut * math.sin(theta), z])
        cols.append(red())

    # Lateral nozzle caps (front/back)
    n_nozzle = int(n_points * 0.20)
    R_nozzle, nz_y, nz_z = 0.05, 0.17, 0.28
    for side in (-1, 1):
        for _ in range(n_nozzle // 2):
            theta = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, R_nozzle)
            # Disc face (cap) in XZ plane at y = ±nz_y
            pts.append([r * math.cos(theta), side * nz_y, nz_z + r * math.sin(theta)])
            cols.append(red())

    pts_arr = np.array(pts, dtype=np.float32)
    col_arr = np.array(cols, dtype=np.uint8)
    return pts_arr, col_arr


def cylinder_points(n_points: int = 2000,
                    radius: float = 0.5,
                    height: float = 1.0,
                    color: tuple = (180, 180, 180),
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on a cylinder of the given radius/height standing
    on the ground (z = 0..height). Used as a generic target seed."""
    rng = random.Random(seed)
    pts, cols = [], []
    cr, cg, cb = color
    for _ in range(n_points):
        theta = rng.uniform(0, 2 * math.pi)
        z     = rng.uniform(0, height)
        pts.append([radius * math.cos(theta), radius * math.sin(theta), z])
        cols.append((cr + rng.randint(-20, 20),
                     cg + rng.randint(-20, 20),
                     cb + rng.randint(-20, 20)))
    arr = np.array(pts, dtype=np.float32)
    col = np.clip(np.array(cols), 0, 255).astype(np.uint8)
    return arr, col


def sphere_points(n_points: int = 2000,
                  radius: float = 0.5,
                  color: tuple = (180, 180, 180),
                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on a sphere centered above the ground at z=radius."""
    rng = random.Random(seed)
    pts, cols = [], []
    cr, cg, cb = color
    for _ in range(n_points):
        # Rejection-sample a unit vector, then scale.
        while True:
            x = rng.uniform(-1, 1); y = rng.uniform(-1, 1); z = rng.uniform(-1, 1)
            r2 = x*x + y*y + z*z
            if 0.01 < r2 <= 1.0:
                break
        r = math.sqrt(r2)
        pts.append([radius * x / r, radius * y / r, radius + radius * z / r])
        cols.append((cr + rng.randint(-20, 20),
                     cg + rng.randint(-20, 20),
                     cb + rng.randint(-20, 20)))
    arr = np.array(pts, dtype=np.float32)
    col = np.clip(np.array(cols), 0, 255).astype(np.uint8)
    return arr, col


_SHAPE_FUNCS = {
    'hydrant':  hydrant_points,
    'cylinder': cylinder_points,
    'sphere':   sphere_points,
}


def gen(output_dir: Path, target: str = 'fire hydrant') -> None:
    transforms_path = output_dir / "transforms.json"
    if not transforms_path.exists():
        print(f"ERROR: transforms.json not found in {output_dir}")
        sys.exit(1)

    # Pull shape + args from the shared targets table when available.
    if _targets is not None:
        cfg   = _targets.get(target)
        shape = cfg['init_shape']
        args  = dict(cfg['init_args'])
    else:
        shape = 'hydrant'
        args  = {}

    fn = _SHAPE_FUNCS.get(shape)
    if fn is None:
        print(f"ERROR: unknown init_shape '{shape}'")
        sys.exit(1)

    pts, cols = fn(n_points=3000, **args)
    print(f"Generated {len(pts)} points using shape='{shape}' for target='{target}'")

    # Recenter the seed cloud at the actual SAM3-estimated target. The shape
    # functions all return points centered at the world origin (because they
    # don't know where the target is); transforms.json carries the runtime
    # target_position so the seed lines up with what was actually filmed.
    with open(transforms_path) as f:
        tf = json.load(f)
    tgt = tf.get('target_position', {'x': 0.0, 'y': 0.0, 'z': 0.0})
    tx, ty, tz = float(tgt['x']), float(tgt['y']), float(tgt['z'])
    if abs(tx) + abs(ty) + abs(tz) > 1e-6:
        print(f"Recentering seed at target ({tx:.3f}, {ty:.3f}, {tz:.3f})")
        pts[:, 0] += tx
        pts[:, 1] += ty
        pts[:, 2] += tz

    ply_path = output_dir / "init_points.ply"
    write_ply(ply_path, pts, cols)
    print(f"Written {len(pts)} points → {ply_path}")

    # Patch transforms.json to reference the point cloud
    tf["ply_file_path"] = "init_points.ply"
    with open(transforms_path, "w") as f:
        json.dump(tf, f, indent=2)
    print(f"Updated transforms.json: ply_file_path = init_points.ply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--target', default='fire hydrant',
                    help='Target prompt (looked up in drone_recon.targets.TARGETS)')
    args = ap.parse_args()
    gen(args.output_dir.expanduser(), args.target)
