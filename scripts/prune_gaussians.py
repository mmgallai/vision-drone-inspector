#!/usr/bin/env python3
"""
Post-training Gaussian pruning for splatfacto checkpoints.

Removes all Gaussians outside a specified world-space bounding box, then
saves a new checkpoint that the nerfstudio viewer can load directly.

Usage:
  python3 scripts/prune_gaussians.py <config.yml> [--xy-radius 1.0] [--z-min -0.2] [--z-max 0.8]
"""

import sys, json, argparse
import numpy as np
import torch
from pathlib import Path


def load_dataparser_transforms(output_dir: Path):
    p = output_dir / 'dataparser_transforms.json'
    d = json.loads(p.read_text())
    R  = np.array([row[:3] for row in d['transform']], dtype=np.float64)
    t  = np.array([row[3]  for row in d['transform']], dtype=np.float64)
    sc = float(d['scale'])
    return R, t, sc


def ns_to_world(means_ns: np.ndarray, R, t, sc) -> np.ndarray:
    # nerfstudio normalized: ns = sc * (R @ world + t)  →  world = R.T @ (ns/sc - t)
    return (R.T @ (means_ns.T / sc - t[:, None])).T


def _load_target_position(output_dir: Path) -> tuple:
    """Read target_position out of transforms.json so the prune box is
    centered on the actual SAM3-estimated target, not always (0,0,0).
    Looks in the data dir written by image_capture (output_dir's grandparent
    in the splatfacto path layout). Falls back to (0,0,0) if missing."""
    # config_path layout:
    #   <recon_output>/3dgs/<recon_output>/splatfacto/<run>/config.yml
    # so the original data dir is two levels up from config_path's parent.
    candidates = [
        output_dir.parent.parent.parent.parent / 'transforms.json',  # recon_output/transforms.json
        output_dir.parent.parent / 'transforms.json',                # fallback
        output_dir / 'transforms.json',
    ]
    for tf_path in candidates:
        if tf_path.exists():
            try:
                tf = json.loads(tf_path.read_text())
                tgt = tf.get('target_position')
                if tgt:
                    print(f'Target read from {tf_path}: '
                          f'({tgt["x"]:.3f}, {tgt["y"]:.3f}, {tgt["z"]:.3f})')
                    return float(tgt['x']), float(tgt['y']), float(tgt['z'])
            except Exception as e:
                print(f'WARN: could not read target from {tf_path}: {e}')
    print('No target_position in transforms.json — defaulting to origin')
    return 0.0, 0.0, 0.0


def prune(config_path: Path, xy_radius: float, z_min: float, z_max: float,
          target_xyz: tuple | None = None):
    # Locate checkpoint
    output_dir = config_path.parent
    # nerfstudio always saves checkpoints to nerfstudio_models/
    model_dir  = output_dir / 'nerfstudio_models'

    ckpts = sorted(model_dir.glob('step-*.ckpt'))
    if not ckpts:
        print(f'ERROR: no checkpoints in {model_dir}'); sys.exit(1)
    ckpt_path = ckpts[-1]
    print(f'Loading  : {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pipeline = ckpt['pipeline']

    means_ns = pipeline['_model.gauss_params.means'].numpy()
    print(f'Gaussians before pruning: {len(means_ns):,}')

    # Convert to world space
    R, t, sc = load_dataparser_transforms(output_dir)
    means_w   = ns_to_world(means_ns, R, t, sc)

    # Recenter the prune box at the SAM3-estimated target. Resolve target
    # position: explicit CLI override → transforms.json → world origin.
    if target_xyz is None:
        tgt_x, tgt_y, tgt_z = _load_target_position(output_dir)
    else:
        tgt_x, tgt_y, tgt_z = target_xyz
        print(f'Target (CLI override): ({tgt_x:.3f}, {tgt_y:.3f}, {tgt_z:.3f})')

    dx = means_w[:, 0] - tgt_x
    dy = means_w[:, 1] - tgt_y
    dz = means_w[:, 2] - tgt_z   # z range is relative to target ground level
    xy = np.sqrt(dx**2 + dy**2)
    keep = (xy <= xy_radius) & (dz >= z_min) & (dz <= z_max)
    print(f'Keeping  : {keep.sum():,}  ({100*keep.mean():.1f}%)  '
          f'[xy≤{xy_radius}m, z∈[{z_min},{z_max}]m around target]')
    print(f'Removing : {(~keep).sum():,} floaters')

    idx = torch.from_numpy(np.where(keep)[0])

    # All per-Gaussian tensors share the same first dimension
    gauss_keys = [k for k in pipeline if '_model.gauss_params.' in k]
    for key in gauss_keys:
        t_orig = pipeline[key]
        pipeline[key] = t_orig[idx]
        print(f'  {key}: {list(t_orig.shape)} → {list(pipeline[key].shape)}')

    # Save pruned checkpoint alongside original
    pruned_path = ckpt_path.with_name(ckpt_path.stem + '_pruned.ckpt')
    torch.save(ckpt, pruned_path)
    print(f'\nSaved    : {pruned_path}')
    print(f'View with: ~/bin/ns-viewer --load-checkpoint {pruned_path} '
          f'--load-config {config_path}')
    return pruned_path, config_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=Path)
    ap.add_argument('--xy-radius', type=float, default=1.0,
                    help='Keep Gaussians within this XY radius from target (m)')
    ap.add_argument('--z-min',    type=float, default=-0.15,
                    help='Minimum Z relative to target ground level (m)')
    ap.add_argument('--z-max',    type=float, default=0.80,
                    help='Maximum Z relative to target ground level (m)')
    ap.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                    default=None,
                    help='Override target position (otherwise read from '
                         'transforms.json target_position field)')
    args = ap.parse_args()
    target = tuple(args.target) if args.target else None
    prune(args.config.expanduser(), args.xy_radius, args.z_min, args.z_max, target)
