"""
Standalone reconstruction pipeline runner.

Use this when you've already captured images (with auto_recon=False, or
you killed the launch before the pipeline finished, or you want to
re-run with different prune settings) and just want to run the 3DGS
pipeline against an existing recon_output/ directory.

Usage:
  python3 -m drone_recon.run_recon ~/recon_output --target "fire hydrant"

Pipeline (mirrors image_capture._run_reconstruction):
  1. gen_init_pointcloud.py   — synthetic seed cloud
  2. ns-train splatfacto      — 30k iterations
  3. prune_gaussians.py       — drop floaters outside target box
  4. ns-export gaussian-splat — portable .ply for any 3DGS viewer
  5. ns-viewer                — live viewer at localhost:7007
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from drone_recon import targets as _targets


_NS_TRAIN  = str(Path.home() / '.local/bin/ns-train')
_NS_VIEWER = str(Path.home() / '.local/bin/ns-viewer')
_NS_EXPORT = str(Path.home() / '.local/bin/ns-export')

# Same env-cleanup logic as image_capture._ns_env. Strip ROS-injected
# /usr/lib/python3/dist-packages so tyro / nerfstudio see the modern
# typing_extensions from user-site.
_USER_SITE   = str(Path.home() / '.local/lib/python3.12/site-packages')
_BAD_PYPATHS = {
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.12/dist-packages',
}


def _ns_env() -> dict:
    env = os.environ.copy()
    parts = [p for p in env.get('PYTHONPATH', '').split(':')
             if p and p not in _BAD_PYPATHS]
    env['PYTHONPATH'] = ':'.join([_USER_SITE] + parts)
    env.pop('PYTHONNOUSERSITE', None)
    return env


def _scripts_dir() -> Path:
    """Resolve the pipeline scripts dir whether running from source or install."""
    candidates = [
        Path(__file__).resolve().parent.parent / 'scripts',     # source tree
        Path('/home/mgallai/ros2_ws/install/drone_recon/share/drone_recon/scripts'),
    ]
    for c in candidates:
        if (c / 'gen_init_pointcloud.py').exists():
            return c
    raise FileNotFoundError('Could not find drone_recon/scripts/')


def run(output_dir: Path, target_prompt: str,
        max_iters: int = 30000,
        skip_train: bool = False,
        skip_viewer: bool = False,
        auto_prune: bool = True) -> int:
    """
    Run the recon pipeline. Returns 0 on full success, non-zero otherwise.
    Logs everything to <output_dir>/reconstruction.log (same path image_capture
    uses, so you can `tail -f` either way).
    """
    output_dir = output_dir.expanduser().resolve()
    if not (output_dir / 'transforms.json').exists():
        print(f'ERROR: {output_dir} has no transforms.json — capture images first')
        return 1

    target_cfg = _targets.get(target_prompt)
    log_path   = output_dir / 'reconstruction.log'
    scripts    = _scripts_dir()

    def step(cmd, label) -> bool:
        print(f'  [{label}] starting...')
        with open(log_path, 'a') as f:
            f.write(f'\n=== {label} ===\n')
            r = subprocess.run(cmd, stdout=f, stderr=f, text=True, env=_ns_env())
        if r.returncode != 0:
            print(f'  [{label}] FAILED (rc={r.returncode}) — see {log_path}')
            return False
        print(f'  [{label}] done')
        return True

    # 1 — synthetic seed cloud
    if not step([sys.executable, str(scripts / 'gen_init_pointcloud.py'),
                 str(output_dir), '--target', target_prompt],
                'gen_init_pointcloud'):
        return 2

    # 2 — splatfacto training
    if not skip_train:
        print(f'  [ns-train splatfacto] starting  (~10-15 min) ...')
        train_cmd = [
            _NS_TRAIN, 'splatfacto',
            '--data',                            str(output_dir),
            '--output-dir',                      str(output_dir / '3dgs'),
            '--max-num-iterations',              str(max_iters),
            '--viewer.quit-on-train-completion', 'True',
        ]
        progress_re = re.compile(r'(\d+)\s+\((\d+\.\d+)%\)')
        with open(log_path, 'a') as logf:
            logf.write('\n=== ns-train splatfacto ===\n')
            proc = subprocess.Popen(train_cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, env=_ns_env())
            last_pct = -1.0
            for line in proc.stdout:
                logf.write(line); logf.flush()
                m = progress_re.search(line)
                if m:
                    pct = float(m.group(2))
                    if pct - last_pct >= 1.0:
                        last_pct = pct
                        bar = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
                        sys.stdout.write(
                            f'\r  3DGS training  [{bar}]  {pct:5.1f}%  '
                            f'(step {m.group(1)}/{max_iters})')
                        sys.stdout.flush()
            proc.wait()
        sys.stdout.write('\n'); sys.stdout.flush()
        if proc.returncode != 0:
            print(f'  [ns-train] FAILED (rc={proc.returncode}) — see {log_path}')
            return 3

    # 3 — find latest config
    configs = sorted((output_dir / '3dgs').glob('**/config.yml'),
                     key=lambda p: p.stat().st_mtime)
    if not configs:
        print('  [recon] ERROR: no config.yml after training')
        return 4
    config_path = configs[-1]

    # 3b — whole-scene splat export (before prune so it includes walls /
    # barrier / surrounding obstacles, not just the target). RGB color
    # mode produces a standard XYZ+RGB PLY any tool can open.
    export_dir = output_dir / 'exports'
    export_dir.mkdir(exist_ok=True)
    print('  [ns-export scene_splat.ply] writing whole-scene PLY ...')
    with open(log_path, 'a') as f:
        f.write('\n=== ns-export gaussian-splat (full scene, RGB) ===\n')
        r_pc = subprocess.run(
            [_NS_EXPORT, 'gaussian-splat',
             '--load-config',     str(config_path),
             '--output-dir',      str(export_dir),
             '--output-filename', 'scene_splat.ply',
             '--ply-color-mode',  'rgb'],
            stdout=f, stderr=f, env=_ns_env())
    if r_pc.returncode != 0:
        print('  [scene_splat.ply] WARN: failed — see reconstruction.log')
    else:
        print(f'  [scene_splat.ply] → {export_dir / "scene_splat.ply"}')

    # 4 — prune to target's bbox (skipped when auto_prune=False; the
    # whole-scene splat then stays canonical for the viewer).
    if auto_prune:
        box = target_cfg['prune_box']
        if not step([sys.executable, str(scripts / 'prune_gaussians.py'),
                     str(config_path),
                     '--xy-radius', str(box['xy_radius']),
                     '--z-min',     str(box['z_min']),
                     '--z-max',     str(box['z_max'])],
                    'prune_gaussians'):
            return 5
    else:
        print('  [prune_gaussians] skipped (auto_prune=False)')

    # 5 — swap pruned ckpt into the canonical step-N.ckpt slot.
    # In auto_prune=False mode there's nothing to swap.
    models_dir = config_path.parent / 'nerfstudio_models'
    if auto_prune:
        pruned_list = sorted(models_dir.glob('*_pruned.ckpt'),
                             key=lambda p: p.stat().st_mtime)
        if not pruned_list:
            print('  [recon] ERROR: no pruned checkpoint produced')
            return 6
        pruned_path = pruned_list[-1]
        canonical   = models_dir / pruned_path.name.replace('_pruned.ckpt', '.ckpt')
        if canonical.exists() and canonical != pruned_path:
            backup = (config_path.parent / 'archive')
            backup.mkdir(exist_ok=True)
            canonical.rename(backup / canonical.name.replace('.ckpt', '_full.ckpt'))
        pruned_path.rename(canonical)
        pruned_path = canonical
        print(f'  [pruned ckpt] → {pruned_path}')
    else:
        full = sorted((p for p in models_dir.glob('step-*.ckpt')
                       if '_pruned' not in p.name),
                      key=lambda p: p.stat().st_mtime)
        if not full:
            print('  [recon] ERROR: no checkpoint found')
            return 6
        pruned_path = full[-1]
        print(f'  [whole-scene ckpt] → {pruned_path}')

    # 6 — export portable PLY (only meaningful when pruned — otherwise
    # it duplicates scene_splat.ply written above).
    if auto_prune:
        if not step([_NS_EXPORT, 'gaussian-splat',
                     '--load-config',     str(config_path),
                     '--output-dir',      str(export_dir),
                     '--output-filename', 'splat_pruned.ply'],
                    'ns-export gaussian-splat (pruned)'):
            print('  [warn] ns-export failed but continuing')

    # 7 — viewer (non-blocking)
    if not skip_viewer:
        print('  [ns-viewer] launching at http://localhost:7007 ...')
        with open(log_path, 'a') as f:
            subprocess.Popen([_NS_VIEWER, '--load-config', str(config_path)],
                             stdout=f, stderr=f, env=_ns_env())

    print('')
    print('=' * 60)
    print(' 3D RECONSTRUCTION COMPLETE')
    print(f'   PLY     → {export_dir / "splat_pruned.ply"}')
    print(f'   Viewer  → http://localhost:7007')
    print(f'   Log     → {log_path}')
    print('=' * 60)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', type=Path,
                    help='Capture directory (must contain transforms.json)')
    ap.add_argument('--target', default='fire hydrant',
                    help='Target prompt — drives prune box + init shape '
                         '(see drone_recon.targets.TARGETS)')
    ap.add_argument('--max-iters', type=int, default=30000,
                    help='ns-train splatfacto iteration count')
    ap.add_argument('--skip-train', action='store_true',
                    help='Skip training (use most recent config.yml)')
    ap.add_argument('--skip-viewer', action='store_true',
                    help='Skip launching ns-viewer')
    ap.add_argument('--no-prune', action='store_true',
                    help='Skip prune_gaussians; keep whole-scene splat as '
                         'canonical (good for room-mapping runs)')
    args = ap.parse_args(argv)
    return run(args.output_dir, args.target,
               max_iters=args.max_iters,
               skip_train=args.skip_train,
               skip_viewer=args.skip_viewer,
               auto_prune=not args.no_prune)


if __name__ == '__main__':
    sys.exit(main())
