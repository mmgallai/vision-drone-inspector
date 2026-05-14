"""
Reconstruction with Depth Anything 3 — joint depth + pose + colored
point cloud from an arbitrary set of views.

Used as the default reconstruction method for whole-scene mapping
(mission_mode=mapping) where splatfacto's tight-orbit assumption breaks
down. See docs/depth_anything_3.md for install instructions and rationale.

Usage (after capturing images via the launch with auto_recon:=false):

    ros2 run drone_recon run_recon_da3 ~/recon_output
    ros2 run drone_recon run_recon_da3 ~/recon_output --output-name myroom.ply

Output: a single colored PLY in <output_dir>/exports/.

This module shells out to the `da3` CLI rather than importing the
package directly. Three reasons:
  1. DA3 installs xformers + a GB-scale model into user-site, which
     shouldn't be loaded by ROS nodes that don't need it.
  2. The CLI is the API contract maintained by the upstream repo —
     less likely to break across DA3 versions.
  3. We can run it under our own clean PYTHONPATH (same trick as
     ns-train) without leaking the dependency into other nodes.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# DA3 lives in its own venv (~/.venvs/da3) with --system-site-packages so it
# inherits user-site torch. But because ROS sources /opt/ros/jazzy/setup.bash
# into the parent shell, /usr/lib/python3/dist-packages ends up earlier in
# sys.path than the venv's site-packages. That makes the system's click
# 8.1.6 (no Choice[T] support) shadow the venv's click 8.3.3, which breaks
# typer at import time. Strip every path that competes with the venv before
# invoking da3.
_USER_SITE   = str(Path.home() / '.local/lib/python3.12/site-packages')
_BAD_PYPATHS = {
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.12/dist-packages',
    '/opt/ros/jazzy/lib/python3.12/site-packages',
}


def _da3_env() -> dict:
    """Build a clean env for the da3 CLI.

    The DA3 venv was made with --system-site-packages so it could reuse
    the user's GPU-enabled torch without redownloading 3 GB. The
    side-effect is that the system dist-packages and ROS site-packages
    dirs land HIGHER in sys.path than the venv's own site-packages — and
    they ship older `click` / `typing_extensions` that crash modern
    typer / tyro at import time. PYTHONPATH entries are prepended to
    sys.path BEFORE the standard dirs, so we explicitly put the venv's
    site-packages first."""
    env = os.environ.copy()
    parts = [p for p in env.get('PYTHONPATH', '').split(':')
             if p and p not in _BAD_PYPATHS]
    venv_sp = str(Path.home() / '.venvs/da3/lib/python3.12/site-packages')
    env['PYTHONPATH'] = ':'.join([venv_sp, _USER_SITE] + parts)
    env.pop('PYTHONNOUSERSITE', None)
    # Cuts allocator fragmentation — last 200 MB of an 8 GB GPU make the
    # difference between OOM and success on the GIANT-LARGE model.
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def _find_da3_cli() -> str | None:
    """Return the absolute path to the `da3` CLI, or None if not found.
    Searched in priority order:
      1. PATH
      2. The dedicated DA3 venv (~/.venvs/da3/bin/da3) — needed because
         DA3 requires numpy<2 which conflicts with the rest of our stack,
         so the recommended install is into an isolated venv.
      3. ~/.local/bin/da3 (a `pip install --user` install)
    """
    p = shutil.which('da3')
    if p:
        return p
    venv_bin = Path.home() / '.venvs/da3/bin/da3'
    if venv_bin.exists():
        return str(venv_bin)
    user_bin = Path.home() / '.local/bin/da3'
    if user_bin.exists():
        return str(user_bin)
    return None


def _images_dir(output_dir: Path) -> Path | None:
    """Pick the directory we should hand to DA3. DA3 only accepts a
    *flat* directory of images (no subdirectories), so when image_capture
    has split captures into `low_ring/` and `high_ring/`, we stage all
    PNGs into a flat `images_da3/` directory using symlinks first.
    """
    images = output_dir / 'images'
    if not images.exists():
        return None

    # Find every image. Prefer a flat layout if it already is flat.
    pngs = sorted(images.glob('*.png'))
    if pngs:
        return images

    # Otherwise stage symlinks to `images_da3/` so DA3 sees a flat dir.
    flat = output_dir / 'images_da3'
    flat.mkdir(exist_ok=True)
    # Clear stale symlinks from prior runs
    for old in flat.glob('*'):
        try:
            old.unlink()
        except OSError:
            pass
    for ring in ('low_ring', 'high_ring'):
        ring_dir = images / ring
        if not ring_dir.exists():
            continue
        for img in sorted(ring_dir.glob('*.png')):
            (flat / f'{ring}__{img.name}').symlink_to(img.resolve())

    if not list(flat.glob('*.png')):
        return None
    return flat


def run(output_dir: Path, output_name: str = 'scene_da3.ply') -> int:
    """
    Run Depth Anything 3 against the captures in <output_dir>/images/.
    Returns 0 on success, non-zero otherwise.
    Logs to <output_dir>/reconstruction.log so the user can `tail -f`
    the same file used by the splatfacto pipeline.
    """
    output_dir = output_dir.expanduser().resolve()

    cli = _find_da3_cli()
    if cli is None:
        sys.stderr.write(
            '[recon] depth_anything_3 not on PATH.\n'
            '        Install per docs/depth_anything_3.md, or pass\n'
            '        recon_method:=splatfacto in the launch.\n')
        return 2

    images = _images_dir(output_dir)
    if images is None:
        sys.stderr.write(f'[recon] no images dir at {output_dir/"images"}\n')
        return 3

    export_dir = output_dir / 'exports'
    export_dir.mkdir(exist_ok=True)
    log_path = output_dir / 'reconstruction.log'

    # Use a dedicated subdir for DA3 — keeps it from colliding with the
    # splatfacto exports and avoids DA3's interactive "clean existing dir?"
    # prompt that would block our subprocess.
    da3_dir = export_dir / 'da3_run'
    if da3_dir.exists():
        # Clean ourselves so DA3 sees an empty target dir and skips the prompt.
        for p in da3_dir.iterdir():
            try:
                if p.is_dir():
                    import shutil as _sh
                    _sh.rmtree(p)
                else:
                    p.unlink()
            except OSError:
                pass

    cmd = [
        cli, 'auto', str(images),
        # GLB (point cloud + camera wireframes) is the format every DA3
        # variant supports. Plain `ply`/`gs_ply` are giant-only. We strip
        # the camera wireframes and convert to a colored PLY below.
        '--export-format', 'glb',
        '--no-show-cameras',
        '--export-dir',    str(da3_dir),
        # Default GIANT-LARGE (1.40B params) eats 7+ GB on its own and OOMs
        # on 8 GB cards. LARGE (0.35B params) fits in ~2 GB and is what the
        # DA3 README recommends for "most use cases".
        '--model-dir',     'depth-anything/DA3-LARGE',
        '--process-res',   '392',
    ]
    print(f'[recon] DA3 starting on {images}')
    print(f'[recon]   {" ".join(cmd)}')
    rc = _stream_run(cmd, log_path,
                     banner='\n=== depth_anything_3 (auto) ===\n'
                            f'  {" ".join(cmd)}\n\n',
                     env=_da3_env(), stdin_input='y\n')
    if rc != 0:
        sys.stderr.write(f'[recon] DA3 failed (rc={rc}) — see {log_path}\n')
        return rc

    # DA3 writes a GLB containing a colored point cloud (and, with
    # --show-cameras, camera wireframes — we disabled those above).
    # Convert it to PLY for downstream Poisson + viewer compatibility.
    glbs = sorted(da3_dir.rglob('*.glb'), key=lambda p: p.stat().st_mtime)
    if not glbs:
        sys.stderr.write(
            f'[recon] DA3 reported success but no GLB found under {da3_dir}\n')
        return 4
    latest_glb = glbs[-1]
    target = export_dir / output_name
    if target.exists():
        target.unlink()
    # trimesh ships in the DA3 venv (DA3 itself uses it for GLB export).
    # Run conversion under that interpreter so we don't drag trimesh into
    # the ROS env.
    venv_python = Path.home() / '.venvs/da3/bin/python'
    conv_code = (
        'import sys, trimesh, numpy as np\n'
        'src, dst = sys.argv[1], sys.argv[2]\n'
        'scene = trimesh.load(src, force="scene")\n'
        'pts = []; cols = []\n'
        'for g in scene.geometry.values():\n'
        '    if isinstance(g, trimesh.PointCloud):\n'
        '        pts.append(np.asarray(g.vertices))\n'
        '        if g.colors is not None and len(g.colors):\n'
        '            cols.append(np.asarray(g.colors)[:, :3])\n'
        'if not pts:\n'
        '    raise SystemExit("no PointCloud in GLB")\n'
        'pts = np.concatenate(pts, axis=0)\n'
        'cols = np.concatenate(cols, axis=0) if cols else None\n'
        'pc = trimesh.PointCloud(pts, colors=cols)\n'
        'pc.export(dst)\n'
        'print(f"[glb2ply] {len(pts):,} points -> {dst}")\n'
    )
    crc = _stream_run([str(venv_python), '-c', conv_code,
                       str(latest_glb), str(target)],
                      log_path,
                      banner='\n=== glb -> ply ===\n')
    if crc != 0 or not target.exists():
        sys.stderr.write(f'[recon] GLB->PLY conversion failed (rc={crc})\n')
        return 5
    print(f'[recon] DA3 wrote → {target}')

    # Bonus: derive a triangle-mesh PLY from the colored point cloud via
    # Open3D's Poisson reconstruction. Open3D ships in the DA3 venv, so
    # we use that interpreter (NOT the system Python) to avoid pulling
    # open3d into the ROS env. Failure is non-fatal — the point cloud is
    # the primary deliverable; the mesh is value-added.
    mesh_path = export_dir / 'scene_mesh.ply'
    venv_python = Path.home() / '.venvs/da3/bin/python'
    poisson_script = None
    for candidate in (
        Path(__file__).resolve().parent.parent / 'scripts' / 'poisson_mesh.py',
        Path('/home/mgallai/ros2_ws/install/drone_recon/share/drone_recon/'
             'scripts/poisson_mesh.py'),
    ):
        if candidate.exists():
            poisson_script = candidate
            break
    if venv_python.exists() and poisson_script is not None:
        print(f'[recon] Poisson-meshing DA3 point cloud → {mesh_path.name}')
        mrc = _stream_run([str(venv_python), str(poisson_script),
                           str(target), str(mesh_path)],
                          log_path,
                          banner='\n=== open3d poisson mesh ===\n')
        if mrc == 0 and mesh_path.exists():
            print(f'[recon] mesh wrote → {mesh_path}')
        else:
            print(f'[recon] mesh export failed (rc={mrc}); see {log_path}')
    else:
        miss = 'venv' if not venv_python.exists() else 'script'
        print(f'[recon] skipping Poisson mesh export ({miss} missing)')

    # Auto-open both Open3D windows (point cloud + mesh) so the user
    # gets immediate visual feedback the moment recon finishes.
    # Non-blocking: spawned with start_new_session so each viewer is
    # detached from this process and from each other.
    _launch_viewers(target, mesh_path if mesh_path.exists() else None,
                    venv_python, log_path)

    return 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stream_run(cmd, log_path: Path, *, banner: str = '',
                env=None, stdin_input: str | None = None) -> int:
    """
    Run a subprocess, streaming its stdout/stderr to BOTH the terminal
    and the recon log file in real time. This is what gives the user
    live progress for DA3's tqdm bars and the [poisson] step messages
    instead of a long opaque silence followed by a wall of log text.

    Returns the subprocess exit code.
    """
    with open(log_path, 'a') as logf:
        if banner:
            logf.write(banner)
            logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_input is not None else None,
            text=True,
            bufsize=1,
            env=env,
        )
        if stdin_input is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_input)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
        return proc.returncode


def _launch_viewers(pcd_path: Path, mesh_path: Path | None,
                    venv_python: Path, log_path: Path) -> None:
    """Pop the DA3 point cloud and the Poisson mesh into separate
    Open3D windows. Non-blocking; failures here are silent (the user
    can always open the PLYs manually from ~/recon_output/exports/)."""
    viewer_script = None
    for candidate in (
        Path(__file__).resolve().parent.parent / 'scripts' / 'view_results.py',
        Path('/home/mgallai/ros2_ws/install/drone_recon/share/drone_recon/'
             'scripts/view_results.py'),
    ):
        if candidate.exists():
            viewer_script = candidate
            break
    if viewer_script is None or not venv_python.exists():
        return

    def _spawn(target: Path, title: str):
        try:
            with open(log_path, 'a') as f:
                subprocess.Popen(
                    [str(venv_python), str(viewer_script),
                     str(target), title],
                    stdout=f, stderr=subprocess.STDOUT,
                    start_new_session=True)
        except OSError as e:
            print(f'[recon] failed to open viewer for {target.name}: {e}')

    if pcd_path.exists():
        _spawn(pcd_path, 'DA3 point cloud')
    if mesh_path is not None:
        _spawn(mesh_path, 'Poisson mesh')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', type=Path,
                    help='Capture directory (must contain images/)')
    ap.add_argument('--output-name', default='scene_da3.ply',
                    help='Filename for the colored PLY in <output_dir>/exports/')
    args = ap.parse_args(argv)
    return run(args.output_dir, args.output_name)


if __name__ == '__main__':
    sys.exit(main())
