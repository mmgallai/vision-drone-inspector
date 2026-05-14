#!/usr/bin/env python3
"""
Image Capture Node — drone_recon
==================================
Saves images and pose metadata when triggered by the mission node.
Output is formatted for 3D Gaussian Splatting (nerfstudio / gsplat).

Output structure:
  ~/recon_output/
    images/
      low_ring/   capture_001_000.0deg.png ...
      high_ring/  capture_025_000.0deg.png ...
    poses.json          — per-image position + orientation + SAM3 metadata
    cameras.json        — camera intrinsics
    transforms.json     — NeRF/3DGS standard camera transform format

Topics subscribed:
  /drone/camera/image_raw  (sensor_msgs/Image)        - live camera feed
  /drone/pose              (geometry_msgs/PoseStamped) - drone world pose
  /mission/capture         (std_msgs/Bool)             - save trigger
  /mission/state           (std_msgs/String)           - current ring (LOW/HIGH)
  /sam3/detected           (std_msgs/Bool)             - SAM3 detection flag
  /sam3/distance           (std_msgs/Float32)          - estimated distance
"""

import os
import json
import math
import re
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from drone_recon._singleton import acquire_singleton
from drone_recon import targets as _targets

# Absolute paths to nerfstudio CLI tools (installed in user site, not on ROS PATH)
_NS_TRAIN  = str(Path.home() / '.local/bin/ns-train')
_NS_VIEWER = str(Path.home() / '.local/bin/ns-viewer')

# ROS sources /usr/lib/python3/dist-packages into PYTHONPATH (and into sys.path
# via .pth files) where typing_extensions 4.10 lives — too old for tyro
# (it needs NoDefault, added in 4.12+). Prepending user-site is not enough on
# its own because the dist-packages entry can still win via sitecustomize/.pth
# ordering, so we also strip the offending entries from PYTHONPATH and disable
# the system site dir for nerfstudio subprocesses.
_USER_SITE = str(Path.home() / '.local/lib/python3.12/site-packages')
_BAD_PYPATHS = {
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.12/dist-packages',
}
def _ns_env():
    env = os.environ.copy()
    parts = [p for p in env.get('PYTHONPATH', '').split(':')
             if p and p not in _BAD_PYPATHS]
    env['PYTHONPATH'] = ':'.join([_USER_SITE] + parts)
    # Ensure user site is enabled even if the parent had it disabled.
    env.pop('PYTHONNOUSERSITE', None)
    return env

# scripts/ installed to share/drone_recon/scripts/ via setup.py data_files
from ament_index_python.packages import get_package_share_directory as _pkg_share
_SCRIPTS_DIR = Path(_pkg_share('drone_recon')) / 'scripts'


# ── Camera intrinsics (must match SDF: 1280×720, hfov=90°) ───────────────────
IMG_W, IMG_H = 1280, 720
HFOV_DEG     = 90.0
FX = FY      = IMG_W / (2.0 * math.tan(math.radians(HFOV_DEG / 2.0)))  # = 640.0
CX, CY       = IMG_W / 2.0, IMG_H / 2.0   # principal point

# Output root (default — overridable via ROS param 'output_dir')
DEFAULT_OUTPUT_DIR = str(Path.home() / 'recon_output')

_BAR = '#' * 60

def _banner(msg: str):
    """Print a highlighted banner directly to stdout (bypasses ROS logger prefix)."""
    sys.stdout.write(f'\n{_BAR}\n# {msg}\n{_BAR}\n\n')
    sys.stdout.flush()

def _print(msg: str):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


class ImageCaptureNode(Node):

    def __init__(self):
        super().__init__('image_capture')

        self.declare_parameter('output_dir',    DEFAULT_OUTPUT_DIR)
        self.declare_parameter('target_prompt', 'fire hydrant')
        # auto_recon=False lets you iterate on the flight without paying the
        # ~15 min 3DGS training tax each time. Run scripts/run_recon.py later
        # against the saved output_dir when you're ready to reconstruct.
        self.declare_parameter('auto_recon',    True)
        # auto_prune=False keeps the full unpruned splat as the canonical
        # output. Useful for whole-scene reconstruction (you still get
        # scene_splat.ply either way, but with auto_prune=False the live
        # ns-viewer also shows the whole scene instead of just the target).
        self.declare_parameter('auto_prune',    True)
        # recon_method: 'both' (default) runs splatfacto (3D Gaussian
        # splat) AND DA3 (colored point cloud + Poisson mesh) so every
        # mission produces TWO deliverables. 'splatfacto' or
        # 'depth_anything_3' force a single method.
        self.declare_parameter('recon_method',  'both')
        OUTPUT_DIR = Path(self.get_parameter('output_dir').value)
        self.output_dir = OUTPUT_DIR
        self.target_prompt = self.get_parameter('target_prompt').value
        self.target_cfg    = _targets.get(self.target_prompt)
        self.auto_recon    = bool(self.get_parameter('auto_recon').value)
        self.auto_prune    = bool(self.get_parameter('auto_prune').value)
        self.recon_method  = self.get_parameter('recon_method').value
        if self.recon_method not in ('splatfacto', 'depth_anything_3', 'both'):
            self.get_logger().warn(
                f'Unknown recon_method "{self.recon_method}" — falling '
                f'back to both (splatfacto + DA3)')
            self.recon_method = 'both'

        # ── Output directories ──────────────────────────────────────────
        (OUTPUT_DIR / 'images' / 'low_ring').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'images' / 'high_ring').mkdir(parents=True, exist_ok=True)

        # Remove stale JSON files from any previous run so they can't corrupt
        # the new mission's output (guards against ghost-node race writes).
        for _stale in ('poses.json', 'transforms.json'):
            _p = OUTPUT_DIR / _stale
            if _p.exists():
                _p.unlink()

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Image,       '/drone/camera/image_raw', self._cb_image,    10)
        self.create_subscription(PoseStamped, '/drone/pose',             self._cb_pose,     10)
        self.create_subscription(PoseStamped, '/mission/target',         self._cb_target,   10)
        self.create_subscription(Bool,        '/mission/capture',        self._cb_capture,  10)
        self.create_subscription(String,      '/mission/state',          self._cb_state,    10)
        self.create_subscription(Bool,        '/sam3/detected',          self._cb_detected, 10)
        self.create_subscription(Float32,     '/sam3/distance',          self._cb_distance, 10)

        # ── State cache ─────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self._latest_img  = None
        self._pose        = None
        # Latest SAM3-estimated target position from /mission/target. The
        # mission node publishes (target_x, target_y, ORBIT_ALT_LOW); we
        # take XY for use as the recon coordinate origin and use 0 for Z
        # since the target sits on the ground.
        self._target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._ring        = 'low_ring'
        self._detected    = False
        self._distance    = 0.0
        self._count       = 0

        # Accumulated metadata for final JSON export
        self._poses_list  = []
        self._frames_list = []   # for transforms.json (3DGS format)
        self._recon_started = False   # guard: only trigger reconstruction once

        self._write_cameras_json()
        self.get_logger().info(
            f' Image capture ready — output: {self.output_dir}')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _cb_image(self,   msg): self._latest_img = msg
    def _cb_pose(self,    msg): self._pose       = msg
    def _cb_detected(self,msg): self._detected   = msg.data
    def _cb_distance(self,msg): self._distance   = msg.data

    def _cb_target(self, msg: PoseStamped):
        # Z=0 because the target sits on the ground (mission_node publishes
        # ORBIT_ALT_LOW for visualization, not the actual ground contact).
        self._target_xyz = (msg.pose.position.x, msg.pose.position.y, 0.0)

    def _cb_state(self, msg: String):
        s = msg.data
        if 'LOW'  in s: self._ring = 'low_ring'
        if 'HIGH' in s: self._ring = 'high_ring'
        if s == 'DONE' and not self._recon_started:
            self._recon_started = True
            if not self.auto_recon:
                _banner('DRONE MISSION COMPLETE  —  '
                        f'{self._count} images captured  →  '
                        'auto_recon=False, skipping reconstruction')
                tool = ('run_recon_da3' if self.recon_method == 'depth_anything_3'
                        else 'run_recon')
                _print(f'  Run later:  ros2 run drone_recon {tool} '
                       f'{self.output_dir}')
                return
            _banner('DRONE MISSION COMPLETE  —  '
                    f'{self._count} images captured  →  starting reconstruction '
                    f'({self.recon_method})')
            if self.recon_method == 'depth_anything_3':
                runner = self._run_reconstruction_da3
            elif self.recon_method == 'splatfacto':
                runner = self._run_reconstruction
            else:                # 'both' (default)
                runner = self._run_reconstruction_both
            threading.Thread(target=runner, daemon=True).start()

    def _cb_capture(self, msg: Bool):
        if not msg.data:
            return
        if self._latest_img is None or self._pose is None:
            self.get_logger().warn('Capture triggered but no image/pose cached yet')
            return

        self._save()  # increments self._count internally on success

    # ──────────────────────────────────────────────────────────────────────
    # Post-mission 3D reconstruction pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _run_reconstruction_both(self):
        """Run DA3 FIRST (~2-5 min), then splatfacto (~10-15 min) on the
        same captures. Final outputs in <output_dir>/exports/:
            scene_da3.ply     — DA3 colored point cloud
            scene_mesh.ply    — Open3D Poisson mesh from DA3
            splat_pruned.ply  — splatfacto, target only
            scene_splat.ply   — splatfacto, whole scene

        Why this order: splatfacto's ns-viewer holds ~4 GB VRAM after
        training and stays alive for the user. With only 8 GB total,
        DA3 (which needs ~3.5 GB) can't fit AFTER splatfacto without
        killing the viewer. Running DA3 first is clean: the `da3`
        subprocess fully exits when done, releasing all its VRAM, then
        splatfacto trains on the full GPU and leaves its viewer running
        for inspection.

        Either pipeline failing logs the error but does NOT abort the
        other — you still get whichever output worked."""
        log = self.get_logger()
        try:
            self._run_reconstruction_da3()
        except Exception as e:
            log.error(f'DA3 pipeline crashed: {e}; continuing to splatfacto')
        try:
            self._run_reconstruction()
        except Exception as e:
            log.error(f'splatfacto pipeline crashed: {e}')

    def _run_reconstruction_da3(self):
        """Reconstruction via Depth Anything 3 — joint depth+pose+point-cloud
        from arbitrary views. Used for mapping-mode runs where splatfacto's
        tight-orbit assumption breaks down. Wraps drone_recon.run_recon_da3
        so the same code path is hit whether triggered automatically or
        invoked manually from the CLI."""
        from drone_recon import run_recon_da3
        log = self.get_logger()
        log.info(f'DA3 reconstruction starting → {self.output_dir}')
        rc = run_recon_da3.run(self.output_dir)
        if rc == 0:
            _banner(
                'DA3 RECONSTRUCTION COMPLETE\n'
                '#\n'
                f'#   PLY  →  {self.output_dir / "exports" / "scene_da3.ply"}\n'
                '#   Drop into https://superspl.at/editor or any PLY viewer.\n'
                '#'
            )
        else:
            log.error(f'DA3 reconstruction failed (rc={rc})')

    def _run_reconstruction(self):
        """
        Full reconstruction pipeline, runs in a daemon thread after mission DONE.

        Steps:
          1. gen_init_pointcloud.py — seed Gaussians on target geometry
          2. ns-train splatfacto    — 30k-iteration 3DGS with live progress bar
          3. prune_gaussians.py     — remove floaters outside target bbox
          4. ns-export gaussian-splat — write a portable .ply for any 3DGS viewer
          5. ns-viewer              — launch interactive viewer at localhost:7007
          6. Firefox                — auto-open viewer URL in browser
        """
        log_path = self.output_dir / 'reconstruction.log'
        out_dir  = str(self.output_dir)
        log = self.get_logger()

        log.info(f'Reconstruction pipeline started — log: {log_path}')

        def step(cmd, label):
            """Run a subprocess, tee stdout+stderr to the log file."""
            _print(f'  [{label}] starting...')
            with open(log_path, 'a') as f:
                f.write(f'\n=== {label} ===\n')
                result = subprocess.run(cmd, stdout=f, stderr=f, text=True,
                                        env=_ns_env())
            if result.returncode != 0:
                _print(f'  [{label}] FAILED (rc={result.returncode}) — see {log_path}')
                log.error(f'Recon [{label}] FAILED rc={result.returncode}')
                return False
            _print(f'  [{label}] done')
            return True

        # 1 — target-shaped Gaussian seed (synthetic point cloud at the
        # SAM3-estimated target position matching the detected object's
        # gross geometry). transforms.json was already written correctly
        # at capture time (with the forward-of-camera filter applied
        # inline), so we no longer need a regen_transforms pass.
        if not step(['python3', str(_SCRIPTS_DIR / 'gen_init_pointcloud.py'),
                     out_dir, '--target', self.target_prompt],
                    'gen_init_pointcloud'):
            return

        # 3 — train with live progress bar
        _print('  [ns-train splatfacto] starting  (~10-15 min) ...')
        _TOTAL_ITERS = 30000
        _progress_re = re.compile(r'(\d+)\s+\((\d+\.\d+)%\)')
        # `--viewer.quit-on-train-completion True` makes ns-train exit after
        # the final step instead of holding the built-in viewer open. Without
        # it, our subprocess.wait() blocks indefinitely and the pipeline
        # never advances to prune_gaussians or our own ns-viewer launch.
        train_cmd = [
            _NS_TRAIN, 'splatfacto',
            '--data',                            out_dir,
            '--output-dir',                      str(self.output_dir / '3dgs'),
            '--max-num-iterations',              str(_TOTAL_ITERS),
            '--viewer.quit-on-train-completion', 'True',
        ]
        with open(log_path, 'a') as logf:
            logf.write('\n=== ns-train splatfacto ===\n')
            proc = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_ns_env(),
            )
            last_pct = -1.0
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                m = _progress_re.search(line)
                if m:
                    pct = float(m.group(2))
                    if pct - last_pct >= 1.0:   # update every 1 %
                        last_pct = pct
                        filled = int(pct / 5)    # 20-char bar
                        bar = '#' * filled + '-' * (20 - filled)
                        sys.stdout.write(
                            f'\r  3DGS training  [{bar}]  {pct:5.1f}%  '
                            f'(step {m.group(1)}/{_TOTAL_ITERS})')
                        sys.stdout.flush()
            proc.wait()
        sys.stdout.write('\n')
        sys.stdout.flush()
        if proc.returncode != 0:
            _print(f'  [ns-train] FAILED (rc={proc.returncode}) — see {log_path}')
            log.error(f'ns-train FAILED rc={proc.returncode}')
            return
        _print('  [ns-train splatfacto] done')

        # 4 — find the freshest config.yml produced by this run
        configs = sorted(
            (self.output_dir / '3dgs').glob('**/config.yml'),
            key=lambda p: p.stat().st_mtime)
        if not configs:
            _print('  [recon] ERROR: no config.yml found after training')
            log.error('No config.yml found after ns-train')
            return
        config_path = configs[-1]

        # 4b — export the whole-scene splat as an RGB PLY BEFORE pruning,
        # so it includes walls / barrier / cones / drums, not just what
        # survives the per-target prune box. ns-export's `pointcloud` mode
        # doesn't work with splatfacto (it needs train_pixel_sampler from
        # NeRF-style datamanagers), so we use gaussian-splat with
        # --ply-color-mode rgb which produces a standard XYZ+RGB PLY that
        # any tool (CloudCompare, Open3D, MeshLab, SuperSplat, …) opens.
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        _print('  [ns-export scene_splat.ply] writing whole-scene PLY ...')
        with open(log_path, 'a') as f:
            f.write('\n=== ns-export gaussian-splat (full scene, RGB) ===\n')
            ns_export = str(Path.home() / '.local/bin/ns-export')
            r = subprocess.run(
                [ns_export, 'gaussian-splat',
                 '--load-config',     str(config_path),
                 '--output-dir',      str(export_dir),
                 '--output-filename', 'scene_splat.ply',
                 '--ply-color-mode',  'rgb'],
                stdout=f, stderr=f, env=_ns_env())
        if r.returncode != 0:
            _print('  [scene_splat.ply] WARN: export failed — see reconstruction.log')
        else:
            _print(f'  [scene_splat.ply] → {export_dir / "scene_splat.ply"}')

        # 5 — prune floaters to the target's bounding box (per-target config
        # so it works for hydrant, car, cone, etc.). Skipped when
        # auto_prune=False so the canonical ckpt stays the whole-scene
        # splat — useful for room-mapping runs.
        if not self.auto_prune:
            _print('  [prune_gaussians] skipped (auto_prune=False) — '
                   'whole-scene splat kept as canonical')
        else:
            box = self.target_cfg['prune_box']
            if not step(['python3', str(_SCRIPTS_DIR / 'prune_gaussians.py'),
                         str(config_path),
                         '--xy-radius', str(box['xy_radius']),
                         '--z-min',     str(box['z_min']),
                         '--z-max',     str(box['z_max'])],
                        'prune_gaussians'):
                return

        # 6 — find the canonical checkpoint to view. When auto_prune=True
        # (default) this is the just-produced *_pruned.ckpt, swapped into
        # the step-NNNNNNNNN.ckpt slot (nerfstudio's loader requires a
        # purely-numeric step name, so the `_full.ckpt` backup gets moved
        # to a sibling `archive/` directory). When auto_prune=False, we
        # just use the unpruned ckpt directly.
        models_dir = config_path.parent / 'nerfstudio_models'
        archive_dir = config_path.parent / 'archive'
        if self.auto_prune:
            pruned_list = sorted(models_dir.glob('*_pruned.ckpt'),
                                 key=lambda p: p.stat().st_mtime)
            if not pruned_list:
                _print('  [recon] ERROR: no pruned checkpoint found')
                log.error('No pruned checkpoint found')
                return
            pruned_path = pruned_list[-1]
        else:
            full_ckpts = sorted(
                (p for p in models_dir.glob('step-*.ckpt')
                 if '_pruned' not in p.name),
                key=lambda p: p.stat().st_mtime)
            if not full_ckpts:
                _print('  [recon] ERROR: no checkpoint found')
                log.error('No checkpoint found')
                return
            pruned_path = full_ckpts[-1]
            _print(f'  [whole-scene ckpt] → {pruned_path}')
        # Swap the pruned ckpt into the canonical slot (only when we
        # actually pruned). Skipped when auto_prune=False — `pruned_path`
        # already points at the canonical full ckpt in that branch.
        if self.auto_prune:
            canonical = models_dir / pruned_path.name.replace('_pruned.ckpt', '.ckpt')
            if canonical.exists() and canonical != pruned_path:
                archive_dir.mkdir(exist_ok=True)
                backup = archive_dir / canonical.name.replace('.ckpt', '_full.ckpt')
                if not backup.exists():
                    canonical.rename(backup)
                else:
                    canonical.unlink()
            pruned_path.rename(canonical)
            pruned_path = canonical
            _print(f'  [pruned ckpt] → {pruned_path}')

        # 7a — export portable .ply for any 3DGS web viewer (SuperSplat,
        # antimatter15, PlayCanvas etc.). Skipped in auto_prune=False mode
        # because that would just duplicate scene_splat.ply.
        if self.auto_prune:
            _print('  [ns-export splat_pruned.ply] writing target-only PLY ...')
            with open(log_path, 'a') as f:
                f.write('\n=== ns-export gaussian-splat (pruned) ===\n')
                ns_export = str(Path.home() / '.local/bin/ns-export')
                r = subprocess.run(
                    [ns_export, 'gaussian-splat',
                     '--load-config',     str(config_path),
                     '--output-dir',      str(export_dir),
                     '--output-filename', 'splat_pruned.ply'],
                    stdout=f, stderr=f, env=_ns_env())
            if r.returncode != 0:
                _print('  [splat_pruned.ply] WARN: export failed — see reconstruction.log')
            else:
                _print(f'  [splat_pruned.ply] → {export_dir / "splat_pruned.ply"}')

        # 7 — launch viewer (non-blocking)
        _print('  [ns-viewer] launching...')
        with open(log_path, 'a') as f:
            subprocess.Popen(
                [_NS_VIEWER,
                 '--load-config', str(config_path)],
                stdout=f, stderr=f,
                env=_ns_env())

        _banner(
            '3D RECONSTRUCTION COMPLETE\n'
            '#\n'
            f'#   Viewer  →  http://localhost:7007\n'
            f'#   Model   →  {pruned_path}\n'
            '#'
        )

        # 8 — open browser
        for browser_cmd in (['firefox', 'http://localhost:7007'],
                            ['xdg-open', 'http://localhost:7007']):
            try:
                subprocess.Popen(browser_cmd,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────

    def _save(self):
        # ── Extract pose ────────────────────────────────────────────────
        p = self._pose.pose.position
        tx, ty, _ = self._target_xyz

        # Compute yaw from drone position toward the SAM3-estimated target.
        # Computing this server-side (instead of trusting the published
        # quaternion) sidesteps a 1-cycle ROS timing lag where the capture
        # trigger races ahead of the matching pose update on the first
        # capture. Now that the target may not be at world origin, the math
        # uses target_xyz instead of (0,0,0).
        dx = tx - p.x
        dy = ty - p.y
        if dx * dx + dy * dy < 1e-4:
            self.get_logger().warn(
                ' [SKIP] capture — drone is at the target, can\'t aim camera')
            return
        yaw = math.atan2(dy, dx)

        # Forward-of-camera filter: drop frames where the camera is not
        # actually pointed at the target. This used to be done in a
        # post-process step (regen_transforms.py) — folding it in here
        # means transforms.json is correct on first write and we no
        # longer need the regen pass.
        # Camera forward is -col2 of the c2w matrix; we already know the
        # camera will be aimed by `yaw` toward the target, so the angular
        # check reduces to: is target ahead of drone in the XY plane?
        # This is always true after the yaw recomputation above, but we
        # also check the published yaw didn't lag — guards the orbit-entry
        # transition where the drone hadn't finished rotating.
        q = self._pose.pose.orientation
        published_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # Wrap to [-π, π] then check magnitude
        yaw_err = (published_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_err) > math.radians(60.0):
            self.get_logger().info(
                f' [SKIP] capture — drone yaw lags target by '
                f'{math.degrees(yaw_err):+.0f}°')
            return

        # ── Decode image (only after we know we're going to keep it) ────
        frame = self.bridge.imgmsg_to_cv2(self._latest_img, desired_encoding='bgr8')

        # Frame passed all filters — bump the count.
        self._count += 1

        # ── Filename ─────────────────────────────────────────────────────
        # Encode orbit angle as drone's azimuth around the target
        angle_deg = math.degrees(math.atan2(-dy, -dx)) % 360.0
        fname = f'capture_{self._count:03d}_{angle_deg:06.1f}deg.png'
        fpath = self.output_dir / 'images' / self._ring / fname
        cv2.imwrite(str(fpath), frame)

        # ── Metadata ─────────────────────────────────────────────────────
        meta = {
            'n':            self._count,
            'file':         str(fpath.relative_to(self.output_dir)),
            'ring':         self._ring,
            'angle_deg':    round(angle_deg, 2),
            'position':     {'x': round(p.x, 4),
                             'y': round(p.y, 4),
                             'z': round(p.z, 4)},
            'yaw_rad':      round(yaw, 5),
            'sam3_detected':self._detected,
            'sam3_dist_m':  round(self._distance, 2),
        }
        self._poses_list.append(meta)

        # ── transforms.json frame (3DGS / NeRF standard) ─────────────────
        # Camera-to-world 4×4 matrix (c2w)
        c2w = self._pose_to_c2w(p.x, p.y, p.z, yaw)
        self._frames_list.append({
            'file_path':        str(fpath.relative_to(self.output_dir)),
            'transform_matrix': c2w.tolist(),
        })

        # ── Write JSONs after every capture ──────────────────────────────
        self._write_poses_json()
        self._write_transforms_json()

        self.get_logger().info(
            f' [SAVED #{self._count:3d}] {self._ring}  {angle_deg:.1f}°  '
            f'SAM3:{"✓" if self._detected else "✗"}  '
            f'→ {fname}')

    # ──────────────────────────────────────────────────────────────────────
    # Pose → camera-to-world matrix
    # ──────────────────────────────────────────────────────────────────────

    def _pose_to_c2w(self, tx, ty, tz, yaw) -> np.ndarray:
        """
        Build a 4×4 camera-to-world matrix (OpenGL/NeRF convention).

        Coordinate derivation from Gazebo SDF:
          SDF camera pose: <pose>0.1 0 0.02 0 +0.5236 0</pose>
          → sensor frame = Ry(+0.5236) relative to drone link
          → Gazebo camera looks along sensor +X

        Gazebo sensor axes in drone link frame:
          sensor +X (look) = [cos p,  0, -sin p]   p=0.5236
          sensor +Y (left) = [0,      1,  0    ]
          sensor +Z (up  ) = [sin p,  0,  cos p]

        OpenCV optical frame (X=right, Y=down, Z=forward) in link:
          optical +X =  -sensor +Y = [0,       -1,  0     ]
          optical +Y =  -sensor +Z = [-sin p,   0, -cos p ]
          optical +Z =  +sensor +X = [ cos p,   0, -sin p ]

        nerfstudio c2w (OpenGL: X=right, Y=up, Z=backward) columns:
          col0 =  optical +X  (right)
          col1 = -optical +Y  (up   = flip image Y)
          col2 = -optical +Z  (back = flip forward)
        """
        p = 0.5236  # SDF camera pitch (POSITIVE, matches <pose>...0.5236...</pose>)

        # Optical axes in drone link frame
        ox = np.array([0.0,          -1.0,  0.0        ])  # right
        oy = np.array([-math.sin(p),  0.0, -math.cos(p)])  # down
        oz = np.array([ math.cos(p),  0.0, -math.sin(p)])  # forward

        # Rotate to world frame by drone yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [0,   0,  1]])
        ox_w = Rz @ ox
        oy_w = Rz @ oy
        oz_w = Rz @ oz

        # Assemble c2w (OpenGL/nerfstudio convention)
        c2w = np.eye(4)
        c2w[:3, 0] =  ox_w   # right
        c2w[:3, 1] = -oy_w   # up   (flip Y: image-down → world-up)
        c2w[:3, 2] = -oz_w   # back (flip Z: forward  → backward)
        c2w[:3, 3] = [tx, ty, tz]
        return c2w

    # ──────────────────────────────────────────────────────────────────────
    # JSON writers
    # ──────────────────────────────────────────────────────────────────────

    def _write_cameras_json(self):
        data = {
            'camera_model': 'PINHOLE',
            'width':  IMG_W, 'height': IMG_H,
            'fx': FX, 'fy': FY,
            'cx': CX, 'cy': CY,
            'hfov_deg': HFOV_DEG,
        }
        with open(self.output_dir / 'cameras.json', 'w') as f:
            json.dump(data, f, indent=2)

    def _write_poses_json(self):
        # Wrap the per-image list with mission-level metadata so downstream
        # scripts can recenter the prune box / init seed at the actual
        # SAM3-estimated target instead of assuming world origin.
        tx, ty, tz = self._target_xyz
        data = {
            'target_prompt':   self.target_prompt,
            'target_position': {'x': tx, 'y': ty, 'z': tz},
            'frames':          self._poses_list,
        }
        with open(self.output_dir / 'poses.json', 'w') as f:
            json.dump(data, f, indent=2)

    def _write_transforms_json(self):
        """
        transforms.json — standard format for nerfstudio / gsplat.
        Can be used directly with: ns-train gaussian-splatting

        We additionally embed `target_position` (a non-standard nerfstudio
        field — nerfstudio ignores unknown top-level keys) so the recon
        scripts (gen_init_pointcloud, prune_gaussians) can recenter at
        the actual target rather than the world origin.
        """
        tx, ty, tz = self._target_xyz
        data = {
            'camera_model':    'OPENCV',
            'fl_x': FX, 'fl_y': FY,
            'cx':   CX, 'cy':   CY,
            'w':    IMG_W, 'h': IMG_H,
            'k1': 0.0, 'k2': 0.0,
            'p1': 0.0, 'p2': 0.0,
            'target_position': {'x': tx, 'y': ty, 'z': tz},
            'frames':          self._frames_list,
        }
        with open(self.output_dir / 'transforms.json', 'w') as f:
            json.dump(data, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    # Refuse to start if another image_capture is already running. Two
    # instances writing into the same output directory was the source of the
    # duplicate-`n` PNGs and the trailing-data poses.json corruption.
    _lock = acquire_singleton('image_capture')   # noqa: F841 (held for lifetime)
    rclpy.init(args=args)
    node = ImageCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
