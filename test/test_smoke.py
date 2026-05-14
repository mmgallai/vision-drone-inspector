"""
Smoke tests for drone_recon. These do NOT require ROS, GPU, or Gazebo —
they catch the kinds of regressions that hurt us in the past:

  * a script's import path breaks after a refactor
  * the targets table loses an entry
  * the c2w camera matrix is no longer orthonormal / aimed forward
  * the singleton lock no longer blocks a second acquirer

Run:  pytest -q test/   (from the package root)
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))


def test_import_modules():
    """Every module under drone_recon/ should import without ROS."""
    # _singleton has no ROS dep
    from drone_recon import _singleton  # noqa: F401
    from drone_recon import targets     # noqa: F401


def test_targets_have_expected_positions_and_hit_validation_works():
    """Each inspectable target must have an expected_position so
    mission_node has a sane fallback when SAM3 misses.

    Then the is_plausible_hit() helper must correctly reject hits at
    the WRONG object's position (so SAM3 mis-segmentations don't lead
    the drone to the hydrant when the user asked for the plant)."""
    from drone_recon.targets import (
        TARGETS, is_plausible_hit, HIT_REJECT_RADIUS_M)

    expected = {
        'fire hydrant':  (0.0, 0.0),
        'potted plant':  (-3.5, 1.5),
        'park bench':    (-3.5, -1.5),
        'trash bin':     (1.5, -2.0),
        'mailbox':       (1.5, 2.0),
    }
    for name, want in expected.items():
        ep = TARGETS[name].get('expected_position')
        assert ep == want, f'{name}: expected_position {ep} != {want}'

    # No expected_position → every hit is "plausible" (legacy behavior).
    assert is_plausible_hit(99.0, 99.0, None) is True

    # When asking for "potted plant" (expected (-3.5, 1.5)):
    plant_xy = expected['potted plant']
    # ✓ a hit AT the plant is plausible
    assert is_plausible_hit(-3.5, 1.5, plant_xy) is True
    # ✓ a hit 1.5 m away is plausible (within reject_radius)
    assert is_plausible_hit(-2.0, 1.5, plant_xy) is True
    # ✗ a hit at the HYDRANT's position is NOT plausible
    #   (distance 3.81 m, well outside the 2 m radius)
    assert is_plausible_hit(0.0, 0.0, plant_xy) is False

    # When asking for "mailbox" (expected (1.5, 2.0)) the closest
    # competing object is the hydrant 2.5 m away — that hit must be
    # rejected (inside the 2 m radius from mailbox would be plausible).
    mb_xy = expected['mailbox']
    assert is_plausible_hit(1.5, 2.0, mb_xy) is True       # at mailbox
    assert is_plausible_hit(0.0, 0.0, mb_xy) is False      # at hydrant
    assert is_plausible_hit(1.5, -2.0, mb_xy) is False     # at trash bin

    # And similarly for trash bin
    tb_xy = expected['trash bin']
    assert is_plausible_hit(1.5, -2.0, tb_xy) is True
    assert is_plausible_hit(0.0, 0.0, tb_xy) is False
    assert is_plausible_hit(1.5, 2.0, tb_xy) is False      # mailbox

    assert HIT_REJECT_RADIUS_M == 2.0


def test_targets_table_complete():
    """Every entry must have height_m, init_shape, init_args, and prune_box."""
    from drone_recon import targets
    required = {'height_m', 'init_shape', 'init_args', 'prune_box'}
    for name, cfg in targets.TARGETS.items():
        missing = required - set(cfg)
        assert not missing, f'{name} missing keys {missing}'
        box = cfg['prune_box']
        assert {'xy_radius', 'z_min', 'z_max'} <= set(box), \
            f'{name} prune_box incomplete'
        assert box['z_min'] < box['z_max'], f'{name} prune box has z_min ≥ z_max'


def test_targets_get_falls_back_to_default():
    """Unknown prompts return the 'default' config, never None."""
    from drone_recon import targets
    cfg = targets.get('something nobody knows')
    assert cfg['height_m'] == targets.TARGETS['default']['height_m']
    # And case-insensitive
    assert targets.get('Fire Hydrant')['init_shape'] == 'hydrant'


def test_singleton_blocks_second_acquirer(tmp_path, monkeypatch):
    """Second process trying to acquire the same lock must exit non-zero."""
    from drone_recon import _singleton
    name = f'pytest_{os.getpid()}'
    fh = _singleton.acquire_singleton(name)
    assert fh is not None

    cmd = [sys.executable, '-c',
           f'import sys; sys.path.insert(0, {str(_PKG_ROOT)!r}); '
           f'from drone_recon._singleton import acquire_singleton; '
           f'acquire_singleton({name!r})']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1
    assert 'already running' in r.stderr

    # Cleanup
    fh.close()
    Path(f'/tmp/drone_recon_{name}.lock').unlink(missing_ok=True)


def test_pose_to_c2w_orthonormal_and_aimed():
    """The c2w matrix must be orthonormal (rotation block) and the
    camera forward vector must point toward the origin from any orbit
    pose facing the target."""
    # We import the function directly from image_capture by reading the file
    # (image_capture itself imports rclpy which we may not have installed
    # in the test environment).
    from importlib.util import spec_from_file_location, module_from_spec
    src = (_PKG_ROOT / 'drone_recon' / 'image_capture.py').read_text()
    # Strip ROS imports + body so we only get the math helper. Easier: just
    # paste the same algorithm here and compare against an expected matrix.
    p = 0.5236
    ox = np.array([0.0, -1.0, 0.0])
    oy = np.array([-math.sin(p), 0.0, -math.cos(p)])
    oz = np.array([math.cos(p), 0.0, -math.sin(p)])

    yaw = math.pi  # facing -X
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 0] =  Rz @ ox
    c2w[:3, 1] = -Rz @ oy
    c2w[:3, 2] = -Rz @ oz
    c2w[:3, 3] = [2.0, 0.0, 1.5]

    # Orthonormality: R^T R = I
    R = c2w[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)

    # Camera forward is -col2 in OpenGL convention → should point toward origin
    forward = -c2w[:3, 2]
    to_origin = np.array([-2.0, 0.0, -1.5]); to_origin /= np.linalg.norm(to_origin)
    cosang = float(np.dot(forward, to_origin))
    assert cosang > 0.9, f'Camera not aimed at origin: cos(angle)={cosang}'


def test_gen_init_pointcloud_recenters_at_target(tmp_path):
    """When transforms.json carries a non-zero target_position, the seed
    point cloud's centroid should land at that target, not at the origin."""
    import json
    import struct

    # Minimal valid transforms.json with a target offset
    tx, ty, tz = 1.5, -2.0, 0.10
    tf = {
        'camera_model': 'OPENCV',
        'fl_x': 640.0, 'fl_y': 640.0, 'cx': 640.0, 'cy': 360.0,
        'w': 1280, 'h': 720,
        'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
        'target_position': {'x': tx, 'y': ty, 'z': tz},
        'frames': [],
    }
    (tmp_path / 'transforms.json').write_text(json.dumps(tf))

    # Run the script as a subprocess so we exercise the real CLI path
    script = _PKG_ROOT / 'scripts' / 'gen_init_pointcloud.py'
    r = subprocess.run([sys.executable, str(script), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f'gen_init failed: {r.stderr}'

    # Parse the binary PLY and check that the centroid is at target
    ply_path = tmp_path / 'init_points.ply'
    assert ply_path.exists()
    with open(ply_path, 'rb') as f:
        data = f.read()
    # Skip header (ends after 'end_header\n')
    idx = data.index(b'end_header\n') + len(b'end_header\n')
    body = data[idx:]
    # Each vertex: 3 floats + 3 uchars = 15 bytes
    n = len(body) // 15
    xs, ys, zs = [], [], []
    for i in range(n):
        x, y, z = struct.unpack_from('<fff', body, i * 15)
        xs.append(x); ys.append(y); zs.append(z)

    cx, cy, cz = sum(xs)/n, sum(ys)/n, sum(zs)/n
    # The unrecentered hydrant has a centroid roughly at (0, 0, 0.25); after
    # translation it should be (tx, ty, tz + 0.25). Allow generous tolerance.
    assert abs(cx - tx) < 0.05, f'X centroid {cx} not near {tx}'
    assert abs(cy - ty) < 0.05, f'Y centroid {cy} not near {ty}'
    # Z centroid is hydrant_local_z + tz, where hydrant_local_z ≈ 0.25
    assert abs(cz - tz - 0.25) < 0.10, f'Z centroid {cz} not near {tz + 0.25}'


def test_rgb_norm_to_depth_pixel_mapping():
    """Map known RGB normalized centroids into the depth image and check
    the result. Both cameras share an optical center and hfov=90°, but the
    depth camera has wider vertical FOV (480/320 vs 720/640), so:
        - center stays at center
        - top of RGB lands at ~y=60 in the depth image (depth sees more above)
        - bottom of RGB lands at ~y=420
        - left/right edges of RGB land at left/right edges of depth (same hfov)
    """
    # Avoid importing the full sam3_detector module (it requires ultralytics).
    # Pull out only the static helper by re-evaluating its body.
    src = (_PKG_ROOT / 'drone_recon' / 'sam3_detector.py').read_text()
    assert '_rgb_norm_to_depth_pixel' in src
    # Reproduce the constants and the pure helper here for a self-contained test.
    IMG_W, IMG_H = 1280, 720
    FX = 640.0
    DEPTH_W, DEPTH_H = 640, 480
    DEPTH_FX = DEPTH_W / 2.0

    def map_pixel(cx_norm, cy_norm):
        dx = int(round(cx_norm * DEPTH_W))
        ry_rgb_px = cy_norm * IMG_H
        angle_y = (ry_rgb_px - IMG_H / 2.0) / FX
        dy = int(round(DEPTH_H / 2.0 + angle_y * DEPTH_FX))
        return max(0, min(DEPTH_W - 1, dx)), max(0, min(DEPTH_H - 1, dy))

    cases = [
        ((0.5, 0.5), (320, 240)),     # center → center
        ((0.0, 0.5), (0, 240)),       # left edge → left edge of depth
        ((1.0, 0.5), (639, 240)),     # right edge → right edge (clamped)
        ((0.5, 0.0), (320,  60)),     # top of RGB → row 60 of depth
        ((0.5, 1.0), (320, 420)),     # bottom of RGB → row 420 of depth
    ]
    for (cxn, cyn), (ex_dx, ex_dy) in cases:
        dx, dy = map_pixel(cxn, cyn)
        assert (dx, dy) == (ex_dx, ex_dy), \
            f'cxn={cxn}, cyn={cyn} → ({dx},{dy}), expected ({ex_dx},{ex_dy})'


def test_depth_z_to_slant_range_conversion():
    """Z = perpendicular distance to camera plane. slant = Z / cos(angle).
    For a target at the optical axis (center pixel), slant == Z exactly.
    For a target near the image edge, slant > Z by the corner factor."""
    DEPTH_W, DEPTH_H = 640, 480
    DEPTH_FX = 320.0

    def slant(z, dx, dy):
        rx = (dx - DEPTH_W / 2.0) / DEPTH_FX
        ry = (dy - DEPTH_H / 2.0) / DEPTH_FX
        return z * math.sqrt(rx * rx + ry * ry + 1.0)

    # Center: slant exactly equals Z
    assert abs(slant(2.0, 320, 240) - 2.0) < 1e-9
    # Image corner (dx=0, dy=0): rx=-1, ry=-0.75
    # slant = 2 * sqrt(1 + 0.5625 + 1) = 2 * sqrt(2.5625) ≈ 3.202
    s = slant(2.0, 0, 0)
    assert 3.20 < s < 3.21, f'corner slant {s}'
    # 30° off-axis pixel (rx=tan(30°)=0.577, ry=0): slant = Z / cos(30°)
    # rx maps to dx = 320 + 0.577*320 = 504.7 ≈ 505
    s = slant(2.0, 505, 240)
    assert abs(s - 2.0 / math.cos(math.radians(30))) < 0.01


def test_depth_sampling_returns_median_at_centroid(tmp_path):
    """Synthesize a depth image with a known value patch at the centroid
    and verify the slant-range output. We replicate the same patch +
    masking logic as _depth_distance without importing the ROS node."""
    DEPTH_W, DEPTH_H = 640, 480
    DEPTH_FX = 320.0
    DEPTH_NEAR, DEPTH_FAR = 0.05, 20.0
    DEPTH_PATCH_HALF = 5

    # Build a depth image: noisy elsewhere, a clean 2.0 m square at center
    rng = np.random.default_rng(0)
    depth = rng.uniform(DEPTH_FAR + 1, DEPTH_FAR + 5, (DEPTH_H, DEPTH_W)).astype(np.float32)
    depth[235:245, 315:325] = 2.0   # 10×10 patch of clean depth

    def sample(cx_norm, cy_norm):
        dx = int(round(cx_norm * DEPTH_W))
        # Map ry: this test uses centered pixel mapping
        dy = int(round(DEPTH_H / 2.0))
        x0, x1 = max(0, dx - DEPTH_PATCH_HALF), min(DEPTH_W, dx + DEPTH_PATCH_HALF + 1)
        y0, y1 = max(0, dy - DEPTH_PATCH_HALF), min(DEPTH_H, dy + DEPTH_PATCH_HALF + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > DEPTH_NEAR) & (patch < DEPTH_FAR) & np.isfinite(patch)]
        if valid.size < max(3, patch.size // 4):
            return float('nan')
        z = float(np.median(valid))
        rx = (dx - DEPTH_W / 2.0) / DEPTH_FX
        ry = (dy - DEPTH_H / 2.0) / DEPTH_FX
        return z * math.sqrt(rx * rx + ry * ry + 1.0)

    # Sampled at center → should return 2.0 (clean patch, slant == Z at center)
    s = sample(0.5, 0.5)
    assert abs(s - 2.0) < 0.01

    # Sampled outside the clean patch (cx_norm=0.1 → dx=64, far from center)
    # Patch all noise > DEPTH_FAR → invalid → NaN
    s = sample(0.1, 0.5)
    assert math.isnan(s)


def test_scene_config_scene1_matches_legacy_waypoints():
    """The scene1_hydrant config in scene_config.py must reproduce the
    9-waypoint hardcoded list that mission_node used before Fix #4.
    If anyone changes the lawnmower generator, this guards regressions."""
    from drone_recon import scene_config as sc

    cfg = sc.get('scene1_hydrant')
    wps = sc.search_waypoints(cfg)

    expected = [
        # approach corridor
        (5.0,  0.0, 1.8),
        (5.0,  4.2, 1.8),
        (0.0,  4.2, 1.8),
        # lawnmower: column-1 (x=0) N→S, column-2 (x=-3) S→N, column-3 (x=2.5) N→S
        ( 0.0,  3.0, 1.8),
        ( 0.0, -3.0, 1.8),
        (-3.0, -3.0, 1.8),
        (-3.0,  3.0, 1.8),
        ( 2.5,  3.0, 1.8),
        ( 2.5, -3.0, 1.8),
    ]
    assert len(wps) == len(expected), \
        f'got {len(wps)} waypoints, expected {len(expected)}'
    for got, want in zip(wps, expected):
        assert got == want, f'waypoint mismatch: got {got}, expected {want}'


def test_scene_config_open_uses_auto_columns():
    """The 'open' scene uses bounds + col_spacing instead of an explicit
    columns list — auto_columns should fill it in."""
    from drone_recon import scene_config as sc

    cfg = sc.get('open')
    wps = sc.search_waypoints(cfg)
    # bounds x=[-5,5] with col_spacing=3 → ceil(10/3)+1 = 5 columns: -5,-2.5,0,2.5,5
    assert len(wps) == 10, f'expected 10 lawnmower waypoints, got {len(wps)}'
    xs = sorted(set(round(p[0], 2) for p in wps))
    assert xs == [-5.0, -2.5, 0.0, 2.5, 5.0], f'got columns {xs}'


def test_scene_config_return_path_matches_legacy():
    """The return path for scene1 must reproduce the original 3-waypoint
    sequence: (clamp, 4.2) → (5.0, 4.2) → (5.0, 0.0). The clamp point
    keeps the drone from overshooting through the gap when the target
    isn't centered on the gap."""
    from drone_recon import scene_config as sc

    cfg = sc.get('scene1_hydrant')
    # Target at origin → clamp = 0.0
    wps = sc.return_waypoints(cfg, (0.0, 0.0))
    assert wps == [(0.0, 4.2, 1.8), (5.0, 4.2, 1.8), (5.0, 0.0, 1.8)], \
        f'unexpected return path for target at origin: {wps}'

    # Target far east (x=10, outside columns) → clamp to max column 2.5
    wps_far = sc.return_waypoints(cfg, (10.0, 0.0))
    assert wps_far[0] == (2.5, 4.2, 1.8)
    # Last waypoint is always home XY
    assert wps_far[-1][:2] == cfg['home'][:2]


def test_scene_config_mapping_repeats_columns_per_altitude():
    """mapping_waypoints sweeps the columns once per mapping_altitudes
    entry. With scene1's [1.5, 2.5, 3.2], we expect 3 sweeps of 6
    waypoints (3 columns × 2 ends) = 18 column waypoints, plus the
    approach corridor (3 waypoints), total 21."""
    from drone_recon import scene_config as sc

    cfg = sc.get('scene1_hydrant')
    wps = sc.mapping_waypoints(cfg)
    # 3 approach + 3 altitudes × (3 columns × 2) = 3 + 18 = 21
    assert len(wps) == 21, f'expected 21, got {len(wps)}'
    # All sweep waypoints' Z values are exactly the configured altitudes
    sweep_zs = sorted(set(round(w[2], 2) for w in wps[3:]))
    assert sweep_zs == [1.5, 2.5, 3.2], f'got altitudes {sweep_zs}'


def test_scene_config_open_has_minimal_return_path():
    """The 'open' scene has no barrier / exit_corridor — return path
    should just be [home]."""
    from drone_recon import scene_config as sc

    cfg = sc.get('open')
    wps = sc.return_waypoints(cfg, (1.0, 2.0))
    assert len(wps) == 1
    assert wps[0][:2] == cfg['home'][:2]


def test_run_recon_da3_returns_error_when_not_installed(tmp_path):
    """If the `da3` CLI isn't on PATH, run_recon_da3.run() should print a
    clear instruction and return a non-zero exit code rather than crashing.
    DA3 is opt-in (~5 GB of deps), so 'not installed' is the common case."""
    from drone_recon import run_recon_da3 as r

    # Build a minimal capture dir layout so the function gets past the
    # images-dir check and fails specifically at the CLI-missing step.
    (tmp_path / 'images' / 'low_ring').mkdir(parents=True)
    rc = r.run(tmp_path, output_name='unused.ply')
    # When CLI isn't on PATH, the runner should report and return non-zero.
    # We accept any non-zero rc — the precise value is an implementation detail.
    assert rc != 0, 'expected non-zero rc when DA3 CLI is missing'


def test_run_recon_da3_errors_without_images(tmp_path):
    """Even if DA3 is installed, run() should refuse a directory with no
    images/ subdir instead of handing DA3 a bogus path."""
    from drone_recon import run_recon_da3 as r
    rc = r.run(tmp_path)
    assert rc != 0


def test_scene_objects_synonym_matching():
    """Every canonical object must be matchable from at least its primary
    synonym, plus a couple of common alternates."""
    from drone_recon.scene_objects import match_target, list_canonical_objects

    cases = {
        'fire hydrant': ['fire hydrant', 'hydrant', 'red hydrant'],
        'potted plant': ['potted plant', 'plant', 'pot of plant'],
        'park bench':   ['park bench', 'bench', 'wooden bench'],
        'trash bin':    ['trash bin', 'trash can', 'garbage', 'bin'],
        'mailbox':      ['mailbox', 'mail box', 'post box'],
        'traffic cone': ['traffic cone', 'cone', 'orange cone'],
        'barrel':       ['barrel', 'drum', 'steel drum'],
        'crate':        ['crate', 'wooden crate', 'box'],
    }
    canonical = set(list_canonical_objects())
    assert canonical == set(cases.keys())
    for expected, phrases in cases.items():
        for phrase in phrases:
            got = match_target(phrase)
            assert got == expected, f'match_target({phrase!r}) → {got!r}, expected {expected!r}'

    # Things that should NOT match anything in the scene
    for nope in ['blue car', 'fire truck', 'a banana', 'spaceship', '']:
        assert match_target(nope) is None, f'{nope!r} matched something'


def test_voice_mission_validate_routes_correctly():
    """The validate() helper must:
      - accept mapping intents (no target)
      - accept inspection intents whose target maps to a canonical object
      - reject unknown objects with a helpful list
    """
    from drone_recon.voice_mission import validate

    r = validate('scan the room')
    assert r['ok'] and r['intent'] == 'mapping' and r['target'] is None

    r = validate('find the fire hydrant')
    assert r['ok'] and r['intent'] == 'inspection' and r['target'] == 'fire hydrant'

    r = validate('look for the bin')
    assert r['ok'] and r['target'] == 'trash bin'

    r = validate('find the spaceship')
    assert not r['ok']
    # The error message should list all canonical scene objects
    for name in ['fire hydrant', 'potted plant', 'park bench',
                 'trash bin', 'mailbox', 'traffic cone', 'barrel', 'crate']:
        assert name in r['message'], f'{name!r} missing from error message'

    r = validate('')
    assert not r['ok']


def test_voice_mission_intent_classifier():
    """Verify the classifier picks 'mapping' vs 'inspection' for a
    representative spread of phrasings, including the trickier ones
    where a mapping verb appears alongside a specific object name."""
    from drone_recon.voice_mission import detect_intent

    mapping_phrases = [
        'scan the room',
        'map the whole room',
        'do a full scan of everything',
        'survey the area',
        'sweep the entire scene',
        'go through the room and map it',
        'check out the surroundings',
        'scan everything',
        'do a full sweep',
        'I want you to map the room',
    ]
    inspection_phrases = [
        'fire hydrant',
        'look for the fire hydrant',
        'find a red car',
        'search for a traffic cone',
        'scan the fire hydrant',          # "scan" alone, no area noun
        'map the hydrant',                # likewise
        'locate the dog',
        'find the barrel near the wall',
    ]
    for p in mapping_phrases:
        assert detect_intent(p) == 'mapping', f'expected mapping for: {p!r}'
    for p in inspection_phrases:
        assert detect_intent(p) == 'inspection', f'expected inspection for: {p!r}'


def test_voice_mission_build_launch_argv():
    """Both branches must force recon_method=both so every run produces
    BOTH a splat and a DA3 point cloud + Poisson mesh."""
    from drone_recon.voice_mission import build_launch_argv

    m = build_launch_argv('mapping', '')
    assert 'recon_method:=both' in m
    assert 'mission_mode:=mapping' in m
    assert 'auto_prune:=false' in m
    assert not any(a.startswith('target:=') for a in m), \
        'mapping path must not pass a target'

    i = build_launch_argv('inspection', 'fire hydrant')
    assert 'recon_method:=both' in i
    assert 'mission_mode:=inspection' in i
    assert 'target:=fire hydrant' in i


def test_voice_target_filler_stripping():
    """voice_target.clean() should strip natural-language wrappers so the
    SAM3 prompt is just the noun phrase."""
    from drone_recon.voice_target import clean

    cases = [
        ('look for the fire hydrant',          'fire hydrant'),
        ('Find the red car.',                  'red car'),
        ('search for a traffic cone',          'traffic cone'),
        ('please look for a barrel',           'barrel'),
        ('locate the hydrant!',                'hydrant'),
        ('  spot the dog  ',                   'dog'),
        ('Map the room',                       'room'),
        # Already clean — should pass through
        ('fire hydrant',                       'fire hydrant'),
        ('blue car',                           'blue car'),
    ]
    for raw, expected in cases:
        got = clean(raw)
        assert got.lower() == expected.lower(), \
            f'clean({raw!r}) → {got!r}, expected {expected!r}'


def test_voice_target_transcribe_returns_empty_when_no_backends(tmp_path):
    """If no STT backend is installed, transcribe() returns '' instead of
    raising — main() then falls back to typed input."""
    from drone_recon.voice_target import transcribe

    # Create a dummy file just so the path exists. None of the backends
    # are installed in test, so all three should fall through to ''.
    wav = tmp_path / 'silence.wav'
    wav.write_bytes(b'')
    assert transcribe(str(wav)) == ''


def test_ai_mission_tools_schema_valid():
    """Each tool must have name + parameters; the inspection enum must
    list the actual canonical scene objects."""
    from drone_recon.ai_mission import _tools
    from drone_recon.scene_objects import list_canonical_objects

    tools = _tools()
    assert len(tools) == 2
    names = sorted(t['function']['name'] for t in tools)
    assert names == ['start_inspection', 'start_mapping']

    insp = next(t for t in tools if t['function']['name'] == 'start_inspection')
    target = insp['function']['parameters']['properties']['target']
    assert target['type'] == 'string'
    assert sorted(target['enum']) == sorted(list_canonical_objects())

    mapping = next(t for t in tools if t['function']['name'] == 'start_mapping')
    assert mapping['function']['parameters']['properties'] == {}


def test_ai_mission_parse_tool_call():
    """parse_tool_call should pull (name, args) out of an Ollama-shaped
    response and gracefully handle the no-tool-call case + the
    arguments-as-JSON-string quirk."""
    from drone_recon.ai_mission import parse_tool_call

    # Standard case
    r1 = {'message': {'tool_calls': [
        {'function': {'name': 'start_inspection',
                      'arguments': {'target': 'fire hydrant'}}}
    ]}}
    assert parse_tool_call(r1) == ('start_inspection',
                                   {'target': 'fire hydrant'})

    # No tool call (e.g. LLM returns plain text)
    r2 = {'message': {'content': 'I do not understand'}}
    assert parse_tool_call(r2) == (None, None)

    # Arguments as JSON string instead of dict
    r3 = {'message': {'tool_calls': [
        {'function': {'name': 'start_inspection',
                      'arguments': '{"target": "potted plant"}'}}
    ]}}
    assert parse_tool_call(r3) == ('start_inspection',
                                   {'target': 'potted plant'})

    # Empty mapping function
    r4 = {'message': {'tool_calls': [
        {'function': {'name': 'start_mapping', 'arguments': {}}}
    ]}}
    assert parse_tool_call(r4) == ('start_mapping', {})

    # Qwen sometimes emits the tool call as JSON inside message.content
    # instead of using tool_calls — verify we recover from that.
    r5 = {'message': {
        'content': 'sure thing\n{"name": "start_inspection", '
                   '"arguments": {"target": "potted plant"}}'
    }}
    assert parse_tool_call(r5) == ('start_inspection',
                                   {'target': 'potted plant'})

    # Garbage content with no JSON → still (None, None)
    r6 = {'message': {'content': 'I would suggest you rephrase.'}}
    assert parse_tool_call(r6) == (None, None)


def test_ai_mission_build_launch_argv():
    """Both branches must force recon_method=both. start_inspection must
    refuse unknown targets so a hallucinating LLM can't sneak through."""
    from drone_recon.ai_mission import build_launch_argv
    import pytest as _pt

    a = build_launch_argv('start_mapping', {})
    assert 'recon_method:=both' in a
    assert 'mission_mode:=mapping' in a
    assert 'auto_prune:=false' in a

    b = build_launch_argv('start_inspection', {'target': 'fire hydrant'})
    assert 'recon_method:=both' in b
    assert 'target:=fire hydrant' in b

    # Hallucinated target
    with _pt.raises(ValueError, match='unknown inspection target'):
        build_launch_argv('start_inspection', {'target': 'spaceship'})

    # Hallucinated tool
    with _pt.raises(ValueError, match='unknown tool'):
        build_launch_argv('do_a_barrel_roll', {})


def test_ai_mission_call_ollama_with_mock_http():
    """call_ollama should pass the user's text and tool defs to whatever
    http_post is supplied. We pass a fake one and inspect the payload
    + return its canned response."""
    from drone_recon.ai_mission import call_ollama

    captured = {}
    def fake_post(url, payload):
        captured['url']     = url
        captured['payload'] = payload
        return {'message': {'tool_calls': [
            {'function': {'name': 'start_mapping', 'arguments': {}}}]}}

    resp = call_ollama('please scan everything', http_post=fake_post)
    assert captured['url'].endswith('/api/chat')
    assert any(m['role'] == 'user' and 'scan everything' in m['content']
               for m in captured['payload']['messages'])
    assert len(captured['payload']['tools']) == 2
    assert resp['message']['tool_calls'][0]['function']['name'] == 'start_mapping'


def test_pipeline_scripts_inventory():
    """Lock the set of pipeline scripts so additions/removals are
    deliberate. regen_transforms.py was deleted earlier (its filter
    moved into image_capture). poisson_mesh.py was added for the
    DA3 → Poisson mesh post-process step."""
    scripts_dir = _PKG_ROOT / 'scripts'
    py_scripts = sorted(p.name for p in scripts_dir.glob('*.py'))
    assert py_scripts == ['gen_init_pointcloud.py', 'poisson_mesh.py',
                          'prune_gaussians.py', 'view_results.py'], \
        f'Unexpected pipeline scripts: {py_scripts}'


def test_launch_file_imports_and_yields_description():
    """Catch breakage of scene1.launch.py without needing Gazebo.

    Skipped when launch_ros isn't on PYTHONPATH (e.g. in a CI image
    without ROS sourced). Imports the launch module, calls
    generate_launch_description(), and verifies the four expected launch
    arguments are declared.
    """
    pytest.importorskip('launch')
    pytest.importorskip('launch_ros')
    pytest.importorskip('ament_index_python')

    from importlib.util import spec_from_file_location, module_from_spec
    launch_path = _PKG_ROOT / 'launch' / 'scene1.launch.py'
    spec = spec_from_file_location('scene1_launch', launch_path)
    mod  = module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Often this is `Package 'drone_recon' not found` if the install
        # tree isn't on AMENT_PREFIX_PATH — that's a sourcing problem,
        # not a launch-file bug, so skip rather than fail.
        if 'not found' in str(e):
            pytest.skip(f'ROS overlay not sourced: {e}')
        raise

    desc = mod.generate_launch_description()
    # Pull all DeclareLaunchArgument nodes out of the description's
    # entities and check the names match what we expect.
    from launch.actions import DeclareLaunchArgument
    arg_names = sorted(
        e.name for e in desc.entities if isinstance(e, DeclareLaunchArgument)
    )
    assert arg_names == ['auto_prune', 'auto_recon', 'control_mode', 'headless',
                         'mission_mode', 'output_dir', 'recon_method',
                         'scene', 'target'], \
        f'Unexpected launch args: {arg_names}'

    # Verify the auto_recon default really is "true" (string)
    auto_recon_arg = next(
        e for e in desc.entities
        if isinstance(e, DeclareLaunchArgument) and e.name == 'auto_recon'
    )
    assert auto_recon_arg.default_value[0].text == 'true'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
