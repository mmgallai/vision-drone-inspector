"""
Per-target configuration shared across the drone_recon package.

Each entry contains everything that used to be hard-coded around the
fire hydrant scene:

  height_m   — assumed real-world height of the object (used by
               sam3_detector to estimate distance from bbox height)
  init_shape — name of the function in gen_init_pointcloud that seeds
               splatfacto with a synthetic-but-target-shaped point cloud
               ('hydrant', 'cylinder', 'sphere')
  init_args  — kwargs for the init shape function
  prune_box  — world-space bounding box for prune_gaussians (XY-radius +
               Z range, target assumed at world origin)

A 'default' entry is used as fallback for any prompt that isn't listed.
"""

TARGETS = {
    'fire hydrant': {
        'height_m':          0.60,
        'orbit_radius':      2.0,
        # Known SDF placement — used by mission_node as a prior so the
        # drone has a sane fallback when SAM3 doesn't confirm during
        # search. Without this, search-exhaustion defaulted target=(0,0)
        # which IS the hydrant — making every "find the X" request land
        # on the hydrant whenever SAM3 missed X. See also
        # mission_node._validate_hit which rejects SAM3 hits more than
        # 2 m from the expected position.
        'expected_position': (0.0, 0.0),
        'init_shape':        'hydrant',
        'init_args':         {},
        'prune_box':         {'xy_radius': 1.0, 'z_min': -0.15, 'z_max': 0.80},
    },
    'potted plant': {
        'height_m':          1.40,    # ~1.2 m model + drone offset
        'orbit_radius':      1.5,     # smaller object, closer orbit
        'expected_position': (-3.5, 1.5),
        'init_shape':        'cylinder',
        'init_args':         {'radius': 0.30, 'height': 1.20, 'color': (60, 130, 60)},
        'prune_box':         {'xy_radius': 0.6, 'z_min': -0.15, 'z_max': 1.60},
    },
    'park bench': {
        'height_m':          0.90,
        'orbit_radius':      2.0,
        'expected_position': (-3.5, -1.5),
        'init_shape':        'cylinder',
        'init_args':         {'radius': 0.80, 'height': 0.90, 'color': (160, 110, 70)},
        'prune_box':         {'xy_radius': 1.2, 'z_min': -0.15, 'z_max': 1.10},
    },
    # Synonym: SAM3 fires more reliably on the bare prompt "bench"
    # than on "park bench" against our wooden-slat primitive — same
    # underlying object, same orbit/init config.
    'bench': {
        'height_m':          0.90,
        'orbit_radius':      2.0,
        'expected_position': (-3.5, -1.5),
        'init_shape':        'cylinder',
        'init_args':         {'radius': 0.80, 'height': 0.90, 'color': (160, 110, 70)},
        'prune_box':         {'xy_radius': 1.2, 'z_min': -0.15, 'z_max': 1.10},
    },
    'trash bin': {
        'height_m':          0.85,
        'orbit_radius':      1.5,
        'expected_position': (1.5, -2.0),
        'init_shape':        'cylinder',
        'init_args':         {'radius': 0.35, 'height': 0.85, 'color': (60, 60, 70)},
        'prune_box':         {'xy_radius': 0.5, 'z_min': -0.15, 'z_max': 1.00},
    },
    'mailbox': {
        'height_m':          1.45,    # post + box
        'orbit_radius':      1.5,
        'expected_position': (1.5, 2.0),
        'init_shape':        'cylinder',
        'init_args':         {'radius': 0.20, 'height': 1.45, 'color': (40, 50, 130)},
        'prune_box':         {'xy_radius': 0.5, 'z_min': -0.15, 'z_max': 1.60},
    },
    'car': {
        'height_m':     1.50,
        'orbit_radius': 3.0,
        'init_shape':   'cylinder',
        'init_args':    {'radius': 1.0, 'height': 1.50, 'color': (60, 60, 200)},
        'prune_box':    {'xy_radius': 3.0, 'z_min': -0.15, 'z_max': 1.80},
    },
    'traffic cone': {
        'height_m':     0.55,
        'orbit_radius': 1.0,
        'init_shape':   'cylinder',
        'init_args':    {'radius': 0.20, 'height': 0.55, 'color': (240, 90, 30)},
        'prune_box':    {'xy_radius': 0.5, 'z_min': -0.15, 'z_max': 0.80},
    },
    'default': {
        'height_m':     0.80,
        'orbit_radius': 2.0,
        'init_shape':   'cylinder',
        'init_args':    {'radius': 0.50, 'height': 1.00, 'color': (180, 180, 180)},
        'prune_box':    {'xy_radius': 1.5, 'z_min': -0.20, 'z_max': 1.50},
    },
}


def get(prompt: str) -> dict:
    """Return the target config for a prompt; case-insensitive; falls back
    to the 'default' entry. The returned dict is a shallow copy — callers
    may mutate it without affecting the table."""
    key = (prompt or '').strip().lower()
    return dict(TARGETS.get(key, TARGETS['default']))


# Maximum allowed XY distance between a SAM3 hit and the target's
# expected position before we treat the hit as a misidentification of
# some OTHER scene object. 2 m chosen because the closest two
# inspectable items (hydrant ↔ mailbox) are 2.5 m apart — 2 m gives
# slack for SAM3 noise while still rejecting hits on the wrong object.
HIT_REJECT_RADIUS_M = 2.0


def is_plausible_hit(tx: float, ty: float,
                     expected_xy: tuple[float, float] | None,
                     reject_radius_m: float = HIT_REJECT_RADIUS_M) -> bool:
    """
    Return True if the SAM3-derived (tx, ty) is plausibly the target.

    Without an expected position (e.g. unknown target / open scene),
    every hit is plausible — falls back to mission_node's existing
    confirm-by-frequency logic.

    With an expected position, hits within `reject_radius_m` are
    accepted and hits farther are rejected. This filters out the
    classic failure mode where SAM3 mis-segments the bright red
    hydrant when the user asked for "potted plant", which used to
    pollute mission_node's averaging buffer with hits at the origin.
    """
    if expected_xy is None:
        return True
    dx = tx - expected_xy[0]
    dy = ty - expected_xy[1]
    return (dx * dx + dy * dy) <= (reject_radius_m * reject_radius_m)
