"""
Per-scene mission configuration.

Each scene defines geometry the mission planner needs but the SDF doesn't
expose: where to spawn, the AABB to sweep during search, the optional
approach/exit corridor through obstacles, orbit parameters, and the
walls/barrier the flight_logger uses for clearance warnings.

The mission_node reads its scene by ROS parameter `scene` (default
`scene1_hydrant`). Add new entries here when you build a new world.

Field summary
─────────────
home              : (x, y, z) drone spawn / land pose
search_altitude   : altitude during search and return
orbit_radius      : XY radius of low/high orbit rings around the target
orbit_alt_low     : low orbit altitude
orbit_alt_high    : high orbit altitude
orbit_speed       : orbit angular speed (rad/s)
columns           : explicit column-x list for the lawnmower (in sweep order).
                    Use this when you want a non-trivial column ordering
                    (e.g. start at the gap entry, sweep outward).
y_range           : (ymin, ymax) for the lawnmower columns
approach_corridor : list of (x, y) waypoints to traverse before the lawnmower
                    starts. Used to enter the search area through obstacles.
exit_corridor     : list of (x, y) waypoints from the orbit center toward
                    home. Used to leave the search area for landing.
walls             : axis-aligned wall x/y values for the flight_logger
                    clearance warnings (xmin, xmax, ymin, ymax).
barrier           : optional dict {x, y_min, y_max} describing a single
                    Y-aligned barrier for the flight_logger to track.
                    Set to None for scenes without a barrier.
"""

import math
from typing import Iterable


SCENES = {
    'scene1_hydrant': {
        'home':              (5.0, 0.0, 0.11),
        # Lowered 2.5 → 1.8 m: at 2.5 m a 0.4 m mailbox or 1.4 m plant
        # filled too few pixels for SAM3 to segment reliably; at 1.8 m
        # they fill ~3× the frame area. Walls are 3.5 m tall so 1.8 m
        # is still plenty of clearance.
        'search_altitude':   1.8,
        'orbit_radius':      2.0,
        'orbit_alt_low':     1.5,
        'orbit_alt_high':    3.0,
        'orbit_speed':       0.15,
        # Object cluster lives in x ∈ [-3.5, 1.5], y ∈ [-3.0, 2.0]. The
        # room is now 20×16 m so we don't need to sweep the full width;
        # 4 columns spaced 2.5 m cover all 5 targets with a wide margin.
        'columns':           [-3.0, -1.0, 1.0, 2.5],
        # y_range expanded to match the larger room, with margin so the
        # drone doesn't graze the south/north walls.
        'y_range':           (-4.0, 4.0),
        # No interior barrier — drone flies straight from takeoff (5,0)
        # to its first column. We keep a one-hop "approach corridor"
        # purely as a defensive idle at search altitude before the sweep
        # starts, so the GZ pose-set has time to settle.
        'approach_corridor': [(5.0, 0.0)],
        # Return path: first hop heads east before climbing/landing at home.
        'exit_corridor_y':   0.0,
        'exit_corridor':     [(5.0, 0.0)],
        'walls':             {'xmin': -10.0, 'xmax': 10.0, 'ymin': -8.0, 'ymax': 8.0},
        'barrier':           None,
        # mapping_altitudes: extra altitudes to sweep AFTER the search-altitude
        # pass when mission_mode='mapping'. Each altitude triggers a full
        # column traversal, so we get photos of the scene from multiple
        # heights instead of a tight orbit around one target.
        'mapping_altitudes': [1.5, 2.5, 3.2],
    },
    # An empty 10x10 world with no walls/barriers — use as a starting point
    # when bringing up a new scene without any obstacles.
    'open': {
        'home':              (0.0, 6.0, 0.11),
        'search_altitude':   2.5,
        'orbit_radius':      2.0,
        'orbit_alt_low':     1.5,
        'orbit_alt_high':    3.0,
        'orbit_speed':       0.15,
        'columns':           None,         # auto-generated from bounds
        'bounds':            {'xmin': -5.0, 'xmax':  5.0, 'ymin': -5.0, 'ymax':  5.0},
        'col_spacing':       3.0,
        'y_range':           (-5.0, 5.0),
        'approach_corridor': [],
        'exit_corridor':     [],
        'walls':             {'xmin': -10.0, 'xmax': 10.0, 'ymin': -10.0, 'ymax': 10.0},
        'barrier':           None,
    },
}


def get(name: str) -> dict:
    """Return a shallow copy of the scene config, or raise KeyError."""
    if name not in SCENES:
        raise KeyError(f'Unknown scene "{name}". Available: {sorted(SCENES)}')
    return dict(SCENES[name])


def column_lawnmower(columns: Iterable[float], ymin: float, ymax: float,
                     altitude: float) -> list:
    """
    Generate a north-south boustrophedon (zig-zag) sweep over the given
    column-X list. The first column flies north→south, the next
    south→north, etc. Returns a list of (x, y, z) tuples.
    """
    wps = []
    direction = -1   # first column flies from ymax to ymin (north → south)
    for x in columns:
        if direction == -1:
            wps.append((x, ymax, altitude))
            wps.append((x, ymin, altitude))
        else:
            wps.append((x, ymin, altitude))
            wps.append((x, ymax, altitude))
        direction = -direction
    return wps


def auto_columns(xmin: float, xmax: float, col_spacing: float) -> list:
    """
    Pick an evenly-spaced column-X list covering [xmin, xmax] with at most
    col_spacing between adjacent columns. Always includes both edges.
    """
    if xmax <= xmin or col_spacing <= 0:
        raise ValueError(f'invalid bounds/spacing: '
                         f'xmin={xmin}, xmax={xmax}, col_spacing={col_spacing}')
    n_cols = max(2, int(math.ceil((xmax - xmin) / col_spacing)) + 1)
    return [xmin + (xmax - xmin) * i / (n_cols - 1) for i in range(n_cols)]


def search_waypoints(cfg: dict) -> list:
    """
    Build the full search waypoint list:  approach corridor → lawnmower
    over the configured columns. Each entry is (x, y, z) at search_altitude.
    """
    alt = cfg['search_altitude']
    wps = [(x, y, alt) for (x, y) in cfg.get('approach_corridor', [])]

    columns = cfg.get('columns')
    if columns is None:
        b = cfg['bounds']
        columns = auto_columns(b['xmin'], b['xmax'], cfg.get('col_spacing', 3.0))

    ymin, ymax = cfg['y_range']
    wps.extend(column_lawnmower(columns, ymin, ymax, alt))
    return wps


def mapping_waypoints(cfg: dict) -> list:
    """
    Build a whole-scene mapping waypoint list: same column lawnmower as
    `search_waypoints`, but repeated at each altitude in
    cfg['mapping_altitudes']. No orbit phase — captures fire continuously
    during the sweep, so the resulting 3DGS sees the room from multiple
    heights instead of a tight ring around one target.

    Falls back to a single sweep at search_altitude if mapping_altitudes
    isn't set.
    """
    altitudes = cfg.get('mapping_altitudes') or [cfg['search_altitude']]
    columns = cfg.get('columns')
    if columns is None:
        b = cfg['bounds']
        columns = auto_columns(b['xmin'], b['xmax'], cfg.get('col_spacing', 3.0))
    ymin, ymax = cfg['y_range']

    wps = [(x, y, alt) for (x, y) in cfg.get('approach_corridor', [])
                       for alt in [cfg['search_altitude']]]
    # Approach corridor stays at search_altitude (don't ramp altitude
    # while transiting obstacles)
    for alt in altitudes:
        wps.extend(column_lawnmower(columns, ymin, ymax, alt))
    return wps


def return_waypoints(cfg: dict, target_xy: tuple) -> list:
    """
    Build the return path from the discovered target back to home:

      1. (optional) one waypoint that pulls the drone north (or wherever
         exit_corridor_y points) to a safe latitude, with X *clamped* to
         the target's X so we leave the search area without overshooting;
      2. each fixed waypoint in exit_corridor (typically routes around a
         barrier or other obstacle);
      3. home.

    For scenes without a barrier (no exit_corridor / exit_corridor_y),
    the path is simply [home].
    """
    alt  = cfg['search_altitude']
    home = cfg['home']
    wps = []

    # Step 1: optional clamp waypoint at exit_corridor_y latitude.
    cy = cfg.get('exit_corridor_y')
    if cy is not None:
        b = cfg.get('bounds')
        if b is not None:
            mid_x = max(b['xmin'], min(b['xmax'], target_xy[0]))
        else:
            cols = cfg.get('columns') or [0.0]
            mid_x = max(min(cols), min(max(cols), target_xy[0]))
        wps.append((mid_x, cy, alt))

    # Step 2: fixed corridor waypoints
    for (x, y) in cfg.get('exit_corridor', []):
        wps.append((x, y, alt))

    # Step 3: home
    wps.append((home[0], home[1], alt))
    return wps
