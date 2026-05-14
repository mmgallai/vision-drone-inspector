#!/usr/bin/env python3
"""
Mission Node — drone_recon
===========================
Autonomous flight state machine for target inspection.

States:
  IDLE → SEARCH → APPROACH → ORBIT_LOW → CLIMB → ORBIT_HIGH → RETURN → LAND → DONE

Drone control: kinematic via `gz service set_pose` (no physics/PX4 needed).
  Position is tracked internally and teleported to Gazebo at 4 Hz.

Topics published:
  /drone/pose          (geometry_msgs/PoseStamped) - current drone world pose
  /mission/state       (std_msgs/String)            - current state name
  /mission/capture     (std_msgs/Bool)              - capture trigger pulse
  /mission/orbit_angle (std_msgs/Float32)           - current orbit angle (deg)

Topics subscribed:
  /sam3/detected    (std_msgs/Bool)     - target visible?
  /sam3/centroid_x  (std_msgs/Float32)  - target centroid X, normalized [0,1]
  /sam3/centroid_y  (std_msgs/Float32)  - target centroid Y, normalized [0,1]
  /sam3/distance    (std_msgs/Float32)  - estimated distance to target (m)
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Twist
from std_msgs.msg import String, Bool, Float32
from ros_gz_interfaces.srv import SetEntityPose

from drone_recon._singleton import acquire_singleton
from drone_recon import scene_config as _scenes
from drone_recon import targets as _targets


# ── Mission parameters ────────────────────────────────────────────────────────
# Scene geometry (waypoints, home, orbit radius/altitudes/speed) is loaded
# from drone_recon.scene_config at runtime via the `scene` ROS parameter.
# Anything below this point is scene-independent control-loop tuning.

WORLD_NAME  = 'recon_world'
MODEL_NAME  = 'x3'

# Capture: one image every 15° → 24 images per ring
CAPTURE_STEP_DEG = 15.0
NUM_CAPTURES     = int(360.0 / CAPTURE_STEP_DEG)  # 24

# How fast the drone moves between waypoints (m per control cycle)
SEARCH_STEP = 1.0 / 20.0   # 1 m/s at 20 Hz — slower so SAM3 (max ~2 Hz) has
                            # time to fire from multiple distinct vantage
                            # points along each column. Triangulation needs
                            # ray spread from different drone positions —
                            # at 2.5 m/s the drone whips past each object
                            # in ~1 s, leaving only 1-2 detections during
                            # the visibility window which gives
                            # near-parallel rays and degenerate solutions.

WAYPOINT_THRESH_XY = 0.4  # m
WAYPOINT_THRESH_Z  = 0.3  # m

# Approach speed only — the approach distance threshold is derived per-scene
# from orbit_radius (orbit_radius + 0.3 m).
APPROACH_SPEED = 1.5    # m/s

# Control loop
CONTROL_HZ     = 20
POSE_UPDATE_DIV = 5     # send set_pose every N cycles → 4 Hz

# Velocity-mode controller gains. Used only when control_mode is
# 'velocity_kinematic'. P gains are per-axis; we add a feedforward term
# equal to the commanded-position rate of change so the drone tracks the
# state-machine trajectory instead of lagging behind a step.
VEL_KP_XY     = 2.0    # 1/s
VEL_KP_Z      = 2.0    # 1/s
VEL_KP_YAW    = 2.0    # 1/s
VEL_MAX_LIN   = 2.5    # m/s — clamp on linear velocity command
VEL_MAX_ANG   = 1.5    # rad/s — clamp on angular velocity command

# ── Target-position estimation (real search) ──────────────────────────────────
# Camera pitch matches SDF <pose>... 0.5236 ...</pose> (30° downward tilt).
# IMG dimensions and focal length must match sam3_detector.py / image_capture.py.
_CAM_PITCH      = 0.5236          # radians
_IMG_H_DIV_FX   = 720.0 / 640.0  # = 1.125; vertical / horizontal normalisation

# Spatial clustering of SAM3 hits: each new (tx,ty) estimate either joins
# an existing cluster (within `_CLUSTER_RADIUS_M` of its centroid) or
# starts a new one. A cluster is confirmed once it contains
# `_CONFIRM_HITS` hits — that proves SAM3 saw the same world XY from
# at least `_CONFIRM_HITS` different drone positions, which is robust
# to false positives that scatter randomly across the room. The cluster
# centroid (median, not mean — robust to per-hit ray-cast noise) becomes
# the orbit target.
#   * `_CLUSTER_RADIUS_M`: max distance from a cluster's centroid for a
#     new hit to be considered "the same object". 0.8 m is loose enough
#     for SAM3 ray-cast noise (~0.3-0.5 m) yet tight enough to separate
#     adjacent objects (closest pair in scene1 is hydrant↔mailbox at
#     2.5 m).
#   * `_MAX_CLUSTERS`: cap on simultaneous candidates we track. Hard cap
#     prevents memory growth in a noisy scene; oldest cluster is dropped.
_CONFIRM_HITS         = 5
# Wider cluster radius so hits from very different drone positions still
# group together for the same physical object. With bearing-only
# triangulation the rough per-hit XY estimate carries ~1-2 m noise
# (depth-camera dependent), and we need same-object hits from across the
# whole search trajectory to merge into one cluster — otherwise each
# vantage point spawns its own near-parallel cluster, none triangulate.
_CLUSTER_RADIUS_M     = 2.0
_MAX_CLUSTERS         = 8
# Minimum average SAM3 score for a cluster to be considered the real
# target. SAM3's confidence on the actually-prompted object is typically
# ≥0.5 in good viewing conditions; a wall, bench, or trash bin that
# SAM3 mis-classifies as "potted plant" tends to score 0.30–0.40.
# This threshold lets the confirm-on-cluster fast path commit only when
# we're genuinely sure; weaker matches still accumulate and the
# search-exhausted fallback picks whichever cluster ended up most
# confident. (Empirical — adjust if sim scores differ from real-world.)
_CONFIRM_MIN_SCORE    = 0.45
# Genuine multi-view confirmation requires the drone to have actually
# moved (or rotated) between sample points. Without this gate, the 20 Hz
# mission timer + 1-2 Hz SAM3 detections cause five timer ticks to
# accumulate five hits from a single SAM3 message — same vantage point,
# zero parallax. Require at least one of these between cluster updates:
_MIN_HIT_TRAVEL_M     = 0.30   # drone moved ≥30 cm in XY, OR
_MIN_HIT_YAW_DELTA_R  = math.radians(10.0)  # yawed ≥10°


# ── State names ───────────────────────────────────────────────────────────────

class State:
    IDLE       = 'IDLE'
    SEARCH     = 'SEARCH'
    APPROACH   = 'APPROACH'
    ORBIT_LOW  = 'ORBIT_LOW'
    CLIMB      = 'CLIMB'
    ORBIT_HIGH = 'ORBIT_HIGH'
    RETURN     = 'RETURN'
    LAND       = 'LAND'
    DONE       = 'DONE'


# ── Node ──────────────────────────────────────────────────────────────────────

class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # ── Scene config (declares all per-world geometry) ──────────────
        self.declare_parameter('scene', 'scene1_hydrant')
        scene_name = self.get_parameter('scene').value
        try:
            self.scene = _scenes.get(scene_name)
        except KeyError as e:
            self.get_logger().error(str(e))
            raise

        # ── Mission mode ────────────────────────────────────────────────
        # 'inspection' (default) — find target with SAM3, orbit it twice,
        #                          land. Produces hydrant-quality close-up
        #                          captures.
        # 'mapping'             — altitude-stratified lawnmower of the
        #                          whole search area, no SAM3 dependence,
        #                          no orbit. Captures continuously. Pair
        #                          with auto_prune=false to get a
        #                          whole-scene splat.
        self.declare_parameter('mission_mode', 'inspection')
        self.mission_mode = self.get_parameter('mission_mode').value
        if self.mission_mode not in ('inspection', 'mapping'):
            self.get_logger().warn(
                f'Unknown mission_mode "{self.mission_mode}" — '
                f'falling back to inspection')
            self.mission_mode = 'inspection'

        # ── Control mode ────────────────────────────────────────────────
        # 'kinematic'           — teleport via SetEntityPose (default; works
        #                         today without any flight controller).
        # 'velocity_kinematic'  — Tier 1 of the flight-controller plan.
        #                         Publishes Twist to /drone/cmd_vel and uses
        #                         gz pose feedback. SDF must include the
        #                         VelocityControl plugin (it does as of
        #                         this commit).
        self.declare_parameter('control_mode', 'kinematic')
        self.control_mode = self.get_parameter('control_mode').value
        if self.control_mode not in ('kinematic', 'velocity_kinematic'):
            self.get_logger().warn(
                f'Unknown control_mode "{self.control_mode}" — '
                f'falling back to kinematic')
            self.control_mode = 'kinematic'

        # Pull the values mission_node uses out of the scene dict so the
        # control loop reads instance attributes instead of module globals.
        self.HOME_X, self.HOME_Y, self.HOME_Z = self.scene['home']
        self.SEARCH_ALT       = self.scene['search_altitude']
        self.ORBIT_ALT_LOW    = self.scene['orbit_alt_low']
        self.ORBIT_ALT_HIGH   = self.scene['orbit_alt_high']
        self.ORBIT_SPEED      = self.scene['orbit_speed']

        # ── Per-target tuning (orbit + expected position) ──────────────
        # Smaller objects (mailbox, plant, bin) get tighter orbits than
        # the default scene radius. Without this, a 0.4 m mailbox at
        # (1.5, 2.0) with the scene's r=2 m orbit would put the drone
        # past the barrier wall at x=3.5 m. Per-target overrides come
        # from targets.py; fall back to the scene value when not set.
        # `expected_position` is the SDF-known location of the target —
        # used as a prior so the drone has a sane fallback when SAM3
        # doesn't confirm during search, AND as a sanity check to reject
        # SAM3 mis-segmentations (e.g. SAM3 calling the red hydrant a
        # "potted plant" because of color/shape similarity).
        self.declare_parameter('target_prompt', 'fire hydrant')
        self.target_prompt = self.get_parameter('target_prompt').value
        target_cfg    = _targets.get(self.target_prompt)
        self.ORBIT_RADIUS = float(
            target_cfg.get('orbit_radius', self.scene['orbit_radius']))
        self.APPROACH_DIST    = self.ORBIT_RADIUS + 0.3
        # Autonomous-localization mode: do NOT consult `expected_position`
        # from targets.py. The drone identifies the object visually with
        # SAM3 and localizes it via depth-camera sampling under the mask.
        # The SDF position is GROUND TRUTH only — using it for plausibility
        # checks or orbit centering is the very cheating we want to avoid.
        # All downstream code already handles `expected_target_xy is None`
        # correctly: the plausibility gate becomes a no-op, the orbit
        # snaps to the SAM3 average, and the search-exhausted fallback
        # leaves `target_x/target_y` at their initial value.
        self.expected_target_xy = None
        if self.mission_mode == 'mapping':
            self.SEARCH_WAYPOINTS = _scenes.mapping_waypoints(self.scene)
        else:
            self.SEARCH_WAYPOINTS = _scenes.search_waypoints(self.scene)

        # ── Publishers ──────────────────────────────────────────────────
        self.pub_pose    = self.create_publisher(PoseStamped, '/drone/pose',          10)
        self.pub_target  = self.create_publisher(PoseStamped, '/mission/target',      10)
        self.pub_state   = self.create_publisher(String,      '/mission/state',       10)
        self.pub_capture = self.create_publisher(Bool,        '/mission/capture',     10)
        self.pub_angle   = self.create_publisher(Float32,     '/mission/orbit_angle', 10)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Bool,    '/sam3/detected',   self._cb_detected,   10)
        self.create_subscription(Float32, '/sam3/centroid_x', self._cb_centroid_x, 10)
        self.create_subscription(Float32, '/sam3/centroid_y', self._cb_centroid_y, 10)
        self.create_subscription(Float32, '/sam3/distance',   self._cb_distance,   10)
        self.create_subscription(Float32, '/sam3/score',      self._cb_score,      10)
        # In velocity_kinematic mode we close the loop on actual gz pose
        # (bridged from /model/x3/pose via ros_gz_bridge as a Pose).
        self.create_subscription(Pose,    '/drone/pose_actual', self._cb_pose_actual, 10)
        # Twist publisher — only used in velocity_kinematic mode. Always
        # created so the topic exists for inspection (e.g. ros2 topic echo).
        self.pub_cmd_vel = self.create_publisher(Twist, '/drone/cmd_vel', 10)

        # ── Drone pose (tracked internally, pushed to Gazebo via set_pose) ──
        self.x   = self.HOME_X
        self.y   = self.HOME_Y
        self.z   = self.HOME_Z
        self.yaw = math.pi   # facing -X toward target on startup
        # Drone pitch (rad, +ve = nose down). Stays at 0 except during
        # ORBIT, where we compute a virtual gimbal so the camera (with
        # its fixed 30° downward mount) always points at the target. At
        # high orbit altitudes a level drone would put the target near
        # the bottom edge of the frame; pitching the body forward keeps
        # the target in image-space center for cleaner SAM3/tracker
        # behaviour and fully-coverage splat captures.
        self.pitch = 0.0

        # ── SAM3 feedback ───────────────────────────────────────────────
        self.sam3_detected = False
        self.sam3_cx       = 0.5
        self.sam3_cy       = 0.5
        self.sam3_dist     = 0.0
        self.sam3_score    = 0.0

        # ── Target position (estimated via SAM3 ray-casting) ────────────
        # In autonomous-only mode we have no SDF prior. Initialize at
        # origin so APPROACH has a sane fallback if search exhausts
        # without locking onto anything — the drone will fly to (0,0)
        # rather than crashing on undefined state.
        self.target_x = 0.0
        self.target_y = 0.0
        self._target_confirmed = False
        # List[List[Tuple[float,float]]] — each inner list is one spatial
        # cluster of plausible same-object hits. We confirm the first
        # cluster to hit `_CONFIRM_HITS` members.
        self._hit_clusters = []
        # Drone pose at the time of the last cluster ingestion. Used by
        # the multi-view gate to drop hits that come from a vantage
        # point we already sampled.
        self._last_hit_x   = None
        self._last_hit_y   = None
        self._last_hit_yaw = None

        # ── Mission state ───────────────────────────────────────────────
        self.state              = State.IDLE
        self.orbit_angle        = 0.0
        self.orbit_start_angle  = 0.0
        self.orbit_alt          = self.ORBIT_ALT_LOW
        self.capture_slots_done = set()
        self.total_captures     = 0
        self.captures_low       = []
        self.captures_high      = []

        # Search / return waypoint indices
        self.search_wp_idx = 0
        self.return_wp_idx = 0
        self._return_wps   = []   # computed when entering RETURN

        # Mapping-mode capture cadence: fire every N seconds during the
        # sweep so the splat has photo coverage of every patch of floor,
        # not just the orbit ring. Sized so a 2.5 m/s sweep gives ~1 image
        # per 3.75 m of travel — overlap depends on camera FOV.
        self._mapping_capture_period = 1.5  # seconds
        self._mapping_last_capture   = 0.0

        # Control helpers
        self.cycle      = 0
        self._start_time = None
        # Pose service client — bridged from /world/<world>/set_pose by
        # ros_gz_bridge. We call it asynchronously so the control loop
        # never blocks waiting for Gazebo to ack. _pose_pending is set to
        # the Future of an in-flight call; we skip the next call while it's
        # still pending instead of stacking up service calls.
        self._pose_cli     = self.create_client(
            SetEntityPose, f'/world/{WORLD_NAME}/set_pose')
        self._pose_pending = None
        self._pose_warned  = False

        # Velocity-mode state. actual_* are populated from /drone/pose_actual.
        # cmd_prev_* tracks the previous commanded position so we can
        # build a feedforward velocity (commanded rate of change).
        self.actual_x   = self.HOME_X
        self.actual_y   = self.HOME_Y
        self.actual_z   = self.HOME_Z
        self.actual_yaw = self.yaw
        self._actual_recv = False
        self._cmd_prev_x   = self.x
        self._cmd_prev_y   = self.y
        self._cmd_prev_z   = self.z
        self._cmd_prev_yaw = self.yaw

        # ── Timer ───────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / CONTROL_HZ, self._loop)

        # Delay start 5 s for Gazebo + bridge to initialize
        self._start_timer = self.create_timer(5.0, self._start)

        self.get_logger().info('─' * 55)
        self.get_logger().info(f' drone_recon Mission Node  (scene={scene_name})')
        self.get_logger().info(f' Orbit: R={self.ORBIT_RADIUS}m  '
                               f'LOW={self.ORBIT_ALT_LOW}m  HIGH={self.ORBIT_ALT_HIGH}m')
        self.get_logger().info(f' Captures: {NUM_CAPTURES}/ring  '
                               f'({CAPTURE_STEP_DEG}° spacing)')
        self.get_logger().info(f' Search: {len(self.SEARCH_WAYPOINTS)}-waypoint lawnmower '
                               f'(real SAM3 detection, confirm={_CONFIRM_HITS} hits)')
        self.get_logger().info('─' * 55)

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _cb_detected(self,   msg): self.sam3_detected = msg.data
    def _cb_centroid_x(self, msg): self.sam3_cx = msg.data
    def _cb_centroid_y(self, msg): self.sam3_cy = msg.data
    def _cb_distance(self,   msg): self.sam3_dist = msg.data
    def _cb_score(self,      msg): self.sam3_score = msg.data

    def _cb_pose_actual(self, msg: Pose):
        """Actual drone pose from gz, bridged via ros_gz_bridge (Pose msg).
        Used by the velocity-mode controller as the closed-loop feedback."""
        self.actual_x = msg.position.x
        self.actual_y = msg.position.y
        self.actual_z = msg.position.z
        # quaternion → yaw (Z-axis rotation)
        q = msg.orientation
        self.actual_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._actual_recv = True

    # ──────────────────────────────────────────────────────────────────────
    # Startup
    # ──────────────────────────────────────────────────────────────────────

    def _start(self):
        self._start_timer.cancel()
        self._start_time = time.time()
        self.get_logger().info(' Gazebo ready — starting mission: SEARCH')
        self._transition(State.SEARCH)

    # ──────────────────────────────────────────────────────────────────────
    # Main control loop (20 Hz)
    # ──────────────────────────────────────────────────────────────────────

    def _loop(self):
        self.cycle += 1

        if   self.state == State.IDLE:       pass
        elif self.state == State.SEARCH:     self._do_search()
        elif self.state == State.APPROACH:   self._do_approach()
        elif self.state == State.ORBIT_LOW:  self._do_orbit()
        elif self.state == State.CLIMB:      self._do_climb()
        elif self.state == State.ORBIT_HIGH: self._do_orbit()
        elif self.state == State.RETURN:     self._do_return()
        elif self.state == State.LAND:       self._do_land()

        self._publish_pose()

        if self.control_mode == 'kinematic':
            if self.cycle % POSE_UPDATE_DIV == 0:
                self._gz_set_pose()
        else:   # 'velocity_kinematic'
            self._publish_cmd_vel()

    # ──────────────────────────────────────────────────────────────────────
    # Target position estimation
    # ──────────────────────────────────────────────────────────────────────

    def _add_to_clusters(self, hit: dict) -> list:
        """Add a SAM3 hit to the matching spatial cluster (or start a new
        one), and return the cluster the hit landed in.

        `hit` is a dict containing:
            xy:     (x, y) — coarse depth-based estimate, used ONLY for
                    grouping hits that point at the same object
            origin: (x, y, z) — drone world position at detection time
            ray:    (dx, dy, dz) — unit ray direction in world frame
            score:  float — SAM3 confidence

        Cluster matching uses only the coarse `xy` estimate; the precise
        target XY comes from triangulating the cluster's `(origin, ray)`
        pairs once the cluster matures (see `_triangulate_rays`).

        We cap total clusters at `_MAX_CLUSTERS` to bound the noisy-scene
        case; when full, we evict the cluster with the lowest avg score
        (least likely to be the real prompted object).
        """
        ex, ey = hit['xy']
        for cluster in self._hit_clusters:
            n = len(cluster)
            cx = sum(p['xy'][0] for p in cluster) / n
            cy = sum(p['xy'][1] for p in cluster) / n
            dx, dy = ex - cx, ey - cy
            if dx * dx + dy * dy <= _CLUSTER_RADIUS_M * _CLUSTER_RADIUS_M:
                cluster.append(hit)
                return cluster
        # No match — start a fresh cluster
        if len(self._hit_clusters) >= _MAX_CLUSTERS:
            def _avg_score(c):
                return sum(p['score'] for p in c) / len(c)
            self._hit_clusters.sort(key=_avg_score)
            self._hit_clusters.pop(0)
        new_cluster = [hit]
        self._hit_clusters.append(new_cluster)
        return new_cluster

    def _compute_world_ray(self) -> tuple | None:
        """
        Pure-geometry helper: convert the latest SAM3 mask centroid
        (`self.sam3_cx`, `self.sam3_cy`) into a world-frame ray
        consisting of (origin, unit_direction). Origin is the drone's
        current world XYZ; direction is the unit vector pointing through
        the SAM3 centroid pixel.

        No depth measurement is consulted — the ray is a pure direction
        derived from camera geometry + drone pose. This is what
        bearing-only triangulation ([_triangulate_rays]) stitches across
        multiple drone vantage points to recover the target's world XY
        without trusting the noisy depth-camera reading.

        The camera coordinate system (matches image_capture.py / SDF):
          optical +X (right)   in drone body = [0,        -1,       0       ]
          optical +Y (down)    in drone body = [-sin(p),   0,      -cos(p)  ]
          optical +Z (forward) in drone body = [ cos(p),   0,      -sin(p)  ]
        where p = `_CAM_PITCH` (30° camera tilt downward).

        Returns None if the centroid is so close to the optical center
        that the ray direction is degenerate.
        """
        p   = _CAM_PITCH
        r_x = (self.sam3_cx - 0.5) * 2.0
        r_y = (self.sam3_cy - 0.5) * _IMG_H_DIV_FX

        # Ray in drone body frame (un-normalized)
        bx = r_x * 0.0    + r_y * (-math.sin(p)) + 1.0 * math.cos(p)
        by = r_x * (-1.0) + r_y * 0.0            + 1.0 * 0.0
        bz = r_x * 0.0    + r_y * (-math.cos(p)) + 1.0 * (-math.sin(p))

        mag = math.sqrt(bx * bx + by * by + bz * bz)
        if mag < 1e-9:
            return None
        bx, by, bz = bx / mag, by / mag, bz / mag

        # Rotate body→world by drone yaw
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        wx = cy * bx - sy * by
        wy = sy * bx + cy * by
        wz = bz

        return (self.x, self.y, self.z), (wx, wy, wz)

    def _estimate_target_pos(self):
        """
        Quick world-XY estimate for the latest SAM3 detection, used for
        SPATIAL CLUSTERING (grouping multi-frame hits that point at the
        same physical object). Combines the bearing ray from
        `_compute_world_ray()` with the SAM3 distance — falls back to a
        ground-plane intersection when distance is missing.

        This estimate is INTENTIONALLY coarse — it inherits the depth-
        camera noise we're trying to escape. The cluster's final XY
        comes from `_triangulate_rays`, which uses only direction (not
        distance) and is therefore much more accurate.
        """
        ray = self._compute_world_ray()
        if ray is None:
            return None
        (ox, oy, oz), (wx, wy, wz) = ray

        dist = self.sam3_dist
        if dist > 0.2:
            tx = ox + dist * wx
            ty = oy + dist * wy
        elif wz < -0.01:
            # Ground-plane intersection fallback (z=0)
            t  = -oz / wz
            tx = ox + t * wx
            ty = oy + t * wy
        else:
            return None

        return tx, ty

    @staticmethod
    def _cluster_ray_spread_deg(cluster: list) -> float:
        """Largest pairwise angle (degrees) between any two ray
        directions in the cluster. 0° → all rays parallel. Used to
        decide whether the cluster has enough viewing diversity for
        bearing triangulation to converge."""
        if len(cluster) < 2:
            return 0.0
        dirs = np.array([p['ray'] for p in cluster], dtype=float)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        # Avoid divide-by-zero on a degenerate ray (shouldn't happen
        # since `_compute_world_ray` already drops those, but be safe).
        norms[norms < 1e-9] = 1.0
        dirs = dirs / norms
        cos_min = 1.0
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                cos_min = min(cos_min, float(np.dot(dirs[i], dirs[j])))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_min))))

    def _target_xy_from_cluster(self, cluster: list) -> tuple:
        """Reduce a cluster of SAM3 hits to a single (target_x, target_y).

        Tries bearing-only triangulation first — it uses only the ray
        directions and is therefore immune to depth-camera noise. But
        triangulation only works when the cluster's rays span a real
        angular range; if the drone observed the target from one
        direction (e.g., looking west the whole time), the rays are
        near-parallel and the LSQ solution is ill-conditioned —
        numerically it collapses near the origin centroid, which is
        wrong by exactly the triangulation baseline.

        We therefore:
          1. Compute angular spread of the cluster's ray directions.
             Reject triangulation if max pairwise angle < 25°.
          2. Run triangulation; reject if the conditioning is poor
             (smallest singular value of the LSQ matrix is tiny).
          3. Reject results outside scene walls.
          4. On any rejection, fall back to the median of the rough
             per-hit XY estimates — same behaviour as the pre-
             triangulation code path.
        """
        rays = [(p['origin'], p['ray']) for p in cluster]
        n = len(cluster)
        cx = sorted(p['xy'][0] for p in cluster)[n // 2]
        cy = sorted(p['xy'][1] for p in cluster)[n // 2]

        if len(rays) >= 2:
            dirs = np.array([r[1] for r in rays], dtype=float)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            # Max pairwise angle between any two rays
            cos_min = 1.0
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    cos_min = min(cos_min, float(np.dot(dirs[i], dirs[j])))
            max_angle_rad = math.acos(max(-1.0, min(1.0, cos_min)))
            if math.degrees(max_angle_rad) < 25.0:
                self.get_logger().warn(
                    f' [triangulate] rays too parallel '
                    f'(max spread {math.degrees(max_angle_rad):.1f}° '
                    f'< 25°) — falling back to median XY')
                return cx, cy

        tri = self._triangulate_rays(rays)
        if tri is None:
            return cx, cy
        tx, ty, _tz = tri
        walls = self.scene.get('walls')
        if walls and not (
            walls['xmin'] - 2 <= tx <= walls['xmax'] + 2 and
            walls['ymin'] - 2 <= ty <= walls['ymax'] + 2
        ):
            self.get_logger().warn(
                f' [triangulate] result ({tx:.2f},{ty:.2f}) outside '
                f'scene walls — falling back to median XY')
            return cx, cy
        return tx, ty

    @staticmethod
    def _triangulate_rays(rays: list) -> tuple | None:
        """Bearing-only triangulation: find the world point closest (in
        least-squares sense) to every supplied ray.

        Args:
            rays: list of ((ox, oy, oz), (dx, dy, dz)) tuples. Direction
                vectors should be unit length.

        Math:
            Each ray contributes a 3x3 projection matrix
                Pᵢ = I − dᵢdᵢᵀ
            which projects vectors onto the plane perpendicular to dᵢ.
            The squared distance from a point p to ray i is
                |Pᵢ (p − oᵢ)|².
            Minimising the sum over all rays gives the closed-form normal
            equation
                (Σᵢ Pᵢ) · p = Σᵢ Pᵢ · oᵢ
            which we solve directly.

            For the unique solution to exist the rays must span enough
            angles (rays of identical direction give a singular matrix).
            Returns None when the matrix is singular (parallel rays) so
            callers can fall back to a centroid.
        """
        if not rays:
            return None
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for (ox, oy, oz), (dx, dy, dz) in rays:
            d = np.array([dx, dy, dz], dtype=float)
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            d /= n
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ np.array([ox, oy, oz], dtype=float)
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        return float(p[0]), float(p[1]), float(p[2])

    # ──────────────────────────────────────────────────────────────────────
    # State handlers
    # ──────────────────────────────────────────────────────────────────────

    def _do_search(self):
        # ── Mapping mode: no SAM3 dependence, capture continuously ───────
        if self.mission_mode == 'mapping':
            now = time.time()
            if (now - self._mapping_last_capture) >= self._mapping_capture_period:
                self._mapping_last_capture = now
                # Use the current search-altitude waypoint angle as a label;
                # mapping captures aren't tied to a "ring" so we just dump
                # them all into low_ring/.
                self._trigger_capture(0.0)

            # Walk the mapping waypoints. When done, return home directly
            # (no APPROACH/ORBIT — there's no target to orbit).
            if self.search_wp_idx >= len(self.SEARCH_WAYPOINTS):
                self.get_logger().info(
                    f' Mapping sweep complete — {self.total_captures} captures')
                self._return_wps   = self._compute_return_waypoints()
                self.return_wp_idx = 0
                self._transition(State.RETURN)
                return

            wp = self.SEARCH_WAYPOINTS[self.search_wp_idx]
            tx, ty, tz = wp
            dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
            dxy = math.hypot(dx, dy)
            if abs(dz) > WAYPOINT_THRESH_Z:
                step_z = min(abs(dz), 0.04)
                self.z += math.copysign(step_z, dz)
                self.yaw = math.atan2(ty - self.y, tx - self.x)
                return
            if dxy > WAYPOINT_THRESH_XY:
                step = min(SEARCH_STEP, dxy)
                self.x += (dx / dxy) * step
                self.y += (dy / dxy) * step
                self.yaw = math.atan2(dy, dx)
                return
            self.search_wp_idx += 1
            if self.search_wp_idx < len(self.SEARCH_WAYPOINTS):
                nxt = self.SEARCH_WAYPOINTS[self.search_wp_idx]
                self.get_logger().info(
                    f' Map WP {self.search_wp_idx}/{len(self.SEARCH_WAYPOINTS)}'
                    f' → ({nxt[0]:.1f},{nxt[1]:.1f},{nxt[2]:.1f})')
            return


        # ── Inspection mode (default) — accumulate SAM3 hits, then APPROACH
        # Vision-only autonomous identification: we trust SAM3's text-
        # prompt segmentation but only commit to a target XY once the
        # same world position has been confirmed from multiple drone
        # vantage points. False positives (a wall mistaken for a plant,
        # a green trash bin called a "potted plant") scatter spatially
        # across views and never form a tight cluster — only the real
        # object accumulates hits at one XY.
        if self.sam3_detected and not self._target_confirmed:
            est = self._estimate_target_pos()
            ray = self._compute_world_ray()
            # Filter phantom hits: SAM3 detector keeps `detected` sticky
            # for 2 s after the last positive frame, but during those
            # silent frames score/cx/cy/dist all read zero. Without this
            # gate, 0-score hits with random ground-plane ray-casts
            # poison the clusters.
            if self.sam3_score < 0.30:
                est = None
            # Multi-view gate: only ingest a hit if the drone has moved
            # or rotated since the last accepted hit. Without this, five
            # 20 Hz timer ticks during a single SAM3 detection burst
            # would all add the SAME ray to the cluster (zero parallax)
            # and trigger a false confirm.
            if est is not None and self._last_hit_x is not None:
                dx = self.x - self._last_hit_x
                dy = self.y - self._last_hit_y
                dyaw = (self.yaw - self._last_hit_yaw + math.pi) % (2 * math.pi) - math.pi
                if (dx * dx + dy * dy < _MIN_HIT_TRAVEL_M ** 2
                        and abs(dyaw) < _MIN_HIT_YAW_DELTA_R):
                    est = None
            if est is not None and ray is not None:
                self._last_hit_x = self.x
                self._last_hit_y = self.y
                self._last_hit_yaw = self.yaw
                hit = {
                    'xy':     (est[0], est[1]),
                    'origin': ray[0],
                    'ray':    ray[1],
                    'score':  float(self.sam3_score),
                }
                cluster = self._add_to_clusters(hit)
                size = len(cluster)
                avg_score = sum(p['score'] for p in cluster) / size
                # Confirm fast path: enough multi-view hits AND SAM3 was
                # confident across them. The score gate kills mis-ID
                # failure modes (bench scores ~0.30, real plant scores
                # ~0.55+). Triangulate the cluster's rays for the
                # precise XY — this skips the noisy depth measurement.
                # Fast confirm only when the cluster has BOTH enough
                # high-score hits AND enough angular ray spread for
                # bearing triangulation to produce a sub-meter answer.
                # Without the spread check, the drone confirms early on
                # near-parallel rays (drone going down a single column,
                # all rays pointing the same way), forcing a fallback
                # to the noisy depth-based median estimate.
                if size >= _CONFIRM_HITS and avg_score >= _CONFIRM_MIN_SCORE:
                    spread_deg = self._cluster_ray_spread_deg(cluster)
                    if spread_deg >= 25.0:
                        tx, ty = self._target_xy_from_cluster(cluster)
                        self.target_x, self.target_y = tx, ty
                        origins = ', '.join(
                            f'({p["origin"][0]:.1f},{p["origin"][1]:.1f})'
                            for p in cluster)
                        self.get_logger().info(
                            f' TARGET CONFIRMED at ({tx:.2f}, {ty:.2f}) m via '
                            f'bearing-triangulation  [{size} hits, '
                            f'avg SAM3 score {avg_score:.2f}, '
                            f'ray spread {spread_deg:.1f}°, '
                            f'origins: {origins}]')
                        self._target_confirmed = True
                        self._transition(State.APPROACH)
                        return
                    # Not enough spread yet — keep searching from other
                    # vantage points so future hits add the missing
                    # parallax.
                    self.get_logger().info(
                        f' Cluster {size}/{_CONFIRM_HITS} mature but '
                        f'ray spread only {spread_deg:.1f}° (need '
                        f'≥25°) — keep searching for a non-parallel '
                        f'view',
                        throttle_duration_sec=2.0)
                else:
                    self.get_logger().info(
                        f' Target candidate — hit ({est[0]:.2f},{est[1]:.2f}) '
                        f'score={self.sam3_score:.2f} '
                        f'from drone ({self.x:.1f},{self.y:.1f},'
                        f'yaw={math.degrees(self.yaw):.0f}°)  '
                        f'joined cluster of {size}/{_CONFIRM_HITS} '
                        f'(avg score {avg_score:.2f}); '
                        f'{len(self._hit_clusters)} cluster(s) tracked',
                        throttle_duration_sec=1.0)

        # ── Waypoint navigation ──────────────────────────────────────────
        if self.search_wp_idx >= len(self.SEARCH_WAYPOINTS):
            # Search exhausted before any cluster met both the size and
            # score thresholds for fast-path confirm. Pick the cluster
            # with the highest average SAM3 score among those with at
            # least 2 hits (a singleton could be a one-frame false
            # positive). If nothing qualifies, target_x/y stay at origin.
            qualified = [c for c in self._hit_clusters if len(c) >= 2]
            if qualified:
                def _avg_score(c):
                    return sum(p['score'] for p in c) / len(c)
                best = max(qualified, key=_avg_score)
                n = len(best)
                avg = _avg_score(best)
                tx, ty = self._target_xy_from_cluster(best)
                self.target_x, self.target_y = tx, ty
                origins = ', '.join(
                    f'({p["origin"][0]:.1f},{p["origin"][1]:.1f})'
                    for p in best)
                self.get_logger().warn(
                    f' Search exhausted — picking best-scoring cluster '
                    f'({tx:.2f},{ty:.2f}) via triangulation, {n} hits, '
                    f'avg score {avg:.2f}, origins: {origins} '
                    f'(no cluster reached {_CONFIRM_HITS} hits with '
                    f'score ≥ {_CONFIRM_MIN_SCORE})')
            else:
                self.get_logger().warn(
                    f' Search exhausted — SAM3 produced no hits at all '
                    f'for "{self.target_prompt}"; defaulting to origin')
            self._transition(State.APPROACH)
            return

        wp = self.SEARCH_WAYPOINTS[self.search_wp_idx]
        tx, ty, tz = wp

        dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
        dxy = math.hypot(dx, dy)

        # Rise to search altitude before moving laterally
        if abs(dz) > WAYPOINT_THRESH_Z:
            step_z = min(abs(dz), 0.04)
            self.z += math.copysign(step_z, dz)
            self.yaw = math.atan2(ty - self.y, tx - self.x)
            return

        if dxy > WAYPOINT_THRESH_XY:
            step = min(SEARCH_STEP, dxy)
            self.x += (dx / dxy) * step
            self.y += (dy / dxy) * step
            self.yaw = math.atan2(dy, dx)
            return

        # Waypoint reached
        self.search_wp_idx += 1
        if self.search_wp_idx < len(self.SEARCH_WAYPOINTS):
            nxt = self.SEARCH_WAYPOINTS[self.search_wp_idx]
            self.get_logger().info(
                f' Search WP {self.search_wp_idx}/{len(self.SEARCH_WAYPOINTS)}'
                f' → ({nxt[0]:.1f},{nxt[1]:.1f},{nxt[2]:.1f})')

    def _do_approach(self):
        tx, ty = self.target_x, self.target_y
        dist = math.hypot(self.x - tx, self.y - ty)

        # The previous version had a "lost SAM3 mid-approach → re-search"
        # branch here, but it created an infinite SEARCH↔APPROACH ping-pong
        # whenever target_x/y came from the expected_position fallback
        # (because in that case SAM3 had never been detecting in the first
        # place — there was no "lost" to detect). With the SDF-known prior
        # we commit to (target_x, target_y) once SEARCH ends and just
        # approach, regardless of whether SAM3 is currently visible. SAM3
        # detection ramps up naturally as the drone closes the distance.

        if dist <= self.APPROACH_DIST:
            entry_angle = math.atan2(self.y - ty, self.x - tx)
            self.orbit_angle        = entry_angle
            self.orbit_start_angle  = entry_angle
            self.orbit_alt          = self.ORBIT_ALT_LOW
            self.capture_slots_done = set()
            self.x = tx + self.ORBIT_RADIUS * math.cos(entry_angle)
            self.y = ty + self.ORBIT_RADIUS * math.sin(entry_angle)
            self.z = self.ORBIT_ALT_LOW
            self.get_logger().info(
                f' Approach complete — starting LOW orbit at R={self.ORBIT_RADIUS}m '
                f'around ({tx:.2f},{ty:.2f})')
            self._transition(State.ORBIT_LOW)
            return

        dx = tx - self.x
        dy = ty - self.y
        d  = math.hypot(dx, dy)
        step = APPROACH_SPEED / CONTROL_HZ
        self.x += (dx / d) * min(step, d)
        self.y += (dy / d) * min(step, d)
        self.z  = self.ORBIT_ALT_LOW
        self.yaw = math.atan2(dy, dx)

    def _do_orbit(self):
        tx, ty = self.target_x, self.target_y
        dt = 1.0 / CONTROL_HZ
        self.orbit_angle += self.ORBIT_SPEED * dt

        self.x = tx + self.ORBIT_RADIUS * math.cos(self.orbit_angle)
        self.y = ty + self.ORBIT_RADIUS * math.sin(self.orbit_angle)
        self.z = self.orbit_alt
        self.yaw = math.atan2(ty - self.y, tx - self.x)

        # ── Virtual camera gimbal ───────────────────────────────────────
        # The camera is statically tilted `_CAM_PITCH` (30°) below
        # horizon in the SDF. With a level drone at high orbit altitude
        # the optical axis hits the ground far past the target — the
        # target slides to the bottom edge of the frame and the OpenCV
        # tracker drifts to whatever's centered (e.g. a nearby cone).
        # Pitch the drone forward by the difference between the
        # geometrically-required tilt-to-target and the static camera
        # mount, so the optical axis points exactly at the target's
        # mid-height every tick of the orbit.
        target_height_z = 0.7  # rough midpoint of all orbital targets
        horiz_to_target = math.hypot(tx - self.x, ty - self.y)
        if horiz_to_target > 1e-3:
            ideal_total_pitch = math.atan2(self.z - target_height_z,
                                           horiz_to_target)
            self.pitch = ideal_total_pitch - _CAM_PITCH
        else:
            self.pitch = 0.0

        angle_deg = math.degrees(self.orbit_angle) % 360.0
        slot = int(angle_deg / CAPTURE_STEP_DEG)
        if slot not in self.capture_slots_done:
            self.capture_slots_done.add(slot)
            self._trigger_capture(angle_deg)

        a_msg = Float32(); a_msg.data = float(angle_deg)
        self.pub_angle.publish(a_msg)

        laps = (self.orbit_angle - self.orbit_start_angle) / (2.0 * math.pi)
        if laps >= 1.0:
            ring = 'LOW' if self.state == State.ORBIT_LOW else 'HIGH'
            n = len(self.captures_low if self.state == State.ORBIT_LOW
                    else self.captures_high)
            self.get_logger().info(
                f' {ring} orbit complete — {n}/{NUM_CAPTURES} captures')
            if self.state == State.ORBIT_LOW:
                self._transition(State.CLIMB)
            else:
                self._return_wps   = self._compute_return_waypoints()
                self.return_wp_idx = 0
                self._transition(State.RETURN)

    def _do_climb(self):
        if self.z < self.ORBIT_ALT_HIGH - 0.05:
            self.z = min(self.z + 0.03, self.ORBIT_ALT_HIGH)
        else:
            self.z                  = self.ORBIT_ALT_HIGH
            self.orbit_alt          = self.ORBIT_ALT_HIGH
            self.orbit_start_angle  = self.orbit_angle
            self.capture_slots_done = set()
            self.get_logger().info(f' Climb complete — altitude {self.z:.1f}m')
            self._transition(State.ORBIT_HIGH)

    def _compute_return_waypoints(self):
        """Build a return path from the discovered target back to home,
        using the scene's exit_corridor (e.g. the gap in a barrier).
        Delegates to scene_config.return_waypoints so the rule is shared
        across scenes."""
        return _scenes.return_waypoints(self.scene,
                                        (self.target_x, self.target_y))

    def _do_return(self):
        if self.return_wp_idx >= len(self._return_wps):
            self._transition(State.LAND)
            return

        wp = self._return_wps[self.return_wp_idx]
        tx, ty, tz = wp

        dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
        dxy = math.hypot(dx, dy)

        # Rise to safe altitude first before moving laterally
        if abs(dz) > WAYPOINT_THRESH_Z:
            step_z = min(abs(dz), 0.04)
            self.z += math.copysign(step_z, dz)
            return

        if dxy > WAYPOINT_THRESH_XY:
            step = min(SEARCH_STEP, dxy)
            self.x += (dx / dxy) * step
            self.y += (dy / dxy) * step
            self.yaw = math.atan2(dy, dx)
            return

        # Waypoint reached
        self.return_wp_idx += 1
        if self.return_wp_idx < len(self._return_wps):
            nxt = self._return_wps[self.return_wp_idx]
            self.get_logger().info(
                f' Return WP {self.return_wp_idx}/{len(self._return_wps)}'
                f' → ({nxt[0]:.1f},{nxt[1]:.1f},{nxt[2]:.1f})')

    def _do_land(self):
        self.x = self.HOME_X
        self.y = self.HOME_Y
        if self.z > 0.2:
            self.z = max(self.z - 0.025, 0.15)
        else:
            self.z = 0.15
            self._mission_done()

    # ──────────────────────────────────────────────────────────────────────
    # Capture + state helpers
    # ──────────────────────────────────────────────────────────────────────

    def _trigger_capture(self, angle_deg: float):
        self.total_captures += 1
        ring = 'LOW' if self.state == State.ORBIT_LOW else 'HIGH'
        meta = {
            'n': self.total_captures, 'ring': ring,
            'angle': round(angle_deg, 1),
            'x': round(self.x, 3), 'y': round(self.y, 3), 'z': round(self.z, 3),
            'yaw': round(self.yaw, 4),
            'sam3_detected': self.sam3_detected,
            'sam3_dist': round(self.sam3_dist, 2),
        }
        if self.state == State.ORBIT_LOW:
            self.captures_low.append(meta)
        else:
            self.captures_high.append(meta)

        tag = '+' if self.sam3_detected else '-'
        self.get_logger().info(
            f' [CAPTURE #{self.total_captures}] {ring} {angle_deg:.1f}°  '
            f'SAM3:{tag}  dist={self.sam3_dist:.1f}m')

        msg = Bool(); msg.data = True
        self.pub_capture.publish(msg)

    def _transition(self, new_state: str):
        self.get_logger().info(f' -- {self.state} -> {new_state} --')
        self.state = new_state
        msg = String(); msg.data = new_state
        self.pub_state.publish(msg)
        # Drop the orbit virtual-gimbal pitch when leaving an orbit. Other
        # states fly level; otherwise the drone would dive to the next
        # waypoint with whatever pitch the orbit last commanded.
        if new_state not in (State.ORBIT_LOW, State.ORBIT_HIGH):
            self.pitch = 0.0

    def _mission_done(self):
        elapsed = time.time() - self._start_time
        self.get_logger().info('')
        self.get_logger().info('=' * 55)
        self.get_logger().info(' MISSION COMPLETE')
        self.get_logger().info(f'  Target pos : ({self.target_x:.2f}, {self.target_y:.2f}) m')
        self.get_logger().info(f'  Duration   : {elapsed:.1f}s')
        self.get_logger().info(f'  Captures   : {self.total_captures}  '
                               f'({len(self.captures_low)} LOW + '
                               f'{len(self.captures_high)} HIGH)')
        self.get_logger().info(f'  Output     : ~/recon_output/')
        self.get_logger().info('=' * 55)
        self._transition(State.DONE)
        # Stop the control loop so this instance no longer calls _gz_set_pose()
        # or publishes /drone/pose. Without this, an old mission_node left over
        # from a previous launch fights the new one for control of the drone.
        self.timer.cancel()

    # ──────────────────────────────────────────────────────────────────────
    # Pose publishing + Gazebo kinematic control
    # ──────────────────────────────────────────────────────────────────────

    def _publish_pose(self):
        now = self.get_clock().now().to_msg()

        # In kinematic mode the commanded pose IS the actual pose (after
        # set_pose propagates), so use self.x/y/z. In velocity mode we
        # publish the actual gz pose so downstream consumers
        # (image_capture, sam3_detector, flight_logger) see where the
        # drone really is. Until the first feedback arrives, fall back
        # to commanded — better to publish something than block startup.
        if self.control_mode == 'velocity_kinematic' and self._actual_recv:
            px, py, pz, pyaw = (self.actual_x, self.actual_y,
                                self.actual_z, self.actual_yaw)
            ppitch = 0.0  # velocity mode doesn't pitch yet
        else:
            px, py, pz, pyaw = self.x, self.y, self.z, self.yaw
            ppitch = self.pitch

        msg = PoseStamped()
        msg.header.stamp    = now
        msg.header.frame_id = 'world'
        msg.pose.position.x = px
        msg.pose.position.y = py
        msg.pose.position.z = pz
        # ZYX intrinsic (yaw → pitch → roll=0) — must match the
        # quaternion built in _gz_set_pose so consumers (image_capture,
        # flight_logger, mission_node._compute_world_ray) see the same
        # orientation we sent to Gazebo.
        cp = math.cos(ppitch * 0.5); sp = math.sin(ppitch * 0.5)
        cy = math.cos(pyaw   * 0.5); sy = math.sin(pyaw   * 0.5)
        msg.pose.orientation.w = cp * cy
        msg.pose.orientation.x = -sp * sy
        msg.pose.orientation.y =  sp * cy
        msg.pose.orientation.z =  cp * sy
        self.pub_pose.publish(msg)

        tgt = PoseStamped()
        tgt.header.stamp    = now
        tgt.header.frame_id = 'world'
        tgt.pose.position.x = self.target_x
        tgt.pose.position.y = self.target_y
        tgt.pose.position.z = self.ORBIT_ALT_LOW
        self.pub_target.publish(tgt)

    def _publish_cmd_vel(self):
        """
        Velocity-kinematic controller (Tier 1).

        Reads:  self.x/y/z/yaw      — commanded trajectory from the state
                                      machine (same nudging behavior as
                                      kinematic mode)
                self.actual_*       — feedback from /drone/pose_actual
        Writes: /drone/cmd_vel      — Twist applied by the gz
                                      VelocityControl plugin to the link

        Output = K_p · (commanded − actual) + d/dt(commanded)
        Linear and angular components clamped to VEL_MAX_*.
        Falls back to a safe zero-Twist while we're still waiting for the
        first actual-pose feedback message.
        """
        msg = Twist()

        if not self._actual_recv:
            # No feedback yet — publish zero so the drone holds. Capture
            # the commanded "previous" so the first feedforward step is sane.
            self._cmd_prev_x   = self.x
            self._cmd_prev_y   = self.y
            self._cmd_prev_z   = self.z
            self._cmd_prev_yaw = self.yaw
            self.pub_cmd_vel.publish(msg)
            return

        dt = 1.0 / CONTROL_HZ

        # Feedforward: commanded position is moving — drone needs that
        # velocity even if error is zero.
        ff_x = (self.x - self._cmd_prev_x) / dt
        ff_y = (self.y - self._cmd_prev_y) / dt
        ff_z = (self.z - self._cmd_prev_z) / dt
        ff_yaw = ((self.yaw - self._cmd_prev_yaw + math.pi)
                  % (2 * math.pi) - math.pi) / dt   # wrap

        # Closed-loop position error → velocity
        vx = VEL_KP_XY  * (self.x - self.actual_x) + ff_x
        vy = VEL_KP_XY  * (self.y - self.actual_y) + ff_y
        vz = VEL_KP_Z   * (self.z - self.actual_z) + ff_z

        yaw_err = (self.yaw - self.actual_yaw + math.pi) % (2 * math.pi) - math.pi
        wz = VEL_KP_YAW * yaw_err + ff_yaw

        # Clamp linear velocity magnitude (preserve direction)
        v_mag = math.sqrt(vx * vx + vy * vy + vz * vz)
        if v_mag > VEL_MAX_LIN:
            scale = VEL_MAX_LIN / v_mag
            vx *= scale; vy *= scale; vz *= scale

        # Clamp angular velocity
        if abs(wz) > VEL_MAX_ANG:
            wz = math.copysign(VEL_MAX_ANG, wz)

        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.linear.z  = float(vz)
        msg.angular.z = float(wz)
        self.pub_cmd_vel.publish(msg)

        self._cmd_prev_x   = self.x
        self._cmd_prev_y   = self.y
        self._cmd_prev_z   = self.z
        self._cmd_prev_yaw = self.yaw

    def _gz_set_pose(self):
        """
        Send a kinematic teleport command to Gazebo via the bridged
        SetEntityPose service. Non-blocking: we track the previous call's
        Future and skip the next call if Gazebo hasn't responded yet,
        instead of queueing up service calls when the world is slow.
        """
        # Skip if a previous call is still in flight — better than stacking
        # up async requests if Gazebo briefly stalls.
        if self._pose_pending is not None and not self._pose_pending.done():
            return

        if not self._pose_cli.service_is_ready():
            if not self._pose_warned:
                self.get_logger().warn(
                    f'/world/{WORLD_NAME}/set_pose service not yet ready — '
                    'is ros_gz_bridge running and the service bridge configured?',
                    once=True)
                self._pose_warned = True
            return

        req = SetEntityPose.Request()
        req.entity.name = MODEL_NAME
        req.entity.type = 2   # ros_gz_interfaces/Entity.MODEL
        req.pose.position.x = float(self.x)
        req.pose.position.y = float(self.y)
        req.pose.position.z = float(self.z)
        # Roll = 0, pitch = self.pitch (positive = nose down, virtual
        # gimbal during orbit), yaw = self.yaw. RPY → quaternion via the
        # standard ZYX intrinsic order (roll-X, pitch-Y, yaw-Z).
        cp = math.cos(self.pitch * 0.5); sp = math.sin(self.pitch * 0.5)
        cy = math.cos(self.yaw   * 0.5); sy = math.sin(self.yaw   * 0.5)
        req.pose.orientation.w = cp * cy
        req.pose.orientation.x = -sp * sy
        req.pose.orientation.y =  sp * cy
        req.pose.orientation.z =  cp * sy
        self._pose_pending = self._pose_cli.call_async(req)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    # Refuse to start if another mission_node is already running. Otherwise
    # both instances race on /world/.../set_pose and the drone wobbles.
    _lock = acquire_singleton('mission_node')   # noqa: F841 (held for lifetime)
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
