#!/usr/bin/env python3
"""
Flight Logger — drone_recon diagnostic tool
============================================
Subscribes to all flight topics and writes a timestamped CSV log.
Also prints real-time warnings when the drone approaches obstacles.

Usage (after sourcing ROS 2):
  ros2 run drone_recon flight_logger [--out ~/flight_log.csv]
or via the launch (the launch wires it in automatically):
  ros2 launch drone_recon scene1.launch.py

Output columns (CSV):
  t_s           — wall-clock seconds since logger started
  ros_t_s       — ROS time (seconds)
  state         — mission state string
  ax ay az      — actual position (m) from /drone/pose
  ayaw          — actual yaw (rad)
  tx ty tz      — target position commanded by state machine
  tyaw          — target yaw
  vx vy vz      — instantaneous velocity (m/s) from finite-differenced pose
  wz            — instantaneous yaw rate (rad/s) from finite-differenced yaw
  speed_xy      — sqrt(vx² + vy²) — horizontal speed (m/s)
  ex ey ez      — position error = target - actual (m)
  dist_target   — 2D distance from drone to target origin (m)
  dist_barrier  — drone X minus barrier X (positive = drone side)
  dist_wall_e   — clearance to east wall  (x= 7.0)
  dist_wall_w   — clearance to west wall  (x=-7.0)
  dist_wall_n   — clearance to north wall (y= 5.0)
  dist_wall_s   — clearance to south wall (y=-5.0)
  near_obstacle — True if any clearance < WARN_DIST
  sam3          — SAM3 detected (0/1)
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool, Float32

from drone_recon import scene_config as _scenes

# ── Logger constants ──────────────────────────────────────────────────────────
WARN_DIST = 0.6   # m — print WARNING when closer than any scene obstacle

# Scene geometry (walls + optional barrier) is loaded from scene_config at
# runtime via the `scene` ROS param. The static module-level constants used
# to live here, but they only worked for scene1_hydrant — moving them into
# the per-scene dict means flight_logger doesn't need a code change to be
# correct for a new world.


def _quat_to_yaw(ox, oy, oz, ow):
    return math.atan2(2.0 * (ow * oz + ox * oy),
                      1.0 - 2.0 * (oy * oy + oz * oz))


def _dist_to_barrier_segment(px: float, py: float,
                             bx: float, by_half: float) -> float:
    """
    Min XY distance from (px, py) to a Y-aligned barrier wall, treating
    the wall as a line segment from (bx, -by_half) to (bx, +by_half).
    Always non-negative. Used by flight_logger; values come from the
    scene's `barrier` config entry (or the call is skipped entirely when
    the scene has no barrier).
    """
    if abs(py) <= by_half:
        return abs(px - bx)
    corner_y = math.copysign(by_half, py)
    return math.hypot(px - bx, py - corner_y)


class FlightLogger(Node):

    def __init__(self, out_path: str, scene_name: str = 'scene1_hydrant'):
        super().__init__('flight_logger')

        self._start_wall = time.time()
        self._out_path   = out_path

        # ── Scene geometry from scene_config ─────────────────────────────
        try:
            self.scene = _scenes.get(scene_name)
        except KeyError as e:
            self.get_logger().error(str(e))
            raise
        self.scene_name = scene_name
        # Walls dict: {xmin, xmax, ymin, ymax}
        self._walls   = self.scene['walls']
        # Optional barrier dict: {x, y_min, y_max} — None for open scenes
        self._barrier = self.scene.get('barrier')

        # ── State ────────────────────────────────────────────────────────
        self.state   = 'UNKNOWN'
        self.sam3    = 0

        # actual
        self.ax = self.ay = self.az = 0.0
        self.ayaw = 0.0
        self.ros_t = 0.0
        self.odom_recv = False

        # target
        self.tx = self.ty = self.tz = 0.0
        self.tyaw = 0.0

        # velocity (finite-differenced from pose; the drone is teleported
        # kinematically, so /drone/cmd_vel is never published — we have to
        # derive speed from successive samples ourselves).
        self.vx = self.vy = self.vz = self.wz = 0.0
        self._prev_t = None
        self._prev_x = self._prev_y = self._prev_z = 0.0
        self._prev_yaw = 0.0

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(PoseStamped,  '/drone/pose',      self._cb_odom,   10)
        self.create_subscription(PoseStamped,  '/mission/target',  self._cb_target, 10)
        self.create_subscription(String,       '/mission/state',   self._cb_state,  10)
        self.create_subscription(Bool,         '/sam3/detected',   self._cb_sam3,   10)

        # ── CSV setup ────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._csv_file = open(out_path, 'w', newline='')
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow([
            't_s', 'ros_t_s', 'state',
            'ax', 'ay', 'az', 'ayaw',
            'tx', 'ty', 'tz', 'tyaw',
            'vx', 'vy', 'vz', 'wz', 'speed_xy',
            'ex', 'ey', 'ez',
            'dist_target', 'dist_barrier',
            'dist_wall_e', 'dist_wall_w', 'dist_wall_n', 'dist_wall_s',
            'near_obstacle', 'sam3',
        ])
        self._csv_file.flush()

        # ── Log timer — 10 Hz ────────────────────────────────────────────
        self._prev_state    = ''
        self._warn_count    = 0   # current near-obstacle burst length
        self._warn_total    = 0   # monotonic total — printed on shutdown
        self.create_timer(0.1, self._log_tick)

        # Auto-stop: once the mission reaches DONE we record a brief tail
        # (so the final landing samples make it into the CSV) and then
        # shut down. Without this the logger ran indefinitely — last run's
        # CSV had 21,584 DONE rows after the mission actually finished.
        self._done_at: float | None = None
        self._done_tail_secs = 5.0

        self.get_logger().info(f'Flight logger started → {out_path}')
        self.get_logger().info(f'WARN threshold: {WARN_DIST} m from any obstacle')
        self.get_logger().info(
            f'Auto-stop: {self._done_tail_secs:.0f}s after first DONE')

    # ──────────────────────────────────────────────────────────────────────

    def _cb_odom(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        new_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        new_t   = (msg.header.stamp.sec +
                   msg.header.stamp.nanosec * 1e-9)

        # Finite-difference velocity from successive poses. Skip the first
        # sample (no prior to diff against) and clamp dt > 1ms to avoid
        # divide-by-zero from coincident timestamps.
        if self._prev_t is not None:
            dt = new_t - self._prev_t
            if dt > 1e-3:
                self.vx = (p.x - self._prev_x) / dt
                self.vy = (p.y - self._prev_y) / dt
                self.vz = (p.z - self._prev_z) / dt
                # Wrap yaw delta to [-π, π] before dividing
                dyaw = (new_yaw - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi
                self.wz = dyaw / dt

        self._prev_t   = new_t
        self._prev_x   = p.x
        self._prev_y   = p.y
        self._prev_z   = p.z
        self._prev_yaw = new_yaw

        self.ax   = p.x
        self.ay   = p.y
        self.az   = p.z
        self.ayaw = new_yaw
        self.ros_t = new_t
        self.odom_recv = True

    def _cb_target(self, msg: PoseStamped):
        self.tx   = msg.pose.position.x
        self.ty   = msg.pose.position.y
        self.tz   = msg.pose.position.z
        q = msg.pose.orientation
        self.tyaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _cb_state(self, msg: String):
        self.state = msg.data

    def _cb_sam3(self, msg: Bool):
        self.sam3 = 1 if msg.data else 0

    # ──────────────────────────────────────────────────────────────────────

    def _log_tick(self):
        if not self.odom_recv:
            return

        t_s = time.time() - self._start_wall

        # Position errors
        ex = self.tx - self.ax
        ey = self.ty - self.ay
        ez = self.tz - self.az

        # Clearances pulled from the scene config so a different world
        # (e.g. 'open' with no barrier) is handled correctly.
        # dist_barrier_signed is kept for the CSV (its sign encodes which
        # side of the wall the drone is on — useful when post-processing).
        # dist_barrier_true is the actual XY distance to the barrier
        # *segment*, used for the proximity warning so we don't alarm on
        # drones safely transiting the gap.
        w = self._walls
        dist_target = math.hypot(self.ax - self.tx, self.ay - self.ty)
        dist_wall_e = w['xmax'] - self.ax
        dist_wall_w = self.ax - w['xmin']
        dist_wall_n = w['ymax'] - self.ay
        dist_wall_s = self.ay - w['ymin']

        clearances = {
            'east wall':  dist_wall_e,
            'west wall':  dist_wall_w,
            'north wall': dist_wall_n,
            'south wall': dist_wall_s,
        }

        if self._barrier is not None:
            bx = self._barrier['x']
            by_half = (self._barrier['y_max'] - self._barrier['y_min']) / 2.0
            dist_barrier_signed = self.ax - bx
            dist_barrier_true   = _dist_to_barrier_segment(
                self.ax, self.ay, bx, by_half)
            clearances['barrier'] = dist_barrier_true
        else:
            # No barrier in this scene — write zeros for the CSV columns
            # so downstream tooling doesn't blow up on missing fields.
            dist_barrier_signed = 0.0
            dist_barrier_true   = float('inf')

        near = any(v < WARN_DIST for v in clearances.values())

        speed_xy = math.hypot(self.vx, self.vy)

        # ── Console output ──────────────────────────────────────────────
        if self.state != self._prev_state:
            print(f'\n[{t_s:7.1f}s] STATE → {self.state}')
            self._prev_state = self.state
            if self.state == 'DONE' and self._done_at is None:
                self._done_at = t_s
                print(f'  Mission DONE — auto-stopping logger in '
                      f'{self._done_tail_secs:.0f}s')

        if near:
            self._warn_count += 1
            self._warn_total += 1
            closest = min(clearances, key=clearances.get)
            print(f'  [WARN {t_s:7.1f}s] NEAR {closest.upper()}: '
                  f'{clearances[closest]:.2f}m  '
                  f'pos=({self.ax:.2f},{self.ay:.2f},{self.az:.2f})  '
                  f'v=({self.vx:.1f},{self.vy:.1f},{self.vz:.1f})  '
                  f'state={self.state}')
        elif self._warn_count > 0 and not near:
            # Just cleared the danger zone
            self._warn_count = 0
            print(f'  [{t_s:7.1f}s] Cleared obstacle')

        # ── CSV row ──────────────────────────────────────────────────────
        self._writer.writerow([
            f'{t_s:.3f}', f'{self.ros_t:.3f}', self.state,
            f'{self.ax:.4f}', f'{self.ay:.4f}', f'{self.az:.4f}',
            f'{self.ayaw:.4f}',
            f'{self.tx:.4f}', f'{self.ty:.4f}', f'{self.tz:.4f}',
            f'{self.tyaw:.4f}',
            f'{self.vx:.3f}', f'{self.vy:.3f}',
            f'{self.vz:.3f}', f'{self.wz:.3f}', f'{speed_xy:.3f}',
            f'{ex:.4f}', f'{ey:.4f}', f'{ez:.4f}',
            f'{dist_target:.3f}',
            f'{dist_barrier_signed:.3f}',
            f'{dist_wall_e:.3f}', f'{dist_wall_w:.3f}',
            f'{dist_wall_n:.3f}', f'{dist_wall_s:.3f}',
            int(near), self.sam3,
        ])
        self._csv_file.flush()

        # ── Auto-stop after DONE tail ───────────────────────────────────
        if self._done_at is not None and (t_s - self._done_at) > self._done_tail_secs:
            print(f'\n[{t_s:7.1f}s] Auto-stop: shutting down flight logger')
            # Trigger clean shutdown of the rclpy spin loop. The destroy_node
            # override will close the CSV and print_summary will be called by
            # main() after spin returns.
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def destroy_node(self):
        self._csv_file.close()
        print(f'\nLog saved → {self._out_path}  '
              f'({self._warn_total} obstacle warnings recorded)')
        super().destroy_node()


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(path: str):
    """Read the CSV and print a flight digest."""
    import csv as _csv
    rows = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    if not rows:
        print('No data recorded.')
        return

    states_seen  = []
    prev_s = ''
    warn_rows    = [r for r in rows if r['near_obstacle'] == '1']
    min_barrier  = min(float(r['dist_barrier']) for r in rows)
    min_wall_e   = min(float(r['dist_wall_e'])  for r in rows)
    min_wall_w   = min(float(r['dist_wall_w'])  for r in rows)
    min_wall_n   = min(float(r['dist_wall_n'])  for r in rows)
    min_wall_s   = min(float(r['dist_wall_s'])  for r in rows)
    max_z        = max(float(r['az'])           for r in rows)
    min_z        = min(float(r['az'])           for r in rows)
    duration     = float(rows[-1]['t_s'])

    for r in rows:
        if r['state'] != prev_s:
            states_seen.append((float(r['t_s']), r['state']))
            prev_s = r['state']

    print('\n' + '='*60)
    print(' FLIGHT LOG SUMMARY')
    print('='*60)
    print(f' Duration       : {duration:.1f} s')
    print(f' Data points    : {len(rows)}  (10 Hz)')
    print()
    print(' State transitions:')
    for ts, s in states_seen:
        print(f'   t={ts:7.1f}s  {s}')
    print()
    # `dist_barrier` in the CSV is the SIGNED X-only distance: + means
    # drone-side of the barrier wall, - means target-side. The drone reaches
    # the target-side legally through the north/south gap, so a negative
    # min_barrier is *not* a violation — it just tells you which side of
    # the barrier the drone reached. The actual proximity check is in the
    # `near_obstacle` flag (1 = drone within WARN_DIST of any obstacle
    # using the true point-to-segment distance).
    side = 'drone-side' if min_barrier > 0 else 'target-side (via gap)'
    print(' Obstacle clearances (minimum recorded):')
    print(f'   Barrier wall (X=3.0)  : closest signed X = {min_barrier:+.2f}m  '
          f'({side})')
    print(f'   East  wall  (X= 7.0)  : {min_wall_e:.2f}m')
    print(f'   West  wall  (X=-7.0)  : {min_wall_w:.2f}m')
    print(f'   North wall  (Y= 5.0)  : {min_wall_n:.2f}m')
    print(f'   South wall  (Y=-5.0)  : {min_wall_s:.2f}m')
    print()
    print(f' Altitude range : {min_z:.2f}m – {max_z:.2f}m')
    print(f' Near-obstacle  : {len(warn_rows)} samples ({len(warn_rows)/len(rows)*100:.1f}%)')
    print('='*60)

    if warn_rows:
        print('\n First 10 near-obstacle events:')
        for r in warn_rows[:10]:
            print(f'  t={float(r["t_s"]):7.1f}s  state={r["state"]:<12s}'
                  f'  pos=({float(r["ax"]):5.2f},{float(r["ay"]):5.2f},{float(r["az"]):5.2f})'
                  f'  barrier={float(r["dist_barrier"]):+.2f}m'
                  f'  cmd=({float(r["vx"]):4.1f},{float(r["vy"]):4.1f},{float(r["vz"]):4.1f})')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Flight diagnostic logger for drone_recon')
    ap.add_argument('--out', default=None,
                    help='CSV output path (default: ~/flight_logs/flight_YYYYMMDD_HHMMSS.csv)')
    ap.add_argument('--summary', metavar='CSV',
                    help='Print summary of an existing log file and exit')
    ap.add_argument('--scene', default='scene1_hydrant',
                    help='Scene name (drone_recon.scene_config.SCENES key). '
                         'Drives wall + barrier geometry for clearance warnings.')
    args = ap.parse_known_args()[0]

    if args.summary:
        print_summary(os.path.expanduser(args.summary))
        return

    if args.out is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out = os.path.expanduser(f'~/flight_logs/flight_{stamp}.csv')

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f'Logging to: {out_path}  (scene={args.scene})')
    print('Ctrl-C to stop and save.\n')

    rclpy.init()
    node = FlightLogger(out_path, scene_name=args.scene)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        # The auto-stop timer may have already shut rclpy down (Fix #7).
        # Calling shutdown again on a closed context raises — guard with ok().
        if rclpy.ok():
            rclpy.shutdown()
        print_summary(out_path)


if __name__ == '__main__':
    main()
