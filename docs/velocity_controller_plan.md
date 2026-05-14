# Velocity-controller / PX4 SITL plan

Current state (as of session 2): the drone is teleported via the
`SetEntityPose` ROS service (bridged from Gazebo) at 4 Hz from
`mission_node._gz_set_pose`. This is kinematic — no rotors spin, no
gravity, no collision response, no realistic dynamics. It works, but
it can't validate flight controllers, motor saturation, or wind handling.

This doc lays out the work to replace teleport with real flight, in
order of increasing scope. Each tier compiles into a working mission;
do them in order, not all at once.

---

## Tier 1 — Velocity-command kinematic (smallest jump) ✅ IMPLEMENTED

Drop `set_pose`, keep gravity off. Gazebo's `VelocityControl` plugin moves
the drone at the commanded velocity each cycle. The drone still floats
(gravity off), but movement is now smooth and collision-aware.

Status (this session):

- ✅ SDF: `VelocityControl` plugin added to `<model name="x3">` in
  `worlds/scene1_hydrant.sdf`. Subscribes to `/model/x3/cmd_vel`.
  PosePublisher now also publishes the model pose explicitly.
- ✅ Bridge: `geometry_msgs/Twist ↔ gz.msgs.Twist` for `/drone/cmd_vel
  → /model/x3/cmd_vel` (ROS→GZ). And `geometry_msgs/Pose ←
  gz.msgs.Pose` for `/drone/pose_actual ← /model/x3/pose` (GZ→ROS).
- ✅ mission_node: `control_mode` ROS param. Default `kinematic` =
  unchanged behavior. `velocity_kinematic` engages the new path:
  - subscribes to `/drone/pose_actual` (closed-loop feedback)
  - publishes `Twist` to `/drone/cmd_vel` at 20 Hz
  - publishes the actual gz pose to `/drone/pose` (so image_capture /
    sam3_detector / flight_logger see real drone position)
  - skips `_gz_set_pose` calls
- ✅ Controller: `vx = K_p · (commanded_x - actual_x) +
  d/dt(commanded_x)` per axis, with clamps (`VEL_MAX_LIN=2.5 m/s`,
  `VEL_MAX_ANG=1.5 rad/s`). Falls back to zero-Twist while waiting
  for the first feedback message.
- ✅ Launch: `control_mode` arg added to `scene1.launch.py`. To engage:
  `ros2 launch drone_recon scene1.launch.py control_mode:=velocity_kinematic`

Tuning notes (do these on the next live run):

- Default gains (`VEL_KP_*=2.0`, `VEL_MAX_LIN=2.5 m/s`) are conservative
  starting points. If the drone overshoots waypoints, reduce K_p; if it
  lags behind the commanded trajectory, raise it.
- Yaw error wraps to `[-π, π]` to handle the discontinuity at ±π. If the
  drone occasionally spins instead of taking the short way around, the
  wrap is wrong for that case — file a repro.
- The state machine still uses `self.x/y/z` for "have we reached the
  waypoint" checks. Since `self.x` is the commanded position (advanced
  by `SEARCH_STEP` each cycle), the kinematic-mode logic is preserved
  — but in velocity mode the drone may lag commanded by 0.2-0.5 m.
  If captures fire before the drone has settled, switch the threshold
  to use `actual_*` instead of `self.x`.

Risk: low. Estimated effort: ½ day. **Done.**

## Tier 2 — Multicopter motor model + velocity controller (real dynamics)

Add gravity. Use gz-sim's `MulticopterMotorModel` for thrust simulation
plus `MulticopterVelocityControl` for the high-level controller. The
drone now responds to physics — needs throttle for hover, can stall, can
roll into a wall.

Changes (in addition to Tier 1):
1. SDF: enable gravity. Add `MulticopterMotorModel` per rotor (4
   instances), each with a thrust constant + spin direction. Add
   `MulticopterVelocityControl` linking to the motor model.
2. Tune the velocity-control gains (PX, PY, PZ, attitude gains) for the
   x3 mass (1.5 kg) and inertia from the SDF.
3. mission_node `_do_orbit` orbit-speed currently is 0.15 rad/s ≈ 0.3
   m/s tangential; tune the velocity-mode P gain so steady-state
   tracking error stays under WAYPOINT_THRESH_XY (0.4 m). Likely 0.5-1.0
   per axis.
4. Add altitude PID — the drone needs to fight gravity to hold z.
5. Capture trigger logic should fire only when the drone is within a
   tolerance of the commanded orbit point — currently uses internal
   pose, which under real dynamics won't match the commanded slot.

Risk: medium. Estimated effort: 1-2 days. Tuning is the long pole.

## Tier 3 — PX4 SITL via MAVROS or px4_msgs

Replace the Tier-2 controller with a full PX4 SITL flight stack. PX4
runs as a separate process, communicates with Gazebo for sensor +
control, and exposes MAVLink. We send setpoints via MAVROS or the
`px4_msgs` ROS bridge.

Changes (replaces Tier 2 controller):
1. PX4-Autopilot built and SITL-launchable (separate apt+make step,
   adds ~5 min build time).
2. SDF model needs PX4-compatible plugins (`OdometryPublisher`,
   `LiftDrag`, sensor topics matching PX4's expectations). Easier to
   start from the official `gz_x500` model and re-skin.
3. mission_node arms / sets OFFBOARD mode / publishes
   `/fmu/in/trajectory_setpoint` (px4_msgs) or `mavros/setpoint_position`
   (MAVROS). Subscribes to `/fmu/out/vehicle_local_position` for state.
4. State machine adapts: SEARCH/APPROACH/ORBIT now produce position
   setpoints; PX4 handles the controller.

Risk: high (PX4 is a separate world). Estimated effort: 2-3 days.

---

## Recommendation

Do Tier 1 first. It's the smallest diff and gives the team a
side-by-side comparison: same mission, kinematic teleport vs velocity
control, in two `control_mode` settings. That comparison will tell you
whether the orbit / search loops are robust enough to graduate to Tier 2,
or whether the state machine itself needs more guardrails first.

Tier 3 only makes sense if you actually need MAVLink integration —
inflight payload integration, real airframe code reuse, or HIL with a
physical Pixhawk. For pure simulation + reconstruction, Tier 2 is enough.

## What is already in place

- `mission_node` already publishes `PoseStamped` to `/drone/pose`, which
  is used by `image_capture`, `flight_logger`, and `sam3_detector`. With
  Tier 1+ this becomes `gz_pose_publisher → /drone/pose` directly (the
  pose-publisher SDF plugin in `worlds/scene1_hydrant.sdf` is already
  there).
- `bridge.yaml` already bridges depth + RGB + LiDAR + IMU, and now
  bridges the `set_pose` service. Adding a `cmd_vel` Twist bridge is one
  more entry.
- `scene_config` already parameterizes orbit gains/altitudes; Tier 2's
  velocity-control gains can be added to the same dict.
- The sync layer (Fix #2) means whatever flight controller produces pose
  doesn't need any timing assumptions in sam3_detector — the synchronizer
  re-pairs by stamp.

## What you should *not* keep when moving to Tier 1+

- The kinematic `_gz_set_pose` call. Either remove or gate it behind
  `control_mode == 'kinematic'`.
- The 4 Hz `POSE_UPDATE_DIV` rate. Velocity command should be 20+ Hz to
  feel responsive.
- The "drone always faces target via yaw=atan2" trick in image_capture
  — under real dynamics yaw lags position; the synced pose msg has the
  actual quaternion so use that.
