"""
Scene 1 Launch — Fire Hydrant
Starts: Gazebo → ros_gz_bridge → sam3_detector → mission_node → image_capture

Works both on a bare Ubuntu host and inside the Docker container.
"""

import os
import importlib.util
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# ── PYTHONPATH for user-installed pip packages ────────────────────────────────
# On the host machine, ultralytics/torch are installed in the user site-packages
# directory. We inject it whenever it exists. The previous version of this
# block also gated on `importlib.util.find_spec('ultralytics') is None`, which
# was wrong: that check runs in the launch process (which auto-loads user-site
# via Python's default), but the spawned `ros2 run` subprocess doesn't, so the
# child saw ModuleNotFoundError even when the launch found the module.
_user_site = os.path.expanduser('~/.local/lib/python3.12/site-packages')
_base_pythonpath = os.environ.get('PYTHONPATH', '')
if os.path.isdir(_user_site):
    _extra_pythonpath = _user_site + ':' + _base_pythonpath
else:
    _extra_pythonpath = _base_pythonpath

# ── SAM3 weights path ─────────────────────────────────────────────────────────
# Default: ~/sam3/sam3.pt. This lives in the user's home so it is NOT wiped
# by `rm -rf build/` during a colcon clean rebuild. The previous default was
# inside ~/ros2_ws/build/drone_recon/sam3/, which was lost when the build dir
# got cleaned out — see docs/ for the re-download instructions.
# Supports override via SAM3_WEIGHTS_PATH env var (set by Docker entrypoint).
_sam3_default = os.environ.get(
    'SAM3_WEIGHTS_PATH',
    os.path.expanduser('~/sam3/sam3.pt'))


def generate_launch_description():
    pkg = get_package_share_directory('drone_recon')
    world_file = os.path.join(pkg, 'worlds', 'scene1_hydrant.sdf')

    headless     = LaunchConfiguration('headless')
    output_dir   = LaunchConfiguration('output_dir')
    target       = LaunchConfiguration('target')
    scene        = LaunchConfiguration('scene')
    auto_recon   = LaunchConfiguration('auto_recon')
    auto_prune   = LaunchConfiguration('auto_prune')
    control_mode = LaunchConfiguration('control_mode')
    mission_mode = LaunchConfiguration('mission_mode')
    recon_method_arg = LaunchConfiguration('recon_method')
    randomize    = LaunchConfiguration('randomize')
    randomize_seed = LaunchConfiguration('seed')

    # If recon_method is the literal string 'auto', run BOTH splatfacto and
    # DA3 on every mission so the user always gets a splat *and* a colored
    # point cloud. Otherwise pass through whatever the user set.
    recon_method = PythonExpression([
        '"both" if "', recon_method_arg, '" == "auto" else ',
        '"', recon_method_arg, '"'
    ])

    # ── Gazebo ───────────────────────────────────────────────────────────────
    # Headless mode: -s flag starts the physics/sensor server without the GUI.
    # The ogre2 renderer still produces camera images via Xvfb or EGL.
    gz_cmd_gui      = ['gz', 'sim', '-r', world_file]
    gz_cmd_headless = ['gz', 'sim', '-r', '-s', world_file]

    from launch.conditions import IfCondition, UnlessCondition

    gz_gui = ExecuteProcess(
        cmd=gz_cmd_gui,
        output='screen',
        condition=UnlessCondition(headless),
        additional_env={'GZ_SIM_RESOURCE_PATH': os.path.expanduser('~/.gz/fuel')},
    )
    gz_headless = ExecuteProcess(
        cmd=gz_cmd_headless,
        output='screen',
        condition=IfCondition(headless),
        additional_env={'GZ_SIM_RESOURCE_PATH': os.path.expanduser('~/.gz/fuel')},
    )

    # ── ros_gz_bridge (start 3 s after Gazebo) ───────────────────────────────
    # Topic bridges live in bridge.yaml. The set_pose service bridge has to
    # be passed as a positional CLI argument because parameter_bridge's YAML
    # parser only understands topic entries.
    # Every node uses `use_sim_time=true` so that timestamps on bridged gz
    # messages (camera, depth, /drone/pose_actual) and locally-published
    # messages (/drone/pose from mission_node, /sam3/* from sam3_detector,
    # CSV rows from flight_logger) all reference the SAME clock. Without
    # this, mission_node's wall-time stamps would never align with gz's
    # sim-time camera stamps, and message_filters in sam3_detector would
    # silently fail to assemble triplets.
    use_sim_time_param = {'use_sim_time': True}
    bridge = TimerAction(period=3.0, actions=[
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            arguments=[
                '/world/recon_world/set_pose@ros_gz_interfaces/srv/SetEntityPose',
                '--ros-args', '-p',
                f'config_file:={os.path.join(pkg, "config", "bridge.yaml")}',
            ],
            parameters=[use_sim_time_param],
            output='screen',
        )
    ])

    # ── Scene randomizer (start 4 s after launch — after bridge is up) ───────
    # Permutes the 5 placeable objects (hydrant/plant/bench/bin/mailbox)
    # among the 5 canonical SDF positions BEFORE the drone enters the
    # search area. The drone has no SDF prior, so each randomized run is
    # a true test of vision-only autonomous identification. Disable with
    # `randomize:=false`; reproduce a specific permutation with `seed:=N`.
    from launch.conditions import IfCondition as _IfCondition
    randomizer = TimerAction(period=4.0, actions=[
        Node(
            package='drone_recon',
            executable='scene_randomizer',
            name='scene_randomizer',
            output='screen',
            parameters=[use_sim_time_param, {
                'seed': randomize_seed,
            }],
            condition=_IfCondition(randomize),
        )
    ])

    # ── SAM3 detector (start 5 s after launch) ───────────────────────────────
    sam3 = TimerAction(period=5.0, actions=[
        Node(
            package='drone_recon',
            executable='sam3_detector',
            name='sam3_detector',
            output='screen',
            additional_env={'PYTHONPATH': _extra_pythonpath},
            parameters=[use_sim_time_param, {
                'target_prompt': target,
                'model_size':    'b',
                # 0.30 picked after the SDF plausibility gate was removed
                # in favor of vision-only localization. At 0.15, SAM3
                # returned every shape vaguely matching the prompt (a
                # red hydrant came back as a "potted plant" sometimes),
                # and without a position-based reject those false hits
                # poisoned the mission_node averaging buffer. 0.30
                # filters out the borderline mis-IDs while still
                # accepting the synthetic-scene's weaker confidences
                # for the actually-prompted object.
                'confidence':    0.30,
                'model_path':    _sam3_default,
            }],
        )
    ])

    # ── Mission node (start 7 s after launch) ────────────────────────────────
    mission = TimerAction(period=7.0, actions=[
        Node(
            package='drone_recon',
            executable='mission_node',
            name='mission_node',
            output='screen',
            parameters=[use_sim_time_param, {
                'scene':         scene,
                'control_mode':  control_mode,
                'mission_mode':  mission_mode,
                'target_prompt': target,
            }],
        )
    ])

    # ── Image capture node (start 7 s after launch) ──────────────────────────
    capture = TimerAction(period=7.0, actions=[
        Node(
            package='drone_recon',
            executable='image_capture',
            name='image_capture',
            output='screen',
            parameters=[use_sim_time_param, {
                'output_dir':    output_dir,
                'target_prompt': target,
                'auto_recon':    auto_recon,
                'auto_prune':    auto_prune,
                'recon_method':  recon_method,
            }],
        )
    ])

    # ── Flight logger (start 6 s after launch — before mission arms) ─────────
    import datetime as _dt
    _log_stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_path  = os.path.expanduser(f'~/flight_logs/flight_{_log_stamp}.csv')
    flight_logger = TimerAction(period=6.0, actions=[
        Node(
            package='drone_recon',
            executable='flight_logger',
            name='flight_logger',
            output='screen',
            arguments=['--out', _log_path, '--scene', scene],
            parameters=[use_sim_time_param],
        )
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run Gazebo server-only (no GUI window). '
                        'Requires Xvfb or a forwarded DISPLAY for camera rendering.'),
        DeclareLaunchArgument(
            'output_dir', default_value=os.path.expanduser('~/recon_output'),
            description='Directory to save captured images and metadata'),
        DeclareLaunchArgument(
            'target', default_value='fire hydrant',
            description='SAM3 text prompt for the target object. Drives '
                        'detection, distance estimation, init point cloud '
                        'shape, and final prune box (see drone_recon/targets.py).'),
        DeclareLaunchArgument(
            'scene', default_value='scene1_hydrant',
            description='Scene name (drone_recon.scene_config.SCENES key). '
                        'Drives mission waypoints, orbit geometry, home '
                        'position, and obstacle clearance warnings.'),
        DeclareLaunchArgument(
            'auto_recon', default_value='true',
            description='Run the 3DGS reconstruction pipeline automatically '
                        'after the mission completes. Set to false when '
                        'iterating on flight quality — re-run later with '
                        '`ros2 run drone_recon run_recon <output_dir>`.'),
        DeclareLaunchArgument(
            'auto_prune', default_value='true',
            description='Prune the splat to the per-target bounding box. '
                        'Set false for whole-scene mapping runs; the live '
                        'viewer then shows the full unpruned splat. '
                        'scene_splat.ply is exported either way.'),
        DeclareLaunchArgument(
            'control_mode', default_value='kinematic',
            description='Drone actuation mode. "kinematic" (default) '
                        'teleports the drone via SetEntityPose. '
                        '"velocity_kinematic" closes the loop on actual '
                        'gz pose and publishes Twist to /drone/cmd_vel '
                        '(Tier 1 of the flight-controller plan; see '
                        'docs/velocity_controller_plan.md).'),
        DeclareLaunchArgument(
            'mission_mode', default_value='inspection',
            description='Flight pattern. "inspection" (default) finds the '
                        'target with SAM3 and orbits it. "mapping" sweeps '
                        'the search area at multiple altitudes for whole-'
                        'scene reconstruction (use with auto_prune:=false).'),
        DeclareLaunchArgument(
            'recon_method', default_value='auto',
            description='Reconstruction method. "auto" (default) runs '
                        'BOTH splatfacto and Depth Anything 3 — every '
                        'mission produces a splat *and* a colored point '
                        'cloud. Force one with "splatfacto", '
                        '"depth_anything_3", or explicit "both". DA3 '
                        'requires a one-time install; see '
                        'docs/depth_anything_3.md.'),
        DeclareLaunchArgument(
            'randomize', default_value='true',
            description='At launch, shuffle which physical object lives '
                        'at each of the five canonical scene slots '
                        '(plant/hydrant/bench/bin/mailbox), so each '
                        'mission is a fresh test of vision-only '
                        'autonomy. Set to "false" to keep the SDF '
                        'defaults.'),
        DeclareLaunchArgument(
            'seed', default_value='0',
            description='Random seed for the scene randomizer. '
                        '0 (default) uses OS entropy — every launch '
                        'is different. Set to any non-zero int to '
                        'reproduce a specific permutation.'),
        gz_gui,
        gz_headless,
        bridge,
        randomizer,
        sam3,
        mission,
        capture,
        flight_logger,
    ])
