"""
Simulation Stack Launch File for Drosophila Connectome Robot (Gazebo + Bridge + Brain Node).

Starts:
  1. Gazebo Fortress Simulator (learning_lab_arena.world)
  2. Four-Wheel Drosophila Robot Spawner
  3. ROS-GZ Bridge (/cmd_vel, /scan, /odom)
  4. Drosophila Whole-Brain Connectome ROS 2 Node (138,639 neurons)

This stack is managed by the SimulationLifecycleSupervisor for Sleep/Wake lifecycle control.
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

try:
    from ament_index_python.packages import get_package_share_directory
    HAVE_AMENT = True
except ImportError:
    HAVE_AMENT = False


def get_pkg_dir(package_name: str) -> str:
    if HAVE_AMENT:
        try:
            return get_package_share_directory(package_name)
        except Exception:
            pass
    pkg_root = Path(__file__).resolve().parents[1]
    return str(pkg_root)


def generate_launch_description():
    pkg_dir = get_pkg_dir("fly_brain_ros")
    world_file = os.path.join(pkg_dir, "worlds", "learning_lab_arena.world")
    model_sdf = os.path.join(pkg_dir, "models", "four_wheel_fly_robot", "model.sdf")
    config_file = os.path.join(pkg_dir, "config", "fly_brain_params.yaml")

    # Resolve canonical checkpoint default dynamically
    default_checkpoint_file = os.environ.get("FLY_BRAIN_CHECKPOINT_PATH", "")
    if not default_checkpoint_file:
        try:
            from fly_brain_ros.checkpoint_config import get_canonical_checkpoint_path
            default_checkpoint_file = get_canonical_checkpoint_path()
        except Exception:
            candidates = [
                os.path.join(pkg_dir, "checkpoints", "fly_brain_checkpoint.json"),
                os.path.join(str(Path(pkg_dir).parent), "checkpoints", "fly_brain_checkpoint.json"),
                "/home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    default_checkpoint_file = cand
                    break
            if not default_checkpoint_file:
                default_checkpoint_file = os.path.join(pkg_dir, "checkpoints", "fly_brain_checkpoint.json")

    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    headless = LaunchConfiguration("headless", default="false")
    checkpoint_file = LaunchConfiguration(
        "checkpoint_file",
        default=default_checkpoint_file
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo simulator headless (-s server only) without GUI client",
    )
    declare_checkpoint_file = DeclareLaunchArgument(
        "checkpoint_file",
        default_value=default_checkpoint_file,
        description="Path to persistent brain checkpoint file for restoration",
    )
    declare_start_asleep = DeclareLaunchArgument(
        "start_asleep",
        default_value="false",
        description="Start robot in persistent ASLEEP state if true",
    )

    # 1. Start Gazebo Fortress Simulator
    gz_args = PythonExpression([
        "'-r -s ' + '", world_file, "' if '", headless, "' == 'true' else '-r ' + '", world_file, "'"
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_pkg_dir("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    # 2. Spawn 4-Wheel Robot
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", model_sdf,
            "-name", "four_wheel_fly_robot",
            "-x", "0.0",
            "-y", "-6.0",
            "-z", "0.15",
            "-Y", "1.5708",
        ],
        output="screen",
    )

    # 3. ROS-GZ Bridge (/cmd_vel, /scan, /odom)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
            "/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        ],
        output="screen",
    )

    start_asleep = LaunchConfiguration("start_asleep", default="false")

    # 4. Drosophila Neural Connectome Controller Node
    fly_brain_node = Node(
        package="fly_brain_ros",
        executable="fly_brain_node",
        name="fly_brain_node",
        parameters=[
            config_file,
            {
                "use_sim_time": use_sim_time,
                "checkpoint_file": checkpoint_file,
                "start_asleep": PythonExpression(["'", start_asleep, "'.lower() in ['true', '1', 'yes']"]),
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_headless,
        declare_checkpoint_file,
        declare_start_asleep,
        gz_sim,
        spawn_robot,
        bridge,
        fly_brain_node,
    ])
