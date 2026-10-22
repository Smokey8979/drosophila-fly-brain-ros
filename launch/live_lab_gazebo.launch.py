"""
Launch Gazebo Fortress Arena, Spawner, ROS-GZ Bridge, and the Drosophila Connectome Controller Node.
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    from ament_index_python.packages import get_package_share_directory
    HAVE_AMENT = True
except ImportError:
    HAVE_AMENT = False


def get_package_share_dir(package_name: str) -> str:
    """Resolves package share directory with clean fallback to local source tree."""
    if HAVE_AMENT:
        try:
            return get_package_share_directory(package_name)
        except Exception:
            pass
    # Fallback to current repository path
    pkg_root = Path(__file__).resolve().parents[1]
    return str(pkg_root)


def generate_launch_description():
    pkg_dir = get_package_share_dir("fly_brain_ros")
    world_file = os.path.join(pkg_dir, "worlds", "learning_lab_arena.world")
    model_sdf = os.path.join(pkg_dir, "models", "four_wheel_fly_robot", "model.sdf")
    config_file = os.path.join(pkg_dir, "config", "fly_brain_params.yaml")

    # Launch configuration arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    headless = LaunchConfiguration("headless", default="false")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo simulator headless without GUI client",
    )

    # 1. Start Gazebo Fortress Simulator via ros_gz_sim
    gz_args = f"-r {world_file}"
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_dir("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    # 2. Spawn 4-Wheel Robot into Gazebo Fortress
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

    # 3. ROS-GZ Bridge (Twist, LaserScan, Odometry)
    # Bridges topics between ROS 2 Humble and Gazebo Fortress
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

    # 4. Launch Drosophila Neural Connectome Controller Node
    fly_brain_node = Node(
        package="fly_brain_ros",
        executable="fly_brain_node",
        name="fly_brain_node",
        parameters=[config_file, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_headless,
        gz_sim,
        spawn_robot,
        bridge,
        fly_brain_node,
    ])
