"""
Unified Single Launch File for Drosophila Connectome Robot Simulation & Live Laboratory.

Starts the complete integrated environment:
  1. Gazebo Fortress 3D Physics Simulator with learning lab arena world
  2. 4-Wheel Differential/Skid-Steer Drosophila Robot Spawner
  3. ROS-GZ Bridge (/cmd_vel, /scan, /odom)
  4. Drosophila Whole-Brain Connectome ROS 2 Node (138,639 neurons)
  5. Interactive Live Laboratory Web Dashboard Server (http://127.0.0.1:5000)

Usage:
  ros2 launch fly_brain_ros live_lab_full.launch.py
  ros2 launch fly_brain_ros live_lab_full.launch.py use_sim_time:=false start_dashboard:=true
  ros2 launch fly_brain_ros live_lab_full.launch.py headless:=true
"""

import os
import sys
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
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

    # Locate web dashboard server script
    web_server_candidates = [
        Path(__file__).resolve().parents[1] / "live_lab" / "web_server.py",
        Path(__file__).resolve().parents[2] / "live_lab" / "web_server.py",
        Path(pkg_dir) / "live_lab" / "web_server.py",
        Path.cwd() / "live_lab" / "web_server.py",
        Path("/home/om/Desktop/robo_fly/live_lab/web_server.py"),
    ]
    web_server_path = None
    for cand in web_server_candidates:
        if cand.exists():
            web_server_path = str(cand.resolve())
            break
    if not web_server_path:
        web_server_path = str(Path(__file__).resolve().parents[1] / "live_lab" / "web_server.py")

    # Launch configuration arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    headless = LaunchConfiguration("headless", default="false")
    start_dashboard = LaunchConfiguration("start_dashboard", default="true")
    dashboard_port = LaunchConfiguration("dashboard_port", default="5000")
    start_asleep = LaunchConfiguration("start_asleep", default="false")

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
    declare_start_dashboard = DeclareLaunchArgument(
        "start_dashboard",
        default_value="true",
        description="Automatically launch browser web dashboard at http://127.0.0.1:5000",
    )
    declare_dashboard_port = DeclareLaunchArgument(
        "dashboard_port",
        default_value="5000",
        description="Port for the interactive web dashboard server",
    )
    declare_start_asleep = DeclareLaunchArgument(
        "start_asleep",
        default_value="false",
        description="Start robot in persistent ASLEEP state if true",
    )

    # 1. Start Gazebo Fortress Simulator via ros_gz_sim
    gz_args = PythonExpression([
        "'-r -s ' + '", world_file, "' if '", headless, "' == 'true' else '-r ' + '", world_file, "'"
    ])

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
        parameters=[
            config_file,
            {
                "use_sim_time": use_sim_time,
                "start_asleep": PythonExpression(["'", start_asleep, "'.lower() in ['true', '1', 'yes']"]),
            },
        ],
        output="screen",
    )

    # 5. Launch Interactive Live Laboratory Web Dashboard Server
    dashboard_process = ExecuteProcess(
        cmd=[
            sys.executable,
            web_server_path,
            "--port",
            dashboard_port,
            "--host",
            "0.0.0.0",
        ],
        output="screen",
        condition=IfCondition(start_dashboard),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_headless,
        declare_start_dashboard,
        declare_dashboard_port,
        declare_start_asleep,
        gz_sim,
        spawn_robot,
        bridge,
        fly_brain_node,
        dashboard_process,
    ])
