"""ROS 2 Launch file for Fly Brain closed-loop robot control."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('fly_brain_ros'),
        'config',
        'fly_brain_params.yaml'
    )

    return LaunchDescription([
        # Main Fly Brain Controller Node
        Node(
            package='fly_brain_ros',
            executable='fly_brain_node',
            name='fly_brain_node',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('/scan', '/scan'),
                ('/cmd_vel', '/cmd_vel'),
                ('/emergency_stop', '/emergency_stop'),
            ]
        ),

        # Mock 4-Wheel Chassis for simulation testing
        Node(
            package='fly_brain_ros',
            executable='mock_chassis',
            name='mock_chassis',
            output='screen',
        )
    ])
