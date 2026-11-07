"""Launch VoiceNavigatorNode and RegionVisualizerNode for tb4_voice_nav."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    own = get_package_share_directory('tb4_voice_nav')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'regions_yaml',
                default_value=os.path.join(own, 'config', 'regions.yaml'),
                description='Path to segmented regions.yaml',
            ),
            DeclareLaunchArgument(
                'at_region_radius',
                default_value='1.5',
                description='Radius (m) within which robot is considered already at origin region',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='true',
            ),
            DeclareLaunchArgument(
                'fastdds_udp',
                default_value='true',
                choices=['true', 'false'],
            ),
            SetEnvironmentVariable(
                'FASTRTPS_DEFAULT_PROFILES_FILE',
                os.path.join(own, 'config', 'fastdds_udp.xml'),
                condition=IfCondition(LaunchConfiguration('fastdds_udp')),
            ),
            Node(
                package='tb4_voice_nav',
                executable='region_visualizer_node',
                name='region_visualizer_node',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'regions_yaml': LaunchConfiguration('regions_yaml'),
                    }
                ],
            ),
            Node(
                package='tb4_voice_nav',
                executable='voice_navigator_node',
                name='voice_navigator_node',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'regions_yaml': LaunchConfiguration('regions_yaml'),
                        'at_region_radius': LaunchConfiguration('at_region_radius'),
                    }
                ],
            ),
        ]
    )
