"""Launch Gazebo maze world, TurtleBot 4, AMCL localization, Nav2, RViz, and RegionVisualizerNode."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    own = get_package_share_directory('tb4_voice_nav')
    sim = get_package_share_directory('turtlebot4_ignition_bringup')
    nav = get_package_share_directory('turtlebot4_navigation')

    def include(root: str, name: str, arguments: dict):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(root, 'launch', name)),
            launch_arguments=arguments.items(),
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'map',
                default_value=os.path.join(own, 'maps', 'map.yaml'),
                description='Full path to occupancy grid map.yaml (maze)',
            ),
            DeclareLaunchArgument(
                'regions_yaml',
                default_value=os.path.join(own, 'config', 'regions.yaml'),
                description='Full path to segmented regions.yaml',
            ),
            DeclareLaunchArgument(
                'localization_params',
                default_value=os.path.join(own, 'config', 'localization.yaml'),
                description='Full path to WSL-tuned AMCL localization.yaml',
            ),
            DeclareLaunchArgument(
                'nav2_params',
                default_value=os.path.join(own, 'config', 'nav2.yaml'),
                description='Full path to Nav2 parameters YAML',
            ),
            DeclareLaunchArgument(
                'fastdds_udp',
                default_value='true',
                choices=['true', 'false'],
                description='Use UDP-only Fast DDS profile to avoid WSL2 shared-memory TF stalls',
            ),
            DeclareLaunchArgument('rviz', default_value='true'),
            DeclareLaunchArgument(
                'gui',
                default_value='true',
                choices=['true', 'false'],
                description='Open the Gazebo client window',
            ),
            SetEnvironmentVariable(
                'FASTRTPS_DEFAULT_PROFILES_FILE',
                os.path.join(own, 'config', 'fastdds_udp.xml'),
                condition=IfCondition(LaunchConfiguration('fastdds_udp')),
            ),
            LogInfo(
                msg='Fast DDS UDP-only profile enabled for WSL2 TF stability',
                condition=IfCondition(LaunchConfiguration('fastdds_udp')),
            ),
            SetEnvironmentVariable(
                'IGN_GUI_PLUGIN_PATH',
                ':'.join(
                    [
                        os.path.join(
                            get_package_share_directory(
                                'turtlebot4_ignition_gui_plugins'
                            ),
                            'lib',
                        ),
                        os.path.join(
                            get_package_share_directory(
                                'irobot_create_ignition_plugins'
                            ),
                            'lib',
                        ),
                    ]
                ),
            ),
            SetEnvironmentVariable(
                'IGN_GAZEBO_RESOURCE_PATH',
                ':'.join(
                    [
                        os.path.join(sim, 'worlds'),
                        os.path.join(
                            get_package_share_directory(
                                'irobot_create_ignition_bringup'
                            ),
                            'worlds',
                        ),
                        os.path.dirname(
                            get_package_share_directory('turtlebot4_description')
                        ),
                    ]
                ),
            ),
            include(
                get_package_share_directory('ros_ign_gazebo'),
                'ign_gazebo.launch.py',
                dict(ign_args=[os.path.join(sim, 'worlds', 'maze.sdf'), ' -r -s -v 2']),
            ),
            ExecuteProcess(
                cmd=[
                    'ign',
                    'gazebo',
                    '-g',
                    '-v',
                    '2',
                    '--gui-config',
                    os.path.join(sim, 'gui', 'standard', 'gui.config'),
                ],
                condition=IfCondition(LaunchConfiguration('gui')),
                output='screen',
            ),
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='clock_bridge',
                arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
            ),
            include(
                sim,
                'turtlebot4_spawn.launch.py',
                dict(
                    world='maze',
                    model='standard',
                    rviz=LaunchConfiguration('rviz'),
                    use_sim_time='true',
                    slam='false',
                    nav2='false',
                    localization='false',
                ),
            ),
            include(
                nav,
                'localization.launch.py',
                dict(
                    map=LaunchConfiguration('map'),
                    params=LaunchConfiguration('localization_params'),
                    use_sim_time='true',
                ),
            ),
            include(
                nav,
                'nav2.launch.py',
                dict(
                    use_sim_time='true',
                    params_file=LaunchConfiguration('nav2_params'),
                ),
            ),
            Node(
                package='tb4_voice_nav',
                executable='region_visualizer_node',
                name='region_visualizer_node',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'regions_yaml': LaunchConfiguration('regions_yaml'),
                    }
                ],
            ),
        ]
    )
