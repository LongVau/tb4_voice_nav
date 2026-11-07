from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'tb4_voice_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.xml'),
        ),
        (
            os.path.join('share', package_name, 'maps'),
            glob('maps/*.pgm') + glob('maps/*.yaml') + glob('maps/*.png'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sumika',
    maintainer_email='sumika@todo.todo',
    description='Map segmentation and Vietnamese PhoWhisper voice navigation for TurtleBot 4',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'map_segmenter = tb4_voice_nav.map_segmenter:main',
            'region_visualizer_node = tb4_voice_nav.region_visualizer_node:main',
            'voice_command_node = tb4_voice_nav.voice_command_node:main',
            'voice_navigator_node = tb4_voice_nav.voice_navigator_node:main',
        ],
    },
)
