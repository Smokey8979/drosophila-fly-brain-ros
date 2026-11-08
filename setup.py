from setuptools import setup
import os
from glob import glob

package_name = 'fly_brain_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, 'live_lab', 'virtual_world'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        ('share/' + package_name + '/models/four_wheel_fly_robot', glob('models/four_wheel_fly_robot/*')),
        ('share/' + package_name + '/live_lab/static', glob('live_lab/static/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboFly Researcher',
    maintainer_email='research@example.com',
    description='ROS 2 integration of Drosophila whole-brain connectome for autonomous 4-wheel robot',
    license='GPL-2.0-or-later',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fly_brain_node = fly_brain_ros.fly_brain_node:main',
            'standalone_demo = fly_brain_ros.standalone_demo:main',
            'mock_chassis = fly_brain_ros.mock_chassis:main',
        ],
    },
)
