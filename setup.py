from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_recon'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'),  glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'),  glob('config/*.yaml')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohamed Gallai',
    maintainer_email='mgallai@binghamton.edu',
    description='Autonomous drone inspection with SAM3 and 3D Gaussian Splatting',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mission_node    = drone_recon.mission_node:main',
            'sam3_detector   = drone_recon.sam3_detector:main',
            'image_capture   = drone_recon.image_capture:main',
            'flight_logger   = drone_recon.flight_logger:main',
            'scene_randomizer = drone_recon.scene_randomizer:main',
            'live_view       = drone_recon.live_view:main',
            'run_recon       = drone_recon.run_recon:main',
            'run_recon_da3   = drone_recon.run_recon_da3:main',
            'voice_target    = drone_recon.voice_target:main',
            'voice_mission   = drone_recon.voice_mission:main',
            'voice_gui       = drone_recon.voice_gui:main',
            'ai_mission      = drone_recon.ai_mission:main',
        ],
    },
)
