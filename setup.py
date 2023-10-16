from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='0.0.0',
    author='UWRobotLearning ',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='uwrobotlearning@cs.uw.edu',
    description='Isaac Gym environments for ground control',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'lxml',
                      'numpy<1.24',
                      'opencv-python-headless',
                      'pybullet',
                      'pygame',
                      'tensorboard',
                      'transformations'
                      ]
)
