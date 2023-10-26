from setuptools import setup, find_packages
import os

pkg_root = os.path.dirname(os.path.realpath(__file__))

def local_dep(dep_name):
	return f"{dep_name}@file://localhost{os.path.join(pkg_root, dep_name)}"

setup(
	name='legged_gym',
	version='0.0.0',
	author='UWRobotLearning ',
	license="BSD-3-Clause",
	packages=find_packages(),
	author_email='uwrobotlearning@cs.uw.edu',
	description='Isaac Gym environments for ground control',
	python_requires="==3.8.*",
	install_requires=['isaacgym',  # needs to be installed manually
					local_dep('rsl_rl'),
					'matplotlib',
					'lxml',
					'numpy<1.24',
					'opencv-python-headless',
					'pybullet',
					'pygame',
					'tensorboard',
					'transformations',
					'hydra-core',
					'pyquaternion',
					'setuptools==59.5.0',
					'termcolor',
					'noise',
					'scikit-learn',
					'tqdm',
					'torch',
					'torchvision',
					'torchaudio'
					],
		extras_require={
			"deploy": [local_dep('robot_deployment')]
		}
)
