from setuptools import setup, find_packages
import os


# This portion allows installing local dependencies (in non-editable mode, meaning
# the changes made wouldn't be reflected in the project unless the packages are re-installed)
# This is not recommended, hence it's not used by default.
# For automatic installation of dependencies, see autosetup.py in the project root.
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
					'rsl_rl',  #local_dep('rsl_rl'),  # needs to be installed manually
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
					'torchaudio',
					# 'pydantic'
					],
		extras_require={
			"deploy": ['robot_deployment']  #local_dep('robot_deployment')]  # needs to be installed manually
		}
)
