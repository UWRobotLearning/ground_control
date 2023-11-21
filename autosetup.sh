#!/usr/bin/env bash

# Moves to the root directory of the project (the directory this file is in).
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

# Checks if isaacgym is installed by pip.
pip show isaacgym &> /dev/null
ISAAC_MISSING=$?

# (also from this point on, we would like to stop the script if an error occurs)
set -e

# If isaacgym hasn't been installed yet and is placed in the same parent folder as 
# this repository, installs it. If the isaacgym folder cannot be found either,
# lets the user know and exits with an error.
if [ $ISAAC_MISSING -eq 0 ]; then
    echo 'isaacgym already installed, skipping'
elif [ ! -d "../isaacgym/python" ]; then
    echo "Cannot find isaacgym, make sure it\'s in the same directory as the package or manually install it."
    exit 1
else
    cd ../isaacgym/python
    pip install -e .
    cd ../../ground_control
fi

# Run setup.py to install everything else needed for this package.
pip install -e .
echo ">>>Autosetup has successfully installed all python dependencies."

# If the command line option "--add_deploy" is added (or any command line option really),
# do the following block of logic.
if [ $# -ge 1 ]; then
   # If not already installed, install Boost, LCM and CMake,
   # needed for building Unitree's SDK interface
   sudo apt install -y libboost-all-dev liblcm-dev cmake
   
   # Go to the Unitree SDK folder, make a new build folder (delete old folder if exists)
   # and build the package. Then move it to robot_deployment, and go back to package root.
   cd robot_deployment/third_party/unitree_legged_sdk
   if [ -d "build" ]; then
       rm -r build
   fi
   mkdir build
   cd build
   cmake ..
   make
   mv robot_interface* ../../..
   cd ../../../..
   
   # Run setup.py to install the 'robot_deployment' sub-package (after building c++ deps).
   pip install -e .[deploy]
   echo ">>>Autosetup has successfully installed all robot_deployment c++ dependencies and the corresponding python package."
fi
