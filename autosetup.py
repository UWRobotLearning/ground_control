import os           # For path processing and navigation
import sys          # For sys.executable, access to the Python path
import subprocess   # For calling commands (pip and apt)
import argparse     # For parsing arguments passed to the script
import shutil       # For moving/manipulating multiple paths/folders

# Ground Control Autosetup Script
# This script completes all necessary steps for setup, or can be used as a reference in manual setup.

# There is one local dependency (isaacgym) that is outside this repo, you should download it and either
# place it in the same parent directory as 'ground_control' (meaning outside the repo) or manually install it
# via pip (by running 'pip install -e .' in isaacgym/python) and then run this script.

# By default, the script sets up everything necessary for train/play on sim, but not deployment. If you would like
# autosetup of deployment, you can run this script as 'python autosetup.py -d' or 'python autosetup.py --deploy', which
# will perform additional steps and re-installs the main repo at the end. With this option, you may be asked for the sudo
# password in order to install some dependencies from apt.

# The main script gets the arguments (parsed by argparse) and performs the entire setup, printing details along the way.
def main(args: argparse.Namespace):
    # Absolute path to the root of this repo (the 'ground_control' folder)
    project_root: os.PathLike = os.path.dirname(os.path.realpath(__file__))

    # Install isaacgym, skip if already installed, throw a FileNotFoundError if it's not installed
    # and cannot be found in the same parent directory.
    print("1: Installing the 'isaacgym' library...")
    local_pip_install('isaacgym', os.path.join(project_root, '../isaacgym/python'))

    # Install rsl_rl (as above, skip if already installed, throw an error if it cannot be found)
    print("2: Installing rsl_rl...")
    local_pip_install('rsl_rl', os.path.join(project_root, 'rsl_rl'))

    # Install this project (ground_control, with all other dependencies except for deployment).
    # Re-install if already installed, throw an error if setup.py not found.
    print("3: Installing 'ground_control' (this project)...")
    local_pip_install('ground_control', project_root, reinstall=True)

    # Finish the script successfully if deployment setup is not enabled.
    if not args.deploy:
        print("Successfully finished (without deployment setup)!")
        return

    # Install apt dependencies (asks user for sudo password), fail if apt fails, otherwise continue.
    print("4: [DEPLOY] Installing Boots, LCM, CMake via apt...")
    subprocess.run("sudo apt install -y libboost-all-dev liblcm-dev cmake", shell=True, check=True)

    # Build the robot_interface package (build changes between devices, hence done for each machine).
    print("5: [DEPLOY] Building unitree_legged_sdk...")
    # Go to the source code directory for the package, make a build folder (remove old one if exists),
    # and compile the source code. Then place the robot_interface package at the root of robot_deployment.
    os.chdir(os.path.join(project_root, 'robot_deployment/third_party/unitree_legged_sdk'))
    if os.path.isdir('./build'):
        shutil.rmtree('./build')
    os.mkdir('build')
    os.chdir('build')
    subprocess.run('cmake ..', shell=True, check=True)
    subprocess.run('make', shell=True, check=True)
    subprocess.run('mv robot_interface* ../../..', shell=True, check=True)

    # Install the robot_deployment package, now that the robot_interface is available. Re-install if needed.
    print("6: [DEPLOY] Install the robot_deployment package...")
    local_pip_install('robot_deployment', os.path.join(project_root, 'robot_deployment'), reinstall=True)

    # Install ground_control with the [deploy] option, which also check for robot_deployment as a dependency
    # (and can include other dependencies in the future if needed, specified in setup.py at project root).
    print("7: [DEPLOY] Re-install 'ground_control' (this project) with deployment option...")
    subprocess.run(
        args=[sys.executable, "-m", "pip", "install", "-e", ".[deploy]"],
        cwd=project_root,
        check=True
    )

    print("Sucessfully finished, including deployment!")

# Given the package name (as accessed via pip) and the absolute path to a local package's root folder (containing setup.py)
# tries to install the package in editable mode from that path. By default, if the package is already installed, then it skips
# the installation and prints a message to the user, this can be overriden by setting reinstall=True. Otherwise, if the package
# cannot be found in the given absolute path, throws a FileNotFoundError with an appropriate message.
def local_pip_install(package_name: str, absolute_path_to_package: os.PathLike, reinstall=False) -> None:
    if not reinstall:
        pip_show = subprocess.run(
            args=[sys.executable, "-m", "pip", "show", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if pip_show.returncode == 0:
            print(f'-Package {package_name} is already installed, skipping...')
            return
    if not os.path.isdir(absolute_path_to_package):
        print(f'-Couldn\'t find the package {package_name} in {absolute_path_to_package}.')
        print('-Make sure it\'s accessible by that path, or manually install it.')
        raise FileNotFoundError
    subprocess.run(
        args=[sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=absolute_path_to_package,
        check=True
    )

# Main script - contains and parses the arguments needed to the main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Ground Control Autosetup",
        description="Autosetup script for the 'ground_control' repo, including setup for deployment",
        epilog="Sets up the 'ground control' repo by installing isaacgym (assumed to be in the same parent directory \
            as the 'ground control' repository) and rsl_rl (in the repo), then installing this repo. It can also install \
            the deployment module if chosen."
    )
    parser.add_argument("-d", "--deploy", action="store_true", help="Installs necessary dependencies for deployment")
    main(parser.parse_args())
