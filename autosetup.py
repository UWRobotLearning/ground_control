import os
import sys
import importlib.util
import subprocess
import argparse
import shutil

def local_pip_install(package_name: str, absolute_path_to_package: os.PathLike, reinstall=False) -> None:
    if not reinstall and importlib.util.find_spec(package_name) is not None:
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

def main(args: argparse.Namespace):
    project_root: os.PathLike = os.path.dirname(os.path.realpath(__file__))
    os.chdir(project_root)

    print("1: Installing the 'isaacgym' library...")
    local_pip_install('isaacgym', os.path.join(project_root, '../isaacgym/python'))

    print("2: Installing rsl_rl...")
    local_pip_install('rsl_rl', os.path.join(project_root, 'rsl_rl'))

    print("3: Installing 'ground_control' (this project)...")
    local_pip_install('ground_control', project_root, reinstall=True)

    if not args.deploy:
        print("Successfully finished (without deployment setup)!")
        return
    
    print("4: [DEPLOY] Installing Boots, LCM, CMake via apt...")
    subprocess.run("sudo apt install -y libboost-all-dev liblcm-dev cmake", shell=True, check=True)

    print("5: [DEPLOY] Building unitree_legged_sdk...")
    os.chdir('robot_deployment/third_party/unitree_legged_sdk')
    if os.path.isdir('./build'):
        shutil.rmtree('./build')
    os.mkdir('build')
    os.chdir('build')
    subprocess.run('cmake ..', shell=True, check=True)
    subprocess.run('make', shell=True, check=True)
    subprocess.run('mv robot_interface* ../../..', shell=True, check=True)

    print("6: [DEPLOY] Install the robot_deployment package...")
    local_pip_install('robot_deployment', os.path.join(project_root, 'robot_deployment'), reinstall=True)

    print("7: [DEPLOY] Re-install 'ground_control' (this project) with deployment option...")
    subprocess.run(
        args=[sys.executable, "-m", "pip", "install", "-e", ".[deploy]"],
        cwd=project_root, 
        check=True
    )

    print("Sucessfully finished, including deployment!")


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