Moved to: [https://UWRobotLearning.github.io/LeggedRobots/a1_control_base](https://UWRobotLearning.github.io/LeggedRobots/a1_control_base)

# ground_control


## Installation instructions

1. Create a conda environment with Python 3.8 (Isaac Gym does not support Python 3.9, only Python >=3.6,<3.9):

`conda create -n a1 python=3.8`

2. Activate the conda environment:
`conda activate a1`

3. Install Jax: 
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Install isaacgym inside the conda environment (Make sure the conda environment is activated)
```bash
cd path/to/isaacgym/python
pip install -e .
```

5. Export the path as follows (replace "/path/to" to the right path where miniconda was installed):
```bash
export LD_LIBRARY_PATH=/path/to/miniconda3/envs/a1/lib
```

6. At this point it's worth checking two things: a) that we can run isaacgym on the gpu, and b) that we can see the gpu on jax.
Run the following:
```bash
cd /path/to/isaacgym/python/examples
python joint_monkey.py
```
You should see Isaac Gym open up, and you should see a message at the beginning of the terminal similar to:
```bash
+++ Using GPU PhysX
...
Physics Device: cuda:0
```

Now to test the GPU on JAX, run the following on the terminal:
```bash
python
```

```python
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)   ## This should print "gpu"
```

If both of these produce the expected output, continue :D

7. Now, let's install RSL_RL:
```bash
cd /path/to/ground_control
cd rsl_rl
pip install -e .
```

8. Now, let's install the ground_control package:
```bash
cd /path/to/ground_control
pip install -e .
```
Note: To solve package dependency issues, we need to downgrade the version of Pydantic to 1.10. However, then the 
class TypeAdapter is not implemented, which is the version that we use. The temporary fix for now is to comment out Pydantic
in train.py, play.py, and anywhere else that it gets used.

9. At this point, you should be able to train a policy on ground_control with the following command:
```bash
cd /path/to/ground_control/..
python ground_control/legged_gym/scripts/train.py headless=true
```
(Note: This will create an `experiment_logs` directory in the parent folder of ground_control)

10. To install the libraries needed for deployment, run the following:
```bash
sudo apt install -y libboost-all-dev liblcm-dev cmake
```

11. Now, let's install the unitree_legged_sdk:
```bash
cd /path/to/ground_control/robot_deployment/third_party/unitree_legged_sdk
```
If the `build/` directory exists, to do a clean install, remove it first:
```bash
rm -r build/
```

Then, run the following:
```bash
mkdir build && cd build
cmake ..
make
```

If this builds successfully, then move the built object as follows:
```bash
mv robot_interface* ../../..
```
Else, you may need to debug this

12. Now, install the robot_deployment package as follows:
```bash
cd /path/to/ground_control/robot_deployment
pip install -e .
```

13. Now, we are going to install tensorflow probability as follows:
```bash
pip install tfp-nightly[jax]
```

Note: this breaks typing-extensions from pydantic
Need to figure out how to either make both of these compatible or maybe use another repo other than tensorflow probability, such as 


Note: One way to fix dependency issues is to use Pydantic <2.0, so for example 1.10, since this is the package that requires a version
of typing extensions which is not compatible with tensorflow_probability


## Observation and Actions Details:

### In LocomotionGym (deployment environment):
1. 
