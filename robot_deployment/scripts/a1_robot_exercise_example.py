"""Example of running A1 robot with position control.

To run:
python -m robot_deployment.robots.a1_robot_exercise_example
"""
import os.path as osp
import numpy as np
import pybullet
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from pybullet_utils import bullet_client
import time
from typing import Tuple

from legged_gym import LEGGED_GYM_ROOT_DIR
from configs.definitions import DeploymentConfig
from robot_deployment.robots import a1
from robot_deployment.robots import a1_robot
from robot_deployment.robots.motors import MotorCommand

@dataclass
class Config:
  use_real_robot: bool = False
  deployment: DeploymentConfig = DeploymentConfig(
    use_real_robot="${use_real_robot}",
    timestep=0.002,
    action_repeat=1,
    reset_time_s=3.,
    num_solver_iterations=30,
    init_position=(0., 0., 0.32),
    init_rack_position=(0., 0., 1.),
    stiffness=dict(joint=100.),
    damping=dict(joint=1.),
    on_rack=False
  )

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def get_action(robot: a1.A1, t):
  mid_action = np.array([0.0, 0.9, -1.8] * 4)
  amplitude = np.array([0.0, 0.2, -0.4] * 4)
  freq = 1.0
  return MotorCommand(desired_position=mid_action +
                      amplitude * np.sin(2 * np.pi * freq * t),
                      kp=robot.motor_group.kps,
                      desired_velocity=np.zeros(robot.num_motors),
                      kd=robot.motor_group.kds)



@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
  connection_mode = pybullet.DIRECT if cfg.deployment.use_real_robot else pybullet.GUI
  p = bullet_client.BulletClient(connection_mode=connection_mode)
  p.setAdditionalSearchPath(osp.join(LEGGED_GYM_ROOT_DIR, 'resources'))
  p.loadURDF("plane.urdf")
  p.setGravity(0.0, 0.0, -9.81)

  robot_ctor = a1_robot.A1Robot if cfg.deployment.use_real_robot else a1.A1
  robot = robot_ctor(pybullet_client=p, sim_conf=cfg.deployment)
  robot.reset()

  for _ in range(10000):
    action = get_action(robot, robot.time_since_reset)
    robot.step(action)
    if not cfg.deployment.use_real_robot:
      time.sleep(cfg.deployment.timestep)
    print(robot.base_orientation_rpy)


if __name__ == "__main__":
  main()
