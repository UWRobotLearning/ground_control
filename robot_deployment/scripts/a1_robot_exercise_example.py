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
from robot_deployment.robots.magic import _find_process_name
import socket
import multiprocessing

@dataclass
class Config:
    use_real_robot: bool = True
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





  #Deleting udp instances does not trigger off the respective modes!

  #Although we deleted the udp port instance FOR HIGH LEVEL, the robot IS STILL SET TO SPORT MODE
  #NEED TO SEND A COMMAND LIKE L2 + B (damping mode) => L1 + L2 + start (Sport mode triggerd off),
  #this will make sure that the A1 is in normal mode (joint standby state).
  #After doing that, then we can create a instance of udp low mode and send commands.

  #If we need to to do reset, We first need to delete the udp low instance, the robot would STILL
  #BE IN NORMAL MODE. NEED TO SEND A COMMAND LIKE L2+B (damping mode) => (run A1_sport_1 exe in the bin of )
  #the sport mode controller

  #After tdoing L1 + L2 + start (Sport mode triggered off), I was able to switch to normal mode
  #and I was able to run the low level reset and commands well!!!

  #It seems after I run keep_program alive, and trigger sports mode and trigger it off,
  #I am still not able to run recovery or any high level commands.
  #THATTS BECAUSE THE PORT 8082 IS STILL BOUND. is it? 

  #So basically after I trigger the sports mode off, I am able to use low level
  #but not high level. 

  #But when I set the robot to sports mode using rc, and then lay it down using 
  #damping mode, I am able to run my High level. Oh that's because it is already set to
  #sport mode :0. 

  ### So I need to check if I am able to trigger sports mode through keep_program_alive.sh
  ### And then use high level. Wait, this doesn't work, right? NOPE This doesn't work.

  ##So here it is going to work:

  #INitially to normal mode, hence can use low level cmds.
  #low level motor cmds => ..=> moment it gets terminated, => trigger sport mode => send recovery..and navigation using A1 sport policy.
  # =>set to damping mode => trigger sport mode off => robot.reset() =>start over

  #after switching to normal mode, 

  """
  
  I'm only familiar with the GO1 robot, but I'd guess they use a similar system.
On the Go1, there is the Raspberry Pi and the Main Control Board, which work together.

To activate Sport Mode, a shell script is run, which then repetitively restarts the sport mode program, in case it fails.
The Sport Mode is handled by the Pi and the Joint control is handled by the MCB.

My guess is, that they've mapped this script to the remote control, to turn it on or off.

The SDK on the other hand connects either to the Pi or the MCB, depending on of you are using high- or low-level control.
The high-level works only if the sport mode is running on the Pi and the low-level control only works, if the sport mode isn't running on the Pi since the ports would otherwise be blocked.

So the UDP messages provided by the SDK are im my eyes not suitable for toggling between normal and sport mode.
You could try to run another program on the Pi (should be automatically started), which receives messages on a different IP port.
And that program could then either start or kill the sport mode process.
  
  """

  #SWITCH LEVEL DOES NOT WORK
  #ALthough if you switch to HIGH when constructing robot_interface for low mode, it DOES NOT WORK
  #But that does not give us the ability to manually switch sports mode and normal mode for high
  # and low level modes.
  
  #Need a way to automate sports mode initially

  #How about using different udp ports for low, and high?



"""
L2 + B; => L1 + L2 + start (for changing modes)
3 modes in 3.3.3 (basic, sport, slam)

udp        0      0 192.168.123.161:8008    192.168.123.10:8007     ESTABLISHED 26029/A1_sport_1
udp        0      0 192.168.123.161:8009    192.168.123.10:8007     ESTABLISHED 26029/A1_sport_1

change

udp        0      0 192.168.123.161:8010    192.168.123.10:8007     ESTABLISHED 19402/A1_sport_2
udp        0      0 192.168.123.161:8082    192.168.123.12:8081     ESTABLISHED 19402/A1_sport_2

change
udp        0      0 192.168.123.161:8008    192.168.123.10:8007     ESTABLISHED 26029/A1_sport_1
udp        0      0 192.168.123.161:8009    192.168.123.10:8007     ESTABLISHED 26029/A1_sport_1
udp     1536      0 192.168.123.161:8010    192.168.123.10:8007     ESTABLISHED 26029/A1_sport_1

udp     1536      0 192.168.123.161:8010    192.168.123.10:8007     ESTABLISHED 10051/A1_sport_1 (Need this!!)


SO IT SEEMS only by commanding through rc, when A1_sport_1 is running, It is able to get the above udp
running, thereby allowing high level commands.

So one way to do this is to keep A1_sport_1 running. And


start with low level reset, so that initial motor positions are already set for
future recovery (involving killing a1_sport_1 exes, and starting high level)


1. low level
    1. keep on sending start sport mode message to a udp port on the PI
        keep on listening until start sport mode message is received on PI

2. terminate
3. Delete previous robot_interface
4. Create High level robot_interface
   1. keep on sending start sport mode message to a udp port on the PI
      keep on listening until start sport mode message is received on PI
   2. keep on querying a port, until a connection is made between PI and MCB
   3. wait for sport mode initialization is done.
   4. Send stop sport mode message to host udp port
   5. close the host udp port.
3. Start recovery and go into damping mode
4. Delete current robot_interface
4. low level
...




"""

def low(cfg, p):
    robot_ctor = a1_robot.A1Robot if cfg.deployment.use_real_robot else a1.A1
    low_robot = robot_ctor(pybullet_client=p, sim_conf=cfg.deployment)
    low_robot.reset()
    low_robot.delete_robot_interface()

def high(cfg, p):
    robot_ctor = a1_robot.A1Robot if cfg.deployment.use_real_robot else a1.A1
    high_robot = robot_ctor(pybullet_client=p, sim_conf=cfg.deployment, mode_type="high")
    high_robot.recover_robot()
    high_robot.delete_robot_interface()
   
   
def initialize(mode_type : str, cfg, p):
    print(mode_type, '\n\n')
    robot_ctor = a1_robot.A1Robot if cfg.deployment.use_real_robot else a1.A1
    robot = robot_ctor(pybullet_client=p, sim_conf=cfg.deployment, mode_type=mode_type)
    if mode_type == "low":
       robot.reset()
       for _ in range(200):
          action = get_action(robot, robot.time_since_reset)
          robot.step(action)
          if not cfg.deployment.use_real_robot:
            time.sleep(cfg.deployment.timestep)
          print(robot.base_orientation_rpy)
    elif mode_type == "high":
       time.sleep(2)
       robot.damping_mode()
       robot.recover_robot()
       time.sleep(2)
       robot.damping_mode()
       
  

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    connection_mode = pybullet.DIRECT if cfg.deployment.use_real_robot else pybullet.GUI
    p = bullet_client.BulletClient(connection_mode=connection_mode)
    p.setAdditionalSearchPath(osp.join(LEGGED_GYM_ROOT_DIR, 'resources'))
    p.loadURDF("plane.urdf")
    p.setGravity(0.0, 0.0, -9.81)
    


    print("starting main.. \n\n")


    process = multiprocessing.Process(target=initialize, args=("low", cfg, p))
    process.start()
    process.join()

    
    process = multiprocessing.Process(target=initialize, args=("high", cfg, p))
    process.start()
    process.join()


    process = multiprocessing.Process(target=initialize, args=("low", cfg, p))
    process.start()
    process.join()

    process = multiprocessing.Process(target=initialize, args=("high", cfg, p))
    process.start()
    process.join()

    process = multiprocessing.Process(target=initialize, args=("high", cfg, p))
    process.start()
    process.join()


if __name__ == "__main__":
  main()
