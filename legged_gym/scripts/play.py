import isaacgym
import logging
import pickle
import cv2
import os.path as osp
import time
import isaac_utils
from isaacgym import gymapi
import math

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

from configs.hydra import *
from legged_gym.envs.a1 import A1
from legged_gym.scripts.train import TrainScriptConfig  # so that loading config pickle file works
from legged_gym.utils.helpers import export_policy_as_jit, get_load_path, get_latest_experiment_path
from legged_gym.utils.logger import Logger
from rsl_rl.runners import OnPolicyRunner

import numpy as np
import torch

web_viewer = isaac_utils.WebViewer()

@dataclass
class PlayScriptConfig:
    checkpoint_root: str = "experiment_logs"
    export_policy: bool = True
    get_commands_from_joystick: bool = True
    num_envs: int = 50
    episode_length_s: float = 200.
    checkpoint: int = -1
    device: str = "cpu"
    num_rows: int = 5
    num_cols: int = 5

    task: TaskConfig = MISSING
    train: TrainConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=PlayScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: PlayScriptConfig):
    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    latest_config_filepath = osp.join(experiment_path, "resolved_config.pkl")
    log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    with open(latest_config_filepath, "rb") as cfg_pkl:
        loaded_cfg = pickle.load(cfg_pkl)
    OmegaConf.resolve(loaded_cfg)

    log.info("2. Printing original config from loaded experiment.")
    print(OmegaConf.to_yaml(loaded_cfg))

    cfg.task = loaded_cfg.task
    cfg.train = loaded_cfg.train

    log.info(f"3. Overriding config parameters with normal play script conditions.")
    cfg.task.env.num_envs = cfg.num_envs
    cfg.task.observation.get_commands_from_joystick = cfg.get_commands_from_joystick
    cfg.task.terrain.num_rows = cfg.num_rows 
    cfg.task.terrain.num_cols = cfg.num_cols 
    cfg.task.sim.device = cfg.device
    cfg.train.device = cfg.device
    cfg.train.runner.checkpoint = cfg.checkpoint
    cfg.task.sim.use_gpu_pipeline = cfg.task.sim.physx.use_gpu = (cfg.task.sim.device != "cpu")
    cfg.task.terrain.curriculum = False
    cfg.task.sim.headless = False
    cfg.task.noise.add_noise = False
    cfg.task.domain_rand.randomize_base_mass = False
    cfg.task.domain_rand.randomize_friction = False
    cfg.task.domain_rand.randomize_gains = False

    log.info(f"4. Preparing environment and runner.")
    task_cfg = cfg.task
    #env: A1 = hydra.utils.instantiate(task_cfg)

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    sim = gym.create_sim(compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=sim_params)
    num_envs = 1
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, 0.0, spacing)

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    envs = []
    cameras = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))

        # add sphere
        pose = gymapi.Transform()
        pose.p, pose.r = gymapi.Vec3(0.0, 0.0, 1.0), gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        gym.create_actor(env, gym.create_sphere(sim, 0.2, None), pose, "sphere", i, 0)

        # add camera
        cam_props = gymapi.CameraProperties()
        cam_props.width, cam_props.height = 300, 300
        cam_handle = gym.create_camera_sensor(env, cam_props)
        gym.set_camera_location(cam_handle, env, gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, 0))

        envs.append(env)
        cameras.append(cam_handle)

    #gym = gymapi.acquire_gym()
    #cam_props = gymapi.CameraProperties()
    #gym.create_camera_sensor(env, cam_props)
    web_viewer.setup(gym, sim, envs, cameras)
    for i in range(100000):
        gym.simulate(sim)
        # render the scene
        web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
    exit()
    env.reset()
    obs = env.get_observations()
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    resume_path = get_load_path(experiment_path, checkpoint=cfg.train.runner.checkpoint)
    log.info(f"5. Loading policy checkpoint from: {resume_path}.")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
        log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    log.info(f"6. Running interactive play script.")
    
    current_time = time.time()
    num_steps = int(cfg.episode_length_s / env.dt)
    for i in range(num_steps):
        actions = policy(obs.detach())
        log.info(f"{actions}")
        
        obs, _, _, _, infos, *_ = env.step(actions.detach())
        
        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
