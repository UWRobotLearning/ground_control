import isaacgym
import logging
import pickle
import os.path as osp

from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
import hydra
from hydra.core.config_store import ConfigStore

from configs.definitions import DeploymentConfig, TaskConfig, TrainConfig
from legged_gym.envs.a1 import A1
from legged_gym.scripts.train import TrainScriptConfig  # so that loading config pickle file works
from legged_gym.utils.helpers import export_policy_as_jit, get_load_path, get_latest_experiment_path
from legged_gym.utils.observation_buffer import ObservationBuffer
from robot_deployment.envs.locomotion_gym_env import LocomotionGymEnv
from rsl_rl.runners import OnPolicyRunner

import numpy as np
import torch

import matplotlib.pyplot as plt

@dataclass
class Config:
    checkpoint_root: str = "experiment_logs"
    checkpoint: int = -1
    export_policy: bool = True
    use_real_robot: bool = False
    get_commands_from_joystick: bool = True
    episode_length_s: float = 100.
    device: str = "cpu"

    deployment: DeploymentConfig = DeploymentConfig(
        use_real_robot="${use_real_robot}",
        get_commands_from_joystick="${get_commands_from_joystick}",
        render=DeploymentConfig.RenderConfig(
            show_gui="${not: ${use_real_robot}}"
        )
    )

    task: TaskConfig = MISSING 
    train: TrainConfig = MISSING

OmegaConf.register_new_resolver("not", lambda b: not b)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    latest_config_filepath = osp.join(experiment_path, "resolved_config.pkl")
    log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    with open(latest_config_filepath, "rb") as cfg_pkl:
        loaded_cfg = pickle.load(cfg_pkl)
    OmegaConf.resolve(loaded_cfg)

    cfg.task = loaded_cfg.task
    cfg.train = loaded_cfg.train

    # override some params for purposes of evaluation
    cfg.task.sim.device = cfg.device
    cfg.task.sim.use_gpu_pipeline = cfg.task.sim.physx.use_gpu = (cfg.task.sim.device != "cpu")
    cfg.task.sim.headless = True
    cfg.train.device = cfg.device
    cfg.train.runner.checkpoint = cfg.checkpoint
    cfg.deployment.timestep = cfg.task.sim.dt * cfg.task.control.decimation / cfg.deployment.action_repeat
    cfg.deployment.init_position = cfg.task.init_state.pos
    cfg.deployment.init_joint_angles = cfg.task.init_state.default_joint_angles
    cfg.deployment.stiffness = cfg.task.control.stiffness
    cfg.deployment.damping = cfg.task.control.damping
    cfg.deployment.action_scale = cfg.task.control.action_scale

    # prepare environment (used to load policy)
    task_cfg = cfg.task
    isaac_env: A1 = hydra.utils.instantiate(task_cfg)

    # load policy
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=isaac_env, _recursive_=False)
    train_cfg = cfg.train
    resume_path = get_load_path(experiment_root, checkpoint=train_cfg.runner.checkpoint)
    log.info(f"Loading model from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=isaac_env.device)

    # export policy as a jit module (used to run it from C++)
    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, experiment_root)
        log.info(f"Exported policy as jit script to: {experiment_root}")


    # create robot environment (either in PyBullet or real world)
    env = LocomotionGymEnv(
        cfg.deployment,
        cfg.task.observation.sensors,
        cfg.task.normalization.obs_scales,
        cfg.task.commands.ranges
    )

    obs, info = env.reset()
    for _ in range(1):
        obs, *_, info = env.step(env.default_motor_angles)

    obs_buf = ObservationBuffer(1, isaac_env.num_obs, task_cfg.observation.history_steps, runner.device)

    all_actions = []
    all_infos = None

    for t in range(int(cfg.episode_length_s / env.robot.control_timestep)):
        # Form observation for policy.
        obs = torch.tensor(obs, device=runner.device).float()
        if t == 0:
            obs_buf.reset([0], obs)
            all_infos = {k: [v.copy()] for k, v in info.items()}
        else:
            obs_buf.insert(obs)
            for k, v in info.items():
                all_infos[k].append(v.copy())

        policy_obs = obs_buf.get_obs_vec(range(task_cfg.observation.history_steps))

        # Evaluate policy and act.
        actions = policy(policy_obs.detach()).detach().cpu().numpy().squeeze()
        actions = task_cfg.control.action_scale*actions + env.default_motor_angles
        all_actions.append(actions)
        obs, _, terminated, _, info = env.step(actions)

        if terminated:
            log.warning("Unsafe, terminating!")
            break

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
