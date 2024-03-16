### ground_control imports
import isaacgym # need to import this before torch
import torch
import logging
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Tuple
import os
from collections import deque
from flax.core import frozen_dict
import time
import statistics
from PIL import Image
import imageio
import cv2

# hydra / config related imports
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
# from pydantic import TypeAdapter  ## Commented out by Mateo because this conflicts with tf_probability

from configs.definitions import TaskConfig, TrainConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig, WITPLocomotionTaskConfig, ResidualLocomotionTaskConfig, ResidualWITPLocomotionTaskConfig
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from configs.hydra import ExperimentHydraConfig

# from legged_gym.envs.a1 import A1
from legged_gym.envs.a1_residual import A1Residual
from legged_gym.utils.helpers import export_policy_as_jit
# from rsl_rl.runners import OnPolicyRunner  ## Commented out by Mateo because we are trying to train with witp code.
from legged_gym.utils.helpers import (set_seed, get_load_path, get_latest_experiment_path, save_resolved_config_as_pkl, save_config_as_yaml,
                                      from_repo_root)
#####

### witp imports
import os
import pickle
import shutil
import jax
import numpy as np
import tqdm

import gymnasium as gym
import wandb
# from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from witp.rl.agents import SACLearner
from witp.rl.data import ReplayBuffer
from witp.rl.evaluation import evaluate
from witp.rl.wrappers import wrap_gym
import ml_collections
from typing import Optional
## DELETE AFTER DEBUGGING
# from jax import config
# config.update("jax_disable_jit", True)
#######
#####
from rsl_rl.runners import OnPolicyRunner

@dataclass
class Flags:
    # env_name: str = 'A1Run-v0'
    # save_dir: str = './tmp/'
    seed: int = 42
    eval_episodes: int = 1
    log_interval: int = 1000
    eval_interval: int = 1000
    video_interval: int = 20
    batch_size: int = 256
    max_steps: int = int(1e5)
    start_training: int = 0 ##int(1e3)
    tqdm: bool = True
    wandb: bool = True
    save_video: bool = False
    action_filter_high_cut: Optional[float] = None
    action_history: int = 1
    control_frequency: int = 20
    utd_ratio: int = 20
    real_robot: bool = False

    def as_dict(self):
        return asdict(self)


## Mateo note: I am going to modify the parameters here to be different than the ones in train.py. Most importantly, num_envs will be
## set to 1 for ease of debugging for now, and might potentially change the task and train configs. Going to change where things get stored.
@dataclass
class TrainResidualScriptConfig:
    """
    A config used with the `train.py` script. Also has top-level
    aliases for commonly used hyperparameters. Feel free to add
    more aliases for your experiments.
    """

    # Uncomment below to launch through joblib or add a different launcher / sweeper
    #defaults: Tuple[Any] = (
    #    "_self_",
    #    {"override hydra/launcher": "joblib"},
    #    
    #)

    seed: int = 1
    torch_deterministic: bool = False
    num_envs: int = 1 # 4096 
    iterations: int = 5000 
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    headless: bool = True
    checkpoint_root: str = from_repo_root("./witp_checkpoints")  ## ""
    logging_root: str = from_repo_root("./witp_experiment_logs")
    checkpoint: int = -1
    export_policy: bool = True
    episode_buffer_len: int = 100
    name: str = ""
    just_pretrained: bool = False
    task: TaskConfig = ResidualWITPLocomotionTaskConfig()
    train: TrainConfig = TrainConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainResidualScriptConfig)


def obs_to_nn_input(observation):
    '''
    Observations will have the shape [n, d] where n is the number of environments and d is the obs dimensionality.
    For now, we will assume n=1, so the only thing we need to do is flatten the array. In the future, we might need to 
    have parallel versions of the agent so that we can maximize the parallelizability of Isaac Gym.

    The observations are returned as torch tensors in the GPU. We want to return these as numpy arrays on the CPU.
    '''
    if isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()
    obs = observation.squeeze()
    # obs = jax.device_put(obs, jax.devices()[1])

    return obs

def action_to_env_input(action):
    '''
    The input action will be an array of shape (12,). However, we need to convert this to a tensor of shape [n, 12], 
    where n will be the number of environments, which we will hardcode to be 1 for now. 

    It doesn't seem like the action needs to be on GPU.
    '''
    action_copy = np.copy(action)
    action = torch.Tensor(action_copy).view(1, -1)
    return action


def create_gif_from_numpy_array(frames_array, filename, fps=30):
    # OpenCV expects the dimensions to be (height, width) instead of (width, height)
    frames_array = frames_array.transpose(0, 2, 3, 1)
    frames_list = [frame for frame in frames_array]
    imageio.mimwrite(filename, frames_list)


@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainResidualScriptConfig) -> None:
    log.info("1. Printing and serializing frozen TrainScriptConfig")
    OmegaConf.resolve(cfg)
    # Type-checking (and other validation if defined) via Pydantic
    # cfg = TypeAdapter(TrainScriptConfig).validate_python(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)
    #save_resolved_config_as_pkl(cfg)

    log.info("2. Initializing Env and Runner with Pretrained Policy")
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    task_cfg = cfg.task
    env: A1Residual = hydra.utils.instantiate(task_cfg)

    experiment_path = cfg.checkpoint_root
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)
    resume_path = get_load_path(experiment_path, checkpoint=cfg.checkpoint)
    runner.load(resume_path)
    log.info(f"2.5 Loading policy checkpoint from: {resume_path}.")
    policy = runner.get_inference_policy(device=env.device)
    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
        log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    # runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)
    log.info(f"4. Preparing environment and runner.")
    
    env.load_pretrained_policy(policy)
    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_critic_observations()
    residual_obs = env.get_residual_observations()

    # if cfg.train.runner.resume_root != "":
    #     experiment_dir = get_latest_experiment_path(cfg.train.runner.resume_root)
    #     resume_path = get_load_path(experiment_dir, checkpoint=cfg.train.runner.checkpoint)
    #     log.info(f"Loading model from: {resume_path}")
    #     runner.load(resume_path)


    ###################################################### 
    # Set up agent
    ###################################################### 
    ## WITP flags:
    FLAGS = Flags()

    ## DROQ configs:
    config = ml_collections.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.discount = 0.99
    config.num_qs = 2
    config.critic_dropout_rate = 0.01
    config.critic_layer_norm = True
    config.tau = 0.005
    config.init_temperature = 0.1
    config.target_entropy = None

    kwargs = dict(config)

    ## Initialize WandB
    if cfg.name == "":
        wandb.init(project='a1_residual')
    else:
        wandb.init(project='a1_residual', name=cfg.name)
    wandb_cfg_dict = {**FLAGS.as_dict(), **kwargs, **dict(cfg)}
    wandb.config.update(wandb_cfg_dict)

    ## Define our own observation space and action space since these are not directly available in 
    ## the ground_control A1 environment:
    ## By default, fine-tuning algorithm will only use the sensors in cfg.task.observation.sensors, NOT the cfg.task.observation.critic_privileged_sensors
    num_obs = env.num_residual_obs
    obs_limit = cfg.task.normalization.clip_observations
    observation_space = gym.spaces.Box(low=-obs_limit, high=obs_limit, shape=(num_obs,))

    num_actions = env.num_actions
    action_limit = cfg.task.normalization.clip_actions
    action_space = gym.spaces.Box(low=-action_limit, high=action_limit, shape=(num_actions,))
    # action_space = gym.spaces.Box(
    #     low=np.array([-0.803, -1.047, -2.697, -0.803, -1.047, -2.697, -0.803, -1.047, -2.697, -0.803, -1.047, -2.697]),
    #     high=np.array([0.803,  4.189, -0.916,  0.803,  4.189, -0.916,  0.803,  4.189, -0.916,  0.803,  4.189, -0.916]),
    #     shape=(num_actions,)
    # )
    sensors =  cfg.task.observation.residual_sensors
    observation_labels = {sensor_name:(0, 0) for sensor_name in sensors}
    observation_labels['motor_pos_unshifted'] = (0, 12)
    observation_labels['motor_vel'] = (12, 24)
    observation_labels['last_action'] = (24, 36)
    observation_labels['base_quat'] = (36, 40)
    observation_labels['base_ang_vel'] = (40, 43)
    observation_labels['base_lin_vel'] = (43, 46)
    replay_buffer = ReplayBuffer(observation_space, action_space,
                                    FLAGS.max_steps, next_observation_space=observation_space,
                                    observation_labels=observation_labels)
    replay_buffer.seed(FLAGS.seed)

    agent = SACLearner.create(FLAGS.seed, observation_space,
                              action_space, **kwargs)

    ###################################################### 
    # Set up replay buffer
    ###################################################### 

    start_i = 0
    ep_counter = 0

    # Set up reward and episode length buffers for logging purposes
    rewards_buffer = deque(maxlen=cfg.episode_buffer_len)
    lengths_buffer = deque(maxlen=cfg.episode_buffer_len)
    current_reward_sum = 0
    current_episode_length = 0
    ep_infos = []

    ep_scene_images = []
    ep_fpv_images = []
    ep_scene_path = '/home/mateo/projects/ground_control/training_videos/scene_residual'
    ep_fpv_path = '/home/mateo/projects/ground_control/training_videos/fpv_residual'
    if cfg.name != "":
        ep_scene_path += "_" + cfg.name
        ep_fpv_path += "_" + cfg.name
    if not os.path.exists(ep_scene_path):
        os.makedirs(ep_scene_path)
    if not os.path.exists(ep_fpv_path):
        os.makedirs(ep_fpv_path)
    ###################################################### 
    # Reset environment
    ###################################################### 
    observation_torch, _, residual_obs_torch = env.reset()
    done = False

    collection_time = 0
    learn_time = 0

    log.info("3. Now Training")
    # runner.learn()

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        ## Convert observation to be of usable form:
        start = time.time()
        observation = obs_to_nn_input(residual_obs_torch)

        if i < FLAGS.start_training:
            action = action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        if cfg.just_pretrained:
            action = np.zeros(12)  ## Sanity check, NEEDS TO BE COMMENTED OUT FOR PROPER PERFORMANCE
        ## Note: Action clipping now happens inside the environment, can be set in config.
        ## Convert action to be of usable form
        env_action = action_to_env_input(action).to(env.device)

        ## Get/set residual in environment (stored as a self.pretrained_action)
        pretrained_action = env.get_pretrained_action(observation_torch)

        ## Environment step
        next_observation_torch, _, reward, done, info, next_residual_obs_torch = env.step(env_action)

        ## Convert next_observation, reward, done, info into usable forms
        next_observation = obs_to_nn_input(next_residual_obs_torch)
        reward = reward.detach().cpu().item()
        done = done.detach().cpu().item()

        if 'episode' in info:
            ep_infos.append(info['episode'])
        current_reward_sum += reward
        current_episode_length += 1

        ## Get images for visualization purposes
        images = env.get_camera_images()
        ep_scene_images.append(np.transpose(images[0][0], (2, 0, 1)))
        ep_fpv_images.append(np.transpose(images[0][1], (2, 0, 1)))

        if not done or ('time_outs' in info and info["time_outs"].item()):
            mask = 1.0
        else:
            time_outs_in_info = 'time_outs' in info
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation,
                 observation_labels=observation_labels))
        observation = next_observation
        observation_torch = next_observation_torch

        stop = time.time()
        collection_time = stop - start

        # Track learning time
        start = stop
        ## This clause will largely be the same!
        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            if "observation_labels" in batch:
                batch = dict(batch)
                batch.pop("observation_labels")
                batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            stop = time.time()
            learn_time = stop - start
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        

        if done:
            observation_torch, _, residual_obs_torch = env.reset()
            done = False
            rewards_buffer.extend([current_reward_sum])
            lengths_buffer.extend([current_episode_length])
            for k, v in info['episode'].items():
                wandb.log({f'training/{k}': v.item()}, step=i)
            wandb.log({'training/cumulative_reward': current_reward_sum}, step=i)
            wandb.log({'training/episode_length': current_episode_length}, step=i)
            wandb.log({'training/mean_cumulative_reward': statistics.mean(rewards_buffer)}, step=i)
            wandb.log({'training/mean_episode_length': statistics.mean(lengths_buffer)}, step=i)

            total_time = collection_time + learn_time
            
            current_reward_sum = 0
            current_episode_length = 0

            if ep_counter % FLAGS.video_interval == 0:  
                ep_scene_images_stacked = np.stack(ep_scene_images, axis=0)
                ep_fpv_images_stacked = np.stack(ep_fpv_images, axis=0)

                wandb.log({"video/scene": wandb.Video(ep_scene_images_stacked, fps=30)}, step=i)
                wandb.log({"video/fpv": wandb.Video(ep_fpv_images_stacked, fps=30)}, step=i)

                ep_scene_filename = os.path.join(ep_scene_path, f'ep_{ep_counter}.gif')
                ep_fpv_filename = os.path.join(ep_fpv_path, f'ep_{ep_counter}.gif')
                create_gif_from_numpy_array(ep_scene_images_stacked, ep_scene_filename, fps=30)
                create_gif_from_numpy_array(ep_fpv_images_stacked, ep_fpv_filename, fps=30)

            ep_scene_images = []
            ep_fpv_images = []
            ep_counter += 1

    log.info("4. Exit Cleanly")
    env.exit()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()

'''
Example command line command:
python train_residual.py checkpoint_root='/home/mateo/projects/experiment_logs/train/2024-02-14_23-01-24' checkpoint=500  headless=True

Note: Currently requires observation spaces to be the same. How can we bypass this?
'''