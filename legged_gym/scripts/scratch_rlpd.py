### ground_control imports
import isaacgym # need to import this before torch
import torch
import logging
import pickle
from dataclasses import dataclass, asdict, field
from typing import Any, Tuple, List, Union
import os
from collections import deque
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

from configs.definitions import TaskConfig, TrainConfig, CollectionConfig, EvalConfig, EnvConfig, ObservationConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig, WITPLocomotionTaskConfig, WITPUnclippedLocomotionTaskConfig, PretrainLocomotionTaskConfig, PreadaptLocomotionTaskConfig, AdaptLocomotionTaskConfig, AugmentedAdaptLocomotionTaskConfig, DownhillAugmentedAdaptLocomotionTaskConfig
from configs.overrides.train import RLPDSACTrainConfig, RLPDREDQTrainConfig, RLPDDroQTrainConfig, IQLAlgorithmConfig, IQLTrainConfig
from configs.hydra import ExperimentHydraConfig

from legged_gym.envs.a1_continual import A1Continual
from rsl_rl.runners import OnPolicyRunner  ## TODO: Change this interface
from legged_gym.utils.helpers import (set_seed, get_load_path, get_latest_experiment_path, save_resolved_config_as_pkl, save_config_as_yaml,
                                      save_config_as_yaml_continual, from_repo_root)
#####

### witp imports
import os
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import pickle
import shutil
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

import gymnasium as gym
import wandb
# from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
## TODO: should change these to come from RLPD or from our own custom path
# from witp.rl.agents import SACLearner
# from witp.rl.data import ReplayBuffer
# from witp.rl.evaluation import evaluate
# from witp.rl.wrappers import wrap_gym
## end TODO
import ml_collections
from typing import Optional
## DELETE AFTER DEBUGGING
# from jax import config
# config.update("jax_disable_jit", True)
#######
#####

## RLPD imports:
from rlpd.agents import SACLearner, IQLLearner
from rlpd.data import ReplayBuffer
from flax.core import frozen_dict
from flax.training import orbax_utils
import orbax.checkpoint as ocp
import optax

@dataclass
class ContinualScriptConfig:
    """
    A config used with the `train_continual.py` script. Also has top-level
    aliases for commonly used hyperparameters. Feel free to add
    more aliases for your experiments.
    """
    name: str = ""
    seed: int = 1
    torch_deterministic: bool = False
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    headless: bool = False
    checkpoint_root: str = ""
    logging_root: str = from_repo_root("../experiment_logs")

    ###################################################### 
    # Pretraining
    ###################################################### 

    pretrain_task: TaskConfig = PretrainLocomotionTaskConfig()
    pretrain_task_eval: TaskConfig = PretrainLocomotionTaskConfig(
        env = EnvConfig(
            num_envs=1,
            episode_length_s=5,
        ) 
    )
    pretrain_train: TrainConfig = TrainConfig()
    save_pretrained: bool = True
    save_pretrained_filename: Optional[str] = "pretrained.pt" ## Copied over from "/home/mateo/projects/experiment_logs/train/2023-12-06_08-46-22/model_5000.pt"
    load_pretrained: Optional[str] = "/home/mateo/projects/experiment_logs/scratch_rlpd/2024-05-01_17-30-13/pretrained.pt" #"/home/mateo/projects/experiment_logs/scratch_rlpd/2024-04-30_13-34-57/pretrained.pt" ##"/home/mateo/projects/experiment_logs/train/2023-12-06_08-46-22/model_5000.pt" ## "pretrained.pt" ## Path to pre-trained_checkpoint

    ###################################################### 
    # Adapt
    ###################################################### 

    preadapt_collection: CollectionConfig = CollectionConfig(
        dataset_size=int(1e6)
    )

    adapt_task_pretrain_obs: TaskConfig = DownhillAugmentedAdaptLocomotionTaskConfig(#AdaptLocomotionTaskConfig(
        env=EnvConfig(
            num_envs=1,
            episode_length_s=5
        ),
        observation=pretrain_task.observation
    )

    adapt_task_adapt_obs: TaskConfig = DownhillAugmentedAdaptLocomotionTaskConfig(#AdaptLocomotionTaskConfig(
        env=EnvConfig(
            num_envs=1,
            episode_length_s=5
        )
    )


    offline_finetuning: IQLTrainConfig = IQLTrainConfig()

    # adapt_train: RLPDSACTrainConfig = RLPDSACTrainConfig()
    # adapt_train: RLPDREDQTrainConfig = RLPDREDQTrainConfig()
    adapt_train: RLPDDroQTrainConfig = RLPDDroQTrainConfig()

    save_preadapt_buffer: bool = True
    save_preadapt_filename: Optional[str] = "preadapt_buffer.pkl"
    load_preadapt_buffer: Optional[str] = "/home/mateo/projects/experiment_logs/scratch_rlpd/2024-05-14_14-46-38/preadapt_buffer.pkl" #"/home/mateo/projects/experiment_logs/scratch_rlpd/2024-04-25_15-56-20/preadapt_buffer.pkl" #"/home/mateo/projects/experiment_logs/train_continual/2024-04-21_20-09-56/preadapt_buffer.pkl"  ## Path to pre-collected_checkpoint
    save_adapt_buffer: bool = True
    save_adapt_buffer_dir: Optional[str] = "buffers"
    save_adapt_agent: bool = True
    save_adapt_agent_dir: Optional[str] = "agents"
    adapt_collection: CollectionConfig = CollectionConfig(
        dataset_size=int(1e4)
    )

    eval: EvalConfig = EvalConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig() ## TODO: is this necessary?


cs = ConfigStore.instance()
cs.store(name="config", node=ContinualScriptConfig)



################################################################################
## TODO: All functions in this block should be moved to utils

def insert_batch_into_replay_buffer(replay_buffer, observations, actions, rewards, dones, next_observations, infos, observation_labels):
    ## Obtain mask
    if 'time_outs' not in infos:  ## No episode was terminated, so should just take into account dones (should all be False)
         masks = torch.logical_not(dones).float()
    else:  ## There was an episode terminated. Masks should be 1 if episode is *not* done or episode was terminated due to timeout, should be 0 if episode was terminated due to MDP end condition.
         masks = torch.logical_or(torch.logical_not(dones), infos["time_outs"]).float()

    ## Convert data to numpy
    observations = observations.cpu().detach().numpy()
    actions = actions.cpu().detach().numpy()
    rewards = rewards.cpu().detach().numpy()
    dones = dones.cpu().detach().numpy()
    next_observations = next_observations.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()

    for i in range(observations.shape[0]):
        replay_buffer.insert(
            dict(observations=observations[i],
                 actions=actions[i],
                 rewards=rewards[i],
                 masks=masks[i],
                 dones=dones[i],
                 next_observations=next_observations[i],
                 observation_labels=observation_labels))
        
def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if k == "observation_labels":
            combined[k] = v
            continue
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0:v.shape[0]] = v
            tmp[v.shape[0]:] = other_dict[k]  ## Might need to shuffle, but if I do, need to make sure I use a random seed.
            # tmp[0::2] = v
            # tmp[1::2] = other_dict[k]
            # tmp = np.concatenate([v, other_dict[k]], axis=0)
            combined[k] = tmp

    return combined

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


def create_gif_from_numpy_array(frames_array, filename, fps=10):
    # OpenCV expects the dimensions to be (height, width) instead of (width, height)
    frames_array = frames_array.transpose(0, 2, 3, 1)
    frames_list = [frame for frame in frames_array]
    imageio.mimwrite(filename, frames_list)
        

def get_distillation_data(datasets: List[ReplayBuffer], size, observation_labels):
    '''
    datasets will be a list of ReplayBuffers. We are going to sample size/len(datasets) samples per dataset, and 
    then combine all the samples into a single dictionary
    '''
    assert len(datasets) > 0, "Length of datasets should be > 0"
    samples_per_dataset = int(size / len(datasets))
    combined = datasets[0].sample_select(samples_per_dataset, include_labels=list(observation_labels.keys()))
    for i in range(1, len(datasets)):
        combined = combine(combined, datasets[i].sample(samples_per_dataset))

    return combined

def normalize_action(action: Union[np.array, torch.Tensor], action_space: gym.spaces.Box):
    '''
    Converts an action from [low, high] to [-1, 1]
    '''
    low, high = action_space.low, action_space.high
    if isinstance(action, torch.Tensor):
        low = torch.from_numpy(low).to(action.device)
        high = torch.from_numpy(high).to(action.device)
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unnormalize_action(action: Union[np.array, torch.Tensor], action_space: gym.spaces.Box):
    '''
    Converts an action from [-1, 1] to [low, high]
    '''
    low, high = action_space.low, action_space.high
    if isinstance(action, torch.Tensor):
        low = torch.from_numpy(low).to(action.device)
        high = torch.from_numpy(high).to(action.device)
    return low + (0.5 * (action + 1.0) * (high - low))

def clip_action(action: Union[np.array, torch.Tensor], action_space: gym.spaces.Box):
    '''
    Clips actions to [low, high] given by action_space
    '''
    low, high = action_space.low, action_space.high
    if isinstance(action, torch.Tensor):
        low = torch.from_numpy(low).to(action.device)
        high = torch.from_numpy(high).to(action.device)
        return torch.clip(action, low, high)
    return np.clip(action, low, high)

################################################################################


def pretrain(train_cfg, env, runner):
    log.info("Pre-training")
    runner.learn()  ## TODO: Change so that it doesn't use Runner anymore
    log.info("Done pre-training")
    return runner.get_inference_policy(device=env.device), runner.alg.actor_critic

def collect(task_cfg, env, policy, dataset_size, obs_processor=lambda x: x, action_processor=lambda x: x, nn_normalized_actions=True):  ## TODO: task_cfg could be replaced by collect config
    '''
    if nn_normalized_actions=True, this means that the actions returned by policy(obs) will be in the range [-1, 1]
    '''
    log.info("Collecting")
    
    ## Initialize Replay Buffer
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=dataset_size, next_observation_space=env.observation_space, observation_labels=env.observation_labels)
    ## Reset environment
    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_critic_observations()
    extra_obs = env.get_extra_observations()

    obs_processed = obs_processor(obs.detach())

    current_time = time.time()
    while len(replay_buffer) < dataset_size:
        raw_actions = policy(obs_processed)
        if nn_normalized_actions:  ## This means action came from an SAC++ policy
            unnormalized_actions = unnormalize_action(raw_actions, env.action_space)
        else:
            unnormalized_actions = clip_action(raw_actions, env.action_space)

        if torch.is_tensor(unnormalized_actions):
            unnormalized_actions = unnormalized_actions.detach()
        actions_processed = action_processor(unnormalized_actions)  ## This is here to interface between pytorch and numpy, to make sure actions are always torch tensors when they go into env
        new_obs, new_critic_obs, rewards, dones, infos, *_ = env.step(actions_processed)
        new_extra_obs = env.get_extra_observations()
        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

        print(f"replay_buffer size: {len(replay_buffer)}")
        buffer_observations = torch.concatenate([critic_obs, extra_obs], dim=-1)  ## Note: critic_obs contains obs, but does not get processed, so it's always a PyTorch tensor
        buffer_new_observations = torch.concatenate([new_critic_obs, new_extra_obs], dim=-1)
        normalized_actions = normalize_action(actions_processed, env.action_space)
        insert_batch_into_replay_buffer(replay_buffer, buffer_observations, normalized_actions, rewards, dones, buffer_new_observations, infos, env.observation_labels)
        obs = new_obs
        critic_obs = new_critic_obs
        extra_obs = new_extra_obs

        obs_processed = obs_processor(obs.detach())

    log.info("Done collecting")
    return replay_buffer

def evaluate(eval_cfg, env, policy, obs_processor=lambda x: x, action_processor=lambda x: x, nn_normalized_actions=True):
    log.info("Evaluating")

    ## Initialize WandB
    if eval_cfg.name is None:
        wandb.init(project=eval_cfg.project_name)
    else:
        wandb.init(project=eval_cfg.project_name, name=eval_cfg.name)
    wandb_cfg_dict = dict(eval_cfg)
    wandb.config.update(wandb_cfg_dict)

    rewards_buffer = deque(maxlen=eval_cfg.episode_buffer_len)
    lengths_buffer = deque(maxlen=eval_cfg.episode_buffer_len)
    current_reward_sum = 0
    current_episode_length = 0

    ep_scene_images = []
    ep_fpv_images = []

    ## Reset environment
    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_critic_observations()
    extra_obs = env.get_extra_observations()

    obs_processed = obs_processor(obs.detach())

    current_time = time.time()
    ep_counter = 0
    ep_start_time = time.time()
    while ep_counter < eval_cfg.eval_episodes:
        raw_actions = policy(obs_processed)
        if nn_normalized_actions:
            unnormalized_actions = unnormalize_action(raw_actions, env.action_space)
        else:
            unnormalized_actions = clip_action(raw_actions, env.action_space)
        if torch.is_tensor(unnormalized_actions):
            unnormalized_actions = unnormalized_actions.detach()
        actions_processed = action_processor(unnormalized_actions)
        new_obs, new_critic_obs, reward, done, info, *_ = env.step(actions_processed)
        new_extra_obs = env.get_extra_observations()
        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

        reward = reward.detach().cpu().item()
        done = done.detach().cpu().item()

        # if 'episode' in info:
        #     ep_infos.append(info['episode'])
        current_reward_sum += reward
        current_episode_length += 1

        ## Get images for visualization purposes
        images = env.get_camera_images()
        ep_scene_images.append(np.transpose(images[0][0], (2, 0, 1)))
        ep_fpv_images.append(np.transpose(images[0][1], (2, 0, 1)))

        obs = new_obs
        critic_obs = new_critic_obs
        extra_obs = new_extra_obs

        obs_processed = obs_processor(obs.detach())

        if done:
            print(f"End of episode! {time.time()-ep_start_time} s.")
            env.reset()
            obs = env.get_observations()
            critic_obs = env.get_critic_observations()
            extra_obs = env.get_extra_observations()

            obs_processed = obs_processor(obs.detach())
            done = False
            rewards_buffer.extend([current_reward_sum])
            lengths_buffer.extend([current_episode_length])
            for k, v in info['episode'].items():
                wandb.log({f'eval/{k}': v.item()}, step=ep_counter)
            wandb.log({'eval/cumulative_reward': current_reward_sum}, step=ep_counter)
            wandb.log({'eval/episode_length': current_episode_length}, step=ep_counter)
            wandb.log({'eval/mean_cumulative_reward': statistics.mean(rewards_buffer)}, step=ep_counter)
            wandb.log({'eval/mean_episode_length': statistics.mean(lengths_buffer)}, step=ep_counter)

            if ep_counter % eval_cfg.video_interval == 0:  
                ep_scene_images_stacked = np.stack(ep_scene_images, axis=0)
                ep_fpv_images_stacked = np.stack(ep_fpv_images, axis=0)

                wandb.log({"video/scene": wandb.Video(ep_scene_images_stacked, fps=10)}, step=ep_counter)
                wandb.log({"video/fpv": wandb.Video(ep_fpv_images_stacked, fps=10)}, step=ep_counter)
            
            current_reward_sum = 0
            current_episode_length = 0
        
            ep_scene_images = []
            ep_fpv_images = []
            ep_start_time = time.time()
            ep_counter += 1

    stats = {
        "mean_cumulative_reward": statistics.mean(rewards_buffer),
        "std_cumulative_reward": statistics.stdev(rewards_buffer),
        "mean_episode_length": statistics.mean(lengths_buffer),
        "std_episode_length": statistics.stdev(lengths_buffer)
    }

    wandb.finish()
    log.info("Done evaluating")
    return stats

def offline_pretrain(adapt_cfg, env, agent, data):
    # import ipdb;ipdb.set_trace()
    log.info("Offline Pre-training with IQL")
    ## Initialize WandB
    if adapt_cfg.name == "":
        wandb.init(project=adapt_cfg.project_name)
    else:
        wandb.init(project=adapt_cfg.project_name, name=adapt_cfg.name)
    wandb_cfg_dict = dict(adapt_cfg)
    wandb.config.update(wandb_cfg_dict)
    for i in tqdm.tqdm(
        range(0, adapt_cfg.pretrain_steps), smoothing=0.1, disable=not adapt_cfg.tqdm
    ):
        offline_batch = data.sample_select(
                adapt_cfg.batch_size, 
                include_labels=list(env.actor_labels.keys())
            )
        
        if "observation_labels" in offline_batch:
            offline_batch = dict(offline_batch)
            offline_batch.pop("observation_labels")
            offline_batch = frozen_dict.freeze(offline_batch)

        agent, update_info = agent.update(offline_batch)

        if i % adapt_cfg.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

    wandb.finish()
    log.info("Done Offline Pre-training with IQL")
    return agent.eval_actions, agent

def adapt(adapt_cfg, env, agent, data, use_utd=True):
    ## Initialize WandB
    if adapt_cfg.name == "":
        wandb.init(project=adapt_cfg.project_name)
    else:
        wandb.init(project=adapt_cfg.project_name, name=adapt_cfg.name)
    wandb_cfg_dict = dict(adapt_cfg)
    wandb.config.update(wandb_cfg_dict)
    wandb_counter = 0

    ###################################################### 
    # Set up replay buffer and offline dataset
    ###################################################### 
    # Set up replay buffer
    start_i = 0
    ep_counter = 0
    replay_buffer = ReplayBuffer(env.actor_observation_space, env.action_space,
                                    adapt_cfg.max_steps, next_observation_space=env.actor_observation_space,
                                    observation_labels=env.actor_labels)
    replay_buffer.seed(adapt_cfg.seed)

    # Set up offline dataset
    if data is not None:
        ds = data
    else:
        ds_dir = adapt_cfg.offline_data_dir
        with open(ds_dir, 'rb') as f:
            ds = pickle.load(f)
    # Set up reward and episode length buffers for logging purposes
    rewards_buffer = deque(maxlen=adapt_cfg.episode_buffer_len)
    lengths_buffer = deque(maxlen=adapt_cfg.episode_buffer_len)
    current_reward_sum = 0
    current_episode_length = 0
    ep_infos = []

    ep_scene_images = []
    ep_fpv_images = []
    ep_scene_path = '/home/mateo/projects/ground_control/training_videos/scene'
    ep_fpv_path = '/home/mateo/projects/ground_control/training_videos/fpv'
    if adapt_cfg.name is not None:
        ep_scene_path += "_" + adapt_cfg.name
        ep_fpv_path += "_" + adapt_cfg.name
    if not os.path.exists(ep_scene_path):
        os.makedirs(ep_scene_path)
    if not os.path.exists(ep_fpv_path):
        os.makedirs(ep_fpv_path)
    ###################################################### 
    # Reset environment
    ###################################################### 

    if adapt_cfg.pretrain_steps > 0:
        log.info("Starting offline pre-training")  
    for i in tqdm.tqdm(
        range(0, adapt_cfg.pretrain_steps), smoothing=0.1, disable=not adapt_cfg.tqdm
    ):
        offline_batch = ds.sample_select(
                adapt_cfg.batch_size * adapt_cfg.utd_ratio, 
                include_labels=list(env.actor_labels.keys())
            )
        
        batch = {}
        if "observation_labels" in offline_batch:
            offline_batch = dict(offline_batch)
            offline_batch.pop("observation_labels")
            offline_batch = frozen_dict.freeze(offline_batch)

        if use_utd:
            agent, update_info = agent.update(offline_batch, adapt_cfg.utd_ratio)
        else:
            agent, update_info = agent.update(offline_batch)

        if i % adapt_cfg.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=wandb_counter)
        wandb_counter += 1

    if adapt_cfg.pretrain_steps > 0:
        log.info("Done offline pre-training")


    observation, _ = env.reset()
    done = False

    collection_time = 0
    learn_time = 0

    log.info("3. Now Training")

    ## Set up linear schedule for offline ratio
    offline_ratio_schedule_fn = optax.linear_schedule(
        init_value=1., 
        end_value=0,
        transition_steps=adapt_cfg.max_steps,
        transition_begin=adapt_cfg.start_training
        )

    for i in tqdm.tqdm(range(start_i, adapt_cfg.max_steps),
                       smoothing=0.1,
                       disable=not adapt_cfg.tqdm):
        ## Convert observation to be of usable form:
        start = time.time()
        observation = obs_to_nn_input(observation)

        ## TODO: Mateo: Remove
        normalized_action, agent = agent.sample_actions(observation)
        unnormalized_action = unnormalize_action(normalized_action, env.action_space)
        ## end TODO: uncomment below
        # if i < adapt_cfg.start_training:
        #     unnormalized_action = env.action_space.sample()
        #     normalized_action = normalize_action(unnormalized_action, env.action_space)
        # else:
        #     normalized_action, agent = agent.sample_actions(observation)
        #     unnormalized_action = unnormalize_action(normalized_action, env.action_space)

        ## Note: Action clipping now happens inside the environment, can be set in config.
        ## Convert action to be of usable form
        env_action = action_to_env_input(unnormalized_action)

        ## Environment step
        next_observation, _, reward, done, info = env.step(env_action)

        ## Convert next_observation, reward, done, info into usable forms
        next_observation = obs_to_nn_input(next_observation)
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
            mask = 1.0  ## truncated so should bootstrap
        else:
            time_outs_in_info = 'time_outs' in info
            mask = 0.0  ## terminated so don't bootstrap
        replay_buffer.insert(
            dict(observations=observation,
                 actions=normalized_action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation,
                 observation_labels=env.actor_labels))
        observation = next_observation

        stop = time.time()
        collection_time = stop - start

        # Track learning time
        start = stop
        offline_ratio_value = offline_ratio_schedule_fn(i)
        ## This clause will largely be the same!
        if i >= adapt_cfg.start_training:
            # import ipdb;ipdb.set_trace()
            assert isinstance(adapt_cfg.utd_ratio, int)
            online_batch = replay_buffer.sample(
                # adapt_cfg.utd_ratio * int(adapt_cfg.batch_size * (1 - adapt_cfg.offline_ratio))
                adapt_cfg.utd_ratio * int(adapt_cfg.batch_size * (1 - offline_ratio_value))
            )
            offline_batch = ds.sample_select(
                # adapt_cfg.utd_ratio * int(adapt_cfg.batch_size * adapt_cfg.offline_ratio),
                adapt_cfg.utd_ratio * int(adapt_cfg.batch_size * offline_ratio_value), 
                include_labels=list(env.actor_labels.keys())
            )

            batch = combine(offline_batch, online_batch)
            if "observation_labels" in batch:
                batch = dict(batch)
                batch.pop("observation_labels")
                batch = frozen_dict.freeze(batch)

            # assert (np.all(batch['actions'] <= 1)) and (np.all(batch['actions'] <= 1)), "Actions going into network are NOT normalized"  ## Note: this will cause issues
            if use_utd:
                agent, update_info = agent.update(offline_batch, adapt_cfg.utd_ratio)
            else:
                agent, update_info = agent.update(offline_batch)

            stop = time.time()
            learn_time = stop - start
            if i % adapt_cfg.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=wandb_counter)
                    wandb.log({f'training/offline_ratio': offline_ratio_value}, step=wandb_counter)

        if done:
            observation, _ = env.reset()
            done = False
            rewards_buffer.extend([current_reward_sum])
            lengths_buffer.extend([current_episode_length])
            for k, v in info['episode'].items():
                wandb.log({f'training/{k}': v.item()}, step=wandb_counter)
            wandb.log({'training/cumulative_reward': current_reward_sum}, step=wandb_counter)
            wandb.log({'training/episode_length': current_episode_length}, step=wandb_counter)
            wandb.log({'training/mean_cumulative_reward': statistics.mean(rewards_buffer)}, step=wandb_counter)
            wandb.log({'training/mean_episode_length': statistics.mean(lengths_buffer)}, step=wandb_counter)

            total_time = collection_time + learn_time
            
            current_reward_sum = 0
            current_episode_length = 0

            if ep_counter % adapt_cfg.video_interval == 0:  
                ep_scene_images_stacked = np.stack(ep_scene_images, axis=0)
                ep_fpv_images_stacked = np.stack(ep_fpv_images, axis=0)

                wandb.log({"video/scene": wandb.Video(ep_scene_images_stacked, fps=10)}, step=wandb_counter)
                wandb.log({"video/fpv": wandb.Video(ep_fpv_images_stacked, fps=10)}, step=wandb_counter)

                ep_scene_filename = os.path.join(ep_scene_path, f'ep_{ep_counter}.gif')
                ep_fpv_filename = os.path.join(ep_fpv_path, f'ep_{ep_counter}.gif')
                create_gif_from_numpy_array(ep_scene_images_stacked, ep_scene_filename, fps=10)
                create_gif_from_numpy_array(ep_fpv_images_stacked, ep_fpv_filename, fps=10)

            ep_scene_images = []
            ep_fpv_images = []
            ep_counter += 1
        wandb_counter += 1

    wandb.finish()

    return agent.eval_actions, agent


def distill(distillation_data, agent):
    ## All the following parameters should go in a config
    lr = 3e-4
    num_epochs = 1
    batch_size = 256
    seed = 42
    ## end of parameters that should go in a config

    key = jax.random.key(seed)
    dataset_size = distillation_data["observations"].shape[0]

    num_batches = int(np.ceil(dataset_size / batch_size))

    @jax.jit
    def mse(params, x_batched, y_batched):
        # Define squared loss for a single pair
        def squared_error(x, y):
            dist = agent.actor.apply_fn({'params': params}, x)
            pred = dist.sample(seed=key)  ## TODO: Verify if this works
            # pred = dist.mode()  ## Alternative, not sure which one is correct
            return jnp.inner(y-pred, y-pred) / 2.0
        # Vectorize single sample version
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
    
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(agent.actor.params)
    loss_grad_fn = jax.value_and_grad(mse)

    import ipdb;ipdb.set_trace()
    ## Make distillation data into batches
    for epoch in range(num_epochs):
        for batch_num in range(num_batches):
            start_idx = batch_num*batch_size
            end_idx = min(batch_num*batch_size+batch_size, dataset_size)
            batch_obs = distillation_data["observations"][start_idx:end_idx, :]
            batch_actions = distillation_data["actions"][start_idx:end_idx, :]
            loss_val, grads = loss_grad_fn(agent.actor.params, batch_obs, batch_actions)
            updates, opt_state = tx.update(grads, opt_state)
            agent.actor.params = optax.apply_updates(agent.actor.params, updates)  ## TODO: Failing, need to unfreeze the TrainState but not sure how

            if batch_num % 1 == 0:
                print(f"Loss at epoch {epoch}, batch {batch_num}/{num_batches} = {loss_val}")

    return agent



@hydra.main(version_base=None, config_name="config")  
def main(cfg: ContinualScriptConfig) -> None:  
    log.info("1. Printing and serializing frozen TrainScriptConfig")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml_continual(cfg)
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    datasets = []

    ############################################################ 
    # Pre-training stage: Pre-train or load checkpoint
    #   1. Set up environment
    #   2. Set up task (aka reward function)
    #   3. Set up agent
    #   4. Train agent with PPO 
    ############################################################

    log.info("Initializing pre-training")
    env: A1Continual = hydra.utils.instantiate(cfg.pretrain_task)  ## Note: env and runner are explicitly initialized outside of pretrain() so that it's explicit which environment we are using and which runner
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.pretrain_train, env=env, _recursive_=False)  ## Note: I don't love the idea of "Runner", I think it's better to have something like an agent with some notation. We should look at jaxrl/stable baselines to see how they do it
    
    if not cfg.load_pretrained:
        pretrained_policy, pretrained_model = pretrain(cfg.pretrain_train, env, runner)
        if cfg.save_pretrained:
            runner.save(cfg.save_pretrained_filename)
    else:
        runner.load(cfg.load_pretrained)
        pretrained_policy  = runner.get_inference_policy(device=env.device)
        pretrained_model = runner.alg.actor_critic

    env.exit()
    log.info("Done with pre-training")

    ############################################################ 
    # Pre-training stage: Data collection w/ adaptation config
    #   1. Change reward function to match finetuning reward
    #   2. Change observation space to match finetuning obs
    #   3. Collect data of pre-trained agent w/ new reward and 
    #      observations  
    ############################################################

    if not cfg.load_preadapt_buffer:
        env: A1Continual = hydra.utils.instantiate(cfg.pretrain_task)
        agent = pretrained_policy 
        preadapt_data = collect(cfg.pretrain_task, env, agent, cfg.preadapt_collection.dataset_size, nn_normalized_actions=False)
        if cfg.save_preadapt_buffer:
            with open(cfg.save_preadapt_filename, 'wb') as f:
                pickle.dump(preadapt_data, f)
        env.exit()
    else:
        with open(cfg.load_preadapt_buffer, 'rb') as f:
            preadapt_data = pickle.load(f)

    datasets.append(preadapt_data)

    ############################################################ 
    # Pre-training stage: Evaluate policy
    #   1. Evaluate pre-trained agent w/ new reward in pretrain 
    #      environment
    #   2. Evaluate pre-trained agent w/new reward in new
    #      environment
    ############################################################

    # env: A1Continual = hydra.utils.instantiate(cfg.pretrain_task_eval)
    # preadapt_stats = evaluate(cfg.eval, env, pretrained_policy, nn_normalized_actions=False)
    # log.info(f"Pre-trained policy in original terrain has stats: {preadapt_stats}")
    # env.exit()

    ############################################################
    # Offline Training Task
    ############################################################
    env: A1Continual = hydra.utils.instantiate(cfg.adapt_task_adapt_obs)
    agent = IQLLearner.create(cfg.offline_finetuning.seed, env.actor_observation_space,
                            env.action_space, **dict(cfg.offline_finetuning.algorithm))
    offline_pretrained_policy, offline_pretrained_model = offline_pretrain(cfg.offline_finetuning, env, agent, preadapt_data)
    env.exit()

    env: A1Continual = hydra.utils.instantiate(cfg.adapt_task_adapt_obs)
    preadapt_stats = evaluate(cfg.eval, env, offline_pretrained_policy, nn_normalized_actions=True, obs_processor=obs_to_nn_input, action_processor=action_to_env_input)
    log.info(f"Pre-trained policy in original terrain has stats: {preadapt_stats}")
    env.exit()

    ############################################################
    # Adaptation Task
    ############################################################

    ## Adaptation with IQL

    env: A1Continual = hydra.utils.instantiate(cfg.adapt_task_adapt_obs)
    # agent = IQLLearner.create(cfg.offline_finetuning.seed, env.actor_observation_space,
                            # env.action_space, **dict(cfg.offline_finetuning.algorithm))
    # offline_pretrained_policy, offline_pretrained_model = offline_pretrain(cfg.offline_finetuning, env, agent, preadapt_data)
    ## Figure out whether either of offline_pretrained_policy or offline_pretrained_model are agents, if not, I can initialize similar to how I use initialize_pretrained_params for SACLearner
    adapt_policy, adapt_model = adapt(cfg.adapt_train, env, offline_pretrained_model, preadapt_data, use_utd=False)
    env.exit()

    ## End of adaptation with IQL

    # env: A1Continual = hydra.utils.instantiate(cfg.adapt_task_adapt_obs)
    # agent = SACLearner.create(cfg.adapt_train.seed, env.actor_observation_space,
    #                         env.action_space, **dict(cfg.adapt_train.algorithm))
    # ## Initialize with pre-trained parameters. TODO There is still a mismatch in that pretraining does not use tanh squashed (this returns nans), whereas adaptation does, there is also a mismatch in whether state_dependent_std is used or not.
    # agent = agent.initialize_pretrained_params(offline_pretrained_model.actor.params, offline_pretrained_model.critic.params)
    
    # log.info(f"Evaluating IQL pre-training on SAC network")
    # offline_initialized_states = evaluate(cfg.eval, env, agent.eval_actions, nn_normalized_actions=True, obs_processor=obs_to_nn_input, action_processor=action_to_env_input)
    # log.info(f"Offline initialized policy in original terrain has stats: {offline_initialized_states}")
    # env.exit()

    # log.info(f"Training with RLPD!")
    # env: A1Continual = hydra.utils.instantiate(cfg.adapt_task_adapt_obs)
    # adapt_policy, adapt_model = adapt(cfg.adapt_train, env, agent, preadapt_data)

    # if cfg.save_adapt_agent:
    #     ## Setting up Orbax checkpointer to save JAX models
    #     orbax_checkpointer = ocp.PyTreeCheckpointer()
    #     save_args = orbax_utils.save_args_from_target(adapt_model)
    #     options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    #     checkpoint_manager = ocp.CheckpointManager(
    #     cfg.save_adapt_agent_dir, orbax_checkpointer, options)
    #     checkpoint_manager.save(0, adapt_model, save_kwargs={'save_args': save_args})
    
    # env.exit()
    
    # del adapt_policy
    # del adapt_model

    ############################################################
    # Exit environment cleanly
    ############################################################
    log.info("4. Exit Cleanly")
    # env.exit()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
