### ground_control imports
import isaacgym # need to import this before torch
import torch
import logging
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Tuple
import os
from collections import deque
import time
import statistics

# hydra / config related imports
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
# from pydantic import TypeAdapter  ## Commented out by Mateo because this conflicts with tf_probability

from configs.definitions import TaskConfig, TrainConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig, WITPLocomotionTaskConfig
from configs.hydra import ExperimentHydraConfig

from legged_gym.envs.a1 import A1
# from rsl_rl.runners import OnPolicyRunner  ## Commented out by Mateo because we are trying to train with witp code.
from legged_gym.utils.helpers import (set_seed, get_load_path, get_latest_experiment_path, save_resolved_config_as_pkl, save_config_as_yaml,
                                      from_repo_root)
#####

### witp imports
import os
import pickle
import shutil

import numpy as np
import tqdm

import gymnasium as gym
# import wandb
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

@dataclass
class Flags:
    # env_name: str = 'A1Run-v0'
    # save_dir: str = './tmp/'
    seed: int = 42
    eval_episodes: int = 1
    log_interval: int = 1000
    eval_interval: int = 1000
    batch_size: int = 256
    max_steps: int = int(1e5)
    start_training: int = int(1e3)
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
class TrainScriptConfig:
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
    headless: bool = False
    checkpoint_root: str = from_repo_root("./witp_checkpoints")  ## ""
    logging_root: str = from_repo_root("./witp_experiment_logs")
    episode_buffer_len: int = 100
    task: TaskConfig = WITPLocomotionTaskConfig()
    train: TrainConfig = TrainConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainScriptConfig)


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

def clip_actions(action):
    '''
    Walk in the Park requires joint limits to learn. Here we clip the action to the right joint limits.
    '''
    INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4)
    ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

    lower_bound = INIT_QPOS - ACTION_OFFSET
    upper_bound = INIT_QPOS + ACTION_OFFSET

    action = np.clip(action, lower_bound, upper_bound)

    return action

def clip_and_norm_actions(action):
    '''
    Walk in the Park requires joint limits to learn. Here we clip the action to the right joint limits, and we also normalize so that
    the action is bounded by -1, 1 approximately.
    '''
    INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4)
    ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

    lower_bound = INIT_QPOS - ACTION_OFFSET
    upper_bound = INIT_QPOS + ACTION_OFFSET

    min_action = -1 + np.zeros((12, ))

    max_action = 1 + np.zeros((12, ))

    action = lower_bound + (upper_bound - lower_bound) * (
            (action - min_action) / (max_action - min_action)
        )
    action = np.clip(action, lower_bound, upper_bound)

    return action

@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainScriptConfig) -> None:
    log.info("1. Printing and serializing frozen TrainScriptConfig")
    OmegaConf.resolve(cfg)
    # Type-checking (and other validation if defined) via Pydantic
    # cfg = TypeAdapter(TrainScriptConfig).validate_python(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)
    #save_resolved_config_as_pkl(cfg)

    log.info("2. Initializing Env and Runner")
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env: A1 = hydra.utils.instantiate(cfg.task)
    # runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

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
    # wandb.init(project='a1')
    # wandb_cfg_dict = {**FLAGS.as_dict(), **kwargs, **dict(cfg)}
    # wandb.config.update(wandb_cfg_dict)

    ## Define our own observation space and action space since these are not directly available in 
    ## the ground_control A1 environment:

    num_obs = env.num_obs
    obs_limit = cfg.task.normalization.clip_observations
    observation_space = gym.spaces.Box(low=-obs_limit, high=obs_limit, shape=(num_obs,))

    num_actions = env.num_actions
    action_limit = cfg.task.normalization.clip_actions
    # action_space = gym.spaces.Box(low=-action_limit, high=action_limit, shape=(num_actions,))
    action_space = gym.spaces.Box(
        low=np.array([-0.803, -1.047, -2.697, -0.803, -1.047, -2.697, -0.803, -1.047, -2.697, -0.803, -1.047, -2.697]),
        high=np.array([0.803,  4.189, -0.916,  0.803,  4.189, -0.916,  0.803,  4.189, -0.916,  0.803,  4.189, -0.916]),
        shape=(num_actions,)
    )

    agent = SACLearner.create(FLAGS.seed, observation_space,
                              action_space, **kwargs)
    
    # import pdb;pdb.set_trace()

    ###################################################### 
    # Set up replay buffer
    ###################################################### 

    start_i = 0
    replay_buffer = ReplayBuffer(observation_space, action_space,
                                    FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    # Set up reward and episode length buffers for logging purposes
    rewards_buffer = deque(maxlen=cfg.episode_buffer_len)
    lengths_buffer = deque(maxlen=cfg.episode_buffer_len)
    current_reward_sum = 0
    current_episode_length = 0
    ep_infos = []

    ep_scene_images = []
    ep_fpv_images = []
    ###################################################### 
    # Reset environment
    ###################################################### 
    observation, _ = env.reset()
    done = False

    collection_time = 0
    learn_time = 0

    log.info("3. Now Training")
    # runner.learn()

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        # import pdb;pdb.set_trace()
        ## Convert observation to be of usable form:
        start = time.time()
        observation = obs_to_nn_input(observation)

        if i < FLAGS.start_training:
            action = action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        ## TODO: Limit action for WITP
        action = clip_and_norm_actions(action)

        ## Convert action to be of usable form
        env_action = action_to_env_input(action)


        ## Environment step
        next_observation, _, reward, done, info = env.step(env_action)
        # print(f"Iteration {i}")
        # print(f"Reward: {reward}")

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

        # ## TODO: I think the logic in the block below is correct now.
        # ### For debugging purposes:
        # if 'time_outs' in info:
        #     print("*********")
        #     print(f"Iteration {i}")
        #     print("time_outs in info")
        #     time_outs = info["time_outs"]
        #     print(f"Info time_outs.item: {time_outs.item()}")
        #     print("*********")

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
                 next_observations=next_observation))
        observation = next_observation

        # ## This clause will likely be different
        # if done:
        #     observation, done = env.reset(), False
        #     for k, v in info['episode'].items():
        #         decode = {'r': 'return', 'l': 'length', 't': 'time'}
        #         wandb.log({f'training/{decode[k]}': v}, step=i)

        stop = time.time()
        collection_time = stop - start

        # Track learning time
        start = stop
        ## This clause will largely be the same!
        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            stop = time.time()
            learn_time = stop - start
            if i % FLAGS.log_interval == 0:
                # print(f"Training step: {i}")
                for k, v in update_info.items():
                    print({f'training/{k}': v})
                    # wandb.log({f'training/{k}': v}, step=i)

        

        if done:
            # print("\n*****DONE*****\n")
            observation, _ = env.reset()
            done = False

            rewards_buffer.extend([current_reward_sum])
            lengths_buffer.extend([current_episode_length])
            # import pdb;pdb.set_trace()
            for k, v in info['episode'].items():
                print({f'training/{k}': v.item()})
                # wandb.log({f'training/{k}': v.item()}, step=i)
            ## Check what info gets logged:
            # import pdb;pdb.set_trace()
            # print(f"Cumulative reward: {current_reward_sum}")
            # print(f"Episode length: {current_episode_length}")
            # wandb.log({'training/cumulative_reward': current_reward_sum}, step=i)
            # wandb.log({'training/episode_length': current_episode_length}, step=i)
            # wandb.log({'training/mean_cumulative_reward': statistics.mean(rewards_buffer)}, step=i)
            # wandb.log({'training/mean_episode_length': statistics.mean(lengths_buffer)}, step=i)

            total_time = collection_time + learn_time
            # wandb.log({'training/cumulative_reward/time': current_reward_sum}, step=total_time)
            # wandb.log({'training/episode_length/time': current_episode_length}, step=total_time)
            # wandb.log({'training/mean_cumulative_reward/time': statistics.mean(rewards_buffer)}, step=total_time)
            # wandb.log({'training/mean_episode_length/time': statistics.mean(lengths_buffer)}, step=total_time)

            
            current_reward_sum = 0
            current_episode_length = 0

            if i % 100 == 0:
                ep_scene_images_stacked = np.stack(ep_scene_images, axis=0)
                ep_fpv_images_stacked = np.stack(ep_fpv_images, axis=0)

                # wandb.log({"video/scene": wandb.Video(ep_scene_images_stacked, fps=10)}, step=i)
                # wandb.log({"video/fpv": wandb.Video(ep_fpv_images_stacked, fps=10)}, step=i)

            ep_scene_images = []
            ep_fpv_images = []

            



        

        ## This will also be different since we don't have an evaluation environment. Can still save checkpoint though
        # if i % FLAGS.eval_interval == 0:
        #     if not FLAGS.real_robot:
        #         eval_info = evaluate(agent,
        #                              eval_env,
        #                              num_episodes=FLAGS.eval_episodes)
        #         for k, v in eval_info.items():
        #             wandb.log({f'evaluation/{k}': v}, step=i)

        #     checkpoints.save_checkpoint(chkpt_dir,
        #                                 agent,
        #                                 step=i + 1,
        #                                 keep=20,
        #                                 overwrite=True)

        #     try:
        #         shutil.rmtree(buffer_dir)
        #     except:
        #         pass

        #     os.makedirs(buffer_dir, exist_ok=True)
        #     with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
        #         pickle.dump(replay_buffer, f)

    log.info("4. Exit Cleanly")
    env.exit()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
