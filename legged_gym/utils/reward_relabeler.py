import isaacgym
import logging
import os.path as osp
import time

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
# from pydantic import TypeAdapter
# from pydantic.dataclasses import dataclass

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig)
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from legged_gym.envs.a1 import A1
from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)
from rsl_rl.runners import OnPolicyRunner
from configs.overrides.rewards import MoveFwdRewardsConfig, WITPLeggedGymRewardsConfig

from witp.rl.data.replay_buffer import ReplayBuffer
import pickle
import torch
import gymnasium as gym
from isaacgym import torch_utils
from math import ceil
import numpy as np

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
        
def human_format(num):  ## From https://stackoverflow.com/a/45846841
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    
class RewardRelabeler:
    def __init__(self, buffer, rewards_cfg, batch_size=512, dt=0.005, device="cuda"):
        self.buffer = buffer
        self.rewards_cfg = rewards_cfg
        self.batch_size = batch_size
        self.dt = dt
        self.device = device
        self.reward_scales = OmegaConf.to_container(self.rewards_cfg.scales) # to allow for keys to be `pop`'d

        self._prepare_reward_function()
        self.relabel_rewards()


    def extract_state(self, batch):
        self.state = {}
        obs_labels = batch["observation_labels"]
        for obs_name in obs_labels:
            self.state[obs_name] = batch["observations"][:,range(*obs_labels[obs_name])]

    def extract_actions(self, batch):
        self.actions = batch["actions"]

    def relabel_rewards(self):
        # import ipdb;ipdb.set_trace()
        len_buffer = len(self.buffer)
        num_batches = ceil(len_buffer/self.batch_size)
        batch_starts = [(k)*self.batch_size for k in range(num_batches)]
        batch_ends   = [min(((k+1)*self.batch_size), len_buffer) for k in range(num_batches)]

        for b in range(num_batches):
            self.rew_buf = torch.zeros(batch_ends[b]-batch_starts[b], device=self.device, dtype=torch.float)  ## TODO: Check if this needs to be + 1
            # reward episode sums
            self.episode_sums = {name: torch.zeros(batch_ends[b]-batch_starts[b], dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}
            batch_sample = self.buffer.sample(self.batch_size, indx=np.arange(batch_starts[b], batch_ends[b]))
            self.extract_state(batch_sample)
            self.extract_actions(batch_sample)
            ## TODO: Need to change the actual reward functions to use the data from the self.state and self.actions dictionaries instead of just from self.____
            self.compute_reward()
            self.buffer.dataset_dict["rewards"][np.arange(batch_starts[b], batch_ends[b])] = self.rew_buf.cpu().numpy()
            

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums andself.critic_obs_buf to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.rew_buf[:] *= self.rewards_cfg.scale_all
        if self.rewards_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # # add termination reward after clipping
        # if "termination" in self.reward_scales:
        #     rew = self._reward_termination() * self.reward_scales["termination"]
        #     self.rew_buf += rew
        #     self.episode_sums["termination"] += rew
            
    def get_relabeled_buffer(self):
        """ Returns the buffer with the relabeled reward. Assumes relabeling has already happened inside __init__
        """
        return self.buffer

    #---------------------------------------------
    #------------ reward functions----------------
    #---------------------------------------------

    #------------ action rewards----------------
    def _reward_action(self):
        # Penalize magnitude of actions
        return -torch.sum(torch.square(self.actions), dim=1)

    # def _reward_action_change(self):
    #     # Penalize changes in actions
    #     return -torch.sum(torch.square(self.action_change), dim=1)

    #------------ base link rewards----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return -torch.square(self.state["base_lin_vel"][:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return -torch.sum(torch.square(self.state["base_ang_vel"][:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return -torch.sum(torch.square(self.state["projected_gravity"][:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     return -torch.square(self.base_heights - self.rewards_cfg.base_height_target)

    #------------ joints rewards----------------
    # def _reward_torques(self):
    #     # Penalize torques
    #     return -torch.sum(torch.square(self.torques), dim=1)

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return -torch.sum(torch.square(self.dof_vel), dim=1)

    # def _reward_dof_accel(self):
    #     # Penalize dof accelerations
    #     return -torch.sum(torch.square(self.dof_accel), dim=1)

    # def _reward_soft_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     return -torch.sum(self.dof_pos_out_of_limits, dim=1)

    # def _reward_soft_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return -torch.sum(self.dof_vel_out_of_limits.clip(min=0., max=1.), dim=1)

    # def _reward_soft_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return -torch.sum(self.torque_out_of_limits.clip(min=0.), dim=1)

    #------------ energy rewards----------------
    # def _reward_power(self):
    #     # Penalize power consumption (mechanical + heat)
    #     return -self.power

    # def _reward_cost_of_transport(self):
    #     # Penalize cost of transport (power / (weight * speed))
    #     return -self.cost_of_transport

    #------------ feet rewards----------------
    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     return self.rew_air_time_on_contact

    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return -torch.sum(self.foot_contact_forces_out_of_limits, dim=1)

    # def _reward_feet_contact_force_change(self):
    #     # penalize foot jerk to prevent large motor backlash
    #     return -torch.sum(self.foot_contact_force_change.square(), dim=1)

    #------------ safety rewards----------------
    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return -self.collisions.float()

    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return -self.termination_buf.float()

    # def _reward_alive(self):
    #     return (1-self.termination_buf.float())

    # def _reward_stumble(self):
    #     # Penalize feet hitting vertical surfaces
    #     return -torch.any(self.stumbles, dim=1).float()

    # def _reward_stand_still(self):
    #     # Penalize motion at zero commands
    #     return -torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    #------------ task rewards----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.state["commands"][:, :2] - self.state["base_lin_vel"][:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.rewards_cfg.tracking_sigma**2)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.state["commands"][:, 2] - self.state["base_ang_vel"][:, 2])
        return torch.exp(-ang_vel_error/self.rewards_cfg.tracking_sigma**2)
    
    #------------ witp rewards----------------
    def _reward_witp_abs_dyaw(self):
        # Penalize any angular velocity
        dyaw_abs = torch.abs(torch.from_numpy(self.state["base_ang_vel"][:, 2])).to(self.device)
        return -dyaw_abs

    def _reward_witp_cos_pitch_times_lin_vel(self):
        # Reward forward velocity times cos(pitch)
        move_speed = 0.5  ## m/s
        roll, pitch, yaw = torch_utils.get_euler_xyz(torch.from_numpy(self.state["base_quat"]).to(self.device))
        cos_pitch = torch.cos(pitch)
        x_velocity = torch.from_numpy(self.state["base_lin_vel"][:, 0]).to(self.device)
        # from dm_control.utils import rewards
        # tolerance = rewards.tolerance(cos_pitch * x_velocity,
        #                        bounds=(move_speed, 2 * move_speed),
        #                        margin=2 * move_speed,
        #                        value_at_margin=0,
        #                        sigmoid='linear')
        cos_pitch_times_vel = cos_pitch * x_velocity
        bounds = (move_speed, 2 * move_speed)
        lower, upper = bounds
        margin = 2*move_speed
        in_bounds = torch.logical_and(lower <= cos_pitch_times_vel, cos_pitch_times_vel <= upper)
        d = torch.where(cos_pitch_times_vel < lower, lower - cos_pitch_times_vel, cos_pitch_times_vel - upper) / margin
        value_at_margin = 0.1
        scale = 1-value_at_margin
        scaled_x = d*scale
        sigmoid = torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        value = torch.where(in_bounds, 1.0, sigmoid)
        return value

# @dataclass
# class CollectScriptConfig:
#     checkpoint_root: str = from_repo_root("../experiment_logs/train")
#     logging_root: str = from_repo_root("../experiment_logs")
#     export_policy: bool = True
#     num_envs: int = 50
#     use_joystick: bool = True
#     episode_length_s: float = 200.
#     checkpoint: int = -1
#     headless: bool = False 
#     device: str = "cpu"

#     hydra: ExperimentHydraConfig = ExperimentHydraConfig()

#     task: TaskConfig = empty_cfg(TaskConfig)(
#         env = empty_cfg(EnvConfig)(
#             num_envs = "${num_envs}"
#         ),
#         observation = empty_cfg(ObservationConfig)(
#             get_commands_from_joystick = "${use_joystick}",
#             sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
#             critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
#             extra_sensors=("base_quat",)
#         ),
#         sim = empty_cfg(SimConfig)(
#             device = "${device}",
#             use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
#             headless = "${headless}",
#             physx = empty_cfg(SimConfig.PhysxConfig)(
#                 use_gpu = "${evaluate_use_gpu: ${task.sim.device}}"
#             )
#         ),
#         terrain = empty_cfg(TerrainConfig)(
#             curriculum = False
#         ),
#         noise = NoNoiseConfig(),
#         domain_rand = NoDomainRandConfig()
#     ) 
#     train: TrainConfig = empty_cfg(TrainConfig)(
#         device = "${device}",
#         log_dir = "${hydra:runtime.output_dir}",
#         runner = empty_cfg(RunnerConfig)(
#             checkpoint="${checkpoint}"
#         )
#     )

# cs = ConfigStore.instance()
# cs.store(name="config", node=CollectScriptConfig)

# @hydra.main(version_base=None, config_name="config")
# def main(cfg: CollectScriptConfig):
def main():
    cfg = OmegaConf.structured(WITPLeggedGymRewardsConfig())
    save_buffer_path = "/home/mateo/projects/experiment_logs/collect/2024-02-28_11-21-39/dataset_2023-12-06_08-46-22_model_5000_1M.pkl"
    with open(save_buffer_path, 'rb') as f:
        loaded_buffer = pickle.load(f)

    relabeler = RewardRelabeler(loaded_buffer, cfg, batch_size=1000000)
    # import ipdb;ipdb.set_trace()

    relabeled_buffer = relabeler.get_relabeled_buffer()
    save_relabeled_path = "/home/mateo/projects/experiment_logs/collect/2024-02-28_11-21-39/dataset_2023-12-06_08-46-22_model_5000_1M_relabeled_witp.pkl"
    with open(save_relabeled_path, 'wb') as f:
        pickle.dump(relabeled_buffer, f)

    # experiment_path = cfg.checkpoint_root
    # latest_config_filepath = osp.join(experiment_path, "resolved_config.yaml")
    # log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    # loaded_cfg = OmegaConf.load(latest_config_filepath)

    # log.info("2. Merging loaded config, defaults and current top-level config.")
    # del(loaded_cfg.hydra) # Remove unpopulated hydra configuration key from dictionary
    # default_cfg = {"task": TaskConfig(), "train": TrainConfig()}  # default behaviour as defined in "configs/definitions.py"
    # merged_cfg = OmegaConf.merge(
    #     default_cfg,  # loads default values at the end if it's not specified anywhere else
    #     loaded_cfg,   # loads values from the previous experiment if not specified in the top-level config
    #     cfg           # highest priority, loads from the top-level config dataclass above
    # )
    # # Resolves the config (replaces all "interpolations" - references in the config that need to be resolved to constant values)
    # # and turns it to a dictionary (instead of DictConfig in OmegaConf). Throws an error if there are still missing values.
    # merged_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    # # Creates a new PlayScriptConfig object (with type-checking and optional validation) using Pydantic.
    # # The merged config file (DictConfig as given by OmegaConf) has to be recursively turned to a dict for Pydantic to use it.
    # # cfg = TypeAdapter(PlayScriptConfig).validate_python(merged_cfg_dict)
    # # cfg = PlayScriptConfig(**merged_cfg_dict)
    # cfg = merged_cfg
    # # Alternatively, you should be able to use "from pydantic.dataclasses import dataclass" and replace the above line with
    # # cfg = PlayScriptConfig(**merged_cfg_dict)
    # log.info(f"3. Printing merged cfg.")
    # print(OmegaConf.to_yaml(cfg))
    # save_config_as_yaml(cfg)

    # log.info(f"4. Preparing environment and runner.")
    # task_cfg = cfg.task
    # env: A1 = hydra.utils.instantiate(task_cfg)
    # env.reset()
    # obs = env.get_observations()
    # critic_obs = env.get_critic_observations()
    # extra_obs = env.get_extra_observations()
    # runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    # resume_path = get_load_path(experiment_path, checkpoint=cfg.train.runner.checkpoint)
    # log.info(f"5. Loading policy checkpoint from: {resume_path}.")
    # runner.load(resume_path)
    # policy = runner.get_inference_policy(device=env.device)

    # if cfg.export_policy:
    #     export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
    #     log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    # log.info(f"6. Instantiating replay buffer.")
    # dataset_size = 1000000#0  ## Need to make this a config param. Rn if I add to config it gets rid of it during merge seems like
    # save_buffer_path = osp.join(cfg.train.log_dir, f"dataset_{resume_path.split('/')[-2]+'_'+resume_path.split('/')[-1][:-3]}_{human_format(dataset_size)}.pkl")
    # num_obs = env.num_critic_obs + env.num_extra_obs
    # obs_limit = cfg.task.normalization.clip_observations
    # observation_space = gym.spaces.Box(low=-obs_limit, high=obs_limit, shape=(num_obs,))

    # num_actions = env.num_actions
    # action_limit = cfg.task.normalization.clip_actions
    # action_space = gym.spaces.Box(low=-action_limit, high=action_limit, shape=(num_actions,))
    # sensors =  cfg.task.observation.sensors + cfg.task.observation.critic_privileged_sensors + cfg.task.observation.extra_sensors
    # observation_labels = {sensor_name:(0, 0) for sensor_name in sensors}
    # observation_labels['projected_gravity'] = (0, 3)
    # observation_labels['commands'] = (3, 6)
    # observation_labels['motor_pos'] = (6, 18)
    # observation_labels['motor_vel'] = (18, 30)
    # observation_labels['last_action'] = (30, 42)
    # observation_labels['yaw_rate'] = (42, 43)
    # observation_labels['base_lin_vel'] = (43, 46)
    # observation_labels['base_ang_vel'] = (46, 49)
    # observation_labels['terrain_height'] = (49, 50)
    # observation_labels['friction'] = (50, 51)
    # observation_labels['base_mass'] = (51, 52)
    # observation_labels['base_quat'] = (52, 56)

    # replay_buffer = ReplayBuffer(observation_space, action_space, capacity=dataset_size, next_observation_space=observation_space, observation_labels=observation_labels)

    # log.info(f"7. Running interactive collect script.")
    # current_time = time.time()
    # num_steps = int(cfg.episode_length_s / env.dt)
    # # for i in range(num_steps):
    # while len(replay_buffer) < dataset_size:
    #     actions = policy(obs.detach())
    #     new_obs, new_critic_obs, rewards, dones, infos, *_ = env.step(actions.detach())
    #     new_extra_obs = env.get_extra_observations()
    #     duration = time.time() - current_time
    #     if duration < env.dt:
    #         time.sleep(env.dt - duration)
    #     current_time = time.time()

    #     print(f"replay_buffer size: {len(replay_buffer)}")
    #     buffer_observations = torch.concatenate([critic_obs, extra_obs], dim=-1)
    #     buffer_new_observations = torch.concatenate([new_critic_obs, new_extra_obs], dim=-1)
    #     insert_batch_into_replay_buffer(replay_buffer, buffer_observations, actions, rewards, dones, buffer_new_observations, infos, observation_labels)
    #     obs = new_obs
    #     critic_obs = new_critic_obs
    #     extra_obs = new_extra_obs

    # log.info(f"8. Save replay buffer here: {save_buffer_path}")
    # with open(save_buffer_path, 'wb') as f:
    #     pickle.dump(replay_buffer, f)

    # with open(save_buffer_path, 'rb') as f:
    #     loaded_buffer = pickle.load(f)

    # log.info("9. Exit Cleanly")
    # env.exit()

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()


'''
Example command to run:

python collect.py num_envs=4096 checkpoint_root='/home/mateo/projects/experiment_logs/train/2023-12-06_08-46-22' checkpoint=4000 use_joystick=False headless=True
'''





