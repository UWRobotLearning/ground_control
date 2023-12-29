import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch
from typing import Dict
from omegaconf import OmegaConf

from legged_gym.utils import observation_buffer
from configs.definitions import (EnvConfig, ObservationConfig, TerrainConfig, CommandsConfig,
                                 InitStateConfig, ControlConfig, AssetConfig, DomainRandConfig,
                                 RewardsConfig, NormalizationConfig, NoiseConfig, ViewerConfig,
                                 SimConfig)


# Base class for RL envs
class BaseEnv:
    sensor_dims: Dict[str, int]

    def __init__(self, env: EnvConfig, observation: ObservationConfig, sim: SimConfig):
        self.sim_cfg = sim
        self.env_cfg = env
        self.observation_cfg = observation
        self.gym = gymapi.acquire_gym()

        self.sim_params = gymapi.SimParams()  # get Isaac-bindings sim_params object which contains default settings
        gymutil.parse_sim_config(OmegaConf.to_container(self.sim_cfg), self.sim_params)  # override defaults in sim_param object
        self.physics_engine = gymapi.SIM_PHYSX
        self.sim_device = self.sim_cfg.device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = self.sim_cfg.headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and self.sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id if not self.headless else -1

        self._parse_cfg()

        self.num_envs = self.env_cfg.num_envs
        self.num_obs = sum([self.sensor_dims[sensor] for sensor in self.observation_cfg.sensors])
        self.num_critic_obs = self.num_obs + sum([self.sensor_dims[sensor] for sensor in self.observation_cfg.critic_privileged_sensors])
        self.num_actions = 12
        self.history_steps = self.observation_cfg.history_steps

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf_history = observation_buffer.ObservationBuffer(
            self.num_envs, self.num_obs,
            self.history_steps, self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device, dtype=torch.float)

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def create_sim(self):
        raise NotImplementedError

    def _parse_cfg(self):
        raise NotImplementedError

    def get_observations(self):
        return self.obs_buf

    def get_critic_observations(self):
        return self.critic_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
