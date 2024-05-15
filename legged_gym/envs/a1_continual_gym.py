import numpy as np
import gymnasium as gym
from gymnasium import spaces
from legged_gym.envs.a1_continual import A1Continual
import torch


class A1Gym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, a1_env: A1Continual=None):
        self._a1_env = a1_env
        assert self._a1_env.num_envs == 1, f"A1Gym wrapper currently only supports num_envs=1, input env has num_envs={self._a1_env.num_envs}"

        self.observation_space = a1_env.actor_observation_space
        self.action_space = a1_env.action_space  ## TODO: Check what this is, and clarify the interface. This should have the actual max position angles that the URDF can handle, and imo this should be unscaled, since I believe SAC from stable baselines will automatically normalize between -1, 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        ## Anything else that is non-standard Gym will go below this line

    def _observation_processor(self, observation: torch.Tensor) -> np.array:
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        obs = observation.squeeze()
        return obs
    
    def _action_processor(self, action: np.array) -> torch.Tensor:
        action = torch.from_numpy(action).unsqueeze(0)
        return action
    
    def _render_frame(self):
        # import ipdb;ipdb.set_trace()
        return self._a1_env.get_camera_images()[0][0]

    def _render_human(self):
        return self._a1_env.render()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  ## Necessary to set self.np_random

        obs, critic_obs = self._a1_env.reset()
        info = {}  ## Unsure if anything should go here

        if self.render_mode == "human":
            self._render_human()
        return self._observation_processor(obs), info
    

    def step(self, action):
        policy_obs, critic_obs, rewards, dones, info = self._a1_env.step(self._action_processor(action))
        
        observation = self._observation_processor(policy_obs)
        reward = rewards.detach().cpu().item()
        done = dones.detach().cpu().item()

        if not done:
            truncated = False
            terminated = False
        elif ('time_outs' in info and info["time_outs"].item()):
            truncated = True
            terminated = False
        else:
            truncated = False
            terminated = True
    
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_human()
        else:
            return NotImplementedError()
        
    def close(self):
        return