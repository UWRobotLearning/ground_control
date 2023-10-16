import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import normalize, quat_mul, quat_conjugate, quat_rotate, quat_rotate_inverse
from legged_gym.envs.a1 import A1
from legged_gym.utils.legmath import angle_axis_from_quat, wrap_to_pi
from rsl_rl.datasets.motion_loader import MocapLoader

from configs.definitions import (TerrainOrDictConfig, CommandsOrDictConfig, InitStateOrDictConfig,
                                 ControlOrDictConfig, AssetOrDictConfig, DomainRandOrDictConfig,
                                 RewardsOrDictConfig, NormalizationOrDictConfig, NoiseOrDictConfig,
                                 ViewerOrDictConfig, SimOrDictConfig)
from configs.overrides.amp import AMPEnvOrDictConfig, AMPObservationOrDictConfig

class A1AMP(A1):
    env_cfg: AMPEnvOrDictConfig
    observation_cfg: AMPObservationOrDictConfig

    def __init__(self, env: AMPEnvOrDictConfig, observation: AMPObservationOrDictConfig,
                 terrain: TerrainOrDictConfig, commands: CommandsOrDictConfig, init_state: InitStateOrDictConfig,
                 control: ControlOrDictConfig, asset: AssetOrDictConfig, domain_rand: DomainRandOrDictConfig,
                 rewards: RewardsOrDictConfig, normalization: NormalizationOrDictConfig, noise: NoiseOrDictConfig,
                 viewer: ViewerOrDictConfig, sim: SimOrDictConfig):
        super().__init__(env, observation, terrain, commands, init_state, control, asset, domain_rand,
                         rewards, normalization, noise, viewer, sim)
        self.amp_loader = MocapLoader(
            motion_files=self.env_cfg.motion_files,
            time_between_frames=self.dt,
            sensors=self.observation_cfg.amp_sensors,
            is_amp=True,
            device=self.device
        )

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._compute_physical_measurements()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_diagnostics()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.viewer_cfg.debug_viz:
            self._draw_debug_vis()

        return env_ids, (env_ids, terminal_amp_states)

    def get_amp_observations(self):
        return self._compute_observations(self.observation_cfg.amp_sensors, normalize=False)

    def _reset_agents(self, env_ids):
        """
        Reset the states of the agents that need to be reseted.
        """
        amp_mask = (np.random.rand(len(env_ids)) <= self.env_cfg.reference_state_initialization_prob)
        amp_env_ids = env_ids[amp_mask]
        std_env_ids = env_ids[~amp_mask]
        if len(amp_env_ids) > 0:
            frames = self.amp_loader.get_full_frame(len(amp_env_ids))
            self._reset_dofs_amp(amp_env_ids, frames)
            self._reset_root_states_amp(amp_env_ids, frames)
        if len(std_env_ids) > 0:
            self._reset_dofs(std_env_ids)
            self._reset_root_states(std_env_ids)

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = MocapLoader.get_joint_pose(frames)
        self.dof_vel[env_ids] = MocapLoader.get_joint_vel(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = MocapLoader.get_root_pos(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = MocapLoader.get_root_rot(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, MocapLoader.get_linear_vel(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, MocapLoader.get_angular_vel(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
