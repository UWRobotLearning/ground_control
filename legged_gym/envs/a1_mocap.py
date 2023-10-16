import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import normalize, quat_mul, quat_conjugate, quat_rotate
from configs.definitions import (TerrainConfig, CommandsConfig, InitStateConfig,
                                 ControlConfig, AssetConfig, DomainRandConfig,
                                 NormalizationConfig, NoiseConfig, ViewerConfig, SimConfig)
from configs.overrides.mocap import MocapEnvConfig, MocapRewardsConfig
from legged_gym.envs.a1 import A1
from legged_gym.utils.legmath import angle_axis_from_quat, wrap_to_pi
from rsl_rl.datasets.motion_loader import MocapLoader

class A1Mocap(A1):
    env_cfg: MocapEnvConfig
    rewards_cfg: MocapRewardsConfig

    def __init__(self, env: MocapEnvConfig, terrain: TerrainConfig,
                 commands: CommandsConfig, init_state: InitStateConfig,
                 control: ControlConfig, asset: AssetConfig,
                 domain_rand: DomainRandConfig, rewards: MocapRewardsConfig,
                 normalization: NormalizationConfig, noise: NoiseConfig,
                 viewer: ViewerConfig, sim: SimConfig):
        super().__init__(env, terrain, commands, init_state, control, asset, domain_rand,
                         rewards, normalization, noise, viewer, sim)
        self._prepare_tracking_functions()
        self.mocap_loader = MocapLoader(motion_files=self.env_cfg.mocap_motion_files, device=self.device, time_between_frames=self.dt, interpolate=False)
        self.mocap_frame = torch.empty((self.num_envs, self.mocap_loader.trajectories_full[0].shape[1]), dtype=torch.float32, device=self.device, requires_grad=False)
        self.origin_offset_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)

    def post_physics_step(self):
        self.time_idx += 1
        env_ids = super().post_physics_step()
        self.mocap_frame = self.mocap_loader.get_full_frame_at_time_idx(self.traj_idx.cpu().numpy(), self.time_idx.cpu().numpy())
        return env_ids

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= (self.time_idx >= self.end_idx)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # clone this since the original buffer will be zeroed out
        ep_lengths = self.episode_length_buf[env_ids].clone()

        super().reset_idx(env_ids)
        for key in self.tracking_error_names:
            error_per_step = self.tracking_episode_sums[key][env_ids] / ep_lengths
            # filter out any NaN elements (due to zero-length episodes)
            error_per_step = error_per_step[~error_per_step.isnan()]
            if len(error_per_step) > 0:
                self.extras["episode"]["error/" + key] = torch.mean(error_per_step)

            self.tracking_episode_sums[key][env_ids] = 0.

    def _reset_agents(self, env_ids):
        mocap_mask = (np.random.rand(len(env_ids)) <= self.env_cfg.reference_state_initialization_prob)
        mocap_env_ids = env_ids[mocap_mask]
        std_env_ids = env_ids[~mocap_mask]

        traj_idx = self.mocap_loader.weighted_traj_idx_sample(len(env_ids))
        if self.env_cfg.reference_state_initialization_at_start:
            time_idx = np.zeros_like(traj_idx)
        else:
            time_idx = self.mocap_loader.traj_time_idx_sample(traj_idx)
        end_idx = self.mocap_loader.trajectory_num_frames[traj_idx]
        frames = self.mocap_loader.get_full_frame_at_time_idx(traj_idx, time_idx)
        self.mocap_frame[env_ids] = frames
        self.traj_idx[env_ids] = torch.tensor(traj_idx, dtype=torch.int64, device=self.device)
        self.time_idx[env_ids] = torch.tensor(time_idx, dtype=torch.int64, device=self.device)
        self.end_idx[env_ids] = torch.tensor(end_idx, dtype=torch.int64, device=self.device)

        if len(mocap_env_ids) > 0:
            self._reset_dofs_mocap(mocap_env_ids, frames[mocap_mask])
            self._reset_root_states_mocap(mocap_env_ids, frames[mocap_mask])
        if len(std_env_ids) > 0:
            self._reset_dofs(std_env_ids)
            self._reset_root_states(std_env_ids)

    def compute_reward(self):
        super().compute_reward()
        for i in range(len(self.tracking_error_names)):
            name = self.tracking_error_names[i]
            self.tracking_episode_sums[name] += self.tracking_error_functions[i]()

    def _compute_privileged_observations(self):
        time = torch.unsqueeze(self.time_idx / self.end_idx, -1)
        return torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            time
        ), dim=-1)

    def _get_obs_size(self, env_cfg: MocapEnvConfig, *ignored):
        privileged_obs_size = 46
        obs_size = privileged_obs_size if env_cfg.observation.base_vel_in_obs else privileged_obs_size-6
        obs_sensors = ("projected_gravity", "motor_angles", "motor_velocities", "last_action", "time")
        if env_cfg.observation.base_vel_in_obs:
            obs_sensors = ("linear_velocity", "angular_velocity") + obs_sensors
        return obs_size, privileged_obs_size, obs_sensors

    def _reset_dofs_mocap(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: Mocap frames to initialize motion with
        """
        self.dof_pos[env_ids] = MocapLoader.get_joint_pose(frames)
        self.dof_vel[env_ids] = MocapLoader.get_joint_vel(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_root_states_mocap(self, env_ids, frames):
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

        self.origin_offset_pos[env_ids, :2] = self.env_origins[env_ids, :2]

    def _init_buffers(self):
        super()._init_buffers()
        self.traj_idx = torch.empty(self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False)
        self.time_idx = torch.clone(self.traj_idx)
        self.end_idx = torch.clone(self.traj_idx)

    def _parse_cfg(self):
        super()._parse_cfg()
        self.tracking_error_names = self.rewards_cfg.tracking_errors

    def _prepare_tracking_functions(self):
        self.tracking_error_functions = []
        for name in self.tracking_error_names:
            name = f"_{name}_tracking_error"
            self.tracking_error_functions.append(getattr(self, name))

        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.tracking_episode_sums = {name: zeros.clone() for name in self.tracking_error_names}

    def _joint_position_tracking_error(self):
        dof_pos_des = MocapLoader.get_joint_pose(self.mocap_frame)
        error = torch.norm(self.dof_pos - dof_pos_des, dim=-1)
        return error

    def _joint_velocity_tracking_error(self):
        dof_vel_des = MocapLoader.get_joint_vel(self.mocap_frame)
        error = torch.norm(self.dof_vel - dof_vel_des, dim=-1)
        return error

    def _foot_position_tracking_error(self):
        foot_position = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        foot_position_des = MocapLoader.get_tar_toe_pos_local(self.mocap_frame)
        error = torch.norm(foot_position - foot_position_des, dim=-1)
        return error

    def _root_position_tracking_error(self):
        root_posn = self.root_states[:, :3]
        root_posn_des = MocapLoader.get_root_pos(self.mocap_frame) + self.origin_offset_pos
        error = torch.norm(root_posn - root_posn_des, dim=-1)
        return error

    def _root_orientation_tracking_error(self):
        root_quat = self.base_quat
        root_quat_des = MocapLoader.get_root_rot(self.mocap_frame)
        root_quat_diff = quat_mul(root_quat_des, quat_conjugate(root_quat))
        root_quat_diff = normalize(root_quat_diff)
        _, root_quat_diff_angle = angle_axis_from_quat(root_quat_diff)
        error = wrap_to_pi(root_quat_diff_angle)
        return error

    def _root_lin_vel_tracking_error(self):
        root_lin = self.base_lin_vel
        root_lin_des = MocapLoader.get_linear_vel(self.mocap_frame)
        error = torch.norm(root_lin - root_lin_des, dim=-1)
        return error

    def _root_ang_vel_tracking_error(self):
        root_ang = self.base_ang_vel
        root_ang_des = MocapLoader.get_angular_vel(self.mocap_frame)
        error = torch.norm(root_ang - root_ang_des, dim=-1)
        return error

    def _reward_joint_position_tracking(self):
        # Reward for matching joint angles
        error = self._joint_position_tracking_error()
        return torch.exp(-self.rewards_cfg.joint_position_tracking_sigma * torch.square(error))

    def _reward_joint_velocity_tracking(self):
        # Reward for matching joint velocities
        error = self._joint_velocity_tracking_error()
        return torch.exp(-self.rewards_cfg.joint_velocity_tracking_sigma * torch.square(error))

    def _reward_foot_position_tracking(self):
        # Reward for matching feet positions
        error = self._foot_position_tracking_error()
        return torch.exp(-self.rewards_cfg.foot_position_tracking_sigma * torch.square(error))

    def _reward_root_pose_tracking(self):
        # Reward for matching the root position and orientation
        posn_error = self._root_position_tracking_error()
        orn_error = self._root_orientation_tracking_error()
        root_pose_error = torch.square(posn_error) + 0.5*torch.square(orn_error)
        return torch.exp(-self.rewards_cfg.root_pose_tracking_sigma * root_pose_error)

    def _reward_root_vel_tracking(self):
        root_lin_error = self._root_lin_vel_tracking_error()
        root_ang_error = self._root_ang_vel_tracking_error()
        root_vel_error = torch.square(root_lin_error) + 0.1*torch.square(root_ang_error)
        return torch.exp(-self.rewards_cfg.root_vel_tracking_sigma * root_vel_error)
