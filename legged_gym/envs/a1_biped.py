import numpy as np
import logging

from isaacgym import torch_utils
from isaacgym import gymtorch

import torch
from scipy.spatial.transform import Rotation  # Ege - for generating random rotations

from legged_gym.envs.a1 import A1
from configs.definitions import (EnvConfig, ObservationConfig, TerrainConfig,
                                 CommandsConfig, InitStateConfig, ControlConfig,
                                 AssetConfig, DomainRandConfig, RewardsConfig,
                                 NormalizationConfig, NoiseConfig, ViewerConfig,
                                 SimConfig, CurriculumConfig, GoalStateConfig)
log = logging.getLogger(__name__)

# TODO: generate from URDF?
# from origin of trunk
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
# from origin of hip joints
HIP_X, HIP_Y = 0.183, 0.047
HIP_OFFSETS = torch.tensor([
    HIP_X,  -HIP_Y,  0.,
    HIP_X,  HIP_Y,   0.,
    -HIP_X, -HIP_Y,  0.,
    -HIP_X, HIP_Y,   0.
])
THIGH_LENGTH = 0.2
CALF_LENGTH = 0.2
HIP_LENGTH = 0.08505

EPISODE_REWARDS = "Rewards"
REWARDS_LOG_NAMES = dict(
    action=f"{EPISODE_REWARDS}/action/action",
    action_change=f"{EPISODE_REWARDS}/action/action_change",
    lin_vel_z=f"{EPISODE_REWARDS}/base/lin_vel_z",
    ang_vel_xy=f"{EPISODE_REWARDS}/base/ang_vel_xy",
    orientation=f"{EPISODE_REWARDS}/base/orientation",
    base_height=f"{EPISODE_REWARDS}/base/base_height",
    power=f"{EPISODE_REWARDS}/energy/power",
    cost_of_transport=f"{EPISODE_REWARDS}/energy/cost_of_transport",
    feet_air_time=f"{EPISODE_REWARDS}/feet/air_time",
    feet_contact_forces=f"{EPISODE_REWARDS}/feet/contact_force",
    feet_contact_force_change=f"{EPISODE_REWARDS}/feet/contact_force_change",
    torques=f"{EPISODE_REWARDS}/joints/torque",
    dof_vel=f"{EPISODE_REWARDS}/joints/velocity",
    dof_accel=f"{EPISODE_REWARDS}/joints/acceleration",
    soft_dof_pos_limits=f"{EPISODE_REWARDS}/joints/angle_out_of_limit",
    soft_dof_vel_limits=f"{EPISODE_REWARDS}/joints/velocity_out_of_limit",
    soft_torque_limits=f"{EPISODE_REWARDS}/joints/torque_out_of_limit",
    collision=f"{EPISODE_REWARDS}/safety/collision",
    stumble=f"{EPISODE_REWARDS}/safety/stumble",
    stand_still=f"{EPISODE_REWARDS}/safety/stand_still",
    termination=f"{EPISODE_REWARDS}/safety/termination",
    alive=f"{EPISODE_REWARDS}/safety/alive",
    tracking_lin_vel=f"{EPISODE_REWARDS}/task/tracking_lin_vel",
    tracking_ang_vel=f"{EPISODE_REWARDS}/task/tracking_ang_vel",
    # Ege - adding log names for rewards related to recovery
    z_axis_orientation=f"{EPISODE_REWARDS}/task/z_axis_orientation",
    xy_drift=f"{EPISODE_REWARDS}/task/xy_drift",
    feet_contact=f"{EPISODE_REWARDS}/task/feet_contact"
)

EPISODE_DIAGNOSTICS = "Diagnostics"
DIAGNOSTICS_LOG_NAMES = dict(
    action=f"{EPISODE_DIAGNOSTICS}/action/action",
    action_change=f"{EPISODE_DIAGNOSTICS}/action/action_change",
    lin_vel_z=f"{EPISODE_DIAGNOSTICS}/base/lin_vel_z",
    ang_vel_xy=f"{EPISODE_DIAGNOSTICS}/base/ang_vel_xy",
    nonflat_orientation=f"{EPISODE_DIAGNOSTICS}/base/nonflat_orientation",
    base_height=f"{EPISODE_DIAGNOSTICS}/base/base_height",
    max_command_x=f"{EPISODE_DIAGNOSTICS}/command/max_command_x",
    power=f"{EPISODE_DIAGNOSTICS}/energy/power",
    cost_of_transport=f"{EPISODE_DIAGNOSTICS}/energy/cost_of_transport",
    feet_air_time=f"{EPISODE_DIAGNOSTICS}/feet/air_time",
    feet_air_time_on_contact=f"{EPISODE_DIAGNOSTICS}/feet/air_time_on_contact",
    feet_contact_forces=f"{EPISODE_DIAGNOSTICS}/feet/contact_force",
    feet_contact_forces_out_of_limits=f"{EPISODE_DIAGNOSTICS}/feet/contact_force_out_of_limit",
    feet_contact_force_change=f"{EPISODE_DIAGNOSTICS}/feet/contact_force_change",
    torques=f"{EPISODE_DIAGNOSTICS}/joints/torque",
    dof_vel=f"{EPISODE_DIAGNOSTICS}/joints/velocity",
    dof_accel=f"{EPISODE_DIAGNOSTICS}/joints/acceleration",
    dof_pos_out_of_limits=f"{EPISODE_DIAGNOSTICS}/joints/angle_out_of_limit",
    dof_vel_out_of_limits=f"{EPISODE_DIAGNOSTICS}/joints/velocity_out_of_limit",
    torque_out_of_limits=f"{EPISODE_DIAGNOSTICS}/joints/torque_out_of_limit",
    collision=f"{EPISODE_DIAGNOSTICS}/safety/collision",
    stumble=f"{EPISODE_DIAGNOSTICS}/safety/stumble",
    termination=f"{EPISODE_DIAGNOSTICS}/safety/termination",
    terrain_level=f"{EPISODE_DIAGNOSTICS}/terrain/level"
)

class A1RecoveryShort(A1):
    def __init__(self, env: EnvConfig, observation: ObservationConfig, terrain: TerrainConfig,
                 commands: CommandsConfig, init_state: InitStateConfig, control: ControlConfig,
                 asset: AssetConfig, domain_rand: DomainRandConfig, rewards: RewardsConfig,
                 normalization: NormalizationConfig, noise: NoiseConfig, viewer: ViewerConfig,
                 sim: SimConfig, curriculum: CurriculumConfig, goal_state: GoalStateConfig):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes PyTorch buffers used during training

        Args:
            cfg (Dict): Environment config file
        """ 
        # Ege
        self.curriculum_cfg = curriculum
        self.goal_state_cfg = goal_state

        super().__init__(env, observation, terrain, commands, init_state, control, asset,
                         domain_rand, rewards, normalization, noise, viewer, sim)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Ege - if an inactive period at start is enabled,
        #       update start_positions when env becomes "active"
        if self.env_cfg.start_inactive_steps > 0:
            starting_idx = (self.episode_length_buf == self.env_cfg.start_inactive_steps)
            self.start_positions[starting_idx] = self.root_states[starting_idx, :3]
        
        clip_actions = self.normalization_cfg.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if self.control_cfg.clip_setpoint:
            if self.control_cfg.control_type == 'P':
                limits = (self.dof_pos_limits.T - self.default_dof_pos) / self.control_cfg.action_scale
                self.actions.clip_(limits[0], limits[1])
            else:
                raise NotImplementedError()

        # step physics and render each frame
        self.render()
        for _ in range(self.control_cfg.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # Ege - if inactive period at start is enabled, disable env actions for that period
            #       the variable "actions" can refer to position targets, thus torques (motor commands)
            #       are zeroed instead to disable any motor force (hence, joint positions aren't affected)
            if self.env_cfg.start_inactive_steps > 0:
                inactive_idx = self.episode_length_buf <= self.env_cfg.num_inactive_steps
                self.torques[inactive_idx] = torch.zeros((torch.sum(inactive_idx), self.torques.shape[1])).to(self.device)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, other_post_physics_outputs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.normalization_cfg.clip_observations
        self.obs_buf.clip_(-clip_obs, clip_obs)
        self.critic_obs_buf.clip_(min=-clip_obs, max=clip_obs)
        self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
        self.obs_buf_history.insert(self.obs_buf)
        policy_obs = self.get_observations()
        critic_obs = self.get_critic_observations()

        infos, self.extras = self.extras, dict()
        return policy_obs, critic_obs, self.rew_buf, self.reset_buf, infos, *other_post_physics_outputs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # Ege - increasing goal length buffer, will call self.check_goal_reach to set some to 0 if necessary
        self.goal_time_buf += 1

        self._compute_physical_measurements()
        self._post_physics_step_callback()

        # Ege - calling self.check_goal_reach to update goal times
        self.check_goal_reach()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_diagnostics()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.viewer_cfg.debug_viz:
            self._draw_debug_vis()

        return env_ids, ()

    # Ege - checks whether each robot is in the goal position
    #       (defined by height, angular position and angular velocity)
    def check_goal_reach(self):
        base_height = self.root_states[:, 2] - self._get_center_heights() #torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        # linear speed of the base (should be as small as possible)
        base_lin_speed = torch.norm(self.root_states[:, 7:10], dim=1)

        base_orientation = self._reward_slope_normal_orientation()

        base_angular_vels = self.root_states[:, 10:13]

        base_xy_distance_traveled = self.xy_drift()

        goal_criteria = torch.ones((6, self.num_envs))

        # binary vector - 1 if robot stays in goal, 0 otherwise
        goal_criteria[0] = base_height <= self.goal_state_cfg.max_height_to_goal_ratio * self.rewards_cfg.base_height_target
        goal_criteria[1] = base_height >= self.goal_state_cfg.min_height_to_goal_ratio * self.rewards_cfg.base_height_target
        #goal_criteria[2] = base_lin_speed <= self.goal_state_cfg.max_speed
        goal_criteria[3] = base_orientation >= self.goal_state_cfg.max_z_deviation
        #goal_criteria[4] = torch.all(torch.abs(base_angular_vels) <= self.goal_state_cfg.max_angular_vel, dim=1)
        goal_criteria[5] = base_xy_distance_traveled <= self.goal_state_cfg.max_xy_distance

        goal_buf = torch.all(goal_criteria, axis=0).to(self.device)
        # set robots' "time in goal" to 0 if they have gotten out of the goal, else don't change
        self.goal_time_buf *= goal_buf

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Ege - disabling reset on contact forces, instead using the goal checking buffer
        #self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # Ege - changed > to >= in the next line.
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        self.reset_buf[:] = self.time_out_buf
        self.termination_buf = torch.logical_and(self.reset_buf, torch.logical_not(self.time_out_buf))
        if self.goal_state_cfg.reset_on_goal:
            self.reset_buf[:] |= self.time_out_buf

    def reset_idx(self, env_ids, log_extras=True):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.terrain_cfg.curriculum:
            self._update_terrain_curriculum(env_ids)
            # Ege - increasing trial buffer
            self.curriculum_trial_buf[env_ids] += 1

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.commands_cfg.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        self._reset_agents(env_ids)
        self._resample_commands(env_ids)

        if self.domain_rand_cfg.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # fill extras
        if log_extras:
            self.extras["episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["episode"][REWARDS_LOG_NAMES[key]] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
                self.episode_sums[key][env_ids] = 0.
            for key in self.diagnostic_sums.keys():
                if key == "feet_air_time_on_contact": # divide by number of contacts over episode
                    diagnostic = self.diagnostic_sums[key][env_ids] / self.contact_count[env_ids]
                    self.contact_count[env_ids] = 0
                else:
                    diagnostic = self.diagnostic_sums[key][env_ids] / self.episode_length_buf[env_ids]
                self.extras["episode"][DIAGNOSTICS_LOG_NAMES[key]] = torch.mean(diagnostic)
                self.diagnostic_sums[key][env_ids] = 0.
            # log additional curriculum info
            if self.terrain_cfg.curriculum:
                self.extras["episode"][DIAGNOSTICS_LOG_NAMES["terrain_level"]] = torch.mean(self.terrain_levels.float())
            if self.commands_cfg.curriculum:
                self.extras["episode"][DIAGNOSTICS_LOG_NAMES["max_command_x"]] = self.command_ranges.lin_vel_x[1]
            # send timeout info to the algorithm
            if self.env_cfg.send_timeouts:
                self.extras["time_outs"] = self.time_out_buf

        # reset buffers
        # Ege - adding goal_time_buf
        for buf in [self.last_actions, self.last_dof_vel, self.last_contacts, self.last_contact_forces,
                    self.last_torques, self.feet_air_time, self.goal_time_buf]:
            buf[env_ids] = torch.zeros_like(buf[env_ids])

        # Ege - rolling success history forward
        self.success_history[env_ids] = torch.roll(self.success_history[env_ids], -1, 1)
        self.success_history[env_ids, -1] = self.success_buf[env_ids]

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    #------------- Callbacks --------------
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state.unsqueeze(0).expand(len(env_ids), -1)
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.custom_origins:
            # x-y position within `pos_noise` meters of the center
            pos_noise = self.init_state_cfg.pos_noise
            self.root_states[env_ids, :2] += torch_utils.torch_rand_float(-pos_noise, pos_noise, (len(env_ids), 2), device=self.device)
        # Ege - setting robot heights based on their position in the terrain, if height samples are defined
        if self.terrain_cfg.mesh_type in ['heightfield', 'trimesh']:
            positions = self.root_states[env_ids, :2].detach().clone()
            positions += self.terrain_cfg.border_size
            positions = (positions/self.terrain_cfg.horizontal_scale).long()
            px = torch.clip(positions[:,0], 0, self.height_samples.shape[0]-2)
            py = torch.clip(positions[:,1], 0, self.height_samples.shape[1]-2)
            height_samples = self.height_samples.float()
            self.root_states[env_ids, 2] = self.base_init_state[2] + height_samples[px, py] * self.terrain_cfg.vertical_scale

        # Ege - randomly initializing root orientation
        if self.domain_rand_cfg.randomize_base_orientation and len(env_ids) > 0:
            root_rot = Rotation.random(len(env_ids))
            uprights = root_rot.apply(np.array([0, 0, 1]))[:, 2] > 0
            if sum(uprights) > 0:
                root_rot[uprights] *= Rotation.from_quat([0,1,0,0])
            self.root_states[env_ids, 3:7] = torch.tensor(root_rot.as_quat(), device=self.device, dtype=torch.float)

        # base velocities
        lin_vel_noise, ang_vel_noise = self.init_state_cfg.lin_vel_noise, self.init_state_cfg.ang_vel_noise
        self.root_states[env_ids, 7:10] =  torch_utils.torch_rand_float(-lin_vel_noise, lin_vel_noise, (len(env_ids), 3), device=self.device)
        self.root_states[env_ids, 10:13] = torch_utils.torch_rand_float(-ang_vel_noise, ang_vel_noise, (len(env_ids), 3), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Ege - resetting start positions, tracked for defining goal states
        self.start_positions[env_ids] = self.root_states[env_ids, :3]

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        # Ege - changing how robots move up/down for testing
        is_moving = self.curriculum_trial_buf >= self.curriculum_cfg.success_rate_mean_window
        # TODO - do we really want to reset curriculum_trial_buf if we are not resetting (is_moving but not in env_ids)?
        self.curriculum_trial_buf[is_moving] = 0
        is_moving = is_moving[env_ids] 
        success_rates = torch.mean(self.success_history[env_ids].type(torch.float), axis=1)
        move_up = success_rates >= self.curriculum_cfg.promotion_success_rate
        move_down = success_rates <= self.curriculum_cfg.demotion_success_rate * ~move_up
        self.terrain_levels[env_ids] += is_moving[env_ids] * (1 * move_up - 1 * move_down)

        #self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    #----------------------------------------
    def _init_buffers(self):
        # Ege - initialize buffers that are specifically helpful to recovery
        # position matrix for each robot's starting position (at that episode)
        self.start_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        # also initializing success buffer (to check whether each agent was successful for that episode)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # TODO: leaving out curriculum stuff for now
        self.success_history = torch.zeros((self.num_envs, self.curriculum_cfg.success_rate_mean_window), device=self.device, dtype=torch.bool)
        self.curriculum_trial_buf = torch.zeros(self.num_envs, device=self.device)
        self.success_per_tile_buf = torch.zeros((self.terrain_cfg.num_rows, self.terrain_cfg.num_cols), device=self.device, dtype=torch.int)
        self.failure_per_tile_buf = torch.zeros((self.terrain_cfg.num_rows, self.terrain_cfg.num_cols), device=self.device, dtype=torch.int)
        self.goal_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        super()._init_buffers()

    def _create_envs(self):
        super()._create_envs()
        self._get_slope_normals()

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.terrain_cfg.max_init_terrain_level if self.terrain_cfg.curriculum else self.terrain_cfg.num_rows-1
            # Ege - implement equal distribution to the specified range of cells
            if self.init_state_cfg.equal_distribution.enabled:
                self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
                self.terrain_types = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
                min_col, max_col = self.init_state_cfg.equal_distribution.col_range
                min_row, max_row = self.init_state_cfg.equal_distribution.row_range
                col_len = max_col + 1 - min_col
                row_len = max_row + 1 - min_row
                terrain_size = row_len * col_len
                for env_number in range(self.num_envs):
                    (i, j) = np.unravel_index((env_number) % terrain_size, (row_len, col_len))
                    self.terrain_levels[env_number] = i + min_row
                    self.terrain_types[env_number] = j + min_col
            else:
                # This part wasn't written by Ege, was outside of the else block originally
                self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
                self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.terrain_cfg.num_cols), rounding_mode='floor').to(torch.long)
            
            self.max_terrain_level = self.terrain_cfg.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = int(np.floor(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_cols))
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.env_cfg.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    # Ege
    def _get_slope_normals(self):
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"] and 'slope_normal' in self.terrain.stats:
            tile_slope_normals = torch.from_numpy(self.terrain.stats['slope_normals']).to(self.device).to(torch.float)
            self.slope_normals = tile_slope_normals[self.terrain_levels, self.terrain_types]
        else:
            slope_normal = torch.tensor([0, 0, 1], device=self.device, requires_grad=False)
            self.slope_normals = slope_normal.unsqueeze(0).repeat(self.num_envs, 1)

    # Ege - samples terrain heights at the xy-position of the centers of robots (specified by env_ids).
    def _get_center_heights(self, env_ids=None):
        if self.terrain_cfg.mesh_type == 'plane':
            result_len = self.num_envs if env_ids is None else len(env_ids)
            return torch.zeros(result_len, device=self.device, requires_grad=False)
        elif self.terrain_cfg.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.root_states[:,:2].detach().clone()
        if env_ids is not None:
            points = points[env_ids]
        points += self.terrain_cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:,0], 0, self.height_samples.shape[0]-2)
        py = torch.clip(points[:,1], 0, self.height_samples.shape[1]-2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights * self.terrain.cfg.vertical_scale
    
    # Ege - calculates distance in the xy-plane from starting position 
    def xy_drift(self, env_ids=None):
        cur_pos, start_pos = self.root_states[:, :2], self.start_positions[:, :2]
        if env_ids is not None:
            cur_pos, start_pos = cur_pos[env_ids], start_pos[env_ids]
        return torch.norm(cur_pos - start_pos, dim=1)

    #---------------------------------------------
    #------------ reward functions----------------
    #---------------------------------------------

    #------------ recovery rewards----------------

    # Ege
    def _reward_z_axis_orientation(self):
        # Reward if orientation close to upright (+z), penalize the opposite.
        rot = Rotation.from_quat(self.base_quat.cpu())
        orientations = rot.apply(np.array([0,0,1]))[:, 2]
        return torch.tensor(orientations).to(self.device)
    
    # Ege
    def _reward_slope_normal_orientation(self):
        # Reward if orientation close to the slope normal in that terrain, penalize the opposite.
        rot = Rotation.from_quat(self.base_quat.cpu())
        orientations = rot.apply(np.array([0,0,1]))
        orientations = torch.tensor(orientations).to(self.device)
        scores = torch.sum(orientations * self.slope_normals, dim=1)
        return scores
        
    # Ege
    def _reward_xy_drift(self):
        # Penalize drifting in xy plane from starting position
        return -torch.square(self.xy_drift())

    # Ege
    def _reward_feet_contact(self):
        # Reward all feet being in contact (hopefully with ground) after recovery
        return torch.all(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 0, dim=1)