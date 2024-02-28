from typing import Tuple
import numpy as np
import os
import logging
from omegaconf import OmegaConf

from isaacgym import torch_utils
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs.base_env import BaseEnv
from legged_gym.utils.gamepad import Gamepad
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.legmath import quat_apply_yaw, wrap_to_pi
import legged_gym.utils.kinematics.urdf as pk
from configs.definitions import (EnvConfig, ObservationConfig, TerrainConfig,
                                 CommandsConfig, InitStateConfig, ControlConfig,
                                 AssetConfig, DomainRandConfig, RewardsConfig,
                                 NormalizationConfig, NoiseConfig, ViewerConfig,
                                 SimConfig)
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

class Hound(BaseEnv):
    def __init__(self, env: EnvConfig, observation: ObservationConfig, terrain: TerrainConfig,
                 commands: CommandsConfig, init_state: InitStateConfig, control: ControlConfig,
                 asset: AssetConfig, domain_rand: DomainRandConfig, rewards: RewardsConfig,
                 normalization: NormalizationConfig, noise: NoiseConfig, viewer: ViewerConfig,
                 sim: SimConfig):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes PyTorch buffers used during training

        Args:
            cfg (Dict): Environment config file
        """ 
        self.env_cfg = env
        self.observation_cfg = observation
        self.terrain_cfg = terrain
        self.commands_cfg = commands
        self.init_state_cfg = init_state
        self.control_cfg = control
        self.asset_cfg = asset
        self.domain_rand_cfg = domain_rand
        self.rewards_cfg = rewards
        self.normalization_cfg = normalization
        self.noise_cfg = noise
        self.viewer_cfg = viewer
        self.sim_cfg = sim
        super().__init__(env, observation, sim)
        self.init_done = False

        self.chain_ee = []
        for ee_name in ["front_right_wheel", "front_left_wheel", "back_right_wheel", "back_left_wheel"]:
            self.chain_ee.append(
                pk.build_serial_chain_from_urdf(
                    open(self.asset_cfg.file).read(), ee_name).to(device=self.device))

        self._get_commands_from_joystick = self.observation_cfg.get_commands_from_joystick
        if self._get_commands_from_joystick:
            self.gamepad = Gamepad(self.command_ranges)

        if not self.headless:
            self.set_camera(self.viewer_cfg.pos, self.viewer_cfg.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_diagnostics()
        self.init_done = True

    def reset(self):
        """ Reset all robots. Additionally, if we use a history of observations,
        reset history of observations to all be the current observation.
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device), log_extras=False)
        self.obs_buf_history.reset(
            torch.arange(self.num_envs, device=self.device),
            self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, critic_obs, *_ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, critic_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
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

    def get_observations(self):
        return self.obs_buf_history.get_obs_vec(np.arange(self.history_steps))

    def get_critic_observations(self):
        policy_obs = self.get_observations()
        return torch.cat((policy_obs, self.critic_obs_buf[..., self.num_obs:]), dim=-1)

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

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.termination_buf = torch.logical_and(self.reset_buf, torch.logical_not(self.time_out_buf))

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
        for buf in [self.last_actions, self.last_dof_vel, self.last_contacts, self.last_contact_forces,
                    self.last_torques, self.feet_air_time]:
            buf[env_ids] = torch.zeros_like(buf[env_ids])
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _reset_agents(self, env_ids):
        """
        Reset the states of the agents that need to be reseted.
        """
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

    def _compute_physical_measurements(self):
        """Fills buffers corresponding to various physical quantities.
        """
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = torch_utils.quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = torch_utils.quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = torch_utils.quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_heights[:] = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.dof_accel[:] = (self.last_dof_vel - self.dof_vel) / self.dt
        self.action_change[:] = self.actions - self.last_actions
        self.collisions[:] = torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        self.stumbles[:] = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
                            > 5*torch.abs(self.contact_forces[:, self.feet_indices, 2]))
        self.foot_contact_forces[:] = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        self.foot_contact_force_change[:] = torch.norm(self.contact_forces[:, self.feet_indices] - self.last_contact_forces[:, self.feet_indices], dim=-1)
        self.dof_pos_out_of_limits[:] = ((self.dof_pos-self.soft_dof_pos_limits[:, 1]).clip(min=0.)
                                         -(self.dof_pos-self.soft_dof_pos_limits[:, 0]).clip(max=0.))
        self.dof_vel_out_of_limits[:] = torch.clip(torch.abs(self.dof_vel) - self.dof_vel_limits*self.rewards_cfg.soft_dof_vel_limit, min=0.)
        self.torque_out_of_limits[:] = torch.clip(torch.abs(self.torques) - self.torque_limits*self.rewards_cfg.soft_torque_limit, min=0.)
        self.foot_contact_forces_out_of_limits[:] = torch.clip(self.foot_contact_forces - self.rewards_cfg.max_contact_force, min=0.)

        # power
        mechanical_power = self.torques*self.dof_vel
        heat = self.asset_cfg.motor_parameter*self.torques**2
        self.power[:] = torch.sum(torch.clip(mechanical_power+heat, min=0.), dim=-1)

        # cost of transport
        gravity = np.linalg.norm(self.sim_cfg.gravity)
        speed = torch.norm(self.base_lin_vel, dim=-1) + 1e-6
        self.cost_of_transport[:] = self.power / (self.robot_mass*gravity*speed)

        # feet air time
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contacts[:] = self.contact_forces[:, self.feet_indices, 2] > self.asset_cfg.contact_force_threshold
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        self.last_contacts[:] = self.contacts
        self.first_contacts[:] = torch.logical_and(self.feet_air_time > 0., contact_filt)
        self.feet_air_time += self.dt
        air_time_offsetted = self.feet_air_time - self.rewards_cfg.foot_air_time_threshold
        self.rew_air_time_on_contact[:] = torch.sum(air_time_offsetted * self.first_contacts, dim=1) # reward only on first contact with ground
        self.rew_air_time_on_contact *= (torch.norm(self.commands[:, :2], dim=1) > 0.1) # no reward for near-zero command
        self.air_time_on_contact[:] = torch.sum(self.feet_air_time * self.first_contacts, dim=1)
        self.contact_count[:] += self.first_contacts.sum(dim=1)
        self.feet_air_time *= torch.logical_not(contact_filt)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.rewards_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_diagnostics(self):
        """ Compute diagnostics
            Calls each diagnostic function, adds each terms to the diagnostic sums
        """
        for i in range(len(self.diagnostic_functions)):
            name = self.diagnostic_names[i]
            self.diagnostic_sums[name] += self.diagnostic_functions[i]()

    def compute_observations(self):
        """ Computes observations
        """

        if self._get_commands_from_joystick:
            lin_vel_x, lin_vel_y, ang_vel, right_bump = self.gamepad.get_command()
            self.commands[:, 0] = lin_vel_x
            self.commands[:, 1] = lin_vel_y
            self.commands[:, 2] = ang_vel

        sensors = self.observation_cfg.sensors + self.observation_cfg.critic_privileged_sensors
        self.critic_obs_buf = self._compute_observations(sensors)
        # add noise if needed
        if self.add_noise:
            obs = self.critic_obs_buf[..., :self.num_obs]
            self.critic_obs_buf[..., :self.num_obs] += (2*torch.rand_like(obs) - 1) * self.noise_scale_vec
        self.obs_buf = self.critic_obs_buf[:, :self.num_obs].clone()

    def _compute_observations(self, sensors: Tuple[str], normalize=True):
        obs_list = []
        for sensor in sensors:
            if sensor == "base_ang_vel":
                scale = self.obs_scales.ang_vel if normalize else 1.
                obs_list.append(self.base_ang_vel * scale)
            elif sensor == "base_lin_vel":
                scale = self.obs_scales.lin_vel if normalize else 1.
                obs_list.append(self.base_lin_vel * scale)
            elif sensor == "base_mass":
                scale = self.obs_scales.base_mass if normalize else 1.
                obs_list.append(self.base_mass.unsqueeze(1) * scale)
            elif sensor == "commands":
                scale = self.commands_scale if normalize else 1.
                obs_list.append(self.commands[..., :self.commands_cfg.num_commands] * scale)
            elif sensor == "d_gain":
                d_gain = (self.randomized_d_gains
                          if self.domain_rand_cfg.randomize_gains
                          else self.d_gains.unsqueeze(0).expand(self.num_envs, -1))
                scale = self.obs_scales.damping if normalize else 1.
                obs_list.append(d_gain * scale)
            elif sensor == "foot_pos":
                obs_list.append(self.foot_positions_in_base_frame(self.dof_pos))
            elif sensor == "friction":
                obs_list.append(self.friction_coeffs.unsqueeze(1))
            elif sensor == "last_action":
                scale = self.obs_scales.last_action if normalize else 1.
                obs_list.append(self.actions.clone() * scale)
            elif sensor == "motor_pos":
                scale = self.obs_scales.dof_pos if normalize else 1.
                obs_list.append((self.dof_pos-self.default_dof_pos) * scale)
            elif sensor == "motor_pos_unshifted":
                scale = self.obs_scales.dof_pos if normalize else 1.
                obs_list.append(self.dof_pos * scale)
            elif sensor == "motor_vel":
                scale = self.obs_scales.dof_vel if normalize else 1.
                obs_list.append(self.dof_vel * scale)
            elif sensor == "projected_gravity":
                obs_list.append(self.projected_gravity)
            elif sensor == "p_gain":
                p_gain = (self.randomized_p_gains
                          if self.domain_rand_cfg.randomize_gains
                          else self.p_gains.unsqueeze(0).expand(self.num_envs, -1))
                scale = self.obs_scales.stiffness if normalize else 1.
                obs_list.append(p_gain * scale)
            elif sensor == "terrain_height":
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)
                if normalize:
                    heights *= self.obs_scales.height_measurements
                obs_list.append(heights)
            elif sensor == "yaw_rate":
                scale = self.obs_scales.ang_vel if normalize else 1.
                obs_list.append(self.base_ang_vel[..., 2:3] * scale)
            elif sensor == "z_pos":
                obs_list.append(self.root_states[..., 2:3])
            else:
                raise ValueError(f"Sensor not recognized: {sensor}")

        obs = torch.cat(obs_list, dim=-1)
        return obs


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.terrain_cfg.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield': 
            log.warn('Terrain created using old heightfield API.')
            self._create_heightfield()
        elif mesh_type=='trimesh':
            log.warn('Terrain created using old trimesh API.')
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    
    def exit(self):
        self.gym.destroy_sim(self.sim)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.domain_rand_cfg.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.domain_rand_cfg.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
                friction_buckets = torch_utils.torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu').squeeze(1)
                self.friction_coeffs[:] = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        else:
            self.friction_coeffs[env_id] = props[0].friction
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.rewards_cfg.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.rewards_cfg.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.domain_rand_cfg.randomize_base_mass:
            rng = self.domain_rand_cfg.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        self.base_mass[env_id] = props[0].mass
        self.robot_mass[env_id] = sum([prop.mass for prop in props])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.commands_cfg.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.commands_cfg.heading_command:
            forward = torch_utils.quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        self.measured_heights = self._get_heights()
        if self.domain_rand_cfg.push_robots and  (self.common_step_counter % self.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_utils.torch_rand_float(self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_utils.torch_rand_float(self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.commands_cfg.heading_command:
            self.commands[env_ids, 3] = torch_utils.torch_rand_float(self.command_ranges.heading[0], self.command_ranges.heading[1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_utils.torch_rand_float(self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.control_cfg.action_scale
        control_type = self.control_cfg.control_type

        if self.domain_rand_cfg.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            desired_pos = actions_scaled + self.default_dof_pos
            if self.control_cfg.clip_setpoint:
                desired_pos.clip_(self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            torques = p_gains*(desired_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_utils.torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = torch.zeros_like(self.dof_vel[env_ids])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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

        # base velocities
        lin_vel_noise, ang_vel_noise = self.init_state_cfg.lin_vel_noise, self.init_state_cfg.ang_vel_noise
        self.root_states[env_ids, 7:10] =  torch_utils.torch_rand_float(-lin_vel_noise, lin_vel_noise, (len(env_ids), 3), device=self.device)
        self.root_states[env_ids, 10:13] = torch_utils.torch_rand_float(-ang_vel_noise, ang_vel_noise, (len(env_ids), 3), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.domain_rand_cfg.max_push_vel_xy
        self.root_states[:, 7:9] = torch_utils.torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges.lin_vel_x[0] = np.clip(self.command_ranges.lin_vel_x[0] - 0.5, -self.commands_cfg.max_curriculum, 0.)
            self.command_ranges.lin_vel_x[1] = np.clip(self.command_ranges.lin_vel_x[1] + 0.5, 0., self.commands_cfg.max_curriculum)


    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self.add_noise = self.noise_cfg.add_noise

        noise_scale_list = []
        noise_scales = self.noise_cfg.noise_scales
        noise_level = self.noise_cfg.noise_level
        for sensor in self.observation_cfg.sensors:
            if sensor == "base_ang_vel":
                sensor_noise_level = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                noise_scale_list.append(torch.full((self.sensor_dims["base_ang_vel"],), sensor_noise_level, device=self.device))
            elif sensor == "base_lin_vel":
                sensor_noise_level = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                noise_scale_list.append(torch.full((self.sensor_dims["base_lin_vel"],), sensor_noise_level, device=self.device))
            elif sensor == "base_mass":
                noise_scale_list.append(torch.zeros(self.sensor_dims["base_mass"], device=self.device))
            elif sensor == "commands":
                noise_scale_list.append(torch.zeros(self.commands_cfg.num_commands, device=self.device))
            elif sensor == "d_gain":
                noise_scale_list.append(torch.zeros(self.sensor_dims["d_gain"], device=self.device))
            elif sensor == "foot_positions":
                noise_scale_list.append(torch.zeros(self.sensor_dims["foot_positions"], device=self.device))
            elif sensor == "friction":
                noise_scale_list.append(torch.zeros(self.sensor_dims["friction"], device=self.device))
            elif sensor == "last_action":
                noise_scale_list.append(torch.zeros(self.sensor_dims["last_action"], device=self.device))
            elif sensor == "motor_pos":
                sensor_noise_level = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                noise_scale_list.append(torch.full((self.sensor_dims["motor_pos"],), sensor_noise_level, device=self.device))
            elif sensor == "motor_vel":
                sensor_noise_level = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                noise_scale_list.append(torch.full((self.sensor_dims["motor_vel"],), sensor_noise_level, device=self.device))
            elif sensor == "projected_gravity":
                sensor_noise_level = noise_scales.gravity * noise_level
                noise_scale_list.append(torch.full((self.sensor_dims["projected_gravity"],), sensor_noise_level, device=self.device))
            elif sensor == "p_gain":
                noise_scale_list.append(torch.zeros(self.sensor_dims["p_gain"], device=self.device))
            elif sensor == "terrain_height":
                height_noise_level = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
                noise_scale_list.append(torch.full((self.sensor_dims["terrain_height"],), height_noise_level, device=self.device))
            elif sensor == "yaw_rate":
                sensor_noise_level = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                noise_scale_list.append(torch.full((self.sensor_dims["yaw_rate"],), sensor_noise_level, device=self.device))
        return torch.cat(noise_scale_list)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        make_zero_tensor = lambda *size, dtype=torch.float: torch.zeros(*size, dtype=dtype, device=self.device, requires_grad=False)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = torch_utils.to_torch(torch_utils.get_axis_params(-1., self.up_axis_idx, dtype=float), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch_utils.to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = make_zero_tensor(self.num_envs, self.num_actions)
        self.p_gains = make_zero_tensor(self.num_actions)
        self.d_gains = self.p_gains.clone()
        self.actions = make_zero_tensor(self.num_envs, self.num_actions)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_contacts = make_zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.contacts = make_zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.first_contacts = make_zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.contact_count = make_zero_tensor(self.num_envs, dtype=torch.int)
        self.commands = make_zero_tensor(self.num_envs, 4) # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = make_zero_tensor(self.num_envs, self.feet_indices.shape[0])
        self.base_lin_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = torch_utils.quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_heights = make_zero_tensor(self.num_envs)
        self.dof_accel = torch.zeros_like(self.dof_vel)
        self.action_change = torch.zeros_like(self.actions)
        self.collisions = make_zero_tensor(self.num_envs, dtype=torch.bool)
        self.stumbles = make_zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.foot_contact_forces = make_zero_tensor(self.num_envs, len(self.feet_indices))
        self.foot_contact_force_change = torch.zeros_like(self.foot_contact_forces)
        self.dof_pos_out_of_limits = torch.zeros_like(self.dof_pos)
        self.dof_vel_out_of_limits = torch.zeros_like(self.dof_vel)
        self.torque_out_of_limits = torch.zeros_like(self.torques)
        self.foot_contact_forces_out_of_limits = torch.zeros_like(self.foot_contact_forces)
        self.power = make_zero_tensor(self.num_envs)
        self.cost_of_transport = make_zero_tensor(self.num_envs)
        self.air_time_on_contact = make_zero_tensor(self.num_envs)
        self.rew_air_time_on_contact = make_zero_tensor(self.num_envs)
        if self.terrain_cfg.measured_points_x is not None and self.terrain_cfg.measured_points_y is not None:
            self.height_points = self._init_height_points()
        self.measured_heights = torch.tensor(0., device=self.device)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.init_state_cfg.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.control_cfg.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.control_cfg.stiffness[dof_name]
                    self.d_gains[i] = self.control_cfg.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.control_cfg.control_type in ["P", "V"]:
                    log.info(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.domain_rand_cfg.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

    def compute_randomized_gains(self, num_envs):
        added_stiffness_range = self.domain_rand_cfg.added_stiffness_range
        added_damping_range = self.domain_rand_cfg.added_damping_range
        p_shift = ((added_stiffness_range[0]-added_stiffness_range[1])*torch.rand(num_envs, self.num_actions, device=self.device)
                   + added_stiffness_range[1])
        d_shift = ((added_damping_range[0]-added_damping_range[1])*torch.rand(num_envs, self.num_actions, device=self.device)
                   + added_damping_range[1])

        return self.p_gains + p_shift, self.d_gains + d_shift

    def _foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        leg_distance = torch.sqrt(THIGH_LENGTH**2 + CALF_LENGTH**2 +
                                2 * THIGH_LENGTH * CALF_LENGTH * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = HIP_LENGTH * l_hip_sign

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        if self.observation_cfg.fast_compute_foot_pos: # about 12x faster with a foot position of 1e-5 meters
            foot_positions = torch.zeros_like(foot_angles)
            for i in range(4):
                foot_positions[:, 3*i:3*i+3].copy_(
                    self._foot_position_in_hip_frame(foot_angles[:, 3*i:3*i+3], l_hip_sign=(-1)**(i+1))
                )
            foot_positions = foot_positions + HIP_OFFSETS.to(self.device)
        else:
            foot_positions = []
            with torch.no_grad():
                for i, chain_ee in enumerate(self.chain_ee):
                    foot_positions.append(
                        chain_ee.forward_kinematics(foot_angles[:, 3*i:3*i+3]).get_matrix()[:, :3, 3]
                    )
            foot_positions = torch.cat(foot_positions, dim=-1)

        return foot_positions

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

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _prepare_diagnostics(self):
        """
        Prepares a list of diagnostics. Looks for self._diagnostic_<DIAGNOSTIC_NAME>.
        """
        self.diagnostic_names = []
        self.diagnostic_functions = []
        for function_name in [name for name in dir(self) if name.startswith("_diagnostic_")]:
            self.diagnostic_names.append(function_name[len("_diagnostic_"):])
            self.diagnostic_functions.append(getattr(self, function_name))
        self.diagnostic_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.diagnostic_names}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_cfg.static_friction
        plane_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        plane_params.restitution = self.terrain_cfg.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain_cfg.horizontal_scale
        hf_params.row_scale = self.terrain_cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain_cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain_cfg.border_size
        hf_params.transform.p.y = -self.terrain_cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.terrain_cfg.static_friction
        hf_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        hf_params.restitution = self.terrain_cfg.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_cfg.static_friction
        tm_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        tm_params.restitution = self.terrain_cfg.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_root = os.path.dirname(self.asset_cfg.file)
        asset_file = os.path.basename(self.asset_cfg.file)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.asset_cfg.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.asset_cfg.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.asset_cfg.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.asset_cfg.flip_visual_attachments
        asset_options.fix_base_link = self.asset_cfg.fix_base_link
        asset_options.density = self.asset_cfg.density
        asset_options.angular_damping = self.asset_cfg.angular_damping
        asset_options.linear_damping = self.asset_cfg.linear_damping
        asset_options.max_angular_velocity = self.asset_cfg.max_angular_velocity
        asset_options.max_linear_velocity = self.asset_cfg.max_linear_velocity
        asset_options.armature = self.asset_cfg.armature
        asset_options.thickness = self.asset_cfg.thickness
        asset_options.disable_gravity = self.asset_cfg.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.asset_cfg.foot_name in s]
        penalized_contact_names = []
        for name in self.asset_cfg.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.asset_cfg.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self.base_mass = torch.zeros(self.num_envs, device=self.device)
        self.robot_mass = torch.zeros(self.num_envs, device=self.device)
        self.friction_coeffs = torch.zeros(self.num_envs, device=self.device)

        base_init_state_list = self.init_state_cfg.pos + self.init_state_cfg.rot + self.init_state_cfg.lin_vel + self.init_state_cfg.ang_vel
        self.base_init_state = torch_utils.to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_utils.torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            a1_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "a1", i, int(not self.asset_cfg.self_collisions), 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, a1_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, a1_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, a1_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(a1_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.terrain_cfg.max_init_terrain_level if self.terrain_cfg.curriculum else self.terrain_cfg.num_rows-1
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

    def _parse_cfg(self):
        #TODO: remove theneed to 'parse_cfg'
        self.dt = self.control_cfg.decimation * self.sim_params.dt
        self.obs_scales = self.normalization_cfg.obs_scales
        self.reward_scales = OmegaConf.to_container(self.rewards_cfg.scales) # to allow for keys to be `pop`'d
        self.command_ranges = self.commands_cfg.ranges
        if self.terrain_cfg.mesh_type not in ['heightfield', 'trimesh']:
            self.terrain_cfg.curriculum = False
        self.max_episode_length_s = self.env_cfg.episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.sensor_dims = dict(
            base_ang_vel=3,
            base_lin_vel=3,
            base_mass=1,
            commands=self.commands_cfg.num_commands,
            d_gain=12,
            foot_pos=12,
            friction=1,
            last_action=12,
            motor_pos=12,
            motor_pos_unshifted=12,
            motor_vel=12,
            projected_gravity=3,
            p_gain=12,
            terrain_height=len(self.terrain_cfg.measured_points_x)*len(self.terrain_cfg.measured_points_y),
            yaw_rate=1,
            z_pos=1
        )

        if self.domain_rand_cfg.push_robots:
            self.push_interval = float(np.ceil(self.domain_rand_cfg.push_interval_s / self.dt))

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if self.terrain_cfg.measured_points_x is None or self.terrain_cfg.measured_points_y is None:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.terrain_cfg.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.terrain_cfg.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.terrain_cfg.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.terrain_cfg.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #---------------------------------------------
    #------------ reward functions----------------
    #---------------------------------------------

    #------------ action rewards----------------
    def _reward_action(self):
        # Penalize magnitude of actions
        return -torch.sum(torch.square(self.actions), dim=1)

    def _reward_action_change(self):
        # Penalize changes in actions
        return -torch.sum(torch.square(self.action_change), dim=1)

    #------------ base link rewards----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return -torch.square(self.base_heights - self.rewards_cfg.base_height_target)

    #------------ joints rewards----------------
    def _reward_torques(self):
        # Penalize torques
        return -torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return -torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_accel(self):
        # Penalize dof accelerations
        return -torch.sum(torch.square(self.dof_accel), dim=1)

    def _reward_soft_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        return -torch.sum(self.dof_pos_out_of_limits, dim=1)

    def _reward_soft_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return -torch.sum(self.dof_vel_out_of_limits.clip(min=0., max=1.), dim=1)

    def _reward_soft_torque_limits(self):
        # penalize torques too close to the limit
        return -torch.sum(self.torque_out_of_limits.clip(min=0.), dim=1)

    #------------ energy rewards----------------
    def _reward_power(self):
        # Penalize power consumption (mechanical + heat)
        return -self.power

    def _reward_cost_of_transport(self):
        # Penalize cost of transport (power / (weight * speed))
        return -self.cost_of_transport

    #------------ feet rewards----------------
    def _reward_feet_air_time(self):
        # Reward long steps
        return self.rew_air_time_on_contact

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return -torch.sum(self.foot_contact_forces_out_of_limits, dim=1)

    def _reward_feet_contact_force_change(self):
        # penalize foot jerk to prevent large motor backlash
        return -torch.sum(self.foot_contact_force_change.square(), dim=1)

    #------------ safety rewards----------------
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return -self.collisions.float()

    def _reward_termination(self):
        # Terminal reward / penalty
        return -self.termination_buf.float()

    def _reward_alive(self):
        return (1-self.termination_buf.float())

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return -torch.any(self.stumbles, dim=1).float()

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return -torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    #------------ task rewards----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.rewards_cfg.tracking_sigma**2)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.rewards_cfg.tracking_sigma**2)

    #-------------------------------------------------
    #------------ diagnostic functions----------------
    #-------------------------------------------------

    #------------ action diagnostics ----------------
    def _diagnostic_action(self):
        return torch.mean(self.actions.abs(), dim=1)

    def _diagnostic_action_change(self):
        return torch.mean(self.action_change.abs(), dim=1)

    #------------ base link diagnostics ----------------
    def _diagnostic_lin_vel_z(self):
        return self.base_lin_vel[:, 2].abs()

    def _diagnostic_ang_vel_xy(self):
        return torch.sum(self.base_ang_vel[:, :2].abs(), dim=1)

    def _diagnostic_nonflat_orientation(self):
        return torch.sum(self.projected_gravity[:, :2].abs(), dim=1)

    def _diagnostic_base_height(self):
       return self.base_heights

    #------------ energy diagnostics ----------------
    def _diagnostic_power(self):
        return self.power

    def _diagnostic_cost_of_transport(self):
        return self.cost_of_transport

    #------------ feet diagnostics ----------------
    def _diagnostic_feet_air_time_on_contact(self):
        return self.air_time_on_contact

    def _diagnostic_feet_contact_forces(self):
        return torch.mean(self.foot_contact_forces, dim=1)

    def _diagnostic_feet_contact_forces_out_of_limits(self):
        return torch.mean(self.foot_contact_forces_out_of_limits, dim=1)

    def _diagnostic_feet_contact_force_change(self):
        return torch.mean(self.foot_contact_force_change, dim=1)

    #------------ joints diagnostics ----------------
    def _diagnostic_torques(self):
        return torch.mean(self.torques.abs(), dim=1)

    def _diagnostic_dof_vel(self):
        return torch.mean(self.dof_vel.abs(), dim=1)

    def _diagnostic_dof_accel(self):
        return torch.mean(self.dof_accel.abs(), dim=1)

    def _diagnostic_dof_pos_out_of_limits(self):
        return torch.mean(self.dof_pos_out_of_limits, dim=1)

    def _diagnostic_dof_vel_out_of_limits(self):
        return torch.mean(self.dof_vel_out_of_limits, dim=1)

    def _diagnostic_torque_out_of_limits(self):
        return torch.mean(self.torque_out_of_limits, dim=1)

    #------------ safety diagnostics ----------------
    def _diagnostic_collision(self):
        return self.collisions.float()

    def _diagnostic_termination(self):
        return self.termination_buf.float()

    def _diagnostic_stumble(self):
        return torch.sum(self.stumbles, dim=1).float()
