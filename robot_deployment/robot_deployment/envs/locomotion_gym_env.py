import gymnasium as gym
from typing import Tuple
import numpy as np
import os.path as osp
import pybullet
from pybullet_utils import bullet_client
import time

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.gamepad import Gamepad
from configs.definitions import DeploymentConfig, NormalizationConfig, CommandsConfig
from robot_deployment.robots.motors import MotorControlMode
from robot_deployment.robots.motors import MotorCommand
from robot_deployment.robots import a1
from robot_deployment.robots import a1_robot


class LocomotionGymEnv(gym.Env):
    """The gym environment for the locomotion tasks."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }
    pybullet_client: bullet_client.BulletClient

    def __init__(
        self,
        config: DeploymentConfig,
        sensors: Tuple[str, ...],
        obs_scales: NormalizationConfig.NormalizationObsScalesConfig,
        command_ranges: CommandsConfig.CommandRangesConfig
    ):
        # set instance variables from arguments
        self.config = config
        self.obs_scales = obs_scales
        self.use_real_robot = config.use_real_robot
        self.get_commands_from_joystick = config.get_commands_from_joystick
        self.command_ranges = command_ranges
        self.sensors = sensors
        self.hard_reset = True
        self.last_frame_time = 0.
        self.env_time_step = self.config.timestep * self.config.action_repeat

        self._setup_robot()
        self.default_motor_angles = self.robot.motor_group.init_positions
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        if self.get_commands_from_joystick:
            self.gamepad = Gamepad(self.command_ranges)
            self.commands = np.array([0., 0., 0.])

    def _setup_robot(self):
        # make the simulator instance
        connection_mode = pybullet.GUI if self.config.render.show_gui and not self.use_real_robot else pybullet.DIRECT
        self.pybullet_client = bullet_client.BulletClient(connection_mode=connection_mode)
        self._reset_sim()

        # construct robot
        robot_ctor = a1_robot.A1Robot if self.use_real_robot else a1.A1
        self.robot = robot_ctor(
            pybullet_client=self.pybullet_client,
            sim_conf=self.config,
            motor_control_mode=MotorControlMode.POSITION
        )

        if self.config.render.show_gui and not self.use_real_robot:
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RENDERING, 1)

        self.clock = lambda: self.robot.time_since_reset
        self.last_action: np.ndarray = None
        self.timesteps = None

    def reset(self):
        if self.hard_reset:
            # clear the simulation world and rebuild the robot interface
            self._reset_sim()

        self.robot.reset(hard_reset=self.hard_reset)
        self.last_action = np.zeros(self.action_space.shape)
        self.timesteps = 0
        if self.config.render.show_gui and not self.use_real_robot:
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RENDERING, 1)

        self.current_time = time.time()
        return self.get_observation(), self.get_full_observation()

    def step(self, action):
        """Step forward the environment, given the action.

        action: 12-dimensional NumPy array of desired motor angles
        """
        clipped_action = np.clip(
            action,
            self.robot.motor_group.min_positions,
            self.robot.motor_group.max_positions
        )
        motor_action = MotorCommand(
            desired_position=clipped_action,
            kp=self.robot.motor_group.kps,
            kd=self.robot.motor_group.kds
        )
        self.robot.step(motor_action)
        if self.config.render.show_gui:
            if not self.use_real_robot:
                duration = time.time() - self.current_time
                if duration < self.robot.control_timestep:
                    time.sleep(self.robot.control_timestep - duration)
                self.current_time = time.time()
            yaw = self.config.render.camera_yaw
            if not self.config.render.fix_camera_yaw:
                yaw += np.degrees(self.robot.base_orientation_rpy[2])
            self.pybullet_client.resetDebugVisualizerCamera(
                cameraDistance=self.config.render.camera_dist,
                cameraYaw=yaw,
                cameraPitch=self.config.render.camera_pitch,
                cameraTargetPosition=self.robot.base_position
            )

        terminated = not self.is_safe
        self.last_action = clipped_action
        self.timesteps += 1
        return self.get_observation(), 0, terminated, False, self.get_full_observation()

    def render(self):
        view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.robot.base_position,
            distance=self.config.render.camera_dist,
            yaw=self.config.render.camera_yaw,
            pitch=self.config.render.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = self.pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.config.render.render_width/self.config.render.render_height,
            nearVal=0.1,
            farVal=100.
        )
        _, _, rgba, _, _ = self.pybullet_client.getCameraImage(
            width=self.config.render.render_width,
            height=self.config.render.render_height,
            renderer=self.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        rgb_array = np.array(rgba)[:, :, :3]
        return rgb_array

    def close(self):
        self.pybullet_client.disconnect()

    def get_observation(self):
        obs_list = []
        for sensor in self.sensors:
            if sensor == "base_ang_vel":
                obs_list.append(self.robot.base_angular_velocity_in_base_frame * self.obs_scales.ang_vel)
            elif sensor == "yaw_rate":
                obs_list.append(self.robot.base_angular_velocity_in_base_frame[[2]] * self.obs_scales.ang_vel)
            elif sensor == "commands":
                if self.get_commands_from_joystick:
                    lin_vel_x, lin_vel_y, ang_vel, right_bump = self.gamepad.get_command()
                    self.commands = np.array([lin_vel_x, lin_vel_y, ang_vel])
                else:
                    raise ValueError("no joystick (or other input) available for commands")
                multiplier = np.array([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel])
                obs_list.append(self.commands * multiplier)
            elif sensor == "motor_pos":
                obs_list.append((self.robot.motor_angles - self.default_motor_angles) * self.obs_scales.dof_pos)
            elif sensor == "motor_vel":
                obs_list.append(self.robot.motor_velocities * self.obs_scales.dof_vel)
            elif sensor == "projected_gravity":
                _, inv_base_orientation = self.pybullet_client.invertTransform(
                    [0, 0, 0], self.robot.base_orientation_quat
                )
                projected_gravity = self.pybullet_client.multiplyTransforms(
                    [0, 0, 0], inv_base_orientation, [0, 0, -1], [0, 0, 0, 1]
                )[0]
                obs_list.append(projected_gravity)
            elif sensor == "last_action":
                obs_list.append((self.last_action - self.default_motor_angles) / self.config.action_scale * self.obs_scales.last_action)
            else:
                raise ValueError(f"Sensor not recognized: {sensor}")

        return np.concatenate(obs_list)

    def get_full_observation(self):
        obs_dict = dict(
            base_angular_velocity_in_base_frame=self.robot.base_angular_velocity_in_base_frame,
            base_position=self.robot.base_position,
            base_orientation_quat=self.robot.base_orientation_quat,
            base_orientation_rpy=self.robot.base_orientation_rpy,
            base_velocity=self.robot.base_velocity,
            base_velocity_in_base_frame=self.robot.base_velocity_in_base_frame,
            foot_contact=self.robot.foot_contacts,
            foot_contact_history=self.robot.foot_contact_history,
            foot_position=self.robot.foot_positions_in_base_frame,
            foot_velocity=self.robot.foot_velocities_in_base_frame,
            motor_angle=self.robot.motor_angles,
            motor_torque=self.robot.motor_torques,
            motor_temperature=self.robot.motor_temperatures,
            motor_velocity=self.robot.motor_velocities,
        )
        return obs_dict

    @property
    def is_safe(self):
        # done
        rot_mat = np.array(
            self.pybullet_client.getMatrixFromQuaternion(self.robot.base_orientation_quat)
        ).reshape((3, 3))
        up_vec = rot_mat[2, 2]
        base_height = self.robot.base_position[2]
        return up_vec > 0.5 and base_height > 0.05

    def _reset_sim(self):
        self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RENDERING, 0)
        self.pybullet_client.resetSimulation()
        self.pybullet_client.setAdditionalSearchPath(osp.join(LEGGED_GYM_ROOT_DIR, 'resources'))
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.config.num_solver_iterations)
        self.pybullet_client.setTimeStep(self.config.timestep)
        self.pybullet_client.setGravity(0, 0, -9.81)
        self.pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

        # ground
        self.ground_id = self.pybullet_client.loadURDF('plane.urdf')
        #self.pybullet_client.changeDynamics(self.ground_id, -1, restitution=0.5)
        #self.pybullet_client.changeDynamics(self.ground_id, -1, lateralFriction=0.5)

    def _build_observation_space(self):
        # TODO
        pass

    def _build_action_space(self):
        """Builds action space corresponding to joint position control"""
        return gym.spaces.Box(
            self.robot.motor_group.min_positions,
            self.robot.motor_group.max_positions
        )
