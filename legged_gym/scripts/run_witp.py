import gymnasium as gym
from typing import Tuple
import numpy as np
import os.path as osp
import pybullet
from pybullet_utils import bullet_client
import time
import jax

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
# from pydantic import TypeAdapter

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.gamepad import Gamepad
from configs.definitions import DeploymentConfig, NormalizationConfig, CommandsConfig
from robot_deployment.robots.motors import MotorControlMode
from robot_deployment.robots.motors import MotorCommand
from robot_deployment.robots import a1
from robot_deployment.robots import a1_robot
from dataclasses import dataclass, field

from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig)
from configs.definitions import (ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig)
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig

from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)

# from witp.rl.agents.agent import Agent
from witp.rl.agents import SACLearner
from witp.configs.droq_config import get_config
from witp.rl.data import ReplayBuffer

from flax.training import checkpoints
import orbax.checkpoint 
from flax.training import orbax_utils
import flax


OmegaConf.register_new_resolver("not", lambda b: not b)
OmegaConf.register_new_resolver("compute_timestep", lambda dt, decimation, action_repeat: dt * decimation / action_repeat)

INIT_JOINT_ANGLES = {
    "1_FR_hip_joint": 0.05,
    "1_FR_thigh_joint": 0.7,
    "1_FR_calf_joint": -1.4,

    "2_FL_hip_joint": 0.05,
    "2_FL_thigh_joint": 0.7,
    "2_FL_calf_joint": -1.4,

    "3_RR_hip_joint": 0.05,
    "3_RR_thigh_joint": 0.7,
    "3_RR_calf_joint": -1.4,

    "4_RL_hip_joint": 0.05,
    "4_RL_thigh_joint": 0.7,
    "4_RL_calf_joint": -1.4
}

@dataclass
class DeployWITPConfig:
    checkpoint_root: str = from_repo_root("../experiment_logs/train")
    logging_root: str = from_repo_root("../experiment_logs")
    checkpoint: int = -1
    export_policy: bool = True
    use_joystick: bool = True
    episode_length_s: float = 200.
    device: str = "cpu"
    use_real_robot: bool = False
    num_envs: int = 1

    # hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    task: TaskConfig = empty_cfg(TaskConfig)(
        env = empty_cfg(EnvConfig)(
            num_envs = "${num_envs}"
        ),
        observation = empty_cfg(ObservationConfig)(
            get_commands_from_joystick = "${use_joystick}",
            sensors = ("motor_pos_unshifted", "motor_vel", "last_action", "base_orientation_quat", "base_ang_vel", "base_velocity")
        ),
        sim = empty_cfg(SimConfig)(
            device = "${device}",
            use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
            headless = True, # meaning Isaac is headless, but not the target for deployment 
            physx = empty_cfg(SimConfig.PhysxConfig)(
                use_gpu = "${evaluate_use_gpu: ${task.sim.device}}"
            )
        ),
        terrain = empty_cfg(TerrainConfig)(
            curriculum = False
        ),
        noise = NoNoiseConfig(),
        domain_rand = NoDomainRandConfig(),
        normalization = empty_cfg(NormalizationConfig)(
            obs_scales = NormalizationConfig.NormalizationObsScalesConfig(
                lin_vel = 1.,
                ang_vel = 1.,
                dof_pos = 1.,
                dof_vel = 1.,
                last_action = 1.,
                base_mass = 1., #0.2
                stiffness = 1., #0.01
                damping = 1.,
            )
        ),
        
        init_state = empty_cfg(InitStateConfig)(
            default_joint_angles = INIT_JOINT_ANGLES
        ),

        control = empty_cfg(ControlConfig)(
            stiffness = dict(joint=20.), # [N*m/rad]
            damping = dict(joint=0.5), # [N*m*s/rad]
        )
    ) 
    train: TrainConfig = empty_cfg(TrainConfig)(
        device = "${device}",
        log_dir = "${hydra:runtime.output_dir}",
        runner = empty_cfg(RunnerConfig)(
            checkpoint="${checkpoint}"
        )
    )
    deployment: DeploymentConfig = DeploymentConfig(
        use_real_robot="${use_real_robot}",
        get_commands_from_joystick="${use_joystick}",
        render=DeploymentConfig.RenderConfig(
            show_gui="${not: ${use_real_robot}}"
        ),
        # render=DeploymentConfig.RenderConfig(
        #     show_gui=False
        # ),
        timestep="${compute_timestep: ${task.sim.dt}, ${task.control.decimation}, ${deployment.action_repeat}}",
        init_position="${task.init_state.pos}",
        init_joint_angles="${task.init_state.default_joint_angles}",
        stiffness="${task.control.stiffness}",
        damping="${task.control.damping}",
        action_scale="${task.control.action_scale}"
    )

cs = ConfigStore.instance()
cs.store(name="config", node=DeployWITPConfig)

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
    
    def witp_action_transform(self, action):
        _INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4)
        ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

        low = _INIT_QPOS - ACTION_OFFSET
        high = _INIT_QPOS + ACTION_OFFSET

        min_norm_action = -1
        max_norm_action = 1
        min_norm_action = (
            np.zeros((12,), dtype=float) + min_norm_action
        )
        max_norm_action = (
            np.zeros((12,), dtype=float) + max_norm_action
        )

        action = low + (high - low) * (
            (action - min_norm_action) / (max_norm_action - min_norm_action)
        )
        action = np.clip(action, low, high)

        return action

    def step(self, action):
        """Step forward the environment, given the action.

        action: 12-dimensional NumPy array of desired motor angles
        """
        # print("\tLocomotionGymEnv.step(action)")
        # print(f"\t\tReceived Action: {action}")
        clipped_action = np.clip(
            action,
            self.robot.motor_group.min_positions,
            self.robot.motor_group.max_positions
        )
        # print(f"\t\tAction limits:")
        # print(f"\t\t\tMin: {self.robot.motor_group.min_positions}")
        # print(f"\t\t\tMax: {self.robot.motor_group.max_positions}")
        # print(f"Clipped Action: {clipped_action}")
        # clipped_action = self.witp_action_transform(clipped_action)
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
            elif sensor == "motor_pos_unshifted":
                obs_list.append(self.robot.motor_angles * self.obs_scales.dof_pos)
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
            elif sensor == "base_orientation_quat":
                ## Note: Since we are going to use a policy trained in mujoco (for now), we need to change the order of this quaternion, since In Isaac Gym/Pybullet, the simulators return [x, y, z, w], but mujoco uses [w, x, y, z]
                pybullet_quat = np.concatenate([self.robot.base_orientation_quat[..., [-1]], self.robot.base_orientation_quat[..., :-1]])
                obs_list.append(pybullet_quat) 
            elif sensor == "base_velocity":
                obs_list.append(self.robot.base_velocity * self.obs_scales.lin_vel)
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
        return gym.spaces.Box(-np.inf, np.inf, (46,), np.float32)
        pass

    def _build_action_space(self):
        """Builds action space corresponding to joint position control"""
        # return gym.spaces.Box(
        #     self.robot.motor_group.min_positions,
        #     self.robot.motor_group.max_positions
        # )
        return gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

@hydra.main(version_base=None, config_name="config")
def main(cfg: DeployWITPConfig):
    default_cfg = {"task": TaskConfig(), "train": TrainConfig()}  # default behaviour as defined in "configs/definitions.py"
    merged_cfg = OmegaConf.merge(
        default_cfg,  # loads default values at the end if it's not specified anywhere else
        cfg           # highest priority, loads from the top-level config dataclass above
    )
    # Resolves the config (replaces all "interpolations" - references in the config that need to be resolved to constant values)
    # and turns it to a dictionary (instead of DictConfig in OmegaConf). Throws an error if there are still missing values.
    merged_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    # Creates a new DeployWITPConfig object (with type-checking and optional validation) using Pydantic.
    # The merged config file (DictConfig as given by OmegaConf) has to be recursively turned to a dict for Pydantic to use it.
    # cfg = TypeAdapter(DeployWITPConfig).validate_python(merged_cfg_dict)
    cfg = merged_cfg
    print(OmegaConf.to_yaml(merged_cfg))

    # import pdb;pdb.set_trace()
    deploy_env = LocomotionGymEnv(
        cfg.deployment,
        cfg.task.observation.sensors,
        cfg.task.normalization.obs_scales,
        cfg.task.commands.ranges
    )

    seed = 42
    kwargs = get_config()
    agent = SACLearner.create(seed, deploy_env.observation_space,
                            deploy_env.action_space, **kwargs)
    
    max_steps = int(1e6)
    replay_buffer = ReplayBuffer(deploy_env.observation_space, deploy_env.action_space,
                                     max_steps)
    replay_buffer.seed(seed)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer() 
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    # ckptr.restore(last_checkpoint, item=agent,
            #   restore_args=flax.training.orbax_utils.restore_args_from_target(agent, mesh=None))

    

    
    chkpt_dir = "/home/rll/projects/walk_in_the_park/saved/checkpoints"
    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
    # orbax_checkpointer.restore(last_checkpoint)
    # agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
    agent = ckptr.restore(last_checkpoint, item=agent,
              restore_args=flax.training.orbax_utils.restore_args_from_target(agent, mesh=None)) # orbax_checkpointer.restore(last_checkpoint, agent)

    # import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
   

    # seed = 42
    # kwargs = get_config()
    # agent = SACLearner.create(seed, deploy_env.observation_space,
    #                         deploy_env.action_space, **kwargs)
    
    # max_steps = int(1e6)
    # replay_buffer = ReplayBuffer(deploy_env.observation_space, deploy_env.action_space,
    #                                  max_steps)
    # replay_buffer.seed(seed)
    
    # import pdb;pdb.set_trace()
    # chkpt_dir = "/home/mateo/projects/walk_in_the_park/successful_run/saved/checkpoints"
    # last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
    # agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

    # import pdb;pdb.set_trace()
    # action, agent = agent.sample_actions(obs)
    # next_observation, reward, done, info = deploy_env.step(action)

    ## Manually setting up observation space and action space
    # import pdb;pdb.set_trace()
    # obs, info = deploy_env.reset()
    # for _ in range(1):
    #     obs, *_, info = deploy_env.step(deploy_env.default_motor_angles)

    # import pdb;pdb.set_trace()
    num_episodes = 1000
    for i in range(num_episodes):
        obs, info = deploy_env.reset()
        for _ in range(1):
            obs, *_, info = deploy_env.step(deploy_env.default_motor_angles)
            done = False
        while not done:
            # action = agent.eval_actions(observation)
            action, agent = agent.sample_actions(obs)
            obs, _, done, _, info = deploy_env.step(action)
        print(f"Done with episode {i}")

        
    # for t in range(int(cfg.episode_length_s / deploy_env.robot.control_timestep)):
    #     # Form observation for policy.

    #     ## Convert observation to format taken by policy
    #     # obs = torch.tensor(obs, device=runner.device).float()
        
    #     ## Insert observation into buffer (initialize buffer first if first timestep)
    #     # if t == 0:
    #     #     obs_buf.reset([0], obs)
    #     #     all_infos = {k: [v.copy()] for k, v in info.items()}
    #     # else:
    #     #     obs_buf.insert(obs)
    #     #     for k, v in info.items():
    #     #         all_infos[k].append(v.copy())

    #     ## If more than one observation needed, get buffer of observations
    #     # policy_obs = obs_buf.get_obs_vec(range(task_cfg.observation.history_steps))

    #     # # Evaluate policy and act.
    #     # actions = policy(policy_obs.detach()).detach().cpu().numpy().squeeze()
    #     # actions = task_cfg.control.action_scale*actions + deploy_env.default_motor_angles
    #     # all_actions.append(actions)

    #     ## Environment step
    #     # obs, _, terminated, _, info = deploy_env.step(actions)

    #     # if terminated:
    #     #     log.warning("Unsafe, terminating!")
    #     #     break

    #     for _ in range(num_episodes):
    #     observation, done = env.reset(), False
    #     while not done:
    #         action = agent.eval_actions(observation)
    #         observation, _, done, _ = env.step(action)

    #     action, agent = agent.sample_actions(obs)
    #     next_observation, reward, done, info = deploy_env.step(action)

    #     mask = 0.0

    #     replay_buffer.insert(
    #         dict(observations=observation,
    #              actions=action,
    #              rewards=reward,
    #              masks=mask,
    #              dones=done,
    #              next_observations=next_observation))
    #     observation = next_observation

    #     if done:
    #         print("Done is True")
    #         break
    #         # observation, done = env.reset(), False

    #     print("step")
    #     # time.sleep(0.1)

    

if __name__=="__main__":
    main()