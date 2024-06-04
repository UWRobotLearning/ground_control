import glob
import numpy as np
from typing import Tuple, Union
from omegaconf import DictConfig
from dataclasses import dataclass, field
from configs.definitions import (EnvConfig, ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig)
from configs.overrides.terrain import TrimeshTerrainConfig, FlatTerrainConfig
from configs.overrides.rewards import LeggedGymRewardsConfig

#########
# Task
#########


@dataclass
class BipedalStandEnvConfig(EnvConfig):
    episode_length_s: float = 10.
    num_inactive_steps: int = 20 # number of steps in the beginning of each episode where the robot is kept inactive
    test_envs_per_tile: int = 10 # number of robots to be placed to each tile in test mode
    test_num_episodes: int = 30


@dataclass
class BipedalStandObservationConfig(ObservationConfig):
    base_vel_in_obs: bool = True


@dataclass
class BipedalStandInitStateConfig(InitStateConfig):
    pos: Tuple[float, float, float]        = (0.0, 0.0, 0.5) # x,y,z [m]
    rot: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0) # x,y,z,w [quat] 
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0) # x,y,z [m/s]
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0) # x,y,z [rad/s]
    lin_vel_noise: float = 0.0
    ang_vel_noise: float = 0.0
    # NOTE: theres a difference in default joint_angles from a1_config.py in old codebase to INIT_JOINT_ANGLES in the current definitions.py
    # hips -> all 0 instead of spread at -0.1, 0.1, thighs -> all 0.9 instead of spread at 0.8, 1., calves -> all -1.8 instead of -1.5
    # Ege
    equal_distribution: dict = field(default_factory=lambda: dict(
        enabled=True,
        row_range=(0,0), #end-inclusive
        col_range=(0,9)  #end-inclusive
    ))



@dataclass
class BipedalStandTaskConfig(TaskConfig):
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.7, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,1.5),
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )


@dataclass
class BipedalStandRewardsConfig(RewardsConfig):
    only_positive_rewards: bool = False 
    base_height_target: float = 0.7 #for standing
    soft_dof_pos_limit: float = 2
    dof_pos_limits: float = 2.0 # Ege - adding penalty for out-of-limit joint positions

    @dataclass
    class BipedalStandRewardsScalesConfig(RewardsConfig.RewardScalesConfig):
        x_axis_orientation: float = 25
        torques: float = 0.001 # Ege - used to be -0.00001
        base_height: float = 1.0
        collision: float = 1
        action_change: float = 0.01
        xy_drift: float = 0.1 # Ege - adding penalty for drifting from start position
        feet_contact: float = 1.   
        lin_vel_z: float = 1
        ang_vel_xy: float = 0.5
        tracking_lin_y_vel: float = 5
        rear_thigh_torques = 1
        front_thigh_torques= 1
        stand_pitch = 4
        lin_vel_x = 100
        lin_vel_y = 5
        stand_still = 10
        rear_motors = 40

    scales: BipedalStandRewardsScalesConfig = BipedalStandRewardsScalesConfig()




#########
# Train
#########

@dataclass
class BipedalStandAlgorithmConfig(AlgorithmConfig):
    _target_: str = "rsl_rl.algorithms.PPO"

@dataclass
class BipedalStandRunnerConfig(RunnerConfig):
    checkpoint: int = 0

@dataclass
class BipedalStandTrainConfig(TrainConfig):
    _target_: str = "rsl_rl.runners.OnPolicyRunner"
    algorithm: BipedalStandAlgorithmConfig = BipedalStandAlgorithmConfig()
    runner: BipedalStandRunnerConfig = BipedalStandRunnerConfig(
        iterations="${oc.select: iterations,5000}"
    )











































































#########
# OLD Task
#########

@dataclass
class AMPTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_amp.A1AMP"
#    env: AMPEnvConfig = AMPEnvConfig(
#        num_envs="${resolve_default_int: 5480, ${num_envs}}"
#    )
#    observation: AMPObservationConfig = AMPObservationConfig(
#        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "friction", "base_mass", "p_gain", "d_gain")
#    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.42),
        default_joint_angles={
            "1_FR_hip_joint": 0.15,
            "1_FR_thigh_joint": 0.55,
            "1_FR_calf_joint": -1.5,

            "2_FL_hip_joint": -0.15,
            "2_FL_thigh_joint": 0.55,
            "2_FL_calf_joint": -1.5,

            "3_RR_hip_joint": 0.15,
            "3_RR_thigh_joint": 0.7,
            "3_RR_calf_joint": -1.5,

            "4_RL_hip_joint": -0.15,
            "4_RL_thigh_joint": 0.7,
            "4_RL_calf_joint": -1.5
        }
    )
    rewards: RewardsConfig = RewardsConfig(
        soft_dof_pos_limit=0.9,
        scales=RewardsConfig.RewardScalesConfig(
            tracking_lin_vel=1.5 * 1. / (0.005 * 6),
            tracking_ang_vel=0.5 * 1. / (0.005 * 6)
        )
    )
    asset: AssetConfig = AssetConfig(
        terminate_after_contacts_on=(
            "base", "1_FR_calf", "2_FL_calf", "3_RR_calf", "4_RL_calf",
            "1_FR_thigh", "2_FL_thigh", "3_RR_thigh", "4_RL_thigh"
        ),
        self_collisions=True
    )
    control: ControlConfig = ControlConfig(
        stiffness=dict(joint=80.),
        damping=dict(joint=1.0),
        decimation=6
    )
    #terrain: TerrainConfig = FlatTerrainConfig()
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.25, 1.75),
        randomize_base_mass=True,
        randomize_gains=True,
        added_stiffness_range=(-8., 8.),
        added_damping_range=(-0.1, 0.1)
    )
    noise: NoiseConfig = NoiseConfig(
        noise_scales=NoiseConfig.NoiseScalesConfig(
            dof_pos=0.03,
            ang_vel=0.3
        )
    )
    commands: CommandsConfig = CommandsConfig(
        heading_command=False,
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1., 2.),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_yaw=(-np.pi/2, np.pi/2),
            heading=(-np.pi, np.pi)
        )
    )

#@dataclass
#class AMPMimicTaskConfig(AMPTaskConfig):
#    env: AMPEnvConfig = AMPEnvConfig(
#        motion_files=("datasets/mocap_motions_a1/trot2.txt",)
#    )
#    observation: AMPObservationConfig = AMPObservationConfig(
#        sensors=("projected_gravity", "motor_pos", "motor_vel", "last_action"),
#        critic_privileged_sensors=("base_lin_vel", "base_ang_vel")
#    )
#    rewards: RewardsConfig = RewardsConfig(
#        scales=RewardsConfig.RewardScalesConfig(
#            tracking_lin_vel=0.,
#            tracking_ang_vel=0.
#        )
#    )

#########
# Train
#########

@dataclass
class AMPAlgorithmConfig(AlgorithmConfig):
    _target_: str = "rsl_rl.algorithms.AMPPPO"
    amp_replay_buffer_size: int = 1_000_000

@dataclass
class AMPRunnerConfig(RunnerConfig):
   amp_reward_coef: float = 2.
   amp_task_reward_lerp: float = 0.3
   amp_discr_hidden_dims: Tuple[int, ...] = (1024, 512)
   num_preload_transitions: int = 2_000_000
   min_normalized_std: Tuple[float, ...] = (0.01, 0.005, 0.01) * 4


@dataclass
class AMPTrainConfig(TrainConfig):
    _target_: str = "rsl_rl.runners.AMPOnPolicyRunner"
    algorithm: AMPAlgorithmConfig = AMPAlgorithmConfig()
    runner: AMPRunnerConfig = AMPRunnerConfig(
        iterations="${resolve_default_int: 50_000, ${iterations}}"
    )


# Aliases
#AMPEnvOrDictConfig = Union[AMPEnvConfig, DictConfig]
#AMPObservationOrDictConfig = Union[AMPObservationConfig, DictConfig]
#AMPTaskOrDictConfig = Union[AMPTaskConfig, DictConfig]
#AMPMimicOrDictConfig = Union[AMPMimicTaskConfig, DictConfig]
#AMPAlgorithmOrDictConfig = Union[AMPAlgorithmConfig, DictConfig]
#AMPRunnerOrDictConfig = Union[AMPRunnerConfig, DictConfig]
#AMPTrainOrDictConfig = Union[AMPTrainConfig, DictConfig]
