import glob
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from configs.definitions import (EnvConfig, ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig)
from configs.overrides.terrain import FlatTerrainConfig
from legged_gym.utils.helpers import from_repo_root

MOTION_FILES = tuple(glob.glob(from_repo_root("datasets/mocap_motions_a1/*")))

#########
# Task
#########

@dataclass
class AMPEnvConfig(EnvConfig):
    reference_state_initialization_prob: float = 0.85
    motion_files: Tuple[str, ...] = MOTION_FILES

@dataclass
class AMPObservationConfig(ObservationConfig):
    amp_sensors: Tuple[str, ...] = ("motor_pos_unshifted", "foot_pos", "base_lin_vel", "base_ang_vel",
                               "motor_vel", "z_pos")

@dataclass
class AMPTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_amp.A1AMP"
    env: AMPEnvConfig = AMPEnvConfig(
        num_envs="${oc.select: num_envs,5480}"
    )
    observation: AMPObservationConfig = AMPObservationConfig(
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "friction", "base_mass", "p_gain", "d_gain")
    )
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
    terrain: TerrainConfig = FlatTerrainConfig()
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

@dataclass
class AMPMimicTaskConfig(AMPTaskConfig):
    env: AMPEnvConfig = AMPEnvConfig(
        motion_files=("datasets/mocap_motions_a1/trot2.txt",)
    )
    observation: AMPObservationConfig = AMPObservationConfig(
        sensors=("projected_gravity", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel")
    )
    rewards: RewardsConfig = RewardsConfig(
        scales=RewardsConfig.RewardScalesConfig(
            tracking_lin_vel=0.,
            tracking_ang_vel=0.
        )
    )

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
        iterations="${oc.select: iterations,50_000}"
    )
