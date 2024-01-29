import numpy as np
from typing import Tuple
from dataclasses import dataclass
from configs.definitions import (ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig, EnvConfig, NormalizationConfig)
from configs.overrides.terrain import FlatTerrainConfig
from configs.overrides.rewards import LeggedGymRewardsConfig, WITPLeggedGymRewardsConfig

#########
# Task
#########

@dataclass
class LocomotionTaskConfig(TaskConfig):
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )


WITP_INIT_JOINT_ANGLES = {
    "1_FR_hip_joint": 0.,
    "1_FR_thigh_joint": 0.9,
    "1_FR_calf_joint": -1.8,

    "2_FL_hip_joint": 0.,
    "2_FL_thigh_joint": 0.9,
    "2_FL_calf_joint": -1.8,

    "3_RR_hip_joint": 0.,
    "3_RR_thigh_joint": 0.9,
    "3_RR_calf_joint": -1.8,

    "4_RL_hip_joint": 0.,
    "4_RL_thigh_joint": 0.9,
    "4_RL_calf_joint": -1.8
}

@dataclass
class WITPLocomotionTaskConfig(TaskConfig):
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = WITPLeggedGymRewardsConfig()
    # observation: ObservationConfig = ObservationConfig(
    #     sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
    #     critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    # )
    observation: ObservationConfig = ObservationConfig(
        sensors=("motor_pos_unshifted", "motor_vel", "last_action", "base_quat", "base_ang_vel", "base_lin_vel"),
        critic_privileged_sensors=(),
    )
    normalization: NormalizationConfig = NormalizationConfig(
        normalize=False
    )
    noise: NoiseConfig = NoiseConfig(
        add_noise=False
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
        default_joint_angles=WITP_INIT_JOINT_ANGLES
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    control: ControlConfig = ControlConfig(
        decimation=10,
        clip_setpoint=True,
        joint_lower_limit=(-0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8),
        joint_upper_limit=(0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0),
        # stiffness=dict(joint=60.),  #20.,
        # damping=dict(joint=10.),  #0.5,
        # stiffness : Dict[str, float] = field(default_factory=lambda: dict(joint=20.)), # [N*m/rad]
        # damping: Dict[str, float] = field(default_factory=lambda: dict(joint=0.5)) # [N*m*s/rad]
    )
    env: EnvConfig = EnvConfig(
        episode_length_s=5
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
