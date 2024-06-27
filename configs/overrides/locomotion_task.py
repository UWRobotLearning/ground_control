import numpy as np
from typing import Tuple
from dataclasses import dataclass
from configs.definitions import (ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig, EnvConfig, NormalizationConfig)
from configs.overrides.terrain import FlatTerrainConfig, TrimeshTerrainConfig, RoughFlatConfig, SmoothUpslopeConfig, RoughDownslopeConfig, StairsUpConfig, StairsDownConfig, DiscreteConfig, RoughFlatHardConfig
from configs.overrides.rewards import LeggedGymRewardsConfig, WITPLeggedGymRewardsConfig, MoveFwdRewardsConfig, SimpleLeggedGymRewardsConfig
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig

#########
# Task
#########

@dataclass
class LocomotionTaskConfig(TaskConfig):
    terrain: TerrainConfig = TrimeshTerrainConfig() # FlatTerrainConfig()
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

@dataclass
class ClippedLocomotionTaskConfig(TaskConfig):
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

@dataclass
class ForwardLocomotionTaskConfig(TaskConfig):
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
            lin_vel_x = (0.49, 0.5), # min max [m/s]
            lin_vel_y = (0., 0.), # min max [m/s]
            ang_vel_yaw = (0., 0.), # min max [rad/s]
            heading = (0., 0.), # min max [rad]
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class ForwardClippedLocomotionTaskConfig(TaskConfig):
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
            lin_vel_x = (0.49, 0.5), # min max [m/s]
            lin_vel_y = (0., 0.), # min max [m/s]
            ang_vel_yaw = (0., 0.), # min max [rad/s]
            heading = (0., 0.), # min max [rad]
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True,
        joint_lower_limit=(-0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8),
        joint_upper_limit=(0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0),
        # stiffness=dict(joint=60.),  #20.,
        # damping=dict(joint=10.),  #0.5,
        # stiffness : Dict[str, float] = field(default_factory=lambda: dict(joint=20.)), # [N*m/rad]
        # damping: Dict[str, float] = field(default_factory=lambda: dict(joint=0.5)) # [N*m*s/rad]
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
        # decimation=10,
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

@dataclass
class WITPUnclippedLocomotionTaskConfig(TaskConfig):
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = WITPLeggedGymRewardsConfig()

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
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
        default_joint_angles=WITP_INIT_JOINT_ANGLES
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
        episode_length_s=5
    )
    # control: ControlConfig = ControlConfig(
    #     decimation=10
    # )



@dataclass
class ResidualLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_residual.A1Residual"
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

    noise = NoNoiseConfig(),  ## TODO: Not sure if needed or not here
    domain_rand = NoDomainRandConfig(),  ## TODO: Not sure if needed or not 

@dataclass
class ResidualWITPLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_residual.A1Residual"
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = WITPLeggedGymRewardsConfig()
    # observation: ObservationConfig = ObservationConfig(
    #     sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
    #     critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    # )
    observation: ObservationConfig = ObservationConfig(
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        residual_sensors=("motor_pos_unshifted", "motor_vel", "last_action", "base_quat", "base_ang_vel", "base_lin_vel"),
    )
    # normalization: NormalizationConfig = NormalizationConfig(
    #     normalize=False
    # )

    noise: NoiseConfig = NoiseConfig(
        add_noise=False
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        randomize_friction=False,
        randomize_base_mass=False,
        randomize_gains=False,
        push_robots=False
    )

    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
        default_joint_angles=WITP_INIT_JOINT_ANGLES
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    control: ControlConfig = ControlConfig(
        # decimation=10,
        clip_setpoint=True,
        joint_lower_limit=(-0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8),
        joint_upper_limit=(0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0),
        # stiffness=dict(joint=60.),  #20.,
        # damping=dict(joint=10.),  #0.5,
    )
    env: EnvConfig = EnvConfig(
        episode_length_s=5
    )

@dataclass
class ResidualWITPUnclippedLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_residual.A1Residual"
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = WITPLeggedGymRewardsConfig()
    # observation: ObservationConfig = ObservationConfig(
    #     sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
    #     critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    # )
    observation: ObservationConfig = ObservationConfig(
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        residual_sensors=("motor_pos_unshifted", "motor_vel", "last_action", "base_quat", "base_ang_vel", "base_lin_vel"),
    )
    # normalization: NormalizationConfig = NormalizationConfig(
    #     normalize=False
    # )

    noise: NoiseConfig = NoiseConfig(
        add_noise=False
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        randomize_friction=False,
        randomize_base_mass=False,
        randomize_gains=False,
        push_robots=False
    )

    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
        default_joint_angles=WITP_INIT_JOINT_ANGLES
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    # control: ControlConfig = ControlConfig(
    #     # decimation=10,
    #     clip_setpoint=True,
    #     joint_lower_limit=(-0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8, -0.15, 0.3, -1.8),
    #     joint_upper_limit=(0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0, 0.25, 1.1, -1.0),
    #     # stiffness=dict(joint=60.),  #20.,
    #     # damping=dict(joint=10.),  #0.5,
    # )
    env: EnvConfig = EnvConfig(
        episode_length_s=5
    )

    
@dataclass
class MoveFwdTaskConfig(TaskConfig):
    terrain: TerrainConfig = RoughFlatConfig()
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
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
        default_joint_angles=WITP_INIT_JOINT_ANGLES
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    control: ControlConfig = ControlConfig(
        # decimation=10,
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


@dataclass
class PretrainLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = TrimeshTerrainConfig()#FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=("terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat", "yaw_rate", "z_pos"),
        history_steps=1
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class HistoriesPretrainLocomotionTaskConfig(PretrainLocomotionTaskConfig):
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=("terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat", "yaw_rate", "z_pos"),
        history_steps=4,
        use_history_for_critic=True
    )

@dataclass
class EvalPretrainLocomotionTaskConfig(PretrainLocomotionTaskConfig):
    env: EnvConfig = EnvConfig(
        num_envs=1, 
        episode_length_s=5
    )

@dataclass
class EvalFwdPretrainLocomotionTaskConfig(EvalPretrainLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )
    rewards: SimpleLeggedGymRewardsConfig = SimpleLeggedGymRewardsConfig()
    
@dataclass
class EvalHistoriesPretrainLocomotionTaskConfig(HistoriesPretrainLocomotionTaskConfig):
    env: EnvConfig = EnvConfig(
        num_envs=1, 
        episode_length_s=5
    )

@dataclass
class EvalHistoriesFwdPretrainLocomotionTaskConfig(EvalHistoriesPretrainLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class CollectPretrainLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = FlatTerrainConfig()#TrimeshTerrainConfig()#FlatTerrainConfig()
    rewards: RewardsConfig = SimpleLeggedGymRewardsConfig()#LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=("terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat", "yaw_rate", "z_pos"),
        history_steps=1,
        use_history_for_critic=False
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
        num_envs=40,
        episode_length_s=5
    )

@dataclass
class CollectFwdPretrainLocomotionTaskConfig(CollectPretrainLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class CollectHistoriesPretrainLocomotionTaskConfig(CollectPretrainLocomotionTaskConfig):
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=("terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat", "yaw_rate", "z_pos"),
        history_steps=4,
        use_history_for_critic=True
    )

@dataclass
class CollectFwdHistoriesPretrainLocomotionTaskConfig(CollectHistoriesPretrainLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class AdaptLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = SimpleLeggedGymRewardsConfig()#LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=(),
        extra_sensors=("base_quat", "yaw_rate", "terrain_height", "friction", "base_mass", "z_pos"),
        history_steps=1
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
            num_envs=1,
            episode_length_s=5,
    )

@dataclass
class HistoriesAdaptLocomotionTaskConfig(AdaptLocomotionTaskConfig):
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=(),
        extra_sensors=("base_quat", "yaw_rate", "terrain_height", "friction", "base_mass", "z_pos"),
        history_steps=4
    )

@dataclass
class AugmentedAdaptLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "base_quat", "yaw_rate", "friction", "base_mass", "z_pos"),
        critic_privileged_sensors=(),
        extra_sensors=("terrain_height",),
        history_steps=1
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
            num_envs=1,
            episode_length_s=5,
    )


@dataclass
class FwdAdaptLocomotionTaskConfig(AdaptLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )


@dataclass
class HistoriesFwdAdaptLocomotionTaskConfig(HistoriesAdaptLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class DownhillAdaptLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = RoughDownslopeConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=(),
        extra_sensors=("base_quat", "yaw_rate", "terrain_height", "friction", "base_mass", "z_pos"),
        history_steps=1
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
            num_envs=1,
            episode_length_s=5,
    )

@dataclass
class DownhillHistoriesAdaptLocomotionTaskConfig(DownhillAdaptLocomotionTaskConfig):
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action"),
        critic_privileged_sensors=(),
        extra_sensors=("base_quat", "yaw_rate", "terrain_height", "friction", "base_mass", "z_pos"),
        history_steps=4
    )
    
@dataclass
class DownhillFwdAdaptLocomotionTaskConfig(DownhillAdaptLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class DownhillHistoriesFwdAdaptLocomotionTaskConfig(DownhillHistoriesAdaptLocomotionTaskConfig):
    commands: CommandsConfig = CommandsConfig(
        use_fixed_commands=True,
        fixed_commands= CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.,
            ang_vel_yaw=0.,
            heading=0.
        )
    )

@dataclass
class DownhillAugmentedAdaptLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = RoughDownslopeConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "base_quat", "yaw_rate", "friction", "base_mass", "z_pos"),
        critic_privileged_sensors=(),
        extra_sensors=("terrain_height",),
        history_steps=1
    )
    control: ControlConfig = ControlConfig(
        clip_setpoint=True
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )
    env: EnvConfig = EnvConfig(
            num_envs=1,
            episode_length_s=5,
    )




@dataclass
class PreadaptLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class RoughFwdLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = RoughFlatConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )


@dataclass
class RoughHardFwdLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = RoughFlatHardConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class SmoothUpslopeLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = SmoothUpslopeConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class RoughDownslopeLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = RoughDownslopeConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class StairsDownLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = StairsDownConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class StairsUpLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = StairsUpConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

@dataclass
class DiscreteLocomotionTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_continual.A1Continual"
    terrain: TerrainConfig = DiscreteConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()  ## It's important that this matches adapt config rewards
    observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
        extra_sensors=("base_quat",)
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        ),
        use_fixed_commands=True,
        fixed_commands=CommandsConfig.FixedCommands(
            lin_vel_x=0.5,
            lin_vel_y=0.0,
            ang_vel_yaw=0.0,
            heading=0.0
        ),
        heading_command=False
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )


# @dataclass
# class AdaptLocomotionTaskConfig(TaskConfig):
#     _target_: str = "legged_gym.envs.a1_continual.A1Continual"
#     env: EnvConfig = EnvConfig(
#         num_envs=1
#     )
#     terrain: TerrainConfig = SmoothUpslopeConfig()
#     rewards: RewardsConfig = LeggedGymRewardsConfig() #WITPLeggedGymRewardsConfig()
#     observation: ObservationConfig = ObservationConfig(  ## It's important that this contains adapt config observations
#         sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate", "base_lin_vel", "base_ang_vel", "base_quat"),
#         critic_privileged_sensors=(),
#         extra_sensors=()
#     )
#     domain_rand: DomainRandConfig = DomainRandConfig(
#         friction_range=(0.4, 2.5),
#         added_mass_range=(-1.5, 2.5),
#         randomize_base_mass=True,
#     )
#     commands: CommandsConfig = CommandsConfig(
#         ranges=CommandsConfig.CommandRangesConfig(
#             lin_vel_x=(-1.,2.5),
#         ),
#         use_fixed_commands=True,
#         fixed_commands=CommandsConfig.FixedCommands(
#             lin_vel_x=0.5,
#             lin_vel_y=0.0,
#             ang_vel_yaw=0.0,
#             heading=0.0
#         ),
#         heading_command=False
#     )
#     init_state: InitStateConfig = InitStateConfig(
#         pos=(0., 0., 0.32),
#     )
#     asset: AssetConfig = AssetConfig(
#         self_collisions=False,
#     )







































































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
