from typing import Tuple
from dataclasses import dataclass
from configs.definitions import EnvConfig, RewardsConfig, SimConfig, TaskConfig
from legged_gym.utils.helpers import from_repo_root

MOTION_FILES = (from_repo_root("datasets/mocap_motions_a1/trot2.txt"),)

@dataclass
class MocapEnvConfig(EnvConfig):
    reference_state_initialization_at_start: bool = False
    reference_state_initialization_prob: float = 0.5
    mocap_motion_files: Tuple[str, ...] = MOTION_FILES

@dataclass
class MocapRewardsConfig(RewardsConfig):
    joint_position_tracking_sigma: float = 5.
    joint_velocity_tracking_sigma: float = 0.0003
    foot_position_tracking_sigma: float = 40.
    root_pose_tracking_sigma: float = 20.
    root_vel_tracking_sigma: float = 2.
    tracking_errors: Tuple[str, ...] = (
        "joint_position", "joint_velocity", "foot_position", "root_position",
        "root_orientation", "root_lin_vel", "root_ang_vel"
    )

    @dataclass
    class MocapRewardsScalesConfig(RewardsConfig.RewardScalesConfig):
        joint_position_tracking: float = 0.5
        joint_velocity_tracking: float = 0.05
        foot_position_tracking: float = 0.2
        root_pose_tracking: float = 0.15
        root_vel_tracking: float = 0.1

    scales: MocapRewardsScalesConfig = MocapRewardsScalesConfig()

@dataclass
class MocapTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_mocap.A1Mocap"
    env: MocapEnvConfig = MocapEnvConfig()
    rewards: MocapRewardsConfig = MocapRewardsConfig()
    sim: SimConfig = SimConfig(
        dt=0.0041925
    )
