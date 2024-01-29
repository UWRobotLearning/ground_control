from dataclasses import dataclass
from configs.definitions import RewardsConfig

@dataclass
class LeggedGymRewardsConfig(RewardsConfig):
    only_positive_rewards: bool = True
    soft_dof_pos_limit: float = 0.9
    scales: RewardsConfig.RewardScalesConfig = RewardsConfig.RewardScalesConfig(
        tracking_lin_vel=1.,
        tracking_ang_vel=0.5,
        lin_vel_z=2.,
        ang_vel_xy=0.05,
        torques=2e-4,
        dof_accel=2.5e-7,
        feet_air_time=1.,
        collision=1.,
        action_change=0.15,
        soft_dof_pos_limits=10.
    )


@dataclass
class WITPLeggedGymRewardsConfig(RewardsConfig):
    only_positive_rewards: bool = False
    soft_dof_pos_limit: float = 0.9
    # scales: RewardsConfig.RewardScalesConfig = RewardsConfig.RewardScalesConfig(
    #     tracking_lin_vel=1.,
    #     tracking_ang_vel=0.5,
    #     lin_vel_z=2.,
    #     ang_vel_xy=0.05,
    #     torques=2e-4,
    #     dof_accel=2.5e-7,
    #     feet_air_time=1.,
    #     collision=1.,
    #     action_change=0.15,
    #     soft_dof_pos_limits=10.
    # )

    witp_reward_scale: float = 10
    base_height_target: float = 0.25

    scales: RewardsConfig.RewardScalesConfig = RewardsConfig.RewardScalesConfig(
        witp_abs_dyaw=0.1,
        witp_cos_pitch_times_lin_vel=1.0,
        # base_height=10.0,
    )