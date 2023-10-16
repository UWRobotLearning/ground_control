import numpy as np
from configs.definitions import NormalizationConfig, DeploymentConfig, CommandsConfig
from robot_deployment.envs.locomotion_gym_env import LocomotionGymEnv

config = DeploymentConfig(
    reset_time_s=0.,
    on_rack=True,
    render=DeploymentConfig.RenderConfig(show_gui=True)
)
sensors = ("motor_angles",)
obs_scales = NormalizationConfig.NormalizationObsScalesConfig()
command_ranges = CommandsConfig.CommandRangesConfig()
env = LocomotionGymEnv(config, sensors, obs_scales, command_ranges)

action_selector_ids = []
num_act = env.action_space.shape[0]
for i in range(num_act):
    joint_name = env.robot.motor_group.motor_joint_names[i]
    low = env.action_space.low[i]
    high = env.action_space.high[i]
    start = (low + high) / 2
    action_selector_id = env.pybullet_client.addUserDebugParameter(
        paramName=f" {i}: {joint_name}",
        rangeMin=low,
        rangeMax=high,
        startValue=start
    )
    action_selector_ids.append(action_selector_id)

env.reset()

act = np.zeros(num_act)
while True:
    for i in range(num_act):
        act[i] = env.pybullet_client.readUserDebugParameter(action_selector_ids[i])
    env.step(act)
