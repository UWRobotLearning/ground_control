import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

from witp.rl.wrappers.single_precision import SinglePrecision
from witp.rl.wrappers.universal_seed import UniversalSeed


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env