from legged_gym.envs.a1_continual import A1Continual
from legged_gym.envs.a1_continual_gym import A1Gym
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
# from sbx import SAC, CrossQ, DroQ
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import hydra
from configs.overrides.locomotion_task import  AugmentedAdaptLocomotionTaskConfig 
from configs.definitions import EnvConfig


if __name__ == "__main__":
    pretrain_task = AugmentedAdaptLocomotionTaskConfig(
        env = EnvConfig(
            num_envs=1
        )
    )
    a1_env = hydra.utils.instantiate(pretrain_task)
    
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    }

    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env():
        env = A1Gym(a1_env=a1_env, render_mode="rgb_array")#gym.make(A1Gym, render_mode="rgb_array")
        env = Monitor(env)
        return env
    
    vec_env = DummyVecEnv([make_env])
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    vec_env = VecVideoRecorder(
        vec_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )


    model = SAC(config["policy_type"], vec_env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
    )

    run.finish()

    obs = vec_env.reset()
    for _ in range(1_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
