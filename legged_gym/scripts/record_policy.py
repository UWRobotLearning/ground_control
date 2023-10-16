import logging
import pickle
import time
import cv2
import os
import os.path as osp

from omegaconf import OmegaConf, MISSING
import hydra
from hydra.core.config_store import ConfigStore

from dataclasses import dataclass
from configs.definitions import TaskConfig, TrainConfig

from legged_gym.envs.a1 import A1
from legged_gym.scripts.train import TrainScriptConfig  # so that loading config pickle file works
from legged_gym.utils.helpers import export_policy_as_jit, get_load_path, get_latest_experiment_path

from rsl_rl.runners import OnPolicyRunner

import numpy as np

@dataclass
class Config:
    checkpoint_root: str = "experiment_logs"
    export_policy: bool = True
    record_frames: bool = False
    video_name: str = "isaac"
    checkpoint: int = -1
    device: str = "cpu"
    get_commands_from_joystick: bool = False
    episode_length_s: float = 20.
    stochastic_actor: bool = False

    task: TaskConfig = MISSING
    train: TrainConfig = MISSING

    @dataclass
    class CameraConfig:
        height: float = 0.45
        rot: float = -30.
        rot_per_sec: float = 0.
        dist_multiplier: float = 1.2
    camera: CameraConfig = CameraConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def play(cfg: Config, experiment_root: str) -> None:
    # prepare environment
    env: A1 = hydra.utils.instantiate(cfg.task)
    env.reset()
    obs = env.get_observations()

    # load policy
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)
    train_cfg = cfg.train
    resume_path = get_load_path(experiment_root, checkpoint=train_cfg.runner.checkpoint)
    logging.info(f"Loading model from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device, with_action_noise=cfg.stochastic_actor)

    # export policy as a jit module (used to run it from C++)
    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, experiment_root)
        logging.info(f"Exported policy as jit script to: {experiment_root}")

    camera_rot = np.radians(cfg.camera.rot)
    camera_rot_per_sec = np.radians(cfg.camera.rot_per_sec)

    num_frames = int(cfg.episode_length_s / env.dt)
    logging.info(f"Gathering {num_frames} frames")
    video = None

    current_time = time.time()
    for _ in range(num_frames):
        actions = policy(obs.detach())
        obs, *_ = env.step(actions.detach())

        # Reset camera position.
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
        camera_relative_position = np.array([np.cos(camera_rot), np.sin(camera_rot), cfg.camera.height])
        camera_relative_position *= cfg.camera.dist_multiplier
        env.set_camera(look_at + camera_relative_position, look_at)

        if cfg.record_frames:
            filename = osp.join(experiment_root, ".img.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(
                    osp.join(experiment_root, f"{cfg.video_name}.mp4"),
                    cv2.VideoWriter_fourcc(*'MP4V'),
                    int(1 / env.dt),
                    (img.shape[1], img.shape[0])
                )
            video.write(img)

        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

    if cfg.record_frames:
        video.release()
        os.remove(osp.join(experiment_root, ".img.png"))

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    latest_config_filepath = osp.join(experiment_path, "resolved_config.pkl")
    logging.info(f"Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    with open(latest_config_filepath, "rb") as cfg_pkl:
        loaded_cfg = pickle.load(cfg_pkl)
    OmegaConf.resolve(loaded_cfg)

    cfg.task = loaded_cfg.task
    cfg.train = loaded_cfg.train

    # override some params for purposes of evaluation
    cfg.task.domain_rand.randomize_friction = False
    cfg.task.domain_rand.push_robots = False
    cfg.task.domain_rand.randomize_gains = False
    cfg.task.domain_rand.randomize_base_mass = False
    cfg.task.env.num_envs = 1
    cfg.task.env.episode_length_s = cfg.episode_length_s
    cfg.task.observation.get_commands_from_joystick = cfg.get_commands_from_joystick
    cfg.task.noise.add_noise = False
    cfg.task.sim.device = cfg.device
    cfg.task.sim.use_gpu_pipeline = cfg.task.sim.physx.use_gpu = (cfg.task.sim.device != "cpu")
    cfg.task.sim.headless = False
    cfg.task.terrain.num_rows = 5
    cfg.task.terrain.num_cols = 5
    cfg.task.terrain.curriculum = False
    cfg.train.device = cfg.device
    cfg.train.runner.checkpoint = cfg.checkpoint

    play(cfg, experiment_path)

if __name__ == "__main__":
    main()
