"""Replay AMP trajectories."""
import cv2
import os
import time

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from configs.overrides.amp import AMPTaskConfig

from isaacgym import gymtorch
from legged_gym.envs.a1_amp import A1AMP

from isaacgym.torch_utils import *

import numpy as np
import torch

@dataclass
class Config:
    record_frames: bool = False
    video_name: str = "mocap"
    task: AMPTaskConfig = AMPTaskConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def play(cfg: Config) -> None:
    env: A1AMP = hydra.utils.instantiate(cfg.task)

    camera_rot = np.radians(-30)
    camera_rot_per_sec = 0

    zero_act = torch.zeros((1, env.num_actions), device=env.device)
    env_ids = torch.tensor([0], device=env.device)
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    amp_loader = env.amp_loader
    video = None

    current_time = time.time()
    for traj_idx in range(len(amp_loader.trajectory_lens)):
        for t in np.arange(0, amp_loader.trajectory_lens[traj_idx] - amp_loader.time_between_frames, env.dt):
            full_frame = amp_loader.get_full_frame_at_time(traj_idx, t)
            root_pos = amp_loader.get_root_pos(full_frame)
            root_orn = amp_loader.get_root_rot(full_frame)
            linear_vel = amp_loader.get_linear_vel(full_frame)
            angular_vel = amp_loader.get_angular_vel(full_frame)
            joint_pos = amp_loader.get_joint_pose(full_frame)
            joint_vel = amp_loader.get_joint_vel(full_frame)
            foot_pos = amp_loader.get_tar_toe_pos_local(full_frame)

            env.root_states[env_ids, :3] = root_pos
            env.root_states[env_ids, 3:7] = root_orn
            env.root_states[env_ids, 7:10] = quat_rotate(root_orn.unsqueeze(0), linear_vel.unsqueeze(0))
            env.root_states[env_ids, 10:13] = quat_rotate(root_orn.unsqueeze(0), angular_vel.unsqueeze(0))
            env.dof_pos[env_ids] = joint_pos
            env.dof_vel[env_ids] = joint_vel

            env.gym.set_actor_root_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )
            env.gym.set_dof_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )

            print("---")
            print(env.get_amp_observations()[0, 12:24])
            print(foot_pos)

            env.step(zero_act.detach())

            # reset camera position
            look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec*env.dt) % (2*np.pi)
            camera_relative_position = 2*np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
            env.set_camera(look_at + camera_relative_position, look_at)

            if cfg.record_frames:
                filename = ".img.png"
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img = cv2.imread(filename)
                if video is None:
                    video = cv2.VideoWriter(
                       f"{cfg.video_name}.mp4",
                       cv2.VideoWriter_fourcc(*"MP4V"),
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
        os.remove(".img.png")

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # override some params
    cfg.task.env.num_envs = 1
    for k in cfg.task.control.stiffness.keys():
        cfg.task.control.stiffness[k] = 0.
    for k in cfg.task.control.damping.keys():
        cfg.task.control.damping[k] = 0.
    cfg.task.sim.headless = False
    cfg.task.sim.device = 'cpu'
    cfg.task.terrain.num_rows = 5
    cfg.task.terrain.num_cols = 5
    cfg.task.terrain.curriculum = False
    cfg.task.noise.add_noise = False
    cfg.task.domain_rand.push_robots = False
    cfg.task.domain_rand.randomize_friction = False
    cfg.task.domain_rand.randomize_gains = False
    cfg.task.domain_rand.randomize_base_mass = False

    play(cfg)

if __name__ == '__main__':
    main()
