import glob
import json
from typing import Optional, Union

import torch
import numpy as np

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util

DEFAULT_OBSERVATIONS = (
    "motor_pos_unshifted",
    "foot_pos",
    "base_lin_vel",
    "base_ang_vel",
    "motor_vel",
    "z_pos"
)

# Data format:
# 0-2: root position
# 3-6: root orientation
# 7-18: joint angles
# 19-30: foot positions in local frame
# 31-33: root linear velocity
# 34-36: root angular velocity
# 37-48: joint velocities
# 49-60: foot velocities in local frame

class MocapLoader:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
            self,
            device,
            time_between_frames,
            sensors=DEFAULT_OBSERVATIONS,
            is_amp=False,
            interpolate=True,
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*'),
            ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.interpolate = interpolate
        self.is_amp = is_amp
        self.sensors = sensors

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MocapLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        MocapLoader.ROOT_ROT_START_IDX:MocapLoader.ROOT_ROT_END_IDX
                    ] = root_rot

                self.trajectories.append(
                    torch.tensor(self.extract_observations(motion_data), dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(torch.tensor(motion_data, dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(motion_data.shape[0])

            print(f"Loaded {traj_len:.2f} sec ({motion_data.shape[0]} steps) of motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample(num_preload_transitions)
            if self.interpolate:
                times = self.traj_time_sample(traj_idxs)
                self.preloaded_s = self.get_full_frame_at_time(traj_idxs, times)
                self.preloaded_s_next = self.get_full_frame_at_time(traj_idxs, times + self.time_between_frames)
            else:
                time_idxs = self.traj_time_idx_sample(traj_idxs)
                self.preloaded_s = self.get_full_frame_at_time_idx(traj_idxs, time_idxs)
                self.preloaded_s_next = self.get_full_frame_at_time_idx(traj_idxs, time_idxs+1)

            print(self.get_joint_pose(self.preloaded_s).mean(dim=0))
            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def extract_observations(self, full_frames: Union[np.ndarray, torch.Tensor]):
        frames_list = []
        for sensor in self.sensors:
            if   sensor == "base_ang_vel":
                frames_list.append(MocapLoader.get_angular_vel(full_frames))
            elif sensor == "base_lin_vel":
                frames_list.append(MocapLoader.get_linear_vel(full_frames))
            elif sensor == "foot_pos":
                frames_list.append(MocapLoader.get_tar_toe_pos_local(full_frames))
            elif sensor == "motor_pos_unshifted":
                frames_list.append(MocapLoader.get_joint_pose(full_frames))
            elif sensor == "motor_vel":
                frames_list.append(MocapLoader.get_joint_vel(full_frames))
            elif sensor == "z_pos":
                base_pos = MocapLoader.get_root_pos(full_frames)
                frames_list.append(base_pos[..., 2:3])

        return np.concatenate(frames_list, axis=-1) if isinstance(full_frames, np.ndarray) else torch.cat(frames_list, dim=-1)

    def weighted_traj_idx_sample(self, size: Optional[int] = None):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx: Union[float, np.ndarray]):
        """Sample random time for either a single trajectory (float input)
        or for multiple trajectories (np.ndarray input)."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        time_sample = np.random.uniform(high=self.trajectory_lens[traj_idx] - subst)
        return np.maximum(0., time_sample)

    def traj_time_idx_sample(self, traj_idx: Union[int, np.ndarray]):
        """Sample random time index for either a single trajectory (float input)
        or for multiple trajectories (np.ndarray input)."""
        high = self.trajectory_num_frames[traj_idx]
        return np.random.randint(low=0, high=high-1)

    def get_trajectory(self, traj_idx: Union[int, np.ndarray]):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx: Union[int, np.ndarray], time: Union[float, np.ndarray]):
        """Returns frame for the given trajectory at the specified time.
        Either a single frame for scalar inputs or batch of frames for
        np.ndarray inputs."""
        full_frame = self.get_full_frame_at_time(traj_idx, time)
        return self.extract_observations(full_frame)

    def get_full_frame_at_time(self, traj_idx: Union[int, np.ndarray], time: Union[float, np.ndarray]):
        """Returns full frame for the given trajectory at the specified time.
        Either a single frame for scalar inputs or batch of frames for
        np.ndarray inputs."""
        scalar_input = not isinstance(traj_idx, np.ndarray)
        if scalar_input:
            traj_idx, time = np.array([traj_idx]), np.array([time])
        p = time / self.trajectory_lens[traj_idx]
        n = self.trajectory_num_frames[traj_idx]
        idx_low, idx_high = np.floor(p*n).astype(int), np.ceil(p*n).astype(int)

        all_frame_starts = torch.zeros(len(traj_idx), self.full_observation_dim, device=self.device)
        all_frame_ends = all_frame_starts.clone()
        for idx in set(traj_idx):
            trajectory = self.trajectories_full[idx]
            traj_mask = (idx == traj_idx)
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p*n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        frame = self.blend_full_frames(all_frame_starts, all_frame_ends, blend) if self.interpolate else all_frame_starts
        if scalar_input:
            frame.squeeze_(0)
        return frame

    def get_frame_at_time_idx(self, traj_idx: Union[int, np.ndarray], time_idx: Union[int, np.ndarray]):
        """Returns frame for the given trajectory at the specified time index.
        Either a single frame for scalar inputs or batch of frames for
        np.ndarray inputs."""
        full_frame = self.get_full_frame_at_time_idx(traj_idx, time_idx)
        return self.extract_observations(full_frame)

    def get_full_frame_at_time_idx(self, traj_idx: Union[int, np.ndarray], time_idx: Union[int, np.ndarray]):
        """Returns full frame for the given trajectory at the specified time index.
        Either a single frame for scalar inputs or batch of frames for
        np.ndarray inputs."""
        scalar_input = not isinstance(traj_idx, np.ndarray)
        if scalar_input:
            traj_idx, time_idx = np.array([traj_idx]), np.array([time_idx])
        frame = torch.empty(len(traj_idx), self.full_observation_dim, device=self.device)
        for idx in set(traj_idx):
            trajectory = self.trajectories_full[idx]
            traj_mask = (idx == traj_idx)
            frame[traj_mask] = trajectory[time_idx[traj_mask]]
        if scalar_input:
            frame.squeeze_(0)
        return frame

    def get_frame(self, num_frames: Optional[int] = None):
        """Returns random frame."""
        frame = self.get_full_frame(num_frames)
        return self.extract_observations(frame)

    def get_full_frame(self, num_frames: Optional[int] = None):
        """Returns random full frame."""
        none_input = num_frames is None
        if none_input:
            num_frames = 1

        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            frame = self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample(num_frames)
            if self.interpolate:
                times = self.traj_time_sample(traj_idxs)
                frame = self.get_full_frame_at_time(traj_idxs, times)
            else:
                time_idxs = self.traj_time_idx_sample(traj_idxs)
                frame = self.get_full_frame_at_time_idx(traj_idxs, time_idxs)

        if none_input:
            frame.squeeze_(0)
        return frame

    def blend_full_frames(self, frame0: torch.Tensor, frame1: torch.Tensor, blend: torch.Tensor):
        """Linearly interpolate between two full frames, including orientation.

        Args:
            frame0: First full frame to be blended corresponds to (blend = 0).
            frame1: Second full frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two full frames.
        """

        root_pos0, root_pos1 = MocapLoader.get_root_pos(frame0), MocapLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = MocapLoader.get_root_rot(frame0), MocapLoader.get_root_rot(frame1)
        joints0, joints1 = MocapLoader.get_joint_pose(frame0), MocapLoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MocapLoader.get_tar_toe_pos_local(frame0), MocapLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MocapLoader.get_linear_vel(frame0), MocapLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MocapLoader.get_angular_vel(frame0), MocapLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MocapLoader.get_joint_vel(frame0), MocapLoader.get_joint_vel(frame1)
        tar_toe_vel_0, tar_toe_vel_1 = MocapLoader.get_tar_toe_vel_local(frame0), MocapLoader.get_tar_toe_vel_local(frame1)

        blend_root_pos = utils.lerp(root_pos0, root_pos1, blend)
        blend_root_rot = utils.quaternion_slerp(root_rot0, root_rot1, blend)
        blend_joints = utils.lerp(joints0, joints1, blend)
        blend_tar_toe_pos = utils.lerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = utils.lerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = utils.lerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = utils.lerp(joint_vel_0, joint_vel_1, blend)
        blend_tar_toe_vel = utils.lerp(tar_toe_vel_0, tar_toe_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel, blend_tar_toe_vel], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.extract_observations(self.preloaded_s[idxs])
                s_next = self.extract_observations(self.preloaded_s_next[idxs])
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample(mini_batch_size)
                times = self.traj_time_sample(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of mocap observations."""
        return self.trajectories[0].shape[1]

    @property
    def full_observation_dim(self):
        """Size of full mocap observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_root_pos(pose):
        return pose[..., MocapLoader.ROOT_POS_START_IDX:MocapLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[..., MocapLoader.ROOT_ROT_START_IDX:MocapLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        return pose[..., MocapLoader.JOINT_POSE_START_IDX:MocapLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local(pose):
        return pose[..., MocapLoader.TAR_TOE_POS_LOCAL_START_IDX:MocapLoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_linear_vel(pose):
        return pose[..., MocapLoader.LINEAR_VEL_START_IDX:MocapLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel(pose):
        return pose[..., MocapLoader.ANGULAR_VEL_START_IDX:MocapLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[..., MocapLoader.JOINT_VEL_START_IDX:MocapLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_tar_toe_vel_local(pose):
        return pose[..., MocapLoader.TAR_TOE_VEL_LOCAL_START_IDX:MocapLoader.TAR_TOE_VEL_LOCAL_END_IDX]
