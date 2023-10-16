import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def angle_axis_from_quat(quat):
    assert quat.shape[-1] == 4
    one = torch.tensor(1., device=quat.device)
    assert torch.isclose(torch.norm(quat, dim=-1), one).all()

    shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)

    axis = quat[:, :3]
    axis_norm = torch.norm(axis, dim=1)

    # TODO: check axis_norm < min_axis_norm

    sin_half_angle = axis_norm
    cos_half_angle = quat[:, 3]
    half_angle = torch.atan2(sin_half_angle, cos_half_angle)
    angle = 2 * half_angle

    axis = axis.view(*shape, 3)
    angle = angle.view(*shape)
    return axis, angle
