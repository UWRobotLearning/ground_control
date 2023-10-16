from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class GaitConfig:
    max_forward_speed: float
    max_side_speed: float
    max_rot_speed: float
    gait_parameters: Tuple[float, float, float, float, float]

    # swing foot settings
    foot_clearance_max: float
    foot_clearance_land: float

    # MPC-related settings
    mpc_body_mass: float = 110 / 9.8
    mpc_body_intertia: np.ndarray = np.array([0.027, 0., 0., 0., 0.057, 0., 0., 0., 0.064]) * 5.
    mpc_foot_friction: float = 0.4
    mpc_weight: Tuple[float, ...] = (1., 1., 0., 0., 0., 10., 0., 0., 0.1, 0.1, 0.1, 0., 0.)


CRAWL_CONFIG = GaitConfig(
    max_forward_speed=0.3,
    max_side_speed=0.2,
    max_rot_speed=0.6,
    gait_parameters=(1.5, np.pi, np.pi/2, np.pi*3/2, 0.26),
    mpc_foot_friction=0.4,
    foot_clearance_max=0.17,
    foot_clearance_land=-0.01
)

TROT_CONFIG = GaitConfig(
    max_forward_speed=1.5,
    max_side_speed=0.5,
    max_rot_speed=1.,
    gait_parameters=(2., np.pi, np.pi, 0., 0.5),
    mpc_foot_friction=0.3,
    foot_clearance_max=0.17,
    foot_clearance_land=-0.01
)

FLYTROT_CONFIG = GaitConfig(
    max_forward_speed=2.5,
    max_side_speed=1.,
    max_rot_speed=1.,
    gait_parameters=(3.6, np.pi, np.pi, 0., 0.58),
    mpc_foot_friction=0.2,
    foot_clearance_max=0.08,
    foot_clearance_land=0.01
)
