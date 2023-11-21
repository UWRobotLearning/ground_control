from typing import Tuple
from dataclasses import dataclass
from configs.definitions import ObservationConfig

@dataclass
class ComplexTerrainObservationConfig(ObservationConfig):
    sensors: Tuple[str, ...] = ("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "terrain_height")
    critic_privileged_sensors: Tuple[str, ...] = ("base_lin_vel", "base_ang_vel")
