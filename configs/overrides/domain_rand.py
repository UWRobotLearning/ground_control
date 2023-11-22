from dataclasses import dataclass
from configs.definitions import DomainRandConfig

@dataclass
class NoDomainRandConfig(DomainRandConfig):
    randomize_friction: bool = False
    randomize_base_mass: bool = False
    push_robots: bool = False
    randomize_gains: bool = False