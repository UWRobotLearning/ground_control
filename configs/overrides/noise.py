from dataclasses import dataclass
from configs.definitions import NoiseConfig

@dataclass
class NoNoiseConfig(NoiseConfig):
    add_noise: bool = False