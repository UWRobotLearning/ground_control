from dataclasses import dataclass, field
from configs.definitions import TerrainConfig
from typing import Tuple, Dict, Any

@dataclass
class FlatTerrainConfig(TerrainConfig):
    mesh_type: str = "plane"
    curriculum: bool = False 

@dataclass
class TrimeshTerrainConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    curriculum: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

ROUGH_FLAT_TERRAIN = {
        "type": "rough",
        "difficulty": 0.5 ## [0.5, 0.75, 0.9]
    }

@dataclass
class RoughFlatConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: ROUGH_FLAT_TERRAIN 
    )

ROUGH_FLAT_TERRAIN_HARD = {
        "type": "rough",
        "difficulty": 0.9 ## [0.5, 0.75, 0.9]
    }

@dataclass
class RoughFlatHardConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: ROUGH_FLAT_TERRAIN_HARD 
    )



SMOOTH_UPSLOPE_TERRAIN = {
        "type": "smooth_up_slope",
        "difficulty": 0.9 ## [0.5, 0.75, 0.9]
    }

@dataclass
class SmoothUpslopeConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: SMOOTH_UPSLOPE_TERRAIN 
    )


ROUGH_DOWNSLOPE_TERRAIN = {
        "type": "rough_down_slope",
        "difficulty": 0.9 ## [0.5, 0.75, 0.9]
    }

@dataclass
class RoughDownslopeConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: ROUGH_DOWNSLOPE_TERRAIN 
    )

STAIRS_UP_TERRAIN = {
        "type": "stairs_up",
        "difficulty": 0.5 ## [0.5, 0.75, 0.9]
    }

@dataclass
class StairsUpConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: STAIRS_UP_TERRAIN 
    )

STAIRS_DOWN_TERRAIN = {
        "type": "stairs_down",
        "difficulty": 0.5 ## [0.5, 0.75, 0.9]
    }

@dataclass
class StairsDownConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: STAIRS_DOWN_TERRAIN 
    )

DISCRETE_TERRAIN = {
        "type": "discrete",
        "difficulty": 0.5 ## [0.5, 0.75, 0.9]
    }

@dataclass
class DiscreteConfig(TerrainConfig):
    mesh_type: str = "trimesh"
    selected: bool = True
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: float = 25. # [m]

    # measured points constitute 1m x 1.6m rectangle (without center line)
    measured_points_x: Tuple[float, ...] = (-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    measured_points_y: Tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    max_init_terrain_level: int = 5
    terrain_length: float = 8.
    terrain_width: float = 8.
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    slope_threshold: float = 0.75

    terrain_kwargs: Dict[str, Any] = field(
        default_factory=lambda: DISCRETE_TERRAIN 
    )