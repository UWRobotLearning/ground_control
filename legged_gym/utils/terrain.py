import numpy as np
from numpy.random import choice
from scipy import interpolate
import noise
import random
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

import os
import time # Ege
from tqdm import tqdm # Ege
import pickle # Ege

from isaacgym import terrain_utils
from configs.definitions import TerrainConfig

class Terrain:
    def __init__(self, cfg: TerrainConfig, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.local_dirname = os.path.dirname(os.path.abspath(__file__))
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

    # ============= Terrain Multiplexing ============================
    #TODO: needs to be cleaned up and clearer
        if hasattr(cfg, 'valley'):
            #TODO: see whats going on with these options. seems messy
            # Ege - added new terrain option
            #self.ready_made_valley_terrain()
            self.valley_terrain()
        elif hasattr(cfg, 'plane_slope'):
            self.slope_terrain(cfg.plane_slope)
        elif cfg.curriculum:
            #self.curriculum()
            #self.ready_made_valley_terrain()  # Ege - added terrain to curriculum setting as well
            #self.valley_terrain()
            self.ready_made_semivalley_terrain()
            # Ege - trying gap terrain
            '''
            terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
            gap_terrain(terrain, 1)
            '''
        elif cfg.selected:
            self.selected_terrain()
        else:
            #self.randomized_terrain()
            self.ready_made_semivalley_terrain()

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_threshold)
    # / ============= Terrain Multiplexing ============================

    def randomized_terrain(self):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            # TODO: eval is dangerous!
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    # Ege - added ability to generate valley-like terrain
    def valley_terrain(self):
        width = self.width_per_env_pixels
        gap_size = int(0 * width)
        slope_size = int(0.5 * width)
        x_left = slope_size
        x_right = x_left + gap_size
        begin_time = time.time()
        roughness = np.zeros((self.cfg.num_rows, self.cfg.num_cols))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain("terrain",
                                width=width,
                                length=width,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
            slope = 0.1 * i
            slope *= self.cfg.horizontal_scale / self.cfg.vertical_scale

            #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
            noise_level = j * np.sqrt(1 + slope ** 2) / 6
            noises = np.random.normal(0, noise_level, size=(2 * x_left, width)) #- (noise_level / 2)

            # flat part at the bottom
            terrain.height_field_raw[x_left : x_right, :] = -1. * slope * slope_size

            for x in range(x_left):
                for y in range(0, width):
                    terrain.height_field_raw[x, y] = (-1. * slope * x) + noises[x][y]
                    terrain.height_field_raw[width - x - 1, y] = (-1. * slope * x) + noises[x + x_left][y]

            if self.cfg.record_roughness:
                roughness[i,j] += residual_variance(terrain.height_field_raw[:x_left, :]) / 2 #terrain_roughness_index(terrain.height_field_raw[:x_left, :]) / 2
                roughness[i,j] += residual_variance(terrain.height_field_raw[x_right:, :]) / 2 # terrain_roughness_index(terrain.height_field_raw[x_right:, :]) / 2

            self.add_terrain_to_map(terrain, i, j)
        print("Time to make valley terrain: {} sec".format(time.time() - begin_time) )
        if self.cfg.record_roughness:
            np.savetxt("roughness.csv", roughness, delimiter=",")

    def ready_made_valley_terrain(self):
        begin_time = time.time()
        width = self.width_per_env_pixels
        slope_size = int(0.5 * width)
        x_left = slope_size
        x_right = x_left
        heightmaps = None
        with open(os.path.join(self.local_dirname, "terrain_tiles.pickle"), "rb") as f:
            heightmaps = pickle.load(f)
        #heightmaps = np.load("utils/terrain.npz")
        roughness = np.zeros((self.cfg.num_rows, self.cfg.num_cols))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain("terrain",
                                width=width,
                                length=width,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

            #start_x = i * width
            #start_y = j * width
            #terrain.height_field_raw[:,:] = terrain_arrs['heightmap'][start_x:start_x+width, start_y:start_y+width] / self.cfg.vertical_scale
            terrain.height_field_raw[:,:] = heightmaps[j][i][:width,:width] / self.cfg.vertical_scale
            if self.cfg.record_roughness:
                roughness[i,j] += residual_variance(terrain.height_field_raw[:x_left, :]) / 2 #terrain_roughness_index(terrain.height_field_raw[:x_left, :]) / 2
                roughness[i,j] += residual_variance(terrain.height_field_raw[x_right:, :]) / 2 # terrain_roughness_index(terrain.height_field_raw[x_right:, :]) / 2

            self.add_terrain_to_map(terrain, i, j)
        print("Time to make valley terrain: {} sec".format(time.time() - begin_time) )
        if self.cfg.record_roughness:
            np.savetxt("roughness.csv", roughness, delimiter=",")
    # Ege
    def slope_terrain(self, slope):
        begin_time = time.time()
        slope *= self.cfg.horizontal_scale / self.cfg.vertical_scale
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            width = self.width_per_env_pixels
            terrain = terrain_utils.SubTerrain("terrain",
                                width=width,
                                length=width,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

            for x in range(width):
                terrain.height_field_raw[x, :] = (i * width + x) * slope
            self.add_terrain_to_map(terrain, i, j)
        print("Time to make slope terrain: {} sec".format(time.time() - begin_time) )

    def semivalley_terrain(self):
        width = self.width_per_env_pixels
        begin_time = time.time()
        roughness = np.zeros((self.cfg.num_rows, self.cfg.num_cols))
        heightmaps = np.zeros((self.cfg.num_rows, self.cfg.num_cols, width, width))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain("terrain",
                                width=width,
                                length=width,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
            slope = 0.1 * i
            slope *= self.cfg.horizontal_scale / self.cfg.vertical_scale

            #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
            noise_level = j * np.sqrt(1 + slope ** 2) / 6
            noises = np.random.normal(0, noise_level, size=(width, width)) #- (noise_level / 2)

            for x in range(width):
                for y in range(0, width):
                    terrain.height_field_raw[x, y] = (-1. * slope * x) + noises[x][y]
            if self.cfg.record_roughness:
                roughness[i,j] += residual_variance(terrain.height_field_raw) #terrain_roughness_index(terrain.height_field_raw) 
            if self.cfg.record_heightmaps:
                heightmaps[i, j, :width, :width] = terrain.height_field_raw
            self.add_terrain_to_map(terrain, i, j)
        print("Time to make valley terrain: {} sec".format(time.time() - begin_time) )
        if self.cfg.record_roughness:
            np.savetxt("roughness.csv", roughness, delimiter=",")
        if self.cfg.record_heightmaps:
            with open(os.path.join(self.local_dirname, "terrain_tiles.pickle"), "wb+") as f:
                pickle.dump(heightmaps, f)

    # Ege 
    def ready_made_semivalley_terrain(self):
        begin_time = time.time()
        width = self.width_per_env_pixels
        slope_size = int(0.5 * width)
        x_left = slope_size
        x_right = x_left
        heightmaps = None
        with open(os.path.join(self.local_dirname, "terrain_tiles.pickle"), "rb") as f:
            heightmaps = pickle.load(f)
        #heightmaps = np.load("utils/terrain.npz")
        roughness = np.zeros((self.cfg.num_rows, self.cfg.num_cols))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain("terrain",
                                width=width,
                                length=width,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

            #start_x = i * width
            #start_y = j * width
            #terrain.height_field_raw[:,:] = terrain_arrs['heightmap'][start_x:start_x+width, start_y:start_y+width] / self.cfg.vertical_scale
            terrain.height_field_raw[:,:] = heightmaps[i][j][:width,:width]
            if self.cfg.record_roughness:
                roughness[i,j] += residual_variance(terrain.height_field_raw) #terrain_roughness_index(terrain.height_field_raw) 
            self.add_terrain_to_map(terrain, i, j)
        print("Time to make ready-made semivalley terrain: {} sec".format(time.time() - begin_time) )
        if self.cfg.record_roughness:
            np.savetxt("roughness.csv", roughness, delimiter=",")

    def make_terrain(self, choice, difficulty):
        # TODO: why does this exist if we also have the terrain multiplexing stuff at the top?
        # difficulty is between 0 and 1
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4 # in radians (affects straight slope and noisy terrain)
        step_height = 0.05 + 0.18 * difficulty # 23 cm height (may want to decrease for blind locomotion, e.g., to 13 cm)
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-self.cfg.terrain_noise_magnitude,
                                                 max_height=self.cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


# ======== Terrain Helper Functions ========

def rotate_to_best_fit_plane(heightmap):
    points = np.stack(([], [], []), axis=1)
    for index, height in np.ndenumerate(heightmap):
        points = np.vstack((points, np.array([index[0], index[1], height])))
    pca = PCA(n_components=2)
    pca.fit(points)
    projected = pca.inverse_transform(pca.transform(points))
    distances = np.linalg.norm(points - projected, axis=1) * np.sign(points[:,2] - projected[:,2])
    rotated_heights = np.zeros_like(heightmap)
    for i in range(points.shape[0]):
        x, y, _ = points[i]
        rotated_heights[int(x), int(y)] = distances[i]
    return rotated_heights

def residual_variance(heightmap):
    heightmap = rotate_to_best_fit_plane(heightmap)
    return np.mean(np.square(heightmap))

# looks at the average difference between each pixel that has 8 neighbors
# with those neighbors, then takes the average of that as a measure of roughness
def terrain_roughness_index(heightmap):
    heightmap = rotate_to_best_fit_plane(heightmap)
    total_diff = 0.0
    for i in range(1, heightmap.shape[0] - 1):
        for j in range(1, heightmap.shape[1] - 1):
            neighbors = np.array([heightmap[i+1, j-1], heightmap[i+1, j], heightmap[i+1, j+1],
                                  heightmap[i-1, j-1], heightmap[i-1, j], heightmap[i-1, j+1],
                                  heightmap[i, j+1], heightmap[i, j-1]])
            total_diff += np.linalg.norm(heightmap[i][j] - neighbors)
    return total_diff / float((heightmap.shape[0] - 1) * (heightmap.shape[1] - 1))
