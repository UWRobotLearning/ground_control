import isaacgym
import logging
import os.path as osp
import time
import os
import torch
import math
import numpy as np

from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from typing import Dict
from pydantic import TypeAdapter

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig,
                                 AssetConfig, InitStateConfig)
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)

from legged_gym.wheeled_gym.hound import Hound

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
gym = gymapi.acquire_gym()

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)
@dataclass
class PlayScriptConfig:
    target: str = "legged_gym.wheeled_gym.hound.Hound"
    # asset_file: str = from_repo_root("resources/mushr_description/robots/mushr_nano.urdf")
    asset_file: str = from_repo_root("resources/mushr_description/robots/racecar-mit-phys.urdf")
    # asset_file: str = from_repo_root("resources/jackal/jackal.urdf")
    checkpoint_root: str = from_repo_root("../experiment_logs/train")
    logging_root: str = from_repo_root("../experiment_logs")
    export_policy: bool = True
    num_envs: int = 50
    use_joystick: bool = True
    episode_length_s: float = 200.
    checkpoint: int = -1
    headless: bool = False 
    device: str = "cpu"
    init_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        'chassis_to_back_left_wheel': 0.0,
        'chassis_to_back_right_wheel': 0.0,
        'chassis_to_front_left_hinge': 0.0,
        'front_left_hinge_to_wheel': 0.0,
        'chassis_to_front_right_hinge': 0.0,
        'front_right_hinge_to_wheel': 0.0,
    })

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    task: TaskConfig = empty_cfg(TaskConfig)(
        _target_ = "${target}",
        asset = empty_cfg(AssetConfig)(
            file = "${asset_file}"
        ),
        env = empty_cfg(EnvConfig)(
            num_envs = "${num_envs}"
        ),
        observation = empty_cfg(ObservationConfig)(
            get_commands_from_joystick = "${use_joystick}"
        ),
        sim = empty_cfg(SimConfig)(
            device = "${device}",
            use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
            headless = "${headless}",
            physx = empty_cfg(SimConfig.PhysxConfig)(
                use_gpu = "${evaluate_use_gpu: ${task.sim.device}}"
            )
        ),
        terrain = empty_cfg(TerrainConfig)(
            curriculum = False
        ),
        init_state = empty_cfg(InitStateConfig)(
            default_joint_angles = "${init_joint_angles}"
        ),
        noise = NoNoiseConfig(),
        domain_rand = NoDomainRandConfig()
    )
    train: TrainConfig = empty_cfg(TrainConfig)(
        device = "${device}",
        log_dir = "${hydra:runtime.output_dir}",
        runner = empty_cfg(RunnerConfig)(
            checkpoint="${checkpoint}"
        ),
    )

cs = ConfigStore.instance()
cs.store(name="config", node=PlayScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: PlayScriptConfig):
    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    latest_config_filepath = osp.join(experiment_path, "resolved_config.yaml")
    log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    loaded_cfg = OmegaConf.load(latest_config_filepath)

    log.info("2. Merging loaded config, defaults and current top-level config.")
    del(loaded_cfg.hydra) # Remove unpopulated hydra configuration key from dictionary
    default_cfg = {"task": TaskConfig(), "train": TrainConfig(), "init_state": InitStateConfig()}  # default behaviour as defined in "configs/definitions.py"
    merged_cfg = OmegaConf.merge(
        default_cfg,  # loads default values at the end if it's not specified anywhere else
        loaded_cfg,   # loads values from the previous experiment if not specified in the top-level config
        cfg           # highest priority, loads from the top-level config dataclass above
    )
    # Resolves the config (replaces all "interpolations" - references in the config that need to be resolved to constant values)
    # and turns it to a dictionary (instead of DictConfig in OmegaConf). Throws an error if there are still missing values.
    merged_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    # Creates a new PlayScriptConfig object (with type-checking and optional validation) using Pydantic.
    # The merged config file (DictConfig as given by OmegaConf) has to be recursively turned to a dict for Pydantic to use it.
    cfg = TypeAdapter(PlayScriptConfig).validate_python(merged_cfg_dict)
    # Alternatively, you should be able to use "from pydantic.dataclasses import dataclass" and replace the above line with
    # cfg = PlayScriptConfig(**merged_cfg_dict)
    log.info(f"3. Printing merged cfg.")
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)

    # Handle codesaving after config has been processed.
    log.info(f"4. Running autocommit/codesave if enabled.")

    log.info(f"5. Preparing environment and runner.")
    task_cfg = cfg.task

    physics_engine = gymapi.SIM_PHYSX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim = gym.create_sim(0, 0, physics_engine, sim_params)

    asset_root = os.path.dirname(cfg.task.asset.file)
    asset_file = os.path.basename(cfg.task.asset.file)
    asset_options = gymapi.AssetOptions()

    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    asset = robot_asset

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # create an array of DOF states that will be used to update the actors
    num_dofs = gym.get_asset_dof_count(asset)
    print(num_dofs)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states['pos']

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props['stiffness']
    dampings = dof_props['damping']
    armatures = dof_props['armature']
    has_limits = dof_props['hasLimits']
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    # Print DOF properties
    for i in range(num_dofs):
        print("DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % stiffnesses[i])
        print("  Damping:  %r" % dampings[i])
        print("  Armature:  %r" % armatures[i])
        print("  Limited?  %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower   %f" % lower_limits[i])
            print("    Upper   %f" % upper_limits[i])


    spacing = 5
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    start_pose = gymapi.Transform()
    env_handle = gym.create_env(sim, env_lower, env_upper, 1)
    hound_handle = gym.create_actor(env_handle, robot_asset, start_pose, "hound", 0, 1)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # cam_pos = gymapi.Vec3(1, 1, 1)
    # cam_target = gymapi.Vec3(-1, -1, -1)
    # gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.
    plane_params.dynamic_friction = 1.
    gym.add_ground(sim, plane_params)

    current_time = time.time()
    dt = 1./60.
    while not gym.query_viewer_has_closed(viewer):
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        rand_actions = torch.rand(6) * .5
        # rand_actions = torch.zeros(6)
        rand_actions[..., [2,4]] = 0.
        print(rand_actions)
        rand_actions = gymtorch.unwrap_tensor(rand_actions)
        gym.set_dof_actuation_force_tensor(sim, rand_actions)
        gym.refresh_dof_state_tensor(sim)
        gym.simulate(sim)

        duration = time.time() - current_time
        if duration < dt:
            time.sleep(dt - duration)
        current_time = time.time()
    # runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    # experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    # resume_path = get_load_path(experiment_path, checkpoint=cfg.train.runner.checkpoint)
    # log.info(f"6. Loading policy checkpoint from: {resume_path}.")
    # runner.load(resume_path)
    # policy = runner.get_inference_policy(device=env.device)

    # if cfg.export_policy:
    #     export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
    #     log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    # log.info(f"7. Running interactive play script.")
    # current_time = time.time()
    # num_steps = int(cfg.episode_length_s / env.dt)
    # for i in range(num_steps):
    #     actions = policy(obs.detach())
    #     obs, _, _, _, infos, *_ = env.step(actions.detach())

    #     duration = time.time() - current_time
    #     if duration < env.dt:
    #         time.sleep(env.dt - duration)
    #     current_time = time.time()

    log.info("8. Exit Cleanly")

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
