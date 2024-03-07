import isaacgym
import logging
import os.path as osp
import time

from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from typing import Dict
from pydantic import TypeAdapter
import torch

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig, ControlConfig,
                                 AssetConfig, InitStateConfig)
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)

from legged_gym.wheeled_gym.hound import Hound

from isaacgym import gymapi, gymutil
gym = gymapi.acquire_gym()

joint_names = ['chassis_to_back_left_wheel',
               'chassis_to_back_right_wheel',
               'front_left_hinge_to_wheel',
               'front_right_hinge_to_wheel',
               'chassis_to_front_left_hinge',
               'chassis_to_front_right_hinge']
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
    init_joint_angles: Dict[str, float] = field(default_factory=lambda: {k:0. for k in joint_names})
    replace_cylinder_with_capsule: bool = False
    control_type: str = "T"
    num_actions: int = 6
    p_gains: Dict[str, float] = field(default_factory=lambda: {k:0.1 for k in joint_names})
    d_gains: Dict[str, float] = field(default_factory=lambda: {k:0.1 for k in joint_names})

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    task: TaskConfig = empty_cfg(TaskConfig)(
        _target_ = "${target}",
        asset = empty_cfg(AssetConfig)(
            file = "${asset_file}",
            replace_cylinder_with_capsule = "${replace_cylinder_with_capsule}"
        ),
        control = empty_cfg(ControlConfig)(
            control_type = "${control_type}",
            # stiffness = "${p_gains}",
            # damping = "${d_gains}"
        ),
        env = empty_cfg(EnvConfig)(
            num_envs = "${num_envs}",
            num_actions = "${num_actions}",
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
        # loaded_cfg,   # loads values from the previous experiment if not specified in the top-level config
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
    env: Hound = hydra.utils.instantiate(task_cfg)
    env.reset()
    obs = env.get_observations()

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
    current_time = time.time()
    num_steps = int(cfg.episode_length_s / env.dt)
    for i in range(num_steps):
        # actions = policy(obs.detach())

        rand_actions = torch.rand(env.actions.shape) * .5
        rand_actions[..., [2,4]] = (torch.rand(env.actions[..., 0].shape)*2 - 1).unsqueeze(-1)
        # rand_actions = torch.zeros(env.actions.shape)
        obs, _, _, _, infos, *_ = env.step(rand_actions)

        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

    log.info("8. Exit Cleanly")
    env.exit()

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
