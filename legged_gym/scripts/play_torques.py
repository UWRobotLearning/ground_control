import isaacgym # need to import this before torch
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Tuple
import os
import time

# hydra / config related imports
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from configs.definitions import TaskConfig, TrainConfig, ControlConfig, AssetConfig
from configs.overrides.terrain import FlatTerrainConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig
from configs.hydra import ExperimentHydraConfig

from legged_gym.envs.a1 import A1
from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import (set_seed, save_resolved_config_as_pkl, 
                                      save_config_as_yaml, from_repo_root)

import torch

@dataclass
class PlayTorquesScriptConfig:
    # Uncomment below to launch through joblib or add a different launcher / sweeper
    #defaults: Tuple[Any] = (
    #    "_self_",
    #    {"override hydra/launcher": "joblib"},
    #    
    #)

    seed: int = 1
    torch_deterministic: bool = False
    num_envs: int = 1
    iterations: int = 5000
    episode_length_s: float = 200.
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    headless: bool = False
    checkpoint_root: str = ""
    logging_root: str = from_repo_root("../experiment_logs")
    torques_root: str = from_repo_root("../recorded_torques.pickle")

    task: TaskConfig = LocomotionTaskConfig(
        control = ControlConfig(
            control_type = "T",
        ),
        terrain = FlatTerrainConfig(),
        asset = AssetConfig(
            terminate_after_contacts_on = ()
        )
    )
    train: TrainConfig = TrainConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=PlayTorquesScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: PlayTorquesScriptConfig) -> None:
    log.info("1. Printing and serializing frozen PlayTorquesScriptConfig")
    OmegaConf.resolve(cfg)
    # Type-checking (and other validation if defined) via Pydantic
    cfg = TypeAdapter(PlayTorquesScriptConfig).validate_python(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)
    #save_resolved_config_as_pkl(cfg)

    log.info(f"2. Preparing environment and runner.")
    task_cfg = cfg.task
    env: A1 = hydra.utils.instantiate(task_cfg)
    env.reset()
    obs = env.get_observations()
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    log.info(f"3. Running interactive play script.")
    current_time = time.time()
    num_steps = int(cfg.episode_length_s / env.dt)
    for i in range(num_steps):
        actions = torch.zeros((1,12))
        obs, _, _, _, infos, *_ = env.step(actions.detach())
        
        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

    log.info("4. Exit Cleanly")
    env.exit()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()