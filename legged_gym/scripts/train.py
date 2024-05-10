import isaacgym # need to import this before torch
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Tuple
import os
import wandb

# hydra / config related imports
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from configs.definitions import TaskConfig, TrainConfig, RunnerConfig, WandBConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig
from configs.hydra import ExperimentHydraConfig

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.a1_recovery_short import A1RecoveryShort
from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import (set_seed, get_load_path, get_latest_experiment_path, save_resolved_config_as_pkl, 
                                      save_config_as_yaml, from_repo_root)




ENV = A1RecoveryShort

@dataclass
class TrainScriptConfig:
    """
    A config used with the `train.py` script. Also has top-level
    aliases for commonly used hyperparameters. Feel free to add
    more aliases for your experiments.
    """

    # Uncomment below to launch through joblib or add a different launcher / sweeper
    #defaults: Tuple[Any] = (
    #    "_self_",
    #    {"override hydra/launcher": "joblib"},
    #    
    #)

    seed: int = 1
    torch_deterministic: bool = False
    num_envs: int = 4096 
    iterations: int = 5000 
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    headless: bool = False
    checkpoint_root: str = ""
    logging_root: str = from_repo_root("../experiment_logs")

    task: TaskConfig = LocomotionTaskConfig()
    train: TrainConfig = TrainConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainScriptConfig) -> None:
    log.info("1. Printing and serializing frozen TrainScriptConfig")
    OmegaConf.resolve(cfg)
    # Type-checking (and other validation if defined) via Pydantic
    cfg = TypeAdapter(TrainScriptConfig).validate_python(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)
    #save_resolved_config_as_pkl(cfg)

    log.info(f"2. Configuring WandB for Experiment Logging")
    wandb_cfg = cfg.train.runner.wandb
    if wandb_cfg.enable:
        wandb_run = wandb.init(
            project=wandb_cfg.project_name, 
            entity=wandb_cfg.entity,
            config=dict(OmegaConf.structured(cfg)),
            job_type='train'
        )
        if wandb_cfg.log_code:
            wandb_run.log_code(
                root=LEGGED_GYM_ROOT_DIR, # save the files in the entire ground_control repo
                # ... only if they end with one of the specified file extensions
                include_fn=lambda path: any([path.endswith(ext) for ext in wandb_cfg.codesave_file_extensions])
            )

    log.info("3. Initializing Env and Runner")
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env: ENV = hydra.utils.instantiate(cfg.task)
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    if cfg.train.runner.resume_root != "":
        experiment_dir = get_latest_experiment_path(cfg.train.runner.resume_root)
        resume_path = get_load_path(experiment_dir, checkpoint=cfg.train.runner.checkpoint)
        log.info(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    log.info("4. Now Training")
    runner.learn()

    log.info("5. Exit Cleanly")
    env.exit()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
