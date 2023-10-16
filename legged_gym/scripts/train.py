import isaacgym # need to import this before torch
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Tuple

# hydra / config related imports
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

#from configs.definitions import TaskConfig
from configs.hydra import HydraTaskConfig
from configs.hydra import HydraLocomotionTaskConfig
from configs.hydra import HydraTrainConfig
from configs.hydra import ExperimentHydraConfig

from legged_gym.envs.a1 import A1
from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import set_seed, get_load_path, get_latest_experiment_path

@dataclass
class TrainScriptConfig:
    """
    A config used with the `train.py` script. Also has top-level
    aliases for commonly used hyperparameters. Feel free to add
    more aliases for your experiments.
    """
    seed: int = 1
    torch_deterministic: bool = False
    num_envs: int = 4096 
    iterations: int = 5000 
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    headless: bool = False
    checkpoint_root: str = ""
    logging_root: str = "experiment_logs"

    task: HydraTaskConfig = HydraLocomotionTaskConfig()
    train: HydraTrainConfig = HydraTrainConfig()

    hydra: ExperimentHydraConfig = ExperimentHydraConfig(
        root_dir_name="${logging_root}",
    )

cs = ConfigStore.instance()
cs.store(name="config", node=TrainScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainScriptConfig) -> None:

    log.info("1. Printing and serializing frozen TrainScriptConfig")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    with open(f"{cfg.train.log_dir}/resolved_config.yaml", "w") as config_file:
        OmegaConf.save(cfg, config_file)
    with open(f"{cfg.train.log_dir}/resolved_config.pkl", "wb") as config_pkl:
        pickle.dump(cfg, config_pkl)
        config_pkl.flush()

    log.info("2. Initializing Env and Runner")
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env: A1 = hydra.utils.instantiate(cfg.task)
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    if cfg.train.runner.resume_root != "":
        experiment_dir = get_latest_experiment_path(cfg.train.runner.resume_root)
        resume_path = get_load_path(experiment_dir, checkpoint=cfg.train.runner.checkpoint)
        log.info(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    log.info("3. Now Training")
    runner.learn()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
