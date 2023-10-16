import isaacgym # need to import this before torch
import logging
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Tuple

# config related imports
from configs.definitions import TaskConfig
from configs.definitions import TrainConfig
from configs.definitions import EnvConfig
from configs.overrides.locomotion_task import LocomotionTaskConfig

from legged_gym.envs.a1 import A1
from rsl_rl.runners import OnPolicyRunner
from legged_gym.utils.helpers import set_seed, get_load_path, get_latest_experiment_path

@dataclass
class TrainScriptConfig:
    """
    Top level attributes are either:
    1. used directly in the script such as "cfg.seed".
    2. aliases for attributes inside nested dataclasses such as "cfg.env.num_envs"
    Aliasing is enabled by using hydra. Without hydra, these attributes
    must be confiugred by directly instantiating the nested classes such as "EnvConfig" below.
    """
    seed: int = 1
    torch_deterministic: bool = False
    rl_device: str = "cuda:0"
    # sim_device: str = "cuda:0"
    # num_envs: int = 4096 
    # iterations: int = 5000 
    # headless: bool = False
    # checkpoint_root: str = ""
    # logging_root: str = "experiment_logs"

    task: TaskConfig = LocomotionTaskConfig(
        env=EnvConfig(
            num_envs=16,
        )
    )
    train: TrainConfig = TrainConfig()

def main():

    # 1. create instance of config object from structure defined above
    cfg = TrainScriptConfig()

    log.info("Initializing Env and Runner") # use log instead of print wherever possible
    set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # 2. Example of being instantiated by passing dataclass->dict demotion and then unpacking
    # Not the best, but useful if there are a lot of terms and your class does not accept dictconfigs
    env = A1(**asdict(cfg.task))

    # 3. Example of being instantiated by passing each 'sub-config' separately
    runner = OnPolicyRunner(
        env=env,
        policy=cfg.train.policy,
        algorithm=cfg.train.algorithm,
        runner=cfg.train.runner,
        log_dir=cfg.train.log_dir,
        device=cfg.rl_device,
    )

    # 4. Resuming from previous checkpoint
    if cfg.train.runner.resume_root != "":
        experiment_dir = get_latest_experiment_path(cfg.train.runner.resume_root)
        resume_path = get_load_path(experiment_dir, checkpoint=cfg.train.runner.checkpoint)
        log.info(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    # 5. Run the training loop defined in rsl_rl: OnPolicyRunner.learn()
    log.info("Now Training")
    runner.learn()

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
