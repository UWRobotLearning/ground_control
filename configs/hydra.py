from dataclasses import dataclass, field
from typing import Dict,List,Any,Tuple
from hydra.conf import HydraConf
from hydra.conf import HelpConf, HydraHelpConf, RuntimeConf
from hydra._internal.core_plugins.basic_launcher import BasicLauncherConf
from hydra._internal.core_plugins.basic_sweeper import BasicSweeperConf
from omegaconf import OmegaConf, MISSING

from configs.definitions import TaskConfig  # Needed to import the OmegaConf resolvers
from legged_gym.utils.helpers import get_script_name


### ======================= Hydra Configuration  =============================

OmegaConf.register_new_resolver("slash_to_dot", lambda dir: dir.replace("/", "."))
OmegaConf.register_new_resolver("get_script_name", get_script_name)

@dataclass
class ExperimentHydraConfig(HydraConf):
    root_dir_name: str = "${from_repo_root: ${oc.select: logging_root,../experiment_logs}}/${get_script_name:}"
    new_override_dirname: str = "${slash_to_dot: ${hydra:job.override_dirname}}"
    run: Dict = field(default_factory=lambda: {
        # A more sophisticated example:
        #"dir": "${hydra:root_dir_name}/${hydra:new_override_dirname}/seed=${seed}/${now:%Y-%m-%d_%H-%M-%S}",
        # Default behavior logs by date and script name:
        "dir": "${hydra:root_dir_name}/${now:%Y-%m-%d_%H-%M-%S}",
        }
    )

    sweep: Dict = field(default_factory=lambda: {
        "dir": "${..root_dir_name}/multirun/${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra:new_override_dirname}",
        }
    )

    job: Dict = field(default_factory=lambda: {
        "config": {
            "override_dirname": {
                "exclude_keys": [
                    "sim_device",
                    "rl_device",
                    "headless",
                ]
            }
        },
        "chdir": True
    })

    # This specifically configures the "joblib" launcher
    # Note: for now, parallelism is disabled (hence n_jobs=1) due to a blocker related to:
    # https://github.com/facebookresearch/hydra/discussions/2186 
    #launcher: Dict = field(default_factory=lambda: {
    #    "n_jobs": "1",
    #    "pre_dispatch": "1*n_jobs",
    #})

    #sweeper: Dict = field(default_factory=lambda: {
    #    "other": "2",
    #    "something": "1*n_jobs",
    #})

# === TODO:====
# ref this issue: https://github.com/facebookresearch/hydra/issues/1830
   # # dataclasses for basic configs are located @ https://github.com/facebookresearch/hydra/tree/main/hydra/_internal/core_plugins
   # # dataclasses for other options are located @ https://github.com/facebookresearch/hydra/tree/main/plugins
   # # Ex:
   # # [Launcher] JobLib: https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_joblib_launcher/hydra_plugins/hydra_joblib_launcher/config.py
   # # [Sweeper] Optuna: https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_optuna_sweeper/hydra_plugins/hydra_optuna_sweeper/config.py
   # launcher: Any = BasicLauncherConf()
   # sweeper: Any = BasicSweeperConf() # ex swap for OptunaConf() 

   # hydra_logging: Dict = field(default_factory=lambda: {
   #     "version": 1,
   # }) 
   # job_logging: Dict = field(default_factory=lambda: {
   #     "version": 1,
   # })
   # runtime: RuntimeConf = RuntimeConf(
   #     version=0.1,
   #     version_base="---",
   #     cwd="",
   #     config_sources=[],
   #     output_dir="",
   # )
   # help: HelpConf = HelpConf(
   #     app_name="nada",
   #     header="---",
   #     footer="---",
   #     template="---",
   # )
   # hydra_help: HydraHelpConf = HydraHelpConf(
   #     hydra_help="nada",
   #     template="---",
   # )


