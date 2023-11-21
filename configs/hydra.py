from dataclasses import dataclass, field
from typing import Dict
from hydra.conf import HydraConf
from omegaconf import OmegaConf

from configs.definitions import TaskConfig  # Needed to import the OmegaConf resolvers
from legged_gym.utils.helpers import get_script_name

### ======================= Hydra Configuration  =============================

OmegaConf.register_new_resolver("slash_to_dot", lambda dir: dir.replace("/", "."))
OmegaConf.register_new_resolver("get_script_name", get_script_name)

@dataclass
class ExperimentHydraConfig(HydraConf):
    root_dir_name: str = "${from_repo_root: ${oc.select: logging_root,../experiment_logs}}"
    load_dir_name: str = "${hydra:root_dir_name}/train"
    new_override_dirname: str = "${slash_to_dot: ${hydra:job.override_dirname}}"
    run: Dict = field(default_factory=lambda: {
        # A more sophisticated example:
        #"dir": "${hydra:root_dir_name}/${hydra:new_override_dirname}/seed=${seed}/${now:%Y-%m-%d_%H-%M-%S}",
        # Default behavior logs by date and script name:
        "dir": "${hydra:root_dir_name}/${get_script_name:}/${now:%Y-%m-%d_%H-%M-%S}",
        }
    )

    sweep: Dict = field(default_factory=lambda: {
        "dir": "${hydra:root_dir_name}",
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
        }
    })

