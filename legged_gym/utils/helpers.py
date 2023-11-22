from legged_gym import LEGGED_GYM_ROOT_DIR
import __main__

import os
import copy
import torch
import numpy as np
import random
import pickle
from pathlib import Path
from typing import Type
from dataclasses import fields

from omegaconf import OmegaConf, MISSING

def set_seed(seed, torch_deterministic=False):
    """set seed across modules"""
    if seed < 0:
        seed = 42 if torch_deterministic else np.random.randint(0, 10_000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(torch_deterministic)
    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    return seed

# Saves the current config file as yaml. Checks cfg.train.log_dir for the folder to save,
# and saves it in that folder under the name resolved_config.yaml, trying to resolve all
# the interpolations (indirect references in the config).
def save_config_as_yaml(cfg):
    with open(f"{cfg.train.log_dir}/resolved_config.yaml", "w") as config_file:
        OmegaConf.save(cfg, config_file, resolve=True)

# Saves the current config file (assumed to be already resolved) as pickle (with the extension .pkl). 
# Checks cfg.train.log_dir for the folder to save, and saves it in that folder under the name resolved_config.pkl.
def save_resolved_config_as_pkl(cfg):
    with open(f"{cfg.train.log_dir}/resolved_config.pkl", "wb") as config_pkl:
        pickle.dump(cfg, config_pkl)
        config_pkl.flush()

# Loads a pickle file and returns it from the path specified.
def load_pkl(path):
    with open(path, "rb") as pkl_file:
        return pickle.load(pkl_file)

# Gets the name of the script that is currently being run (used by Hydra for logging location).
def get_script_name():
    return Path(__main__.__file__).stem

# If the given path is not absolute (after resolving user parameters), returns an absolute path
# that is equivalent to the given relative path from the repository root (ground_control).
def from_repo_root(path):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, path))

# Sets all the fields of a dataclass as OmegaConf.MISSING. Used to specify a config dataclass
# where fields that are not specified are taken from defaults or an external file.
def empty_cfg(class_ref: Type):
    args = {key.name: MISSING for key in fields(class_ref)}
    def init(**kwargs):
        args.update(kwargs)
        return class_ref(**args)
    return init

def get_load_path(root, checkpoint=-1):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_path = os.path.join(root, model)
    return load_path

def get_latest_experiment_path(root: str) -> str:
    latest_config_filepath = max(Path(root).rglob("resolved_config.yaml"), key=lambda f: f.stat().st_ctime)
    return latest_config_filepath.parent.as_posix()

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'exported_policy.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


