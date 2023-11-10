from legged_gym import LEGGED_GYM_ROOT_DIR

import os
import copy
import torch
import numpy as np
import random
from pathlib import Path

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

def parse_path(path):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
       return path
    return os.path.join(LEGGED_GYM_ROOT_DIR, path)


def get_load_path(root, checkpoint=-1):
    root = parse_path(root)
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_path = os.path.join(root, model)
    return load_path

def get_latest_experiment_path(root: str) -> str:
    root = parse_path(root)
    latest_config_filepath = max(Path(root).rglob("resolved_config.yaml"), key=lambda f: f.stat().st_ctime)
    return latest_config_filepath.parent.as_posix()

def export_policy_as_jit(actor_critic, path):
    path = parse_path(path)
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
        path = parse_path(path)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


