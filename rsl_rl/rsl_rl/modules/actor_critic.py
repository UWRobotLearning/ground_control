import logging
import numpy as np
from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from rsl_rl.modules.utils import orthogonal_init

log = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        init_noise_std: float,
        fixed_std: bool,
        ortho_init: bool,
        last_actor_layer_scaling: float,
        actor_hidden_dims: Tuple[int, ...],
        critic_hidden_dims: Tuple[int, ...],
        activation: str
    ):
        super().__init__()

        activation_str = activation
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l+1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # orthogonal initialization from Stable-Baselines3
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py
        if ortho_init:
            # ELU gain calculated from experiments with https://discuss.pytorch.org/t/calculate-gain-tanh/20854/3
            gain = 1.12 if activation_str == "elu" else nn.init.calculate_gain(activation_str)
            module_gains = {
                self.actor[:-1]: gain,
                self.critic[:-1]: gain,
                self.critic[-1]: 1.
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.actor[-1].apply(partial(self.init_weights, gain=last_actor_layer_scaling))

        log.info(f"Actor MLP: {self.actor}")
        log.info(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        log_std = np.log(init_noise_std) * torch.ones(num_actions)
        self.log_std = torch.tensor(log_std) if fixed_std else nn.Parameter(log_std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1.):
        """
        Orthogonal initialization
        """
        if isinstance(module, nn.Linear):
            orthogonal_init(module, gain=gain)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        return self.log_std.exp()
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "leaky_relu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
