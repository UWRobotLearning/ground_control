from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

@dataclass
class Metrics:
    values: Tuple[float, ...]
    logger_names: Tuple[str, ...]
    write_names: Tuple[str, ...]

class PPO:
    actor_critic: ActorCritic
    def __init__(
        self,
        actor_critic: ActorCritic,
        value_loss_coef: float,
        use_clipped_value_loss: bool,
        clip_param: float,
        entropy_coef: float,
        num_learning_epochs: int,
        num_mini_batches: int,
        learning_rate: float,
        max_learning_rate: float,
        min_learning_rate: float,
        schedule: str,
        gamma: float,
        lam: float,
        desired_kl: float,
        max_grad_norm: float,
        device: str
    ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        with torch.no_grad():
            self.transition.actions = self.actor_critic.act(obs)
            self.transition.values = self.actor_critic.evaluate(critic_obs)
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor_critic.action_mean
            self.transition.action_sigma = self.actor_critic.action_std
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_critic_obs)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl_div = 0
        mean_clip_fraction = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                should_continue, kl_mean = self._check_trust_region(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)
                if not should_continue:
                    break
                surrogate_loss, clip_fraction = self._surrogate_loss(advantages_batch, actions_log_prob_batch, old_actions_log_prob_batch)
                value_loss = self._value_function_loss(returns_batch, value_batch, target_values_batch)

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                num_updates += 1
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_kl_div += kl_mean.item()
                mean_clip_fraction += clip_fraction.item()

        # Explained variance
        values_pred, values_true = self.storage.values.flatten(), self.storage.returns.flatten()
        explained_variance = 1 - torch.var(values_true - values_pred) / torch.var(values_true)

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl_div /= num_updates
        mean_clip_fraction /= num_updates
        self.storage.clear()

        return Metrics(
            values=(mean_value_loss, mean_surrogate_loss, mean_kl_div, mean_clip_fraction, explained_variance, num_updates),
            logger_names=("Value function loss", "Surrogate loss", "KL divergence", "Clip fraction",
                          "Value function explained variance", "Number of policy updates"),
            write_names=("Loss/value_function", "Loss/surrogate", "Policy/kl", "Policy/clip_fraction",
                         "Loss/value_function_explained_variance", "Policy/num_updates")
        )

    def _surrogate_loss(self, advantages: torch.Tensor, actions_log_prob: torch.Tensor, old_actions_log_prob: torch.Tensor):
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        with torch.no_grad():
            clip_fraction = torch.mean((torch.abs(ratio-1) > self.clip_param).float())
        return surrogate_loss, clip_fraction

    def _value_function_loss(self, returns: torch.Tensor, values: torch.Tensor, target_values: torch.Tensor):
        value_losses = torch.square(values - returns)
        if self.use_clipped_value_loss:
            value_clipped = target_values + (values - target_values).clamp(-self.clip_param, self.clip_param)
            value_losses_clipped = torch.square(value_clipped - returns)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = value_losses.mean()
        return value_loss

    def _check_trust_region(self, mu: torch.Tensor, sigma: torch.Tensor, old_mu: torch.Tensor, old_sigma: torch.Tensor):
        with torch.inference_mode():
            delta_log_prob = (torch.log(sigma/old_sigma + 1e-5)
                              + (torch.square(old_sigma)
                              + torch.square(old_mu - mu)) / (2.*torch.square(sigma))
                              - 0.5)
            kl = torch.sum(delta_log_prob, axis=-1)
            kl_mean = torch.mean(kl)

        if self.desired_kl is None:
            return True, kl_mean

        if self.schedule == "adaptive":
            if kl_mean > 2.*self.desired_kl:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate/1.5)
            elif kl_mean < self.desired_kl/2. and kl_mean > 0.:
                self.learning_rate = min(self.max_learning_rate, self.learning_rate*1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            return True, kl_mean
        elif kl_mean > 1.5*self.desired_kl: # early stopping
            return False, kl_mean
