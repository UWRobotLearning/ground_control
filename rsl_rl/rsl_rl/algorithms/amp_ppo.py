import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple
from typing import Optional

from rsl_rl.algorithms.ppo import PPO, Metrics
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import MocapLoader
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils.utils import Normalizer

class AMPPPO(PPO):
    actor_critic: ActorCritic
    def __init__(
        self,
        actor_critic: ActorCritic,
        discriminator: AMPDiscriminator,
        amp_data: MocapLoader,
        amp_normalizer: Normalizer,
        num_learning_epochs: int,
        num_mini_batches: int,
        clip_param: float,
        gamma: float,
        lam: float,
        value_loss_coef: float,
        entropy_coef: float,
        learning_rate: float,
        max_learning_rate: float,
        min_learning_rate: float,
        max_grad_norm: float,
        use_clipped_value_loss: bool,
        schedule: str,
        desired_kl: float,
        device: float,
        amp_replay_buffer_size: int,
        min_std: Optional[torch.Tensor] = None,
    ):
        super().__init__(actor_critic, value_loss_coef, use_clipped_value_loss, clip_param,
                         entropy_coef, num_learning_epochs, num_mini_batches, learning_rate,
                         max_learning_rate, min_learning_rate, schedule, gamma, lam, desired_kl,
                         max_grad_norm, device)

        self.min_log_std = min_std.log() if min_std is not None else None

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Optimizer for policy and discriminator.
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'}]
        self.optimizer = optim.Adam(params, lr=learning_rate)


    def act(self, obs, critic_obs, amp_obs):
        self.amp_transition.observations = amp_obs
        return super().act(obs, critic_obs)
    
    def process_env_step(self, rewards, dones, infos, amp_obs):
        super().process_env_step(rewards, dones, infos)

        # Record the transition for AMP
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
    
    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_kl_div = 0
        mean_clip_fraction = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                should_continue, kl_mean = self._check_trust_region(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)
                if not should_continue:
                    break
                surrogate_loss, clip_fraction = self._surrogate_loss(advantages_batch, actions_log_prob_batch, old_actions_log_prob_batch)
                value_loss = self._value_function_loss(returns_batch, value_batch, target_values_batch)
                amp_loss, grad_pen_loss, policy_d, expert_d = self._amp_loss(sample_amp_policy, sample_amp_expert)

                # Compute total loss.
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    amp_loss + grad_pen_loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if not self.actor_critic.fixed_std and self.min_log_std is not None:
                    self.actor_critic.log_std.data.clamp_(min=self.min_log_std)
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_log_std)

                num_updates += 1
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                mean_kl_div += kl_mean.item()
                mean_clip_fraction += clip_fraction.item()

        # Explained variance
        values_pred, values_true = self.storage.values.flatten(), self.storage.returns.flatten()
        explained_variance = 1 - torch.var(values_true - values_pred) / torch.var(values_true)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_kl_div /= num_updates
        mean_clip_fraction /= num_updates
        self.storage.clear()

        return Metrics(
            values=(mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred,
                    mean_expert_pred, mean_kl_div, mean_clip_fraction, explained_variance, num_updates),
            logger_names=("Value function loss", "Surrogate loss", "AMP loss", "AMP grad pen loss",
                          "AMP mean policy pred", "AMP mean expert pred", "KL divergence", "Clip fraction",
                          "Value function explained variance", "Number of policy updates"),
            write_names=("Loss/value_function", "Loss/surrogate", "Loss/AMP", "Loss/AMP_grad", "AMP/policy_pred",
                         "AMP/expert_pred", "Policy/kl", "Policy/clip_fraction",
                         "Loss/value_function_explained_variance", "Policy/num_updates")
        )

    def _amp_loss(
            self,
            sample_amp_policy: Tuple[torch.Tensor, torch.Tensor],
            sample_amp_expert: Tuple[torch.Tensor, torch.Tensor]
        ):
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                    self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                    self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = (expert_loss + policy_loss) / 2.
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10.)

            return amp_loss, grad_pen_loss, policy_d, expert_d
