import time
import os
from collections import deque
import hydra

from torch.utils.tensorboard import SummaryWriter
import torch

from configs.definitions import PolicyConfig
from configs.overrides.amp import (AMPAlgorithmConfig, AMPRunnerConfig)

from legged_gym.envs.a1_amp import A1AMP
from rsl_rl.algorithms import AMPPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import MocapLoader
from rsl_rl.utils.utils import Normalizer

class AMPOnPolicyRunner(OnPolicyRunner):
    env: A1AMP
    cfg: AMPRunnerConfig
    alg_cfg: AMPAlgorithmConfig
    alg: AMPPPO

    def __init__(self,
        env: A1AMP,
        policy: PolicyConfig,
        algorithm: AMPAlgorithmConfig,
        runner: AMPRunnerConfig,
        log_dir: str,
        device: str
    ):
        super().__init__(env, policy, algorithm, runner, log_dir, device)

    def make_alg(self, actor_critic: ActorCritic) -> AMPPPO:
        amp_data = MocapLoader(
            motion_files=self.env.env_cfg.motion_files,
            time_between_frames=self.env.dt,
            sensors=self.env.observation_cfg.amp_sensors,
            is_amp=True,
            preload_transitions=True,
            num_preload_transitions=self.cfg.num_preload_transitions,
            device=self.device
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg.amp_reward_coef,
            self.cfg.amp_discr_hidden_dims,
            self.device,
            self.cfg.amp_task_reward_lerp
        ).to(self.device)
        min_std = (
            torch.tensor(self.cfg.min_normalized_std, device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        return hydra.utils.instantiate(
            self.alg_cfg,
            actor_critic=actor_critic,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            min_std=min_std
        )

    def learn(self, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        critic_obs = self.env.get_critic_observations()
        amp_obs = self.env.get_amp_observations()
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=self.cfg.episode_buffer_len)
        lenbuffer = deque(maxlen=self.cfg.episode_buffer_len)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + self.cfg.iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, critic_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()

                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            metrics = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += self.cfg.iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cpu')
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        return super().load(path, load_optimizer)
