import time
import logging
import os
from collections import deque
import statistics
import hydra
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter
import torch

from configs.definitions import (PolicyConfig, AlgorithmConfig, RunnerConfig)

from legged_gym.envs.base_env import BaseEnv
from rsl_rl.algorithms.ppo import PPO, Metrics
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

log = logging.getLogger(__name__)

class OnPolicyRunner:
    cfg: RunnerConfig
    alg_cfg: AlgorithmConfig
    policy_cfg: PolicyConfig

    def __init__(
        self,
        env: BaseEnv,
        policy: PolicyConfig,
        algorithm: AlgorithmConfig,
        runner: RunnerConfig,
        log_dir: str,
        device: str
    ):
        resolve = lambda cfg, cfg_type: cfg if isinstance(cfg, cfg_type) else OmegaConf.to_object(cfg)
        self.cfg = resolve(runner, RunnerConfig)
        self.alg_cfg = resolve(algorithm, AlgorithmConfig)
        self.policy_cfg = resolve(policy, PolicyConfig)
        self.device = device
        self.env = env

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        num_actor_obs = self.env.num_obs * self.env.history_steps
        num_critic_obs = num_actor_obs + (self.env.num_critic_obs - self.env.num_obs)
        log.info(f"num_actor_obs = {num_actor_obs}, num_critic_obs = {num_critic_obs}")
        #TODO: address the 'dynamic actor_critic' limitation
        actor_critic = ActorCritic(
            num_actor_obs = num_actor_obs,
            num_critic_obs = num_critic_obs,
            num_actions = self.env.num_actions,
            init_noise_std = self.policy_cfg.init_noise_std,
            fixed_std = self.policy_cfg.fixed_std,
            ortho_init = self.policy_cfg.ortho_init,
            last_actor_layer_scaling = self.policy_cfg.last_actor_layer_scaling,
            actor_hidden_dims = self.policy_cfg.actor_hidden_dims,
            critic_hidden_dims = self.policy_cfg.critic_hidden_dims,
            activation = self.policy_cfg.activation,
        ).to(self.device)
        #self.alg = self.make_alg(actor_critic) #replaced with below line
        self.alg = PPO(
            actor_critic = actor_critic,
            value_loss_coef = self.alg_cfg.value_loss_coef,
            use_clipped_value_loss = self.alg_cfg.use_clipped_value_loss,
            clip_param = self.alg_cfg.clip_param,
            entropy_coef = self.alg_cfg.entropy_coef,
            num_learning_epochs = self.alg_cfg.num_learning_epochs,
            num_mini_batches = self.alg_cfg.num_mini_batches,
            learning_rate = self.alg_cfg.learning_rate,
            max_learning_rate = self.alg_cfg.max_learning_rate,
            min_learning_rate = self.alg_cfg.min_learning_rate,
            schedule = self.alg_cfg.schedule,
            gamma = self.alg_cfg.gamma,
            lam = self.alg_cfg.lam,
            desired_kl = self.alg_cfg.desired_kl,
            max_grad_norm = self.alg_cfg.max_grad_norm,
            device = self.device
        )

        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [num_critic_obs], [self.env.num_actions])

        _, _ = self.env.reset()

    def make_alg(self, actor_critic: ActorCritic) -> PPO:
            return hydra.utils.instantiate(
                self.alg_cfg,
                actor_critic=actor_critic,
                device=self.device
        )

    def learn(self, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        critic_obs = self.env.get_critic_observations()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

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
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos, *_ = self.env.step(actions)
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

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

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']

        mean_std = self.alg.actor_critic.std.mean().item()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self._write_scalar_metrics(locs, mean_std, fps)
        self._print_scalar_metrics(locs, mean_std, fps, width, pad)

    def _print_scalar_metrics(self, locs, mean_std, fps, width, pad):
        iteration_time = locs['collection_time'] + locs['learn_time']
        header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + self.cfg.iterations} \033[0m "
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor_list = []
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor_list.append(ep_info[key].to(self.device))
                if len(infotensor_list) > 0:
                    infotensor = torch.cat(infotensor_list)
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(key, value, locs['it'])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string = (f"""{'#' * width}\n"""
                      f"""{header.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")
        metrics: Metrics = locs['metrics']
        for value, logger_name in zip(metrics.values, metrics.logger_names):
            logger_name += ':'
            log_string += f"{logger_name:>{pad}} {value:.4f}\n"
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
        if len(locs['rewbuffer']) > 0:
            log_string += (f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                           f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               self.cfg.iterations - locs['it']):.1f}s\n""")
        print(log_string)

    def _write_scalar_metrics(self, locs, mean_std, fps):
        iteration = locs['it']
        metrics: Metrics = locs['metrics']
        for value, write_name in zip(metrics.values, metrics.write_names):
            self.writer.add_scalar(write_name, value, iteration)
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std, iteration)
        self.writer.add_scalar('Perf/total_fps', fps, iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)


    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cpu')
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None, with_action_noise=False):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act if with_action_noise else self.alg.actor_critic.act_inference
