import isaacgym
import logging
import os.path as osp
import time

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
# from pydantic import TypeAdapter
# from pydantic.dataclasses import dataclass

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig)
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from legged_gym.envs.a1 import A1
from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)
from rsl_rl.runners import OnPolicyRunner


from witp.rl.data.replay_buffer import ReplayBuffer
import pickle
import torch
import gymnasium as gym

def insert_batch_into_replay_buffer(replay_buffer, observations, actions, rewards, dones, next_observations, infos, observation_labels):
    ## Obtain mask
    if 'time_outs' not in infos:  ## No episode was terminated, so should just take into account dones (should all be False)
         masks = torch.logical_not(dones).float()
    else:  ## There was an episode terminated. Masks should be 1 if episode is *not* done or episode was terminated due to timeout, should be 0 if episode was terminated due to MDP end condition.
         masks = torch.logical_or(torch.logical_not(dones), infos["time_outs"]).float()

    ## Convert data to numpy
    observations = observations.cpu().detach().numpy()
    actions = actions.cpu().detach().numpy()
    rewards = rewards.cpu().detach().numpy()
    dones = dones.cpu().detach().numpy()
    next_observations = next_observations.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()

    for i in range(observations.shape[0]):
        replay_buffer.insert(
            dict(observations=observations[i],
                 actions=actions[i],
                 rewards=rewards[i],
                 masks=masks[i],
                 dones=dones[i],
                 next_observations=next_observations[i],
                 observation_labels=observation_labels))
        
def human_format(num):  ## From https://stackoverflow.com/a/45846841
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    


@dataclass
class CollectScriptConfig:
    checkpoint_root: str = from_repo_root("../experiment_logs/train")
    logging_root: str = from_repo_root("../experiment_logs")
    export_policy: bool = True
    num_envs: int = 50
    use_joystick: bool = True
    episode_length_s: float = 200.
    checkpoint: int = -1
    headless: bool = False 
    device: str = "cpu"

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    task: TaskConfig = empty_cfg(TaskConfig)(
        env = empty_cfg(EnvConfig)(
            num_envs = "${num_envs}"
        ),
        observation = empty_cfg(ObservationConfig)(
            get_commands_from_joystick = "${use_joystick}",
            sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
            critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
            extra_sensors=("base_quat",)
        ),
        sim = empty_cfg(SimConfig)(
            device = "${device}",
            use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
            headless = "${headless}",
            physx = empty_cfg(SimConfig.PhysxConfig)(
                use_gpu = "${evaluate_use_gpu: ${task.sim.device}}"
            )
        ),
        terrain = empty_cfg(TerrainConfig)(
            curriculum = False
        ),
        noise = NoNoiseConfig(),
        domain_rand = NoDomainRandConfig()
    ) 
    train: TrainConfig = empty_cfg(TrainConfig)(
        device = "${device}",
        log_dir = "${hydra:runtime.output_dir}",
        runner = empty_cfg(RunnerConfig)(
            checkpoint="${checkpoint}"
        )
    )

cs = ConfigStore.instance()
cs.store(name="config", node=CollectScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: CollectScriptConfig):
    experiment_path = cfg.checkpoint_root
    latest_config_filepath = osp.join(experiment_path, "resolved_config.yaml")
    log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    loaded_cfg = OmegaConf.load(latest_config_filepath)

    log.info("2. Merging loaded config, defaults and current top-level config.")
    del(loaded_cfg.hydra) # Remove unpopulated hydra configuration key from dictionary
    default_cfg = {"task": TaskConfig(), "train": TrainConfig()}  # default behaviour as defined in "configs/definitions.py"
    merged_cfg = OmegaConf.merge(
        default_cfg,  # loads default values at the end if it's not specified anywhere else
        loaded_cfg,   # loads values from the previous experiment if not specified in the top-level config
        cfg           # highest priority, loads from the top-level config dataclass above
    )
    # Resolves the config (replaces all "interpolations" - references in the config that need to be resolved to constant values)
    # and turns it to a dictionary (instead of DictConfig in OmegaConf). Throws an error if there are still missing values.
    merged_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    # Creates a new PlayScriptConfig object (with type-checking and optional validation) using Pydantic.
    # The merged config file (DictConfig as given by OmegaConf) has to be recursively turned to a dict for Pydantic to use it.
    # cfg = TypeAdapter(PlayScriptConfig).validate_python(merged_cfg_dict)
    # cfg = PlayScriptConfig(**merged_cfg_dict)
    cfg = merged_cfg
    # Alternatively, you should be able to use "from pydantic.dataclasses import dataclass" and replace the above line with
    # cfg = PlayScriptConfig(**merged_cfg_dict)
    log.info(f"3. Printing merged cfg.")
    print(OmegaConf.to_yaml(cfg))
    save_config_as_yaml(cfg)

    log.info(f"4. Preparing environment and runner.")
    task_cfg = cfg.task
    env: A1 = hydra.utils.instantiate(task_cfg)
    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_critic_observations()
    extra_obs = env.get_extra_observations()
    runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=env, _recursive_=False)

    resume_path = get_load_path(experiment_path, checkpoint=cfg.train.runner.checkpoint)
    log.info(f"5. Loading policy checkpoint from: {resume_path}.")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
        log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    log.info(f"6. Instantiating replay buffer.")
    dataset_size = 1000000#0  ## Need to make this a config param. Rn if I add to config it gets rid of it during merge seems like
    save_buffer_path = osp.join(cfg.train.log_dir, f"dataset_{resume_path.split('/')[-2]+'_'+resume_path.split('/')[-1][:-3]}_{human_format(dataset_size)}.pkl")
    num_obs = env.num_critic_obs + env.num_extra_obs
    obs_limit = cfg.task.normalization.clip_observations
    observation_space = gym.spaces.Box(low=-obs_limit, high=obs_limit, shape=(num_obs,))

    num_actions = env.num_actions
    action_limit = cfg.task.normalization.clip_actions
    action_space = gym.spaces.Box(low=-action_limit, high=action_limit, shape=(num_actions,))
    sensors =  cfg.task.observation.sensors + cfg.task.observation.critic_privileged_sensors + cfg.task.observation.extra_sensors
    observation_labels = {sensor_name:(0, 0) for sensor_name in sensors}
    observation_labels['projected_gravity'] = (0, 3)
    observation_labels['commands'] = (3, 6)
    observation_labels['motor_pos'] = (6, 18)
    observation_labels['motor_vel'] = (18, 30)
    observation_labels['last_action'] = (30, 42)
    observation_labels['yaw_rate'] = (42, 43)
    observation_labels['base_lin_vel'] = (43, 46)
    observation_labels['base_ang_vel'] = (46, 49)
    observation_labels['terrain_height'] = (49, 50)
    observation_labels['friction'] = (50, 51)
    observation_labels['base_mass'] = (51, 52)
    observation_labels['base_quat'] = (52, 56)

    replay_buffer = ReplayBuffer(observation_space, action_space, capacity=dataset_size, next_observation_space=observation_space, observation_labels=observation_labels)

    log.info(f"7. Running interactive collect script.")
    current_time = time.time()
    num_steps = int(cfg.episode_length_s / env.dt)
    # for i in range(num_steps):
    while len(replay_buffer) < dataset_size:
        actions = policy(obs.detach())
        new_obs, new_critic_obs, rewards, dones, infos, *_ = env.step(actions.detach())
        new_extra_obs = env.get_extra_observations()
        duration = time.time() - current_time
        if duration < env.dt:
            time.sleep(env.dt - duration)
        current_time = time.time()

        print(f"replay_buffer size: {len(replay_buffer)}")
        buffer_observations = torch.concatenate([critic_obs, extra_obs], dim=-1)
        buffer_new_observations = torch.concatenate([new_critic_obs, new_extra_obs], dim=-1)
        insert_batch_into_replay_buffer(replay_buffer, buffer_observations, actions, rewards, dones, buffer_new_observations, infos, observation_labels)
        obs = new_obs
        critic_obs = new_critic_obs
        extra_obs = new_extra_obs

    log.info(f"8. Save replay buffer here: {save_buffer_path}")
    with open(save_buffer_path, 'wb') as f:
        pickle.dump(replay_buffer, f)

    with open(save_buffer_path, 'rb') as f:
        loaded_buffer = pickle.load(f)

    log.info("9. Exit Cleanly")
    env.exit()

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()


'''
Example command to run:

python collect.py num_envs=4096 checkpoint_root='/home/mateo/projects/experiment_logs/train/2023-12-06_08-46-22' checkpoint=4000 use_joystick=False headless=True
'''