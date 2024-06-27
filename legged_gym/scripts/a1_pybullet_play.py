import logging

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
# from pydantic import TypeAdapter

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig)
from configs.definitions import DeploymentConfig
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
from legged_gym.utils.observation_buffer import ObservationBuffer
from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
                                      empty_cfg, from_repo_root, save_config_as_yaml)
from robot_deployment.envs.sim_locomotion_gym_env import SimLocomotionGymEnv
import torch
import numpy as np

OmegaConf.register_new_resolver("not", lambda b: not b)
OmegaConf.register_new_resolver("compute_timestep", lambda dt, decimation, action_repeat: dt * decimation / action_repeat)

# INIT_JOINT_ANGLES = {
#     "1_FR_hip_joint": 0.,
#     "1_FR_thigh_joint": 0.0,#0.9,
#     "1_FR_calf_joint": 0.0,#-1.8,

#     "2_FL_hip_joint": 0.,
#     "2_FL_thigh_joint": 0.9,
#     "2_FL_calf_joint": -1.8,

#     "3_RR_hip_joint": 0.,
#     "3_RR_thigh_joint": 0.9,
#     "3_RR_calf_joint": -1.8,

#     "4_RL_hip_joint": 0.,
#     "4_RL_thigh_joint": 0.9,
#     "4_RL_calf_joint": -1.8
# }

INIT_JOINT_ANGLES = {
    "1_FR_hip_joint": -0.1,
    "1_FR_thigh_joint": 0.8,
    "1_FR_calf_joint": -1.5,

    "2_FL_hip_joint": 0.1,
    "2_FL_thigh_joint": 0.8,
    "2_FL_calf_joint": -1.5,

    "3_RR_hip_joint": -0.1,
    "3_RR_thigh_joint": 1.,
    "3_RR_calf_joint": -1.5,

    "4_RL_hip_joint": 0.1,
    "4_RL_thigh_joint": 1.,
    "4_RL_calf_joint": -1.5
}

@dataclass
class DeployScriptConfig:
    checkpoint_root: str = from_repo_root("../experiment_logs/train")
    logging_root: str = from_repo_root("../experiment_logs")
    export_policy: bool = True
    use_joystick: bool = True
    episode_length_s: float = 200.
    checkpoint: int = -1
    device: str = "cpu"
    use_real_robot: bool = False

    hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    task: TaskConfig = empty_cfg(TaskConfig)(
        env = empty_cfg(EnvConfig)(
            num_envs = "${num_envs}"
        ),
        observation = empty_cfg(ObservationConfig)(
            get_commands_from_joystick = "${use_joystick}"
        ),
        sim = empty_cfg(SimConfig)(
            device = "${device}",
            use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
            headless = True, # meaning Isaac is headless, but not the target for deployment 
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
    deployment: DeploymentConfig = DeploymentConfig(
        use_real_robot="${use_real_robot}",
        get_commands_from_joystick="${use_joystick}",
        render=DeploymentConfig.RenderConfig(
            show_gui="${not: ${use_real_robot}}"
        ),
        timestep="${compute_timestep: ${task.sim.dt}, ${task.control.decimation}, ${deployment.action_repeat}}",
        init_position="${task.init_state.pos}",
        init_joint_angles=INIT_JOINT_ANGLES,
        stiffness="${task.control.stiffness}",
        damping="${task.control.damping}",
        action_scale="${task.control.action_scale}"
    )

cs = ConfigStore.instance()
cs.store(name="config", node=DeployScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: DeployScriptConfig):
    default_cfg = {"task": TaskConfig(), "train": TrainConfig()}  # default behaviour as defined in "configs/definitions.py"
    merged_cfg = OmegaConf.merge(
        default_cfg,  # loads default values at the end if it's not specified anywhere else
        cfg           # highest priority, loads from the top-level config dataclass above
    )
    cfg = merged_cfg
    log.info(f"Printing merged cfg.")
    print(OmegaConf.to_yaml(cfg))

    # create robot environment (either in PyBullet or real world)
    deploy_env = SimLocomotionGymEnv(
        cfg.deployment,
        cfg.task.observation.sensors,
        cfg.task.normalization.obs_scales,
        cfg.task.commands.ranges
    )

    obs, info = deploy_env.reset()
    for _ in range(1):
        obs, *_, info = deploy_env.step(deploy_env.default_motor_angles)

    num_obs = obs.shape[0]
    device = cfg.train.device
    obs_buf = ObservationBuffer(1, num_obs, cfg.task.observation.history_steps, device)

    all_actions = []
    all_infos = None

    log.info(f"7. Running the inference loop.")
    for t in range(int(cfg.episode_length_s / deploy_env.robot.control_timestep)):
        # Form observation for policy.
        obs = torch.tensor(obs, device=device).float()
        if t == 0:
            obs_buf.reset([0], obs)
            all_infos = {k: [v.copy()] for k, v in info.items()}
        else:
            obs_buf.insert(obs)
            for k, v in info.items():
                all_infos[k].append(v.copy())

        policy_obs = obs_buf.get_obs_vec(range(cfg.task.observation.history_steps))

        # Evaluate policy and act.
        actions = np.zeros((12,))
        actions = cfg.task.control.action_scale*actions + deploy_env.default_motor_angles
        all_actions.append(actions)
        obs, _, terminated, _, info = deploy_env.step(actions)

        if terminated:
            log.warning("Unsafe, terminating!")
            # break


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
