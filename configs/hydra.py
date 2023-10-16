from dataclasses import dataclass, field
from typing import Dict
from hydra.conf import HydraConf
from omegaconf import OmegaConf
from configs.definitions import * #TODO: can we get rid of this asterisk?
from configs.overrides.locomotion_task import LocomotionTaskConfig

### ======================= Hydra Configuration  =============================
@dataclass
class ExperimentHydraConfig(HydraConf):
    root_dir_name: str = "logs"
    load_dir_name: str = "${hydra:root_dir_name}/train"
    new_override_dirname: str = "${slash_to_dot: ${hydra:job.override_dirname}}"
    run: Dict = field(default_factory=lambda: {
        # A more sophisticated example:
        #"dir": "${hydra:root_dir_name}/${hydra:new_override_dirname}/seed=${seed}/${now:%Y-%m-%d_%H-%M-%S}",
        # Just log by date:
        "dir": "${hydra:root_dir_name}/${now:%Y-%m-%d_%H-%M-%S}",
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

OmegaConf.register_new_resolver("slash_to_dot", lambda dir: dir.replace("/", "."))

### =============== Task Definition Configuration + Hydra Features  ============
# TODO: move the __target__-enabled classes to here to keep the main configs "non-hydra"
# These are the original 'definition' classes but they include bonus functionality

@dataclass
class HydraEnvConfig(EnvConfig):
    num_envs: int = "${resolve_default_int: 4096, ${num_envs}}" # number of envs

# resolving functions
OmegaConf.register_new_resolver("evaluate_max_gpu_contact_pairs", lambda num_envs: 2**23 if num_envs < 8000 else 2**24)
OmegaConf.register_new_resolver("evaluate_use_gpu", lambda device: device.startswith('cuda'))

@dataclass
class HydraSimConfig(SimConfig):
    device: str = "${sim_device}"
    headless: bool = "${headless}"
    use_gpu_pipeline: bool = "${evaluate_use_gpu: ${task.sim.device}}"

    @dataclass
    class HydraPhysxConfig:
        use_gpu: bool = "${evaluate_use_gpu: ${task.sim.device}}"
        max_gpu_contact_pairs: int = "${evaluate_max_gpu_contact_pairs: ${task.env.num_envs}}" # registered by OmegaConf
    physx: HydraPhysxConfig = HydraPhysxConfig()

@dataclass
class HydraTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1.A1"
    env: HydraEnvConfig = HydraEnvConfig()
    sim: HydraSimConfig = HydraSimConfig()


### =============== Train Definition Configuration + Hydra Features  ============
@dataclass
class HydraPolicyConfig(PolicyConfig):
    _target_: str = "rsl_rl.modules.ActorCritic"

@dataclass
class HydraAlgorithmConfig(AlgorithmConfig):
    _target_: str = "rsl_rl.algorithms.PPO"

@dataclass
class HydraRunnerConfig(RunnerConfig):
    iterations: int = "${resolve_default_int: 1500, ${iterations}}" # number of policy updates
    resume_root: str = "${checkpoint_root}"

@dataclass
class HydraTrainConfig(TrainConfig):
    _target_: str = "rsl_rl.runners.OnPolicyRunner"
    device: str = "${rl_device}"
    log_dir: str = "${hydra:run.dir}"
    policy: HydraPolicyConfig = HydraPolicyConfig()
    algorithm: HydraAlgorithmConfig = HydraAlgorithmConfig()
    runner: HydraRunnerConfig = HydraRunnerConfig()

# ======================== Overrides + Hydra Features ==============================
#TODO: need diff solution
from configs.overrides.terrain import TrimeshTerrainConfig, FlatTerrainConfig
from configs.overrides.rewards import LeggedGymRewardsConfig

@dataclass
class HydraLocomotionTaskConfig(HydraTaskConfig):
    _target_: str = "legged_gym.envs.a1.A1"
    env: HydraEnvConfig = HydraEnvConfig()
    sim: HydraSimConfig = HydraSimConfig()

    terrain: TerrainConfig = FlatTerrainConfig()
    rewards: RewardsConfig = LeggedGymRewardsConfig()
    observation: ObservationConfig = ObservationConfig(
        sensors=("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action", "yaw_rate"),
        critic_privileged_sensors=("base_lin_vel", "base_ang_vel", "terrain_height", "friction", "base_mass"),
    )
    domain_rand: DomainRandConfig = DomainRandConfig(
        friction_range=(0.4, 2.5),
        added_mass_range=(-1.5, 2.5),
        randomize_base_mass=True,
    )
    commands: CommandsConfig = CommandsConfig(
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(-1.,2.5),
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos=(0., 0., 0.32),
    )
    asset: AssetConfig = AssetConfig(
        self_collisions=False,
    )

