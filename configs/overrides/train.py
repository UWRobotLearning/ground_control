from dataclasses import dataclass, field, asdict
from configs.definitions import TrainConfig
from typing import Tuple, Dict, Any, Optional, Sequence



### ======================= Train Configs =============================

@dataclass
class PolicyConfig:
    _target_: str = "rsl_rl.modules.ActorCritic"
    init_noise_std: float = 0.25
    fixed_std: bool = False
    ortho_init: bool = True
    last_actor_layer_scaling: float = 0.01
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu" # elu, relu, selu, leaky_relu, tanh, sigmoid

@dataclass
class AlgorithmConfig:
    _target_: str = "rsl_rl.algorithms.PPO"
    value_loss_coef: float = 1.
    use_clipped_value_loss: bool = False
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4 # minibatch size = num_envs*nsteps / nminibatches
    learning_rate: float = 1e-3
    max_learning_rate: float = 1e-2
    min_learning_rate: float = 0.
    schedule: str = 'adaptive' # adaptive, fixed
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.

@dataclass
class RunnerConfig:
    num_steps_per_env: int = 24 # per iteration
    iterations: int = "${oc.select: iterations,1500}" # number of policy updates

    # logging
    save_interval: int = 50 # check for potential saves every this many iterations
    episode_buffer_len: int = 100 # window of previous episodes to consider for average rewards/lengths

    # load and resume
    resume_root: str = ""
    checkpoint: int = -1 # -1 = last saved model

    # Logging with Weights and Biases
    use_wandb: bool = True
    log_videos: bool = True
    video_frequency: Optional[int] = 1 ## How often to save videos, every <frequency> episodes

# @dataclass
# class RLPDPolicyConfig:
#     a: Optional[bool] = None

@dataclass
class RLPDAlgorithmConfig:
    ## Present in td_config
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims: Tuple[int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    num_qs: int = 2
    critic_layer_norm: bool = False

    ## Present in SAC
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    backup_entropy: bool = True
    critic_weight_decay: Optional[float] = None

    ## REDQ/DroQ/RLPD config
    num_min_qs: Optional[int] = None
    critic_dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    use_critic_resnet: bool = False   

    ## Gradient clipping
    gradient_clipping_norm: Optional[float] = None

# @dataclass
# class RLPDRunnerConfig:
#     a: Optional[bool] = None

@dataclass
class IQLAlgorithmConfig:
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims: Tuple[int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    temperature: float = 0.1 # Same as A_scaling
    actor_weight_decay: Optional[float] = None
    critic_weight_decay: Optional[float] = None
    value_weight_decay: Optional[float] = None
    critic_layer_norm: Optional[bool] = True#False
    value_layer_norm: Optional[bool] = True#False
    num_qs: int = 2
    num_min_qs: Optional[int]= None
    num_vs: int = 1
    num_min_vs: Optional[int] = None

@dataclass
class IQLTrainConfig:
    algorithm: IQLAlgorithmConfig = IQLAlgorithmConfig()
    name: Optional[str] = None
    project_name: str = "iql_pretraining"
    seed: int = 42
    log_interval: int = 100
    batch_size: int = 256
    pretrain_steps: int = 100000
    tqdm: bool = True

    def as_dict(self):
        return asdict(self)


@dataclass
class RLPDSACTrainConfig:
    algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
        gradient_clipping_norm=None,
        critic_layer_norm=True
    )
    name: Optional[str] = None
    project_name: str = "rlpd_locomotion_sac"
    offline_ratio: float = 0#0.5
    seed: int = 42
    eval_episodes: int = 1
    log_interval: int = 100
    eval_interval: int = 1000
    video_interval: int = 20
    batch_size: int = 256
    max_steps: int = int(1e5)
    start_training: int = 1000
    pretrain_steps: int = 0#50000
    tqdm: bool = True
    wandb: bool = True
    save_video: bool = False
    checkpoint_model: bool = False
    checkpoint_buffer: bool = False
    action_filter_high_cut: Optional[float] = None
    action_history: int = 1
    control_frequency: int = 20
    utd_ratio: int = 1#20
    real_robot: bool = False
    offline_data_dir: Optional[str] = None 
    episode_buffer_len: int = 100  ## TODO: Check which of these parameters are unused and delete

    def as_dict(self):
        return asdict(self)
    
@dataclass
class RLPDDroQTrainConfig:
    algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
        critic_dropout_rate=0.01,
        critic_layer_norm=True,
        init_temperature=0.1,#1.0,  ## TODO: Try 0.1
    )
    name: Optional[str] = None
    project_name: str = "rlpd_locomotion_droq"
    offline_ratio: float = 0.5
    seed: int = 42
    eval_episodes: int = 1
    log_interval: int = 1000
    eval_interval: int = 1000
    video_interval: int = 20
    batch_size: int = 256
    max_steps: int = int(1e5)
    start_training: int = 1000 #int(1e4)
    pretrain_steps: int = 0#25000
    tqdm: bool = True
    wandb: bool = True
    save_video: bool = False
    checkpoint_model: bool = False
    checkpoint_buffer: bool = False
    action_filter_high_cut: Optional[float] = None
    action_history: int = 1
    control_frequency: int = 20
    utd_ratio: int = 20
    real_robot: bool = False
    offline_data_dir: Optional[str] = None 
    episode_buffer_len: int = 100

    def as_dict(self):
        return asdict(self)
    
@dataclass
class RLPDREDQTrainConfig:
    algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
        num_qs=10,
        num_min_qs=2,
        critic_layer_norm=True,
        init_temperature=1.0,
    )
    name: Optional[str] = None
    project_name: str = "rlpd_locomotion_redq"
    offline_ratio: float = 0.5
    seed: int = 42
    eval_episodes: int = 1
    log_interval: int = 1000
    eval_interval: int = 1000
    video_interval: int = 20
    batch_size: int = 256
    max_steps: int = int(1e5)
    start_training: int = 5000 #int(1e4)
    pretrain_steps: int = 0#5000
    tqdm: bool = True
    wandb: bool = True
    save_video: bool = False
    checkpoint_model: bool = False
    checkpoint_buffer: bool = False
    action_filter_high_cut: Optional[float] = None
    action_history: int = 1
    control_frequency: int = 20
    utd_ratio: int = 20
    real_robot: bool = False
    offline_data_dir: Optional[str] = None 
    episode_buffer_len: int = 100

    def as_dict(self):
        return asdict(self)
