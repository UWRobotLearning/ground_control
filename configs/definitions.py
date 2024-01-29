from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np

from legged_gym.utils.helpers import from_repo_root

INIT_JOINT_ANGLES = {
    "1_FR_hip_joint": 0.,
    "1_FR_thigh_joint": 0.9,
    "1_FR_calf_joint": -1.8,

    "2_FL_hip_joint": 0.,
    "2_FL_thigh_joint": 0.9,
    "2_FL_calf_joint": -1.8,

    "3_RR_hip_joint": 0.,
    "3_RR_thigh_joint": 0.9,
    "3_RR_calf_joint": -1.8,

    "4_RL_hip_joint": 0.,
    "4_RL_thigh_joint": 0.9,
    "4_RL_calf_joint": -1.8
}

### ======================= Task Configs =============================


@dataclass
class EnvConfig:
    num_envs: int = "${oc.select: num_envs,4096}" # number of envs, obtained from top-level, defaults to 4096
    env_spacing: float = 3. # not used with heightfields/trimeshes
    send_timeouts: bool = True # send out time information to the algorithm
    episode_length_s: float = 20. # episode length in seconds

@dataclass
class ObservationConfig:
    get_commands_from_joystick: bool = "${oc.select: get_commands_from_joystick,False}" # whether to get velocity commands from joystick
    history_steps: int = 1 # number of steps of history to include in policy observation
    sensors: Tuple[str, ...] = ("projected_gravity", "commands", "motor_pos", "motor_vel", "last_action")
    critic_privileged_sensors: Tuple[str, ...] = ("base_lin_vel", "base_ang_vel", "terrain_height")
    fast_compute_foot_pos: bool = True # if True, about 12x faster with a foot position error of 1e-5 meters

@dataclass
class TerrainConfig:
    mesh_type: str = 'plane' # one of [None, 'plane', 'heightfield', 'trimesh', 'valley']
    curriculum: bool = False
    static_friction: float = 1.
    dynamic_friction: float = 1.
    restitution: float = 0.
    measured_points_x: Tuple[float] = (0.,)
    measured_points_y: Tuple[float] = (0.,)
    horizontal_scale: Optional[float] = None # [m]
    vertical_scale: Optional[float] = None # [m]
    selected: bool = False # select a unique terrain type and pass all arguments
    terrain_kwargs: Optional[Dict[str, Any]] = None # dict of arguments for selected terrain
    max_init_terrain_level: Optional[int] = None # starting curriculum state
    terrain_length: Optional[float] = None
    terrain_width: Optional[float] = None
    terrain_noise_magnitude: Optional[float] = None
    terrain_smoothness: Optional[float] = None
    num_rows: Optional[int] = None # number of terrain rows (levels)
    num_cols: Optional[int] = None # number of terrain columns (types)
    border_size: Optional[float] = None # [m]
    terrain_proportions: Optional[Tuple[float, float, float, float, float]] = None # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    slope_threshold: Optional[float] = None # trimesh only; slopes above this threshold will be corrected to vertical surfaces

@dataclass
class CommandsConfig:
    curriculum: bool = False
    max_curriculum: float = 1.

    num_commands: int = 3
    resampling_time: float = 10. # time before commands are changed [s]
    heading_command: bool = True # if True, compute ang vel command from heading error

    @dataclass
    class CommandRangesConfig:
        lin_vel_x: Tuple[float, float] = (-1., 1.) # min max [m/s]
        lin_vel_y: Tuple[float, float] = (-1., 1.) # min max [m/s]
        ang_vel_yaw: Tuple[float, float] = (-1., 1.) # min max [rad/s]
        heading: Tuple[float, float] = (-np.pi, np.pi) # min max [rad]
    ranges: CommandRangesConfig = CommandRangesConfig()

@dataclass
class InitStateConfig:
    pos: Tuple[float, float, float] = (0., 0., 0.32) # x, y, z [m]
    rot: Tuple[float, float, float, float] = (0., 0., 0., 1.) # x, y, z, w [quat]
    lin_vel: Tuple[float, float, float] = (0., 0., 0.) # x, y, z [m/s]
    ang_vel: Tuple[float, float, float] = (0., 0., 0.) # x, y, z [rad/s]
    pos_noise: float = 1.
    lin_vel_noise: float = 0.5
    ang_vel_noise: float = 0.5
    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: INIT_JOINT_ANGLES # target angles [rad] when action = 0.
    )

@dataclass
class ControlConfig:
    control_type: str = "P" # P: position, V: velocity, T: torques

    # PD drive parameters
    # Configures all joint
    # see: paper blah for justification
    stiffness : Dict[str, float] = field(default_factory=lambda: dict(joint=20.)) # [N*m/rad]
    damping: Dict[str, float] = field(default_factory=lambda: dict(joint=0.5)) # [N*m*s/rad]

    action_scale: float = 1.
    clip_setpoint: bool = False
    joint_lower_limit: Optional[Tuple[float, float, float, float, float, float, float, float, float, float, float, float]] = None  ## Joint position limits
    joint_upper_limit: Optional[Tuple[float, float, float, float, float, float, float, float, float, float, float, float]] = None  ## Joint position limits

    # number of control action updates @ sim dt per policy dt
    decimation: int = 4

# registering resolver for adding the repository root (ground_control) to local paths
OmegaConf.register_new_resolver("from_repo_root", from_repo_root)

@dataclass
class AssetConfig:
    file: str = "${from_repo_root: resources/a1.urdf}"
    foot_name: str = "foot" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on: Tuple[str, ...] = ("thigh", "calf")
    terminate_after_contacts_on: Tuple[str, ...] = ("base",)
    disable_gravity: bool = False
    collapse_fixed_joints: bool = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    fix_base_link: bool = False # fix the base of the robot
    default_dof_drive_mode: int = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    self_collisions: bool = True # whether to enable/disable bitwise filter
    replace_cylinder_with_capsule: bool = True # replace collision cylinders with capsules, leads to faster/more stable simulation
    flip_visual_attachments: bool = True # Some .obj meshes must be flipped from y-up to z-up
    contact_force_threshold: float = 1. # force threshold (in newtons) on foot for reporting contact

    density: float = 0.001
    angular_damping: float = 0.
    linear_damping: float = 0.
    max_angular_velocity: float = 1000.
    max_linear_velocity: float = 1000.
    armature: float = 0.
    thickness: float = 0.01
    motor_parameter: float = 0.3

@dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.25)
    randomize_base_mass: bool = False
    added_mass_range: Tuple[float, float] = (-1., 1.)
    push_robots: bool = True
    push_interval_s: float = 15.
    max_push_vel_xy: float = 1.
    randomize_gains: bool = False
    added_stiffness_range: Tuple[float, float] = (-5., 5.)
    added_damping_range: Tuple[float, float] = (-0.05, 0.05)

@dataclass
class RewardsConfig:
    only_positive_rewards: bool = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma: float = 0.5 # tracking reward = exp(-error^2 / sigma^2)

    # percentage of urdf limits, values above this limit are penalized
    soft_dof_pos_limit: float = 1.
    soft_dof_vel_limit: float = 1.
    soft_torque_limit: float = 1.

    base_height_target: float = 0.25
    max_contact_force: float = 100. # forces above this value are penalized
    foot_air_time_threshold: float = 0.5 # [s]

    @dataclass
    class RewardScalesConfig:
        lin_vel_z: float = 0.
        ang_vel_xy: float = 0.
        orientation: float = 0.
        base_height: float = 0.
        torques: float = 0.
        power: float = 0.
        cost_of_transport: float = 0.
        dof_vel: float = 0.
        dof_accel: float = 0.
        action: float = 0.
        action_change: float = 0.
        collision: float = 0.
        termination: float = 0.
        alive: float = 0.
        soft_dof_pos_limits: float = 0.
        soft_torque_limits: float = 0.
        tracking_lin_vel: float = 0.
        tracking_ang_vel: float = 0.
        feet_air_time: float = 0.
        stumble: float = 0.
        stand_still: float = 0.
        feet_contact_forces: float = 0.
        feet_contact_force_change: float = 0.
        witp_abs_dyaw: float = 0.
        witp_cos_pitch_times_lin_vel: float = 0.
    scales: RewardScalesConfig = RewardScalesConfig()

@dataclass
class NormalizationConfig:
    normalize: bool = True
    clip_observations: float = 100.
    clip_actions: float = 100.

    @dataclass
    class NormalizationObsScalesConfig:
        lin_vel: float = 2.
        ang_vel: float = 0.25
        dof_pos: float = 1.
        dof_vel: float = 0.05
        height_measurements: float = 5.
        last_action: float = 1.
        base_mass: float = 0.2
        stiffness: float = 0.01
        damping: float = 1.
        base_quat: float = 1.
    obs_scales: NormalizationObsScalesConfig = NormalizationObsScalesConfig()

@dataclass
class NoiseConfig:
    add_noise: bool = True
    noise_level: float = 1. # scales other values

    @dataclass
    class NoiseScalesConfig:
        dof_pos: float = 0.01
        dof_vel: float = 1.5
        lin_vel: float = 0.1
        ang_vel: float = 0.2
        gravity: float = 0.05
        height_measurements: float = 0.1
    noise_scales: NoiseScalesConfig = NoiseScalesConfig()

@dataclass
class ViewerConfig:
    ref_env: int = 0
    pos: Tuple[float, float, float] = (10., 0., 6.) # [m]
    lookat: Tuple[float, float, float] = (11., 5., 3.) # [m]
    debug_viz: bool = False

# resolving functions
OmegaConf.register_new_resolver("evaluate_max_gpu_contact_pairs", lambda num_envs: 2**23 if num_envs < 8000 else 2**24)
OmegaConf.register_new_resolver("evaluate_use_gpu", lambda device: device.startswith('cuda'))

@dataclass
class SimConfig:
    device: str = "${oc.select: sim_device,cuda:0}"
    headless: bool = "${oc.select: headless,False}"
    dt: float = 0.005
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0., 0., -9.81) # [m/s^2]
    up_axis: int = 1 # 0 is y, 1 is z
    use_gpu_pipeline: bool = "${evaluate_use_gpu: ${task.sim.device}}"

    @dataclass
    class PhysxConfig:
        num_threads: int = 10
        solver_type: int = 1 # 0: pdgs, 1: tgs
        use_gpu: bool = "${evaluate_use_gpu: ${task.sim.device}}"
        num_subscenes: int = 0
        num_position_iterations: int = 4
        num_velocity_iterations: int = 0
        contact_offset: float = 0.01 # [m]
        rest_offset: float = 0. # [m]
        bounce_threshold_velocity: float = 0.5 # [m/s]
        max_depenetration_velocity: float = 1. # [m/s]
        default_buffer_size_multiplier: int = 5
        contact_collection: int = 2 # 0: never, 1: last substep, 2: all substeps
        max_gpu_contact_pairs: int = "${evaluate_max_gpu_contact_pairs: ${task.env.num_envs}}"
    physx: PhysxConfig = PhysxConfig()


@dataclass
class TaskConfig:
    _target_: str = "legged_gym.envs.a1.A1"
    env: EnvConfig = EnvConfig()
    observation: ObservationConfig = ObservationConfig()
    terrain: TerrainConfig = TerrainConfig()
    commands: CommandsConfig = CommandsConfig()
    init_state: InitStateConfig = InitStateConfig()
    control: ControlConfig = ControlConfig()
    asset: AssetConfig = AssetConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardsConfig = RewardsConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    noise: NoiseConfig = NoiseConfig()
    viewer: ViewerConfig = ViewerConfig()
    sim: SimConfig = SimConfig()

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

@dataclass
class TrainConfig:
    _target_: str = "rsl_rl.runners.OnPolicyRunner"
    policy: PolicyConfig = PolicyConfig()
    algorithm: AlgorithmConfig = AlgorithmConfig()
    runner: RunnerConfig = RunnerConfig()

    device: str = "${oc.select: rl_device,cuda:0}"
    log_dir: str = "${hydra:runtime.output_dir}"


@dataclass
class DeploymentConfig:
    use_real_robot: bool = False
    get_commands_from_joystick: bool = False
    action_repeat: int = 10
    timestep: float = 0.001
    reset_time_s: float = 3.
    num_solver_iterations: int = 9
    init_position: Tuple[float, float, float] = (0., 0., 0.32)
    init_rack_position: Tuple[float, float, float] = (0., 0., 1.)
    init_joint_angles: Dict[str, float] = field(default_factory=lambda: INIT_JOINT_ANGLES)
    stiffness: Dict[str, float] = field(default_factory=lambda: dict(joint=50.)) # [N*m/rad]
    damping: Dict[str, float] = field(default_factory=lambda: dict(joint=0.5)) # [N*m*s/rad]
    action_scale: float = 1.
    on_rack: bool = False

    @dataclass
    class RenderConfig:
        show_gui: bool = False
        camera_dist: float = 1.
        camera_yaw: float = 30.
        camera_pitch: float = -30.
        render_width: int = 480
        render_height: int = 360
        fix_camera_yaw: bool = True

    render: RenderConfig = RenderConfig()
