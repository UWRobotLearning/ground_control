from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from configs.overrides import observation
from configs.overrides import policy
from configs.overrides import rewards
from configs.overrides import terrain
from configs.overrides import amp
from configs.overrides import mocap

NONE = -1 # will be filled in by default value unless overriden by user

# adapted from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/__init__.py#L11
OmegaConf.register_new_resolver("resolve_default_int", lambda default, arg: default if arg == NONE else arg)

cs = ConfigStore.instance()
cs.store(group="task", name="amp", node=amp.AMPTaskConfig)
cs.store(group="task", name="amp_mimic", node=amp.AMPMimicTaskConfig)
cs.store(group="task", name="mocap", node=mocap.MocapTaskConfig)
cs.store(group="train", name="amp", node=amp.AMPTrainConfig)
cs.store(group="train/policy", name="lstm", node=policy.LSTMPolicyConfig)
cs.store(group="task/observation", name="complex_terrain", node=observation.ComplexTerrainObservationConfig)
cs.store(group="task/rewards", name="legged_gym", node=rewards.LeggedGymRewardsConfig)
cs.store(group="task/terrain", name="flat", node=terrain.FlatTerrainConfig)
cs.store(group="task/terrain", name="trimesh", node=terrain.TrimeshTerrainConfig)
