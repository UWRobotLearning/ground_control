from dataclasses import dataclass
from configs.definitions import PolicyConfig

@dataclass
class LSTMPolicyConfig(PolicyConfig):
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
