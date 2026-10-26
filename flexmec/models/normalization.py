from ._backend import *
from .settings import _env_flag

class ChannelLayerNorm(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm(channels)

    def forward(self, x):
        # Conv1d output layout: (B, C, L). LayerNorm is applied on channel dim C.
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
