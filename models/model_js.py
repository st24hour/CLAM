import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

# JS
from torch.cuda.amp import autocast
# from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from functools import partial


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),
            nn.Dropout(dropout)
    ) 

    def forward(self, x):
        return self.net(x)
# def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
#     inner_dim = int(dim * expansion_factor)
#     return nn.Sequential(
#         dense(dim, inner_dim),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         dense(inner_dim, dim),
#         nn.Dropout(dropout)
#     )

class MLP_mixer(nn.Module):
    def __init__(self, input_dim, dim, num_patches, depth, num_classes, expansion_factor_patch = 0.25, expansion_factor = 0.5, dropout = 0.):
        super(MLP_mixer, self).__init__()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.input_dim = input_dim
        self.to_embed = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim)
        )
        self.mixer = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor_patch, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    # @autocast
    def forward(self, x):
        x = self.to_embed(x)
        return self.mixer(x)


def mlp_mixer_s0(input_dim=1024, dim=1024, num_patches=3000, depth=10, num_classes=2, expansion_factor_patch = 0.25, expansion_factor = 0.5, dropout = 0.):
    return MLP_mixer(
        input_dim=input_dim,
        dim=dim, 
        num_patches=num_patches,
        depth=depth, 
        num_classes=num_classes, 
        expansion_factor_patch = expansion_factor_patch, 
        expansion_factor = expansion_factor,
        dropout = dropout
    )