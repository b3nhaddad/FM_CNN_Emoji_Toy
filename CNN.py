import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, dim)


class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # FiLM: project conditioning to per-channel scale + shift
        self.film  = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]

        h = gamma * self.norm1(x) + beta   # FiLM modulation
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h                        # residual


class CNN_FM(nn.Module):
    """
    Flow matching velocity network.
    Conditioning: CLIP text embedding (512-d) + sinusoidal time embedding,
    injected at every ResBlock via FiLM (scale/shift after GroupNorm).
    """
    def __init__(self, embed_dim=512, channels=128, n_blocks=4,
                 time_dim=128, cond_dim=256):
        super().__init__()
        self.time_dim = time_dim

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # CLIP embedding projection
        self.emb_proj = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_conv  = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks      = nn.ModuleList(
            [ResBlock(channels, cond_dim) for _ in range(n_blocks)]
        )
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x, t, emb):
        # build conditioning vector
        t_emb = sinusoidal_embedding(t, self.time_dim)     # (B, time_dim)
        cond  = self.time_mlp(t_emb) + self.emb_proj(emb) # (B, cond_dim)

        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_conv(h)
