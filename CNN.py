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
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film  = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]

        h = gamma * self.norm1(x) + beta
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class SelfAttention(nn.Module):
    """Attention at 16x16 resolution (avg-pool down, attend, upsample back)."""
    def __init__(self, channels, num_heads=8, downsample=4):
        super().__init__()
        self.downsample = downsample
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        small = F.avg_pool2d(h, self.downsample)
        Hs, Ws = small.shape[2], small.shape[3]
        seq = small.view(B, C, Hs * Ws).transpose(1, 2)
        out, _ = self.attn(seq, seq, seq)
        out = out.transpose(1, 2).view(B, C, Hs, Ws)
        out = F.interpolate(out, size=(H, W), mode='nearest')
        return x + out


class CNN_FM(nn.Module):
    """
    Flow matching velocity network.
    Conditioning: CLIP text embedding (512-d) + sinusoidal time embedding,
    injected at every ResBlock via FiLM. Self-attention at the midpoint.
    """
    def __init__(self, embed_dim=512, channels=256, n_blocks=8,
                 time_dim=256, cond_dim=512):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.emb_proj = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_conv = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks     = nn.ModuleList(
            [ResBlock(channels, cond_dim) for _ in range(n_blocks)]
        )
        self.mid_attn   = SelfAttention(channels, num_heads=8, downsample=4)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x, t, emb):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        cond  = self.time_mlp(t_emb) + self.emb_proj(emb)

        h = self.input_conv(x)
        mid = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            h = block(h, cond)
            if i == mid - 1:
                h = self.mid_attn(h)
        return self.output_conv(h)
