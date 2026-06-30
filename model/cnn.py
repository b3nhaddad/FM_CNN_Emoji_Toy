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


class CrossAttention(nn.Module):
    """
    Cross-attention that conditions spatial features on a CLIP embedding.

    The CLIP token (B, cond_dim) is projected to match the spatial channel
    count and treated as a single key/value token.  Spatial features
    (B, C, H, W) are flattened into (B, H*W, C) as queries.
    """
    def __init__(self, channels, cond_dim, num_heads=8):
        super().__init__()
        num_groups = min(32, channels // 16) if channels >= 32 else 1
        self.norm     = nn.GroupNorm(num_groups, channels)
        self.clip_proj = nn.Linear(cond_dim, channels)
        self.attn     = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, clip_emb):
        B, C, H, W = x.shape
        h  = self.norm(x)
        q  = h.flatten(2).transpose(1, 2)               # (B, H*W, C)
        kv = self.clip_proj(clip_emb).unsqueeze(1)       # (B, 1, C)
        out, _ = self.attn(q, kv, kv)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + out


class CNN_FM(nn.Module):
    """
    U-Net flow-matching velocity network for 64x64 RGB emoji generation.

    Encoder (3 levels) → Bottleneck (cross-attn + self-attn) → Decoder (3 levels).
    Every ResBlock is conditioned via FiLM on time + CLIP embedding.
    Cross-attention at the bottleneck injects global CLIP context.
    """
    def __init__(self, in_channels=3, embed_dim=512, base_channels=128,
                 time_dim=256, cond_dim=512):
        super().__init__()
        self.time_dim = time_dim
        ch  = base_channels        # 128
        ch1 = base_channels * 2    # 256
        ch2 = base_channels * 4    # 512

        # ── conditioning ────────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.emb_proj = nn.Linear(embed_dim, cond_dim, bias=False)

        # ── input projection ─────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # ── encoder ──────────────────────────────────────────────────────────
        # Level 0: 128 ch, 64×64 → downsample to 32×32
        self.enc0  = nn.ModuleList([ResBlock(ch, cond_dim),
                                    ResBlock(ch, cond_dim)])
        self.down0 = nn.Conv2d(ch, ch1, 3, stride=2, padding=1)

        # Level 1: 256 ch, 32×32 → downsample to 16×16
        self.enc1  = nn.ModuleList([ResBlock(ch1, cond_dim),
                                    ResBlock(ch1, cond_dim)])
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)

        # Level 2: 512 ch, 16×16 → downsample to 8×8
        self.enc2  = nn.ModuleList([ResBlock(ch2, cond_dim),
                                    ResBlock(ch2, cond_dim)])
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)

        # ── bottleneck: 512 ch, 8×8 ──────────────────────────────────────────
        self.bot_pre  = nn.ModuleList([ResBlock(ch2, cond_dim),
                                       ResBlock(ch2, cond_dim)])
        self.cross_attn = CrossAttention(ch2, cond_dim, num_heads=8)
        self.self_attn  = SelfAttention(ch2, num_heads=8, downsample=4)
        self.bot_post = nn.ModuleList([ResBlock(ch2, cond_dim),
                                       ResBlock(ch2, cond_dim)])

        # ── decoder ──────────────────────────────────────────────────────────
        # Level 2: upsample 8→16, concat skip2 (512) → 1024 → 256
        self.up2_proj = nn.Conv2d(ch2 + ch2, ch1, 1)
        self.dec2     = nn.ModuleList([ResBlock(ch1, cond_dim),
                                       ResBlock(ch1, cond_dim)])

        # Level 1: upsample 16→32, concat skip1 (256) → 512 → 128
        self.up1_proj = nn.Conv2d(ch1 + ch1, ch, 1)
        self.dec1     = nn.ModuleList([ResBlock(ch, cond_dim),
                                       ResBlock(ch, cond_dim)])

        # Level 0: upsample 32→64, concat skip0 (128) → 256 → 128
        self.up0_proj = nn.Conv2d(ch + ch, ch, 1)
        self.dec0     = nn.ModuleList([ResBlock(ch, cond_dim),
                                       ResBlock(ch, cond_dim)])

        # ── output ───────────────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t, emb):
        """
        Args:
            x   : (B, 3, 64, 64)  noisy image
            t   : (B,)             timestep in [0, 1]
            emb : (B, 512)         CLIP text embedding (zeros → unconditional)
        Returns:
            velocity field (B, 3, 64, 64)
        """
        t_emb = sinusoidal_embedding(t, self.time_dim)
        cond  = self.time_mlp(t_emb) + self.emb_proj(emb)

        h = self.input_conv(x)                           # (B, 128, 64, 64)

        # encoder
        for block in self.enc0:
            h = block(h, cond)
        skip0 = h                                         # (B, 128, 64, 64)
        h = self.down0(h)                                 # (B, 256, 32, 32)

        for block in self.enc1:
            h = block(h, cond)
        skip1 = h                                         # (B, 256, 32, 32)
        h = self.down1(h)                                 # (B, 512, 16, 16)

        for block in self.enc2:
            h = block(h, cond)
        skip2 = h                                         # (B, 512, 16, 16)
        h = self.down2(h)                                 # (B, 512,  8,  8)

        # bottleneck
        for block in self.bot_pre:
            h = block(h, cond)
        h = self.cross_attn(h, emb)
        h = self.self_attn(h)
        for block in self.bot_post:
            h = block(h, cond)

        # decoder
        h = F.interpolate(h, scale_factor=2, mode='nearest')   # (B, 512, 16, 16)
        h = torch.cat([h, skip2], dim=1)                       # (B,1024, 16, 16)
        h = self.up2_proj(h)                                    # (B, 256, 16, 16)
        for block in self.dec2:
            h = block(h, cond)

        h = F.interpolate(h, scale_factor=2, mode='nearest')   # (B, 256, 32, 32)
        h = torch.cat([h, skip1], dim=1)                       # (B, 512, 32, 32)
        h = self.up1_proj(h)                                    # (B, 128, 32, 32)
        for block in self.dec1:
            h = block(h, cond)

        h = F.interpolate(h, scale_factor=2, mode='nearest')   # (B, 128, 64, 64)
        h = torch.cat([h, skip0], dim=1)                       # (B, 256, 64, 64)
        h = self.up0_proj(h)                                    # (B, 128, 64, 64)
        for block in self.dec0:
            h = block(h, cond)

        return self.output_conv(h)                              # (B,   3, 64, 64)
