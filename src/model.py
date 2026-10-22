"""U-Net kecil untuk memprediksi noise eps(x_t, t, y) (T3.4-T3.8).

Komponen:
    sinusoidal_embedding  : t (B,) -> (B, dim)
    TimestepEmbedding     : sinusoidal -> MLP
    ResBlock              : GroupNorm-SiLU-Conv x2 + injeksi embedding (waktu + kelas)
    AttentionBlock        : self-attention multi-head di atas piksel (H*W token)
    UNet                  : encoder-decoder dengan skip connection
Label y: 0..num_classes-1, indeks num_classes = token null (classifier-free guidance).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.utils import Config


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Embedding posisi sinusoidal untuk timestep.

    t: (B,) int/float -> (B, dim). Separuh pertama sin, separuh kedua cos, dengan frekuensi
    geometrik dari 1 sampai 1/max_period.
    """
    if dim % 2 != 0:
        raise ValueError("dim embedding harus genap")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )  # (half,)
    args = t.float()[:, None] * freqs[None, :]  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class TimestepEmbedding(nn.Module):
    """Sinusoidal(t) -> Linear -> SiLU -> Linear. Output (B, emb_dim)."""

    def __init__(self, base_dim: int, emb_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.base_dim))  # (B, emb_dim)


class ResBlock(nn.Module):
    """Residual block dengan embedding gabungan (waktu + kelas) ditambahkan per channel."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # blok mulai sebagai identitas -> training stabil
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W), emb: (B, emb_dim)
        h = self.conv1(F.silu(self.norm1(x)))  # (B, out_ch, H, W)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]  # broadcast ke (H, W)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h  # (B, out_ch, H, W)


class AttentionBlock(nn.Module):
    """Self-attention multi-head: setiap piksel (H*W token) memperhatikan semua piksel lain."""

    def __init__(self, channels: int, groups: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels harus habis dibagi num_heads")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape  # x: (B, C, H, W)
        head_dim = c // self.num_heads
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)  # masing-masing (B, heads, d, N), N = H*W
        scores = torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(head_dim)  # (B, heads, N, N)
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class ResAttnBlock(nn.Module):
    """ResBlock, opsional diikuti AttentionBlock (dipakai pada resolusi tertentu)."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, groups, use_attn: bool):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout, groups)
        self.attn = AttentionBlock(out_ch, groups) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    """Conv stride 2: (B, C, H, W) -> (B, C, H/2, W/2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor x2 lalu conv: (B, C, H, W) -> (B, C, 2H, 2W)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """U-Net class-conditional untuk memprediksi noise.

    forward(x, t, y): x (B, C, H, W), t (B,) long di [0, T), y (B,) long di [0, num_classes]
    (num_classes = token null) atau None (= semua null). Output eps: (B, C, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        norm_groups: int,
        num_classes: int,
        image_size: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes
        emb_dim = base_channels * 4

        self.time_embed = TimestepEmbedding(base_channels, emb_dim)
        self.class_embed = nn.Embedding(num_classes + 1, emb_dim)  # +1 untuk token null
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        def block(cin, cout, res):
            return ResAttnBlock(cin, cout, emb_dim, dropout, norm_groups, res in attn_resolutions)

        # Encoder: catat channel setiap keluaran untuk skip connection
        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch, res = base_channels, image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(block(ch, out_ch, res))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(Downsample(ch))
                res //= 2
                skip_channels.append(ch)

        # Bottleneck
        self.mid_res1 = ResBlock(ch, ch, emb_dim, dropout, norm_groups)
        self.mid_attn = AttentionBlock(ch, norm_groups)
        self.mid_res2 = ResBlock(ch, ch, emb_dim, dropout, norm_groups)

        # Decoder: setiap level memakai num_res_blocks + 1 blok (satu per skip yang tersimpan)
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(block(ch + skip_channels.pop(), out_ch, res))
                ch = out_ch
            if level != 0:
                self.up.append(Upsample(ch))
                res *= 2
        assert not skip_channels, "jumlah skip encoder/decoder tidak cocok"

        self.norm_out = nn.GroupNorm(norm_groups, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None):
        if y is None:
            y = torch.full((x.shape[0],), self.null_class, device=x.device, dtype=torch.long)
        emb = self.time_embed(t) + self.class_embed(y)  # (B, emb_dim)

        h = self.conv_in(x)  # (B, base, H, W)
        skips = [h]
        for module in self.down:
            h = module(h, emb) if isinstance(module, ResAttnBlock) else module(h)
            skips.append(h)

        h = self.mid_res2(self.mid_attn(self.mid_res1(h, emb)), emb)

        for module in self.up:
            if isinstance(module, ResAttnBlock):
                h = module(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = module(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # (B, C, H, W)


def build_model(cfg: Config) -> UNet:
    """Buat U-Net dari config (bagian `data` dan `model`)."""
    m = cfg.model
    return UNet(
        in_channels=cfg.data.channels,
        base_channels=m.base_channels,
        channel_mults=list(m.channel_mults),
        num_res_blocks=m.num_res_blocks,
        attn_resolutions=list(m.attn_resolutions),
        dropout=m.dropout,
        norm_groups=m.norm_groups,
        num_classes=cfg.data.num_classes,
        image_size=cfg.data.image_size,
    )


def count_parameters(model: nn.Module) -> int:
    """Jumlah parameter yang bisa dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
