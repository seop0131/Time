"""
Conditional 1D Diffusion model for short time-series windows.

설계 요지:
  - Input/output: (B, W=10, C=4) 1초 윈도우
  - Conditional: 11-class label embedding으로 동작 클래스 조건부
  - Backbone: 1D U-Net (encoder/decoder + skip)
  - Noise schedule: linear β, T=1000
  - Loss: ε prediction MSE (DDPM 표준)
  - Sampling: DDIM (빠른 추론, 50 step)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusion step t를 sinusoidal embedding으로 변환."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    """time + class embedding이 더해지는 conv residual block."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, emb):
        # x: (B, in_ch, W), emb: (B, emb_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """
    소형 1D U-Net. window=10에 맞춰 down/up sample 한 번씩만 적용.

    구조:
      입력 (B, C=4, W=10)
        ↓ conv  → (B, 64, 10)
        ↓ Res   → (B, 128, 10)        skip1
        ↓ down  → (B, 128, 5)
        ↓ Res   → (B, 256, 5)         bottleneck
        ↑ up    → (B, 128, 10)
        ↑ Res   → (B, 128, 10)        (+ skip1)
        ↓ conv  → (B, C=4, W=10)
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 11, base_ch: int = 64):
        super().__init__()
        emb_dim = base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # +1 unconditional 토큰 (classifier-free guidance용)
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.time_dim = base_ch
        self.in_conv = nn.Conv1d(in_channels, base_ch, kernel_size=3, padding=1)

        self.down1 = ResBlock1D(base_ch, base_ch * 2, emb_dim)
        self.pool = nn.AvgPool1d(kernel_size=2)

        self.mid = ResBlock1D(base_ch * 2, base_ch * 4, emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1_res = ResBlock1D(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim)

        self.out_norm = nn.GroupNorm(min(8, base_ch * 2), base_ch * 2)
        self.out_conv = nn.Conv1d(base_ch * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, C) noisy input
        t: (B,) diffusion timestep
        c: (B,) class index (0..num_classes-1) 또는 num_classes(uncond)
        returns: predicted noise (B, W, C)
        """
        # (B, W, C) → (B, C, W)
        x = x.permute(0, 2, 1)

        # embedding
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(c)
        emb = t_emb + c_emb

        h = self.in_conv(x)             # (B, base, W)
        s1 = self.down1(h, emb)         # (B, 2*base, W)
        h = self.pool(s1)               # (B, 2*base, W/2)
        h = self.mid(h, emb)            # (B, 4*base, W/2)
        h = self.up1(h)                 # (B, 4*base, W)
        # window가 홀수면 길이 mismatch 가능 → s1과 길이 맞춤
        if h.size(-1) != s1.size(-1):
            h = F.interpolate(h, size=s1.size(-1), mode="nearest")
        h = torch.cat([h, s1], dim=1)
        h = self.up1_res(h, emb)        # (B, 2*base, W)
        h = self.out_conv(F.silu(self.out_norm(h)))  # (B, C, W)

        return h.permute(0, 2, 1)       # (B, W, C)


class GaussianDiffusion:
    """
    DDPM forward / reverse 과정. β linear schedule.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: torch.device = torch.device("cpu")):
        self.num_timesteps = num_timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """forward: x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε."""
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_a * x0 + sqrt_1ma * noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, class_idx: torch.Tensor, num_steps: int = 50,
                    guidance: float = 0.0, num_classes: int = 11):
        """
        DDIM reverse process (η=0, deterministic).
        guidance: classifier-free guidance scale (0=무가이드, 1.5~3 권장).
        """
        B = shape[0]
        x = torch.randn(shape, device=self.device)
        t_seq = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=self.device).long()

        for i in range(num_steps):
            t_cur = t_seq[i].repeat(B)
            t_nxt = t_seq[i + 1].repeat(B)
            a_cur = self.alphas_cumprod[t_cur][:, None, None]
            a_nxt = self.alphas_cumprod[t_nxt][:, None, None]

            # ε prediction with classifier-free guidance
            if guidance > 0:
                eps_cond = model(x, t_cur, class_idx)
                eps_uncond = model(x, t_cur, torch.full_like(class_idx, num_classes))
                eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_cur, class_idx)

            # 예측된 x0
            x0_pred = (x - torch.sqrt(1.0 - a_cur) * eps) / torch.sqrt(a_cur)
            # DDIM step
            x = torch.sqrt(a_nxt) * x0_pred + torch.sqrt(1.0 - a_nxt) * eps

        return x
