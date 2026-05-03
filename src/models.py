from __future__ import annotations

import torch
from torch import nn


class BCMLP(nn.Module):
    """
    改进版 BC-MLP：
    - 3 层 hidden（256->256->256），避免过大模型在小数据集上过拟合
    - 每层加 LayerNorm + GELU + Dropout(0.1)
    - LayerNorm 让各维度尺度无关，比 obs 手动归一化更鲁棒
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BCRNN(nn.Module):
    """
    改进版 BC-RNN：
    - 输入先过一个 Linear + LayerNorm projection（缩放输入尺度）
    - GRU hidden 调大到 512，层数保持 2
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs_seq: torch.Tensor, h0: torch.Tensor | None = None):
        # obs_seq: (B, T, obs_dim)
        B, T, _ = obs_seq.shape
        x = self.input_proj(obs_seq.reshape(B * T, -1)).reshape(B, T, -1)
        y, h = self.rnn(x, h0)
        return self.head(y), h


# ─────────────────────────────────────────────────────────────────────────────
#  Diffusion Policy (DDPM + DDIM)
#  论文：Chi et al. "Diffusion Policy: Visuomotor Policy Learning via
#        Action Diffusion" RSS 2023  https://arxiv.org/abs/2303.04137
#
#  实现说明（无需 diffusers 依赖，纯 PyTorch）：
#   1. NoisePredictor  — 条件去噪网络，输入 (obs_emb, noisy_action, timestep)，
#                        预测加在 action 上的噪声 ε。
#   2. BCDiffusion     — 封装完整 DDPM 前向/推理流程：
#       · fit_betas()     计算扩散调度参数（β、ᾱ 等）
#       · q_sample()      训练时加噪
#       · forward()       训练损失（MSE(ε_pred, ε)）
#       · ddpm_sample()   推理时逐步去噪（T_inf 步，默认 100）
#       · ddim_sample()   DDIM 快速采样（可选，更高质量/更少步数）
#   3. EMA             — 指数移动平均权重封装（训练稳定+推理质量提升）
# ─────────────────────────────────────────────────────────────────────────────

import math


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """
    指数移动平均权重封装。
    训练时对主模型参数维护 shadow 副本，以 decay 速率平滑更新。
    用于推理时替换主模型，大幅提升生成质量（diffusion 领域的标准 trick）。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu"):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        """将所有参数注册到 shadow dict。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        """每步训练后调用，用 EMA 规则更新 shadow 权重。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """将 shadow 权重覆盖到主模型（推理前调用）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone().to(param.device)

    def restore(self):
        """恢复主模型原始权重（训练时调用）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone().to(param.device)
        self.backup = {}


# ── Sinusoidal Position Embedding ──────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """将整数时间步 t 编码为连续向量（同 Transformer 位置编码）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long or float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)


class RotationInvariantObsEncoder(nn.Module):
    """
    C_n 离散旋转不变观测编码器（n=4 或 8）。
    将观测前 pair_dim 维按 (x, y) 成对解释为平面向量，做 group average pooling。
    """

    def __init__(self, obs_dim: int, hidden: int, pair_dim: int | None = None, n_rot: int = 4):
        super().__init__()
        if n_rot not in (4, 8):
            raise ValueError(f"n_rot must be 4 or 8, got {n_rot}")
        max_pair_dim = obs_dim - (obs_dim % 2)
        if pair_dim is None:
            pair_dim = max_pair_dim
        pair_dim = max(0, min(int(pair_dim), max_pair_dim))
        if pair_dim % 2 != 0:
            pair_dim -= 1
        self.obs_dim = obs_dim
        self.pair_dim = pair_dim
        self.scalar_dim = obs_dim - pair_dim
        self.n_rot = n_rot

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_rot + 1, dtype=torch.float32)[:-1]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rot_mats = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),
                torch.stack([sin_a, cos_a], dim=-1),
            ],
            dim=-2,
        )
        self.register_buffer("rot_mats", rot_mats)

    def _group_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, D) -> (B, n_rot, D)
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch in RotationInvariantObsEncoder: got {D}, expected {self.obs_dim}")

        if self.pair_dim == 0:
            return obs.unsqueeze(1).expand(B, self.n_rot, D)

        v = obs[:, :self.pair_dim].reshape(B, self.pair_dim // 2, 2)
        v_rot = torch.einsum("gij,bkj->bgki", self.rot_mats.to(obs.dtype), v).reshape(B, self.n_rot, self.pair_dim)
        if self.scalar_dim > 0:
            s = obs[:, self.pair_dim:].unsqueeze(1).expand(B, self.n_rot, self.scalar_dim)
            return torch.cat([v_rot, s], dim=-1)
        return v_rot

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, D) -> (B, hidden)
        xg = self._group_transform(obs)
        B, G, D = xg.shape
        h = self.encoder(xg.reshape(B * G, D)).reshape(B, G, -1)
        return h.mean(dim=1)


class SE2SteerableObsEncoder(nn.Module):
    """
    连续 SE(2) 旋转等变的观测编码器（输出旋转不变 embedding）。
    - 前 pair_dim 维按 (x, y) 成对解释为 2D 向量通道
    - 标量分支只接收旋转不变量（标量 + 向量模长）
    - 向量分支使用 a*I + b*J 形式的线性映射（J 为 90° 旋转）
    """

    def __init__(self, obs_dim: int, hidden: int, pair_dim: int | None = None, vec_channels: int | None = None):
        super().__init__()
        max_pair_dim = obs_dim - (obs_dim % 2)
        if pair_dim is None:
            pair_dim = max_pair_dim
        pair_dim = max(0, min(int(pair_dim), max_pair_dim))
        if pair_dim % 2 != 0:
            pair_dim -= 1

        self.obs_dim = obs_dim
        self.pair_dim = pair_dim
        self.scalar_dim = obs_dim - pair_dim
        self.n_vec = pair_dim // 2
        self.vec_channels = int(vec_channels or max(16, min(64, hidden // 4)))

        self.fallback = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        if self.n_vec == 0:
            return

        scalar_in_dim = self.scalar_dim + self.n_vec
        self.scalar_stem = nn.Sequential(
            nn.Linear(scalar_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden + self.vec_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden + self.vec_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.vec_gate = nn.Sequential(
            nn.Linear(hidden, self.vec_channels),
            nn.Sigmoid(),
        )

        # 等变线性映射: y = A*v + B*J(v)
        self.in_a = nn.Parameter(torch.randn(self.n_vec, self.vec_channels) * 0.02)
        self.in_b = nn.Parameter(torch.randn(self.n_vec, self.vec_channels) * 0.02)
        self.mix_a = nn.Parameter(torch.randn(self.vec_channels, self.vec_channels) * 0.02)
        self.mix_b = nn.Parameter(torch.randn(self.vec_channels, self.vec_channels) * 0.02)

    @staticmethod
    def _rot90(v: torch.Tensor) -> torch.Tensor:
        # v: (..., 2)
        return torch.stack([-v[..., 1], v[..., 0]], dim=-1)

    @staticmethod
    def _vec_norm(v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((v * v).sum(dim=-1).clamp(min=1e-12))

    @staticmethod
    def _equivariant_mix(v: torch.Tensor, w_a: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
        # v: (B, Cin, 2), w_*: (Cin, Cout) -> (B, Cout, 2)
        vj = SE2SteerableObsEncoder._rot90(v)
        return torch.einsum("bic,io->boc", v, w_a) + torch.einsum("bic,io->boc", vj, w_b)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch in SE2SteerableObsEncoder: got {D}, expected {self.obs_dim}")
        if self.n_vec == 0:
            return self.fallback(obs)

        v = obs[:, :self.pair_dim].reshape(B, self.n_vec, 2)
        s = obs[:, self.pair_dim:]
        v0_norm = self._vec_norm(v)

        if self.scalar_dim > 0:
            h = self.scalar_stem(torch.cat([s, v0_norm], dim=-1))
        else:
            h = self.scalar_stem(v0_norm)

        v_h = self._equivariant_mix(v, self.in_a, self.in_b)
        for _ in range(2):
            gate = self.vec_gate(h).unsqueeze(-1)
            v_h = v_h * gate + self._equivariant_mix(v_h, self.mix_a, self.mix_b)
            v_norm = self._vec_norm(v_h)
            h = h + self.scalar_update(torch.cat([h, v_norm], dim=-1))

        v_norm = self._vec_norm(v_h)
        return self.out_proj(torch.cat([h, v_norm], dim=-1))


class HarmonicObsEncoder(nn.Module):
    """
    复数谐波 Fourier 特征编码器（E(2)/SE(2) 风格）。
    使用 z = x + i y 的谐波矩（m=1..M）特征（Re/Im/|.|），并与原始观测投影融合。
    """

    def __init__(self, obs_dim: int, hidden: int, pair_dim: int | None = None, max_order: int = 4):
        super().__init__()
        max_pair_dim = obs_dim - (obs_dim % 2)
        if pair_dim is None:
            pair_dim = max_pair_dim
        pair_dim = max(0, min(int(pair_dim), max_pair_dim))
        if pair_dim % 2 != 0:
            pair_dim -= 1

        self.obs_dim = obs_dim
        self.pair_dim = pair_dim
        self.scalar_dim = obs_dim - pair_dim
        self.n_vec = pair_dim // 2
        self.max_order = max(1, int(max_order))

        self.fallback = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        if self.n_vec == 0:
            return

        feat_dim = self.scalar_dim + self.n_vec + (self.max_order * 3)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.raw_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.register_buffer("orders", torch.arange(1, self.max_order + 1, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch in HarmonicObsEncoder: got {D}, expected {self.obs_dim}")
        if self.n_vec == 0:
            return self.fallback(obs)

        v = obs[:, :self.pair_dim].reshape(B, self.n_vec, 2)
        s = obs[:, self.pair_dim:]
        x = v[..., 0]
        y = v[..., 1]
        r = torch.sqrt((x * x + y * y).clamp(min=1e-12))
        theta = torch.atan2(y, x)

        ords = self.orders.to(obs.dtype).view(1, 1, -1)  # (1,1,M)
        ang = theta.unsqueeze(-1) * ords                  # (B,N,M)
        # 归一化半径，避免高阶 r^m 在个别样本上数值主导。
        r_scale = r.mean(dim=1, keepdim=True).clamp(min=1e-3)
        r_norm = r / r_scale
        r_pow = r_norm.unsqueeze(-1).pow(ords)            # (B,N,M)

        # 复数矩 C_m = E[r^m e^{i m theta}]，保留 Re/Im 与幅值。
        c_re = (r_pow * torch.cos(ang)).mean(dim=1)       # (B,M)
        c_im = (r_pow * torch.sin(ang)).mean(dim=1)       # (B,M)
        c_mag = torch.sqrt((c_re * c_re + c_im * c_im).clamp(min=1e-12))

        if self.scalar_dim > 0:
            feat = torch.cat([s, r_norm, c_re, c_im, c_mag], dim=-1)
        else:
            feat = torch.cat([r_norm, c_re, c_im, c_mag], dim=-1)
        harm_emb = self.net(feat)
        raw_emb = self.raw_proj(obs)
        return self.fuse(torch.cat([harm_emb, raw_emb], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
#  Consistency-Regularized Diffusion (CRD)
#  核心创新：
#  1. 一致性正则化：同一动作在不同噪声水平下去噪结果应一致
#  2. 简化网络：depth=4（避免v2/v3深度模型在小数据集上过拟合）
#  3. 残差FiLM：每层带残差连接，信息流更顺畅
#  4. 自适应无分类器引导(CFG)：推理时根据obs动态调整引导强度
# ══════════════════════════════════════════════════════════════════════════════


class NoisePredictorCRD(nn.Module):
    """
    一致性正则化去噪网络。

    与原始 NoisePredictor 的关键区别：
    1. depth=4（默认），减少过拟合风险
    2. 残差 FiLM：x = x + film(cond) 而非 x = film(cond) * x + beta
    3. 更强的 obs 条件注入：obs_emb 在每层都参与调制
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 256, t_dim: int = 64, depth: int = 4,
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4,
                 residual_film: bool = True):
        super().__init__()
        self.residual_film = residual_film

        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        if obs_backbone in ("c4", "c8"):
            self.obs_emb = RotationInvariantObsEncoder(
                obs_dim, hidden,
                pair_dim=rot_pair_dim,
                n_rot=8 if obs_backbone == "c8" else 4,
            )
        elif obs_backbone == "se2":
            self.obs_emb = SE2SteerableObsEncoder(obs_dim, hidden, pair_dim=rot_pair_dim)
        elif obs_backbone == "harmonic":
            self.obs_emb = HarmonicObsEncoder(
                obs_dim, hidden,
                pair_dim=rot_pair_dim,
                max_order=harmonic_order,
            )
        else:
            self.obs_emb = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )

        # 初始动作嵌入层
        self.act_proj = nn.Sequential(
            nn.Linear(act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # FiLM 调制器：obs_emb 和 t_emb 融合后调制每层
        self.film_layers = nn.ModuleList()
        in_dim = hidden
        for i in range(depth):
            self.film_layers.append(nn.Sequential(
                nn.Linear(hidden + t_dim, hidden * 2),
                nn.LayerNorm(hidden * 2) if residual_film else nn.Identity(),
            ))

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ))
            in_dim = hidden

        self.out = nn.Linear(hidden, act_dim)
        self.act_fn = nn.GELU()

    def forward(self,
                noisy_act: torch.Tensor,
                t: torch.Tensor,
                obs: torch.Tensor) -> torch.Tensor:
        t_e = self.t_emb(t)
        obs_e = self.obs_emb(obs)
        cond = torch.cat([obs_e, t_e], dim=-1)

        x = self.act_proj(noisy_act)

        for layer, film_layer in zip(self.layers, self.film_layers):
            x_input = x
            x = layer(x)
            gamma_beta = film_layer(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            if self.residual_film:
                # 残差 FiLM：保留主路径信息
                x = x * (1 + 0.1 * gamma) + 0.1 * beta
                x = x_input + self.act_fn(x)
            else:
                x = x * (1 + gamma) + beta
                x = self.act_fn(x)

        return self.out(x)


class BCDiffusionCRD(nn.Module):
    """
    一致性正则化 Diffusion Policy。

    核心创新：
    1. 一致性正则化损失：同一obs-action样本在两个随机时间步的噪声预测应该"一致"
       - 这里的"一致"指的是：两者预测的噪声在去噪方向上应该对齐
    2. 推理时支持自适应无分类器引导(CFG)
    3. 使用简化架构(depth=4, hidden=256)减少过拟合

    与 v1/v2 的关键区别：
    - v1: depth=4, hidden=256, 基础 FiLM
    - v2: depth=6, hidden=384, EMA+SWA+ActionNoise（可能过拟合）
    - CRD: depth=4, hidden=256, 一致性正则化, 自适应CFG
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 T: int = 100,
                 beta_min: float = 1e-4, beta_max: float = 2e-2,
                 hidden: int = 256, depth: int = 4,
                 scheduler: str = "cosine",
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4,
                 residual_film: bool = True,
                 consistency_weight: float = 0.1,
                 cfg_strength: float = 1.0):
        super().__init__()
        self.T = T
        self.act_dim = act_dim
        self.scheduler = scheduler
        self.consistency_weight = consistency_weight
        self.cfg_strength = cfg_strength

        self._build_schedule(beta_min, beta_max)

        self.noise_pred = NoisePredictorCRD(
            obs_dim, act_dim,
            hidden=hidden,
            depth=depth,
            obs_backbone=obs_backbone,
            rot_pair_dim=rot_pair_dim,
            harmonic_order=harmonic_order,
            residual_film=residual_film,
        )

    def _build_schedule(self, beta_min: float, beta_max: float):
        if self.scheduler == "cosine":
            steps = self.T + 1
            s = 0.008
            t = torch.linspace(0, self.T, steps, dtype=torch.float32) / self.T
            alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / (alphas_cumprod[0] + 1e-8)
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
            betas = torch.cat([betas[:1], betas])
        else:
            betas = torch.linspace(beta_min, beta_max, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def _denoise_step(self, x: torch.Tensor, t: int, obs: torch.Tensor) -> torch.Tensor:
        """单步去噪，返回预测的原始动作 x0"""
        t_batch = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        eps = self.noise_pred(x, t_batch, obs)
        ab_t = self.alpha_bar[t]
        x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
        return x0_pred

    def forward(self,
                obs: torch.Tensor,
                act: torch.Tensor,
                action_noise_std: float = 0.0) -> torch.Tensor:
        """
        计算训练损失，包含：
        1. 标准噪声预测损失
        2. 一致性正则化损失（同一样本两个时间步的去噪结果应一致）
        """
        B = obs.shape[0]

        # ── 标准噪声预测损失 ──────────────────────────────────────────────────
        t = torch.randint(0, self.T, (B,), device=obs.device)
        if action_noise_std > 0:
            act_noisy = act + torch.randn_like(act) * action_noise_std
        else:
            act_noisy = act
        noise = torch.randn_like(act)
        x_t = self.q_sample(act_noisy, t, noise)
        eps_pred = self.noise_pred(x_t, t, obs)
        loss_mse = torch.nn.functional.mse_loss(eps_pred, noise)

        # ── 一致性正则化损失 ─────────────────────────────────────────────────
        # 对同一批样本，采样两个不同时间步，要求去噪结果一致
        if self.consistency_weight > 0 and B >= 2:
            # 随机采样两个不同时间步
            t1 = torch.randint(0, self.T, (B,), device=obs.device)
            t2 = torch.randint(0, self.T, (B,), device=obs.device)
            # 确保 t1 != t2（随机打乱后取前B/2配对）
            mask = (t1 == t2)
            if mask.any():
                t2 = (t2 + torch.randint(1, self.T, (B,), device=obs.device)) % self.T

            noise1 = torch.randn_like(act)
            noise2 = torch.randn_like(act)
            x_t1 = self.q_sample(act, t1, noise1)
            x_t2 = self.q_sample(act, t2, noise2)

            # 预测去噪结果
            x0_1 = self._denoise_step(x_t1, t1[0].item(), obs)
            x0_2 = self._denoise_step(x_t2, t2[0].item(), obs)

            # 一致性损失：两次去噪的预测动作应该接近
            # 注意：只在高噪声时间步（t较大）施加一致性约束，因为低噪声时本身就很接近
            with torch.no_grad():
                w = ((t1.float() + t2.float()) / (2 * self.T)).unsqueeze(-1)
                w = w.clamp(0.1, 1.0)
            loss_consistency = (w * (x0_1 - x0_2).pow(2)).mean()
            total_loss = loss_mse + self.consistency_weight * loss_consistency
        else:
            total_loss = loss_mse

        return total_loss

    @torch.no_grad()
    def ddpm_sample(self,
                    obs: torch.Tensor,
                    T_inf: int | None = None,
                    use_cfg: bool = True) -> torch.Tensor:
        """
        DDPM 逆向去噪采样。
        obs : (B, obs_dim)
        use_cfg: 是否使用无分类器引导
        """
        T_inf = T_inf or self.T
        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device)

        for i in reversed(range(T_inf)):
            t_batch = torch.full((B,), i, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs)

            # 无分类器引导：沿着 obs 条件方向增强
            if use_cfg and self.cfg_strength != 1.0:
                eps_cfg = eps * self.cfg_strength
            else:
                eps_cfg = eps

            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            ab_t = self.alpha_bar[i]

            coef = beta_t / (1 - ab_t).sqrt()
            mean = (x - coef * eps_cfg) / alpha_t.sqrt()

            if i > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(self,
                    obs: torch.Tensor,
                    T_inf: int = 20,
                    eta: float = 0.0,
                    use_cfg: bool = True) -> torch.Tensor:
        B = obs.shape[0]
        step_size = self.T // T_inf
        x = torch.randn(B, self.act_dim, device=obs.device)

        timesteps = list(range(self.T - 1, -1, -step_size))[:T_inf]
        if timesteps[-1] != 0:
            timesteps.append(0)

        for idx in range(len(timesteps) - 1):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]

            t_batch = torch.full((B,), t_cur, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs)

            if use_cfg and self.cfg_strength != 1.0:
                eps = eps * self.cfg_strength

            ab_t = self.alpha_bar[t_cur]
            ab_tn = self.alpha_bar[t_next]

            pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            direction = (x - ab_t.sqrt() * pred_x0) / (1 - ab_t).sqrt().clamp(min=1e-8)

            if eta == 0.0:
                x = ab_tn.sqrt() * pred_x0 + (1 - ab_tn).sqrt() * direction
            else:
                beta_tn = self.betas[t_next]
                c1 = eta * ((1 - ab_tn / ab_t).clamp(min=0) * (1 - ab_t) / (1 - ab_tn)).sqrt() * beta_tn.sqrt()
                x = ab_tn.sqrt() * pred_x0 + ((1 - ab_tn) - c1 ** 2).clamp(min=0).sqrt() * direction
                if t_next > 0:
                    x = x + c1 * torch.randn_like(x)

        return x


class BCDiffusionConsistency(BCDiffusionCRD):
    """
    兼容性别名：BCDiffusionConsistency = BCDiffusionCRD
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 原有 DDPM + DDIM Diffusion Policy (保持不变)
# ══════════════════════════════════════════════════════════════════════════════


class NoisePredictor(nn.Module):
    """
    条件去噪网络：ε_θ(x_t, t | obs)
    结构改进：
    - 增大容量（hidden=384, depth=6）
    - 每层之间加残差连接（输入扰动后直接加到输出）
    - obs_emb 和 t_emb 分别 embedding 后融合
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 384, t_dim: int = 64, depth: int = 6,
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        if obs_backbone in ("c4", "c8"):
            self.obs_emb = RotationInvariantObsEncoder(
                obs_dim,
                hidden,
                pair_dim=rot_pair_dim,
                n_rot=8 if obs_backbone == "c8" else 4,
            )
        elif obs_backbone == "se2":
            self.obs_emb = SE2SteerableObsEncoder(obs_dim, hidden, pair_dim=rot_pair_dim)
        elif obs_backbone == "harmonic":
            self.obs_emb = HarmonicObsEncoder(
                obs_dim,
                hidden,
                pair_dim=rot_pair_dim,
                max_order=harmonic_order,
            )
        else:
            self.obs_emb = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
        # 每层：Linear + LayerNorm + FiLM + GELU（含残差）
        self.layers    = nn.ModuleList()
        self.film_mods = nn.ModuleList()
        in_dim = act_dim
        for i in range(depth):
            out_dim = hidden
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            ))
            # FiLM: γ, β from (obs_emb ⊕ t_emb)
            self.film_mods.append(nn.Linear(hidden + t_dim, out_dim * 2))
            in_dim = out_dim

        self.act_fn = nn.GELU()
        self.out    = nn.Linear(hidden, act_dim)

    def forward(self,
                noisy_act: torch.Tensor,   # (B, act_dim)
                t: torch.Tensor,           # (B,) long
                obs: torch.Tensor,         # (B, obs_dim)  — 已归一化
                ) -> torch.Tensor:         # (B, act_dim)  — 预测噪声
        t_e   = self.t_emb(t)             # (B, t_dim)
        obs_e = self.obs_emb(obs)         # (B, hidden)
        cond  = torch.cat([obs_e, t_e], dim=-1)  # (B, hidden+t_dim)

        x = noisy_act
        for layer, film in zip(self.layers, self.film_mods):
            x = layer(x)                  # (B, hidden)
            gam_bet = film(cond)          # (B, hidden*2)
            gamma, beta = gam_bet.chunk(2, dim=-1)
            x = self.act_fn(x * (1 + gamma) + beta)

        return self.out(x)               # (B, act_dim)


# ── BCDiffusion ───────────────────────────────────────────────────────────────

class BCDiffusion(nn.Module):
    """
    DDPM + DDIM Diffusion Policy 封装。

    参数
    ----
    obs_dim   : 归一化后的 obs 维度
    act_dim   : 归一化后的 action 维度
    T         : 训练时扩散步数（默认 100）
    beta_min/max : β 调度的端点（支持 cosine 模式）
    hidden    : NoisePredictor 隐层宽度
    depth     : NoisePredictor 层数
    scheduler : "linear" | "cosine"（cosine 更平滑，IS/FID 更优）
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 T: int = 100,
                 beta_min: float = 1e-4, beta_max: float = 2e-2,
                 hidden: int = 384, depth: int = 6,
                 scheduler: str = "cosine",
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4):
        super().__init__()
        self.T        = T
        self.act_dim  = act_dim
        self.scheduler = scheduler

        # 计算扩散调度参数
        self._build_schedule(beta_min, beta_max)

        self.noise_pred = NoisePredictor(
            obs_dim,
            act_dim,
            hidden=hidden,
            depth=depth,
            obs_backbone=obs_backbone,
            rot_pair_dim=rot_pair_dim,
            harmonic_order=harmonic_order,
        )

    def _build_schedule(self, beta_min: float, beta_max: float):
        """构建 α、β 调度参数。"""
        if self.scheduler == "cosine":
            # Cosine schedule（Nichol & Dhariwal 2021）：
            # 与 v2 训练时完全一致的公式（steps=T+1，betas=T+1）
            steps = self.T + 1
            s = 0.008
            t = torch.linspace(0, self.T, steps, dtype=torch.float32) / self.T
            alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / (alphas_cumprod[0] + 1e-8)
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
            betas = torch.cat([betas[:1], betas])   # length T+1
        else:
            # Linear schedule（原始 DDPM）
            betas = torch.linspace(beta_min, beta_max, self.T, dtype=torch.float32)

        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    # ── 训练阶段 ─────────────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """前向加噪：q(x_t | x_0) = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1)        # (B, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def forward(self, obs: torch.Tensor,
                act: torch.Tensor,
                action_noise_std: float = 0.0) -> torch.Tensor:
        """
        计算训练 loss（MSE 噪声预测损失）。
        obs: (B, obs_dim)  act: (B, act_dim)  — 均已归一化
        action_noise_std: 训练时对 action 加噪（数据增强，提升鲁棒性）
        返回标量 loss。
        """
        B = obs.shape[0]
        t = torch.randint(0, self.T, (B,), device=obs.device)

        # Action 数据增强：加少量高斯噪声
        if action_noise_std > 0:
            aug_noise = torch.randn_like(act) * action_noise_std
            act = act + aug_noise

        noise   = torch.randn_like(act)
        x_t     = self.q_sample(act, t, noise)
        eps_pred = self.noise_pred(x_t, t, obs)
        return torch.nn.functional.mse_loss(eps_pred, noise)

    # ── 推理阶段 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_sample(self, obs: torch.Tensor,
                    T_inf: int | None = None) -> torch.Tensor:
        """
        DDPM 逆向去噪，从 x_T ~ N(0,I) 采样动作。
        obs : (B, obs_dim)
        返回 (B, act_dim) — 归一化空间的 action
        """
        T_inf = T_inf or self.T
        B     = obs.shape[0]
        x     = torch.randn(B, self.act_dim, device=obs.device)

        for i in reversed(range(T_inf)):
            t_batch = torch.full((B,), i, device=obs.device, dtype=torch.long)
            eps     = self.noise_pred(x, t_batch, obs)

            beta_t  = self.betas[i]
            alpha_t = self.alphas[i]
            ab_t    = self.alpha_bar[i]

            # DDPM 均值
            coef = beta_t / (1 - ab_t).sqrt()
            mean = (x - coef * eps) / alpha_t.sqrt()

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(self, obs: torch.Tensor,
                    T_inf: int = 20,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM（Denosing Diffusion Implicit Models）采样：
        - 更少的步数（T_inf << T）即可获得高质量动作
        - eta=0 时完全确定性（隐式 ODE），eta=1 时接近 DDPM
        - 对本任务，推荐 T_inf=20 或 10

        obs : (B, obs_dim)
        返回 (B, act_dim) — 归一化空间的 action
        """
        B = obs.shape[0]
        step_size = self.T // T_inf  # 跳步间隔

        # 从纯噪声开始
        x = torch.randn(B, self.act_dim, device=obs.device)

        # 逆序遍历采样的 timesteps
        timesteps = list(range(self.T - 1, -1, -step_size))[:T_inf]
        if timesteps[-1] != 0:
            timesteps.append(0)

        for idx in range(len(timesteps) - 1):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]

            t_batch = torch.full((B,), t_cur, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs)

            ab_t = self.alpha_bar[t_cur]
            ab_tn = self.alpha_bar[t_next]

            # DDIM 一步转移
            # x_{t_next} = sqrt(ab_tn) * pred_x0 + sqrt(1-ab_tn) * direction
            pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            direction = (x - ab_t.sqrt() * pred_x0) / (1 - ab_t).sqrt().clamp(min=1e-8)

            if eta == 0.0:
                # DDIM（确定性，无随机性）
                x = ab_tn.sqrt() * pred_x0 + (1 - ab_tn).sqrt() * direction
            else:
                # DDIM+DDPM 插值：引入受控的随机性
                beta_tn = self.betas[t_next]
                c1 = eta * ((1 - ab_tn / ab_t).clamp(min=0) * (1 - ab_t) / (1 - ab_tn)).sqrt() * beta_tn.sqrt()
                x = ab_tn.sqrt() * pred_x0 + ((1 - ab_tn) - c1 ** 2).clamp(min=0).sqrt() * direction
                if t_next > 0:
                    x = x + c1 * torch.randn_like(x)

        return x


class TemporalFusionNoisePredictor(nn.Module):
    """
    双分支噪声预测器：
    1) 单步分支：原始 FiLM-MLP（保持 diffusion 单步建模能力）
    2) 时序分支：TransformerEncoder 编码 obs 序列
    3) 路由门控：按样本动态融合两个分支的 eps 预测
    """

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden: int = 384,
                 t_dim: int = 64,
                 depth: int = 6,
                 seq_len: int = 8,
                 tf_layers: int = 2,
                 tf_heads: int = 4,
                 tf_dropout: float = 0.1,
                 router_hidden: int = 128,
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.obs_backbone = obs_backbone
        self.base_branch = NoisePredictor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            t_dim=t_dim,
            depth=depth,
            obs_backbone=obs_backbone,
            rot_pair_dim=rot_pair_dim,
            harmonic_order=harmonic_order,
        )

        self.t_emb = SinusoidalPosEmb(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if obs_backbone in ("c4", "c8", "se2", "harmonic"):
            if obs_backbone in ("c4", "c8"):
                self.obs_encoder = RotationInvariantObsEncoder(
                    obs_dim,
                    hidden,
                    pair_dim=rot_pair_dim,
                    n_rot=8 if obs_backbone == "c8" else 4,
                )
            elif obs_backbone == "se2":
                self.obs_encoder = SE2SteerableObsEncoder(obs_dim, hidden, pair_dim=rot_pair_dim)
            else:
                self.obs_encoder = HarmonicObsEncoder(
                    obs_dim,
                    hidden,
                    pair_dim=rot_pair_dim,
                    max_order=harmonic_order,
                )
            self.cur_obs_proj = nn.Identity()
            self.seq_in = nn.Identity()
        else:
            self.obs_encoder = None
            self.cur_obs_proj = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.seq_in = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
            )
        self.noisy_proj = nn.Sequential(
            nn.Linear(act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=tf_heads,
            dim_feedforward=hidden * 4,
            dropout=tf_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.temporal_norm = nn.LayerNorm(hidden)

        self.temporal_head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, act_dim),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 1),
        )

    def _encode_temporal(self, obs_seq: torch.Tensor, obs_mask: torch.Tensor | None) -> torch.Tensor:
        # obs_seq: (B, L, obs_dim), obs_mask: (B, L) True=valid
        B, L, _ = obs_seq.shape
        if L > self.seq_len:
            obs_seq = obs_seq[:, -self.seq_len:, :]
            if obs_mask is not None:
                obs_mask = obs_mask[:, -self.seq_len:]
            L = self.seq_len
        if obs_mask is None:
            obs_mask = torch.ones(B, L, device=obs_seq.device, dtype=torch.bool)

        if self.obs_encoder is not None:
            x = self.obs_encoder(obs_seq.reshape(B * L, -1)).reshape(B, L, -1)
        else:
            x = self.seq_in(obs_seq)
        x = x + self.pos_emb[:, :L, :]
        x = self.temporal_encoder(x, src_key_padding_mask=~obs_mask)
        x = self.temporal_norm(x)

        valid_len = obs_mask.long().sum(dim=1).clamp(min=1)
        last_idx = (valid_len - 1).view(B, 1, 1).expand(-1, 1, x.size(-1))
        return x.gather(1, last_idx).squeeze(1)  # (B, hidden)

    def forward(self,
                noisy_act: torch.Tensor,
                t: torch.Tensor,
                obs: torch.Tensor,
                obs_seq: torch.Tensor,
                obs_mask: torch.Tensor | None = None) -> torch.Tensor:
        eps_base = self.base_branch(noisy_act, t, obs)

        time_ctx = self.t_proj(self.t_emb(t))
        if self.obs_encoder is not None:
            obs_ctx = self.cur_obs_proj(self.obs_encoder(obs))
        else:
            obs_ctx = self.cur_obs_proj(obs)
        temp_ctx = self._encode_temporal(obs_seq, obs_mask)
        noisy_ctx = self.noisy_proj(noisy_act)

        eps_temp = self.temporal_head(torch.cat([noisy_ctx, obs_ctx, time_ctx, temp_ctx], dim=-1))
        gate = torch.sigmoid(self.router(torch.cat([obs_ctx, time_ctx, temp_ctx], dim=-1)))
        return (1.0 - gate) * eps_base + gate * eps_temp


class BCDiffusionTemporal(BCDiffusion):
    """
    扩展版 Diffusion：多步输入、单步输出。
    在标准 diffusion 单步分支上，增加 Transformer 时序分支并做路由融合。
    """

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 T: int = 100,
                 beta_min: float = 1e-4,
                 beta_max: float = 2e-2,
                 hidden: int = 384,
                 depth: int = 6,
                 scheduler: str = "cosine",
                 seq_len: int = 8,
                 tf_layers: int = 2,
                 tf_heads: int = 4,
                 tf_dropout: float = 0.1,
                 router_hidden: int = 128,
                 obs_backbone: str = "mlp",
                 rot_pair_dim: int | None = None,
                 harmonic_order: int = 4):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=T,
            beta_min=beta_min,
            beta_max=beta_max,
            hidden=hidden,
            depth=depth,
            scheduler=scheduler,
            obs_backbone=obs_backbone,
            rot_pair_dim=rot_pair_dim,
            harmonic_order=harmonic_order,
        )
        self.seq_len = seq_len
        self.noise_pred = TemporalFusionNoisePredictor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            depth=depth,
            seq_len=seq_len,
            tf_layers=tf_layers,
            tf_heads=tf_heads,
            tf_dropout=tf_dropout,
            router_hidden=router_hidden,
            obs_backbone=obs_backbone,
            rot_pair_dim=rot_pair_dim,
            harmonic_order=harmonic_order,
        )

    def forward(self,
                obs: torch.Tensor,
                act: torch.Tensor,
                obs_seq: torch.Tensor,
                obs_mask: torch.Tensor | None = None,
                action_noise_std: float = 0.0) -> torch.Tensor:
        B = obs.shape[0]
        t = torch.randint(0, self.T, (B,), device=obs.device)
        if action_noise_std > 0:
            act = act + torch.randn_like(act) * action_noise_std
        noise = torch.randn_like(act)
        x_t = self.q_sample(act, t, noise)
        eps_pred = self.noise_pred(x_t, t, obs, obs_seq, obs_mask)
        return torch.nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def ddpm_sample(self,
                    obs: torch.Tensor,
                    obs_seq: torch.Tensor,
                    obs_mask: torch.Tensor | None = None,
                    T_inf: int | None = None) -> torch.Tensor:
        T_inf = T_inf or self.T
        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device)

        for i in reversed(range(T_inf)):
            t_batch = torch.full((B,), i, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs, obs_seq, obs_mask)
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            ab_t = self.alpha_bar[i]

            coef = beta_t / (1 - ab_t).sqrt()
            mean = (x - coef * eps) / alpha_t.sqrt()
            if i > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self,
                    obs: torch.Tensor,
                    obs_seq: torch.Tensor,
                    obs_mask: torch.Tensor | None = None,
                    T_inf: int = 20,
                    eta: float = 0.0) -> torch.Tensor:
        B = obs.shape[0]
        step_size = self.T // T_inf
        x = torch.randn(B, self.act_dim, device=obs.device)

        timesteps = list(range(self.T - 1, -1, -step_size))[:T_inf]
        if timesteps[-1] != 0:
            timesteps.append(0)

        for idx in range(len(timesteps) - 1):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]

            t_batch = torch.full((B,), t_cur, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs, obs_seq, obs_mask)
            ab_t = self.alpha_bar[t_cur]
            ab_tn = self.alpha_bar[t_next]

            pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            direction = (x - ab_t.sqrt() * pred_x0) / (1 - ab_t).sqrt().clamp(min=1e-8)

            if eta == 0.0:
                x = ab_tn.sqrt() * pred_x0 + (1 - ab_tn).sqrt() * direction
            else:
                beta_tn = self.betas[t_next]
                c1 = eta * ((1 - ab_tn / ab_t).clamp(min=0) * (1 - ab_t) / (1 - ab_tn)).sqrt() * beta_tn.sqrt()
                x = ab_tn.sqrt() * pred_x0 + ((1 - ab_tn) - c1 ** 2).clamp(min=0).sqrt() * direction
                if t_next > 0:
                    x = x + c1 * torch.randn_like(x)
        return x

