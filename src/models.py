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


class C4InvariantObsEncoder(nn.Module):
    """
    C4 (0/90/180/270 deg) 离散旋转不变观测编码器。
    将观测前 pair_dim 维按 (x, y) 成对解释为平面向量，对四个旋转共享同一编码器后做 group average。
    """

    def __init__(self, obs_dim: int, hidden: int, pair_dim: int | None = None):
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

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        rot_mats = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],    # 0
                [[0.0, -1.0], [1.0, 0.0]],   # 90
                [[-1.0, 0.0], [0.0, -1.0]],  # 180
                [[0.0, 1.0], [-1.0, 0.0]],   # 270
            ],
            dtype=torch.float32,
        )
        self.register_buffer("rot_mats", rot_mats)

    def _group_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, D) -> (B, 4, D)
        B, D = obs.shape
        if D != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch in C4InvariantObsEncoder: got {D}, expected {self.obs_dim}")

        if self.pair_dim == 0:
            return obs.unsqueeze(1).expand(B, 4, D)

        v = obs[:, :self.pair_dim].reshape(B, self.pair_dim // 2, 2)
        v_rot = torch.einsum("gij,bkj->bgki", self.rot_mats.to(obs.dtype), v).reshape(B, 4, self.pair_dim)
        if self.scalar_dim > 0:
            s = obs[:, self.pair_dim:].unsqueeze(1).expand(B, 4, self.scalar_dim)
            return torch.cat([v_rot, s], dim=-1)
        return v_rot

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, D) -> (B, hidden)
        xg = self._group_transform(obs)
        B, G, D = xg.shape
        h = self.encoder(xg.reshape(B * G, D)).reshape(B, G, -1)
        return h.mean(dim=1)


# ── Noise Predictor ────────────────────────────────────────────────────────────

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
                 c4_pair_dim: int | None = None):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        if obs_backbone == "c4":
            self.obs_emb = C4InvariantObsEncoder(obs_dim, hidden, pair_dim=c4_pair_dim)
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
                 c4_pair_dim: int | None = None):
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
            c4_pair_dim=c4_pair_dim,
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
                 c4_pair_dim: int | None = None):
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
            c4_pair_dim=c4_pair_dim,
        )

        self.t_emb = SinusoidalPosEmb(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if obs_backbone == "c4":
            self.obs_encoder = C4InvariantObsEncoder(obs_dim, hidden, pair_dim=c4_pair_dim)
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
                 c4_pair_dim: int | None = None):
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
            c4_pair_dim=c4_pair_dim,
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
            c4_pair_dim=c4_pair_dim,
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

