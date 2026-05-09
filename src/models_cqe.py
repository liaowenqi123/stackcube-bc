from __future__ import annotations

import math

import torch
from torch import nn

from models import EMA, SinusoidalPosEmb


class QuasiEquivariantObsEncoder(nn.Module):
    """
    Contextual Quasi-Equivariant (CQE) 观测编码：
    - 绝对分支 h_abs：保留机器人绝对状态偏置
    - 相对分支 h_rel：提取平面几何关系特征
    - 门控融合 h = alpha*h_rel + (1-alpha)*h_abs
    - 可学习群作用器 T_phi(h, g)：建模“近似对称关系”
    """

    def __init__(self, obs_dim: int, hidden: int, rot_pair_dim: int | None = None, trans_pairs: int = 2):
        super().__init__()
        max_pair_dim = obs_dim - (obs_dim % 2)
        if rot_pair_dim is None:
            rot_pair_dim = max_pair_dim
        rot_pair_dim = max(0, min(int(rot_pair_dim), max_pair_dim))
        if rot_pair_dim % 2 != 0:
            rot_pair_dim -= 1

        self.obs_dim = obs_dim
        self.rot_pair_dim = rot_pair_dim
        self.n_vec = rot_pair_dim // 2
        self.scalar_dim = obs_dim - rot_pair_dim
        self.trans_pairs = max(0, min(int(trans_pairs), self.n_vec))

        self.abs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        rel_feat_dim = self.scalar_dim + self.n_vec + 4
        self.rel_encoder = nn.Sequential(
            nn.Linear(rel_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(obs_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # T_phi(h, g): g = [sin(theta), cos(theta), tx, ty]
        self.group_adapter = nn.Sequential(
            nn.Linear(hidden + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = obs[:, :self.rot_pair_dim].reshape(obs.shape[0], self.n_vec, 2) if self.n_vec > 0 else obs.new_zeros(obs.shape[0], 0, 2)
        s = obs[:, self.rot_pair_dim:]
        return v, s

    def _relative_features(self, obs: torch.Tensor) -> torch.Tensor:
        v, s = self._split(obs)
        if self.n_vec == 0:
            return torch.cat([s, torch.zeros(obs.shape[0], 4, device=obs.device, dtype=obs.dtype)], dim=-1)

        norms = torch.sqrt((v * v).sum(dim=-1).clamp(min=1e-12))  # (B, N)
        center = v.mean(dim=1)                                     # (B, 2)
        centered = v - center.unsqueeze(1)
        spread = torch.sqrt((centered * centered).sum(dim=-1).clamp(min=1e-12)).mean(dim=1, keepdim=True)  # (B,1)
        avg_norm = norms.mean(dim=1, keepdim=True)                 # (B,1)
        return torch.cat([s, norms, center, spread, avg_norm], dim=-1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        h_abs = self.abs_encoder(obs)
        h_rel = self.rel_encoder(self._relative_features(obs))
        alpha = torch.sigmoid(self.gate(obs))
        return alpha * h_rel + (1.0 - alpha) * h_abs

    def predict_group_action(self, h: torch.Tensor, theta: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
        g = torch.stack([torch.sin(theta), torch.cos(theta), tx, ty], dim=-1)
        delta = self.group_adapter(torch.cat([h, g], dim=-1))
        return h + delta

    def transform_obs(self, obs: torch.Tensor, theta: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
        if self.n_vec == 0:
            return obs
        v, s = self._split(obs)
        c = torch.cos(theta).view(-1, 1, 1)
        sn = torch.sin(theta).view(-1, 1, 1)
        x = v[..., 0:1]
        y = v[..., 1:2]
        xr = c * x - sn * y
        yr = sn * x + c * y
        vr = torch.cat([xr, yr], dim=-1)
        if self.trans_pairs > 0:
            vr[:, :self.trans_pairs, 0] = vr[:, :self.trans_pairs, 0] + tx.view(-1, 1)
            vr[:, :self.trans_pairs, 1] = vr[:, :self.trans_pairs, 1] + ty.view(-1, 1)
        return torch.cat([vr.reshape(obs.shape[0], self.rot_pair_dim), s], dim=-1)


class CQENoisePredictor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 384, depth: int = 6, t_dim: int = 64,
                 rot_pair_dim: int | None = None, trans_pairs: int = 2):
        super().__init__()
        self.obs_enc = QuasiEquivariantObsEncoder(obs_dim, hidden, rot_pair_dim=rot_pair_dim, trans_pairs=trans_pairs)
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.layers = nn.ModuleList()
        self.film = nn.ModuleList()
        in_dim = act_dim
        for _ in range(depth):
            self.layers.append(nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden)))
            self.film.append(nn.Linear(hidden + t_dim, hidden * 2))
            in_dim = hidden
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, act_dim)

    def forward(self, noisy_act: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        h = self.obs_enc.encode(obs)
        te = self.t_emb(t)
        cond = torch.cat([h, te], dim=-1)
        x = noisy_act
        for layer, fm in zip(self.layers, self.film):
            x = layer(x)
            gamma, beta = fm(cond).chunk(2, dim=-1)
            x = self.act(x * (1 + gamma) + beta)
        return self.out(x)


class BCQuasiEquivDiffusion(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 T: int = 100, beta_min: float = 1e-4, beta_max: float = 2e-2,
                 hidden: int = 384, depth: int = 6, scheduler: str = "cosine",
                 rot_pair_dim: int | None = None, trans_pairs: int = 2):
        super().__init__()
        self.T = T
        self.act_dim = act_dim
        self.scheduler = scheduler
        self._build_schedule(beta_min, beta_max)
        self.noise_pred = CQENoisePredictor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            depth=depth,
            rot_pair_dim=rot_pair_dim,
            trans_pairs=trans_pairs,
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

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def _sample_group(self, B: int, device: torch.device, theta_max_deg: float, trans_max: float):
        theta_max = math.radians(theta_max_deg)
        theta = (torch.rand(B, device=device) * 2 - 1) * theta_max
        tx = (torch.rand(B, device=device) * 2 - 1) * trans_max
        ty = (torch.rand(B, device=device) * 2 - 1) * trans_max
        return theta, tx, ty

    def forward(self, obs: torch.Tensor, act: torch.Tensor, action_noise_std: float = 0.0,
                sym_lambda: float = 0.1, id_lambda: float = 0.05,
                theta_max_deg: float = 180.0, trans_max: float = 0.05):
        B = obs.shape[0]
        t = torch.randint(0, self.T, (B,), device=obs.device)
        if action_noise_std > 0:
            act = act + torch.randn_like(act) * action_noise_std
        noise = torch.randn_like(act)
        x_t = self.q_sample(act, t, noise)
        eps_pred = self.noise_pred(x_t, t, obs)
        core_loss = torch.nn.functional.mse_loss(eps_pred, noise)

        theta, tx, ty = self._sample_group(B, obs.device, theta_max_deg=theta_max_deg, trans_max=trans_max)
        obs_g = self.noise_pred.obs_enc.transform_obs(obs, theta, tx, ty)
        h = self.noise_pred.obs_enc.encode(obs)
        h_g = self.noise_pred.obs_enc.encode(obs_g)
        h_g_pred = self.noise_pred.obs_enc.predict_group_action(h, theta, tx, ty)
        sym_loss = torch.nn.functional.mse_loss(h_g_pred, h_g.detach())

        z = torch.zeros(B, device=obs.device, dtype=obs.dtype)
        h_id = self.noise_pred.obs_enc.predict_group_action(h, z, z, z)
        id_loss = torch.nn.functional.mse_loss(h_id, h.detach())

        total = core_loss + sym_lambda * sym_loss + id_lambda * id_loss
        return total, core_loss.detach(), sym_loss.detach(), id_loss.detach()

    @torch.no_grad()
    def ddpm_sample(self, obs: torch.Tensor, T_inf: int | None = None) -> torch.Tensor:
        T_inf = T_inf or self.T
        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device)
        for i in reversed(range(T_inf)):
            t_batch = torch.full((B,), i, device=obs.device, dtype=torch.long)
            eps = self.noise_pred(x, t_batch, obs)
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            ab_t = self.alpha_bar[i]
            coef = beta_t / (1 - ab_t).sqrt()
            mean = (x - coef * eps) / alpha_t.sqrt()
            x = mean + beta_t.sqrt() * torch.randn_like(x) if i > 0 else mean
        return x

    @torch.no_grad()
    def ddim_sample(self, obs: torch.Tensor, T_inf: int = 20, eta: float = 0.0) -> torch.Tensor:
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
