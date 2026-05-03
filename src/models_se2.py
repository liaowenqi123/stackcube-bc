"""
SE(2) Equivariant Model for StackCube task.

核心思想：
1. FK计算末端位置 → ee_pos (x,y,z)
2. 从观测中提取方块A、方块B、目标位置 → (cubeA, cubeB, goal)
3. 这4个点在水平面(x,y)上形成SE(2)等变结构
4. 机械臂本身的偏置通过z轴和joint signals单独处理

模型结构：
- FK分支：从关节角计算ee_pos
- 几何编码分支：将(ee, cubeA, cubeB, goal)编码为SE(2)等变特征
- 合并分支：联合其他特征（joint velocities, previous action等）
- 输出：action
"""

import math
import torch
from torch import nn
import numpy as np


class SE2FeatureExtractor(nn.Module):
    """
    SE(2)等变特征提取器。
    
    输入: (B, 4, 3) = [ee_pos, cubeA_pos, cubeB_pos, goal_pos] (每个点x,y,z)
    输出: (B, D) SE(2)不变特征
    
    不变性通过以下方式实现：
    - 成对距离（旋转不变）
    - 高度z（本身旋转不变）
    - 相对角度编码（通过sin/cos编码）
    """
    
    def __init__(self, hidden_dim=128, n_pts=4):
        super().__init__()
        self.n_pts = n_pts
        
        # 成对特征维数: n_pts * (n_pts-1) / 2 对 * 3 (dx, dy, dz)
        n_pairs = n_pts * (n_pts - 1) // 2
        pair_feat_dim = n_pairs * 3  # dx, dy, dz for each pair
        
        # 每点的z和原始xyz
        single_feat_dim = n_pts * 3  # x, y, z for each point
        
        total_feat_dim = pair_feat_dim + single_feat_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
    def forward(self, points):
        """
        Args:
            points: (B, 4, 3) [ee, cubeA, cubeB, goal] in (x,y,z)
        Returns:
            (B, D) SE(2) invariant features
        """
        B = points.shape[0]
        
        # 成对相对向量 (SE(2)中dx,dy是等变的，dz是不变的)
        pair_feats = []
        for i in range(self.n_pts):
            for j in range(i+1, self.n_pts):
                rel = points[:, i] - points[:, j]  # (B, 3) = (dx, dy, dz)
                pair_feats.append(rel)
        
        pair_feats = torch.cat(pair_feats, dim=-1)  # (B, n_pairs * 3)
        
        # 原始xyz
        single_feats = points.reshape(B, -1)  # (B, 4*3)
        
        # 合并
        feats = torch.cat([pair_feats, single_feats], dim=-1)
        
        return self.encoder(feats)


class SE2NoisePredictor(nn.Module):
    """
    SE(2)等变噪声预测器（用于Diffusion Policy）。
    
    输入:
      x: (B, act_dim) noisy action
      t: (B,) timestep
      obs_dict: dict with keys:
        - 'joint_pos': (B, 7)
        - 'joint_vel': (B, 7)  
        - 'cube_info': (B, 30) 或其他维度的立方体信息
        - 'prev_action': (B, 2)
        - 'ee_pos': (B, 3)
        - 'cubeA_pos': (B, 3)
        - 'cubeB_pos': (B, 3)
        - 'goal_pos': (B, 3)
    
    输出:
      noise_pred: (B, act_dim)
    """
    
    def __init__(self, obs_dim, act_dim, hidden=256, depth=4, joint_dim=7):
        super().__init__()
        
        # SE(2)等变特征提取
        self.se2_enc = SE2FeatureExtractor(hidden_dim=128)
        
        # 关节编码（处理机械臂偏置）
        self.joint_enc = nn.Sequential(
            nn.Linear(joint_dim * 2, 64),  # joint_pos + joint_vel
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # 时间步编码
        self.time_enc = TimeEmbedding()
        
        # 动作编码
        self.act_enc = nn.Linear(act_dim, 64)
        
        # 合并网络
        total_dim = 128 + 64 + 64 + 64  # se2 + joint + time + action
        layers = []
        for i in range(depth):
            in_dim = total_dim if i == 0 else hidden
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden, act_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t, obs_dict):
        """
        x: (B, act_dim) noisy action
        t: (B,) timestep
        obs_dict: dict
        """
        # SE(2)特征
        points = torch.stack([
            obs_dict['ee_pos'],      # (B, 3)
            obs_dict['cubeA_pos'],   # (B, 3)
            obs_dict['cubeB_pos'],   # (B, 3)
            obs_dict['goal_pos'],    # (B, 3)
        ], dim=1)  # (B, 4, 3)
        se2_feat = self.se2_enc(points)  # (B, 128)
        
        # 关节特征
        joint_feat = self.joint_enc(torch.cat([obs_dict['joint_pos'], obs_dict['joint_vel']], dim=-1))
        
        # 时间特征
        t_feat = self.time_enc(t)
        
        # 动作特征
        a_feat = self.act_enc(x)
        
        # 合并
        combined = torch.cat([se2_feat, joint_feat, t_feat, a_feat], dim=-1)
        
        return self.net(combined)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim=64, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        t: (B,) timestep values in [0, T]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class BCDiffusionSE2(nn.Module):
    """
    SE(2) Equivariant Diffusion Policy模型。
    
    对现有BCDiffusion的改进：
    - 使用FK计算的几何位置做SE(2)等变推理
    - 同时保留关节信息处理机械臂偏置
    """
    
    def __init__(self, obs_dim=48, act_dim=8, T=100, hidden=256, depth=4,
                 beta_min=1e-4, beta_max=2e-2, scheduler='cosine',
                 joint_dim=7, cube_dim=30):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.scheduler = scheduler
        
        # 构建噪声预测器（SE(2)等变版本）
        self.noise_predictor = SE2NoisePredictor(
            obs_dim=obs_dim, act_dim=act_dim, 
            hidden=hidden, depth=depth,
            joint_dim=joint_dim
        )
        
        # 初始化扩散参数
        self.fit_betas()
        
    def fit_betas(self):
        """初始化beta schedule和预计算参数"""
        if self.scheduler == 'cosine':
            # Cosine schedule
            steps = self.T + 1
            s = 0.008
            t = torch.linspace(0, self.T, steps)
            f = torch.cos((t / self.T + s) / (1 + s) * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            # Linear schedule
            betas = torch.linspace(self.beta_min, self.beta_max, self.T)
        
        self.register_buffer('betas', betas)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        
    def q_sample(self, x0, t, noise=None):
        """前向加噪过程: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, noise
    
    def forward(self, obs_dict, acts):
        """训练前向：计算MSE损失"""
        B = acts.shape[0]
        device = acts.device
        
        # 随机采样timestep
        t = torch.randint(0, self.T, (B,), device=device).float()
        
        # 采样噪声
        noise = torch.randn_like(acts)
        
        # 加噪
        x_t, _ = self.q_sample(acts, t.long(), noise)
        
        # 预测噪声
        noise_pred = self.noise_predictor(x_t, t, obs_dict)
        
        # MSE损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def ddim_sample(self, obs_dict, T_inf=20, eta=0.0):
        """
        DDIM采样（使用与原版相同的算法）。
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = next(iter(obs_dict.values())).device
        
        x = torch.randn(B, self.act_dim, device=device)
        
        step_size = self.T // T_inf
        timesteps = list(range(self.T - 1, -1, -step_size))[:T_inf]
        if timesteps[-1] != 0:
            timesteps.append(0)
        
        for idx in range(len(timesteps) - 1):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps = self.noise_predictor(x, t_batch.float(), obs_dict)
            
            ab_t = self.alpha_bars[t_cur]
            ab_tn = self.alpha_bars[t_next]
            
            pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)  # 防止发散
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


def build_obs_dict(obs, ee_pos, cubeA_pos, cubeB_pos, goal_pos):
    """
    从原始48维观测和几何位置构建obs_dict。
    
    Obs layout: 
    [0:8] = qpos
    [8:16] = qvel  
    [16:18] = prev_action
    [18:48] = cube_info (推测: cubeA_pos+quat, cubeB_pos+quat, goal_pos+quat)
    """
    return {
        'joint_pos': obs[:, :7],
        'joint_vel': obs[:, 7:14] if obs.shape[-1] >= 14 else obs[:, 8:15],
        'prev_action': obs[:, 16:18] if obs.shape[-1] >= 18 else torch.zeros_like(obs[:, :2]),
        'ee_pos': ee_pos,
        'cubeA_pos': cubeA_pos,
        'cubeB_pos': cubeB_pos,
        'goal_pos': goal_pos,
    }
