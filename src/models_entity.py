"""
Entity-Aware Diffusion Policy: 双分支架构

分支1 (Standard): 标准NoisePredictor处理全量48维观测
分支2 (Entity):    提取(ee, cubeA, cubeB, goal)四个实体→Cross-Attention→实体关系特征
融合: 两个分支的特征concat后预测噪声

这相当于"在v3之上加了一个实体关系支路"。
"""
from __future__ import annotations

import math
import torch
from torch import nn

from models import NoisePredictor, SinusoidalPosEmb, BCDiffusion


class EntityEncoder(nn.Module):
    """轻量实体编码器：比v1缩小5倍，避免过拟合"""
    
    def __init__(self, entity_dim=7, n_entities=4, hidden=384, ent_hidden=64, n_heads=2):
        super().__init__()
        self.n_entities = n_entities
        
        # 每个实体投影到小维度
        self.entity_proj = nn.Sequential(
            nn.Linear(entity_dim, ent_hidden),
            nn.LayerNorm(ent_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, n_entities, ent_hidden) * 0.02)
        
        # 单层MultiheadAttention（轻量cross-attention，不用完整Transformer）
        self.cross_attn = nn.MultiheadAttention(
            ent_hidden, num_heads=n_heads, dropout=0.2,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(ent_hidden)
        self.attn_drop = nn.Dropout(0.2)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(ent_hidden, ent_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(ent_hidden * 2, ent_hidden),
        )
        self.ffn_norm = nn.LayerNorm(ent_hidden)
        
        # 输出投影到main分支的hidden维
        self.out_proj = nn.Linear(ent_hidden, hidden)
    
    def forward(self, obs):
        entities = torch.stack([
            obs[:, 18:25], obs[:, 25:32],
            obs[:, 32:39], obs[:, 39:46],
        ], dim=1)  # (B, 4, 7)
        
        tok = self.entity_proj(entities) + self.pos_embed  # (B, 4, 64)
        
        # Self-attention（所有实体交互）
        attn_out, _ = self.cross_attn(tok, tok, tok)
        tok = self.attn_norm(tok + self.attn_drop(attn_out))
        
        # FFN
        tok = self.ffn_norm(tok + self.ffn(tok))
        
        # 池化 → 投影到主分支维度
        pooled = tok.mean(dim=1)  # (B, 64)
        return self.out_proj(pooled)  # (B, 384)


class NoisePredictorEntity(nn.Module):
    """
    双分支噪声预测器:
    - 标准分支: 沿用v3的NoisePredictor逻辑（obs_emb + FiLM）
    - 实体分支: EntityEncoder (Cross-Attention)
    - 融合: 两个分支的特征相加，然后进入标准流程
    """
    
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 384, t_dim: int = 64, depth: int = 6):
        super().__init__()
        
        # ── 分支1: 标准观测编码 ──
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        # 标准MLP backbone（和v3一样）
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        
        # ── 分支2: 实体关系编码（Cross-Attention） ──
        self.entity_encoder = EntityEncoder(
            entity_dim=7, n_entities=4,
            hidden=hidden, ent_hidden=64, n_heads=2,
        )
        
        # 融合门控: 自适应融合两个分支
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        
        # ── 标准去噪层（和v3完全一样） ──
        self.layers    = nn.ModuleList()
        self.film_mods = nn.ModuleList()
        in_dim = act_dim
        for i in range(depth):
            out_dim = hidden
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            ))
            self.film_mods.append(nn.Linear(hidden + t_dim, out_dim * 2))
            in_dim = out_dim
        
        self.act_fn = nn.GELU()
        self.out    = nn.Linear(hidden, act_dim)
    
    def forward(self, noisy_act, t, obs):
        # ── 时间编码 ──
        t_e = self.t_emb(t)  # (B, t_dim)
        
        # ── 分支1: 标准obs编码 ──
        obs_e = self.obs_emb(obs)  # (B, hidden)
        
        # ── 分支2: 实体关系编码 ──
        entity_e = self.entity_encoder(obs)  # (B, hidden)
        
        # ── 自适应融合 ──
        gate = self.fusion_gate(torch.cat([obs_e, entity_e], dim=-1))  # (B, hidden)
        fused = gate * obs_e + (1 - gate) * entity_e  # (B, hidden)
        
        # ── 条件（和v3一样） ──
        cond = torch.cat([fused, t_e], dim=-1)
        
        # ── FiLM去噪层（和v3完全一样） ──
        x = noisy_act
        for layer, film in zip(self.layers, self.film_mods):
            x = layer(x)
            gam_bet = film(cond)
            gamma, beta = gam_bet.chunk(2, dim=-1)
            x = self.act_fn(x * (1 + gamma) + beta)
        
        return self.out(x)


class BCDiffusionEntity(BCDiffusion):
    """Entity-Aware版本，继承BCDiffusion，只替换噪声预测器"""
    
    def __init__(self, obs_dim: int, act_dim: int,
                 T: int = 100, hidden: int = 384, depth: int = 6,
                 beta_min: float = 1e-4, beta_max: float = 2e-2,
                 scheduler: str = "cosine"):
        super().__init__(
            obs_dim=obs_dim, act_dim=act_dim, T=T,
            hidden=hidden, depth=depth,
            beta_min=beta_min, beta_max=beta_max,
            scheduler=scheduler,
        )
        # 替换为Entity-Aware噪声预测器
        self.noise_pred = NoisePredictorEntity(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=hidden, depth=depth,
        )
        self.algo = "diffusion_entity"
