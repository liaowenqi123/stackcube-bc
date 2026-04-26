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
