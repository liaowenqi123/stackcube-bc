from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_npz_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def split_idx(n: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


class RunningNormalizer:
    """
    对 obs 或 act 做 z-score 归一化，并剔除常量维度（std < eps）。
    计算完毕后可序列化为 dict，方便存入 checkpoint。
    """

    def __init__(self, eps: float = 1e-8, const_thresh: float = 1e-3):
        self.eps = eps
        self.const_thresh = const_thresh
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        # valid_mask[i] = True 表示该维度有效（非常量）
        self.valid_mask: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RunningNormalizer":
        """x: (N, D) float32"""
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.valid_mask = self.std >= self.const_thresh
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "call fit() first"
        # 先选有效维度，再归一化
        xv = x[:, self.valid_mask]
        std_v = self.std[self.valid_mask]
        return ((xv - self.mean[self.valid_mask]) / np.maximum(std_v, self.eps)).astype(np.float32)

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """x: (D,) — 用于 eval 时单步推理"""
        assert self.mean is not None
        xv = x[self.valid_mask]
        std_v = self.std[self.valid_mask]
        return ((xv - self.mean[self.valid_mask]) / np.maximum(std_v, self.eps)).astype(np.float32)

    @property
    def out_dim(self) -> int:
        assert self.valid_mask is not None
        return int(self.valid_mask.sum())

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "valid_mask": self.valid_mask.tolist(),
            "eps": self.eps,
            "const_thresh": self.const_thresh,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "RunningNormalizer":
        obj = cls(eps=d["eps"], const_thresh=d["const_thresh"])
        obj.mean = np.array(d["mean"], dtype=np.float32)
        obj.std = np.array(d["std"], dtype=np.float32)
        obj.valid_mask = np.array(d["valid_mask"], dtype=bool)
        return obj
