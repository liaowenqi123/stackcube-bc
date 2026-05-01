"""
train_bcdiffusion.py
====================
Diffusion Policy (DDPM + DDIM) 训练脚本。

改进点（v2 vs v1）：
  1. EMA 权重平均（decay=0.999）— 大幅提升推理质量
  2. Cosine Beta Schedule — 更平滑的噪声调度
  3. Action Noise Augmentation — 训练时对 action 加噪，提升鲁棒性
  4. SWA（随机权重平均）— 最终几个 epoch 平均权重
  5. 支持 scheduler="cosine"|"linear"
  6. hidden=384, depth=6（增大模型容量）
  7. 更长训练（500 epochs）

用法示例（PowerShell）：
    python src\train_bcdiffusion.py `
        --dataset .\data\processed\stackcube_rl_state.npz `
        --outdir .\outputs\diffusion_v2 `
        --epochs 500 `
        --batch-size 512 `
        --lr 1e-4 `
        --hidden 384 `
        --depth 6 `
        --ema-decay 0.999 `
        --scheduler cosine `
        --action-noise-std 0.01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models import BCDiffusion, BCDiffusionTemporal, EMA


class TemporalStepDataset(Dataset):
    """
    构建多步输入、单步监督样本：
    输入: obs_seq (L, D), obs_mask (L,)
    监督: 当前时刻 obs_t, act_t
    """

    def __init__(self, obs: np.ndarray, acts: np.ndarray,
                 starts: np.ndarray, lengths: np.ndarray, seq_len: int):
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for s, l in zip(starts.tolist(), lengths.tolist()):
            ep_obs = obs[s:s + l]
            ep_act = acts[s:s + l]
            for t in range(l):
                left = max(0, t - seq_len + 1)
                seq = ep_obs[left:t + 1]
                n = seq.shape[0]
                seq_pad = np.zeros((seq_len, obs.shape[1]), dtype=np.float32)
                mask = np.zeros((seq_len,), dtype=np.bool_)
                seq_pad[-n:] = seq
                mask[-n:] = True
                self.samples.append((
                    seq_pad.astype(np.float32),
                    mask,
                    ep_obs[t].astype(np.float32),
                    ep_act[t].astype(np.float32),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, mask, obs_t, act_t = self.samples[idx]
        return (
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.from_numpy(obs_t),
            torch.from_numpy(act_t),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           required=True)
    p.add_argument("--outdir",            required=True)
    p.add_argument("--epochs",            type=int,   default=500)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--weight-decay",      type=float, default=1e-5)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    # ── Diffusion 超参 ──────────────────────────────────────────────────────
    p.add_argument("--T",               type=int,   default=100,
                   help="DDPM 训练扩散步数")
    p.add_argument("--beta-min",        type=float, default=1e-4)
    p.add_argument("--beta-max",        type=float, default=2e-2)
    p.add_argument("--hidden",          type=int,   default=384,
                   help="NoisePredictor 隐层宽度")
    p.add_argument("--depth",           type=int,   default=6,
                   help="NoisePredictor 层数")
    p.add_argument("--scheduler",       type=str,   default="cosine",
                   choices=["linear", "cosine"],
                   help="Beta schedule：cosine 更平滑（推荐）")
    p.add_argument("--temporal", action="store_true",
                   help="启用时序分支（Transformer）进行多步输入单步输出")
    p.add_argument("--seq-len",         type=int,   default=8,
                   help="时序输入长度 L（仅 temporal 模式生效）")
    p.add_argument("--tf-layers",       type=int,   default=2,
                   help="TransformerEncoder 层数（仅 temporal 模式）")
    p.add_argument("--tf-heads",        type=int,   default=4,
                   help="Transformer 多头数（仅 temporal 模式）")
    p.add_argument("--tf-dropout",      type=float, default=0.1,
                   help="Transformer dropout（仅 temporal 模式）")
    p.add_argument("--router-hidden",   type=int,   default=128,
                   help="路由门控隐藏层宽度（仅 temporal 模式）")
    p.add_argument("--obs-backbone",    type=str,   default="mlp",
                   choices=["mlp", "c4"],
                   help="观测编码骨干：mlp（默认）或 c4（离散旋转不变）")
    p.add_argument("--c4-pair-dim",     type=int,   default=-1,
                   help="C4 模式下按(x,y)成对处理的前缀维度，-1 表示自动取最大偶数维")
    # ── EMA ────────────────────────────────────────────────────────────────
    p.add_argument("--ema-decay",       type=float, default=0.999,
                   help="EMA decay 系数（0.999 推荐，0=禁用 EMA）")
    # ── 数据增强 ──────────────────────────────────────────────────────────
    p.add_argument("--action-noise-std", type=float, default=0.01,
                   help="训练时对 action 加噪的标准差（0=禁用，推荐 0.01）")
    # ── SWA ────────────────────────────────────────────────────────────────
    p.add_argument("--swa-start",       type=int,   default=450,
                   help="SWA 开始 epoch（推荐总 epoch 数的最后 10%%）")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    data  = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)
    starts = data.get("ep_starts")
    lengths = data.get("ep_lengths")

    if args.temporal:
        if starts is None or lengths is None:
            raise KeyError("temporal mode requires ep_starts and ep_lengths in dataset npz")
        n_ep = len(starts)
        ep_train, ep_val = split_idx(n_ep, val_ratio=0.1, seed=args.seed)
        train_mask = np.zeros(len(x_raw), dtype=bool)
        for s, l in zip(starts[ep_train].tolist(), lengths[ep_train].tolist()):
            train_mask[s:s + l] = True
        print(f"Dataset (temporal): {len(ep_train)} train episodes / {len(ep_val)} val episodes")
    else:
        train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=args.seed)
        train_mask = np.zeros(len(x_raw), dtype=bool)
        train_mask[train_idx] = True
        print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")
    print(f"obs_dim={x_raw.shape[1]}  act_dim={y_raw.shape[1]}")

    # ── 归一化（fit 仅在训练集）───────────────────────────────────────────
    obs_norm = RunningNormalizer(const_thresh=1e-3)
    obs_norm.fit(x_raw[train_mask])
    act_norm = RunningNormalizer(const_thresh=0.0)
    act_norm.fit(y_raw[train_mask])

    x = obs_norm.transform(x_raw)
    y = act_norm.transform(y_raw)

    if args.temporal:
        train_ds = TemporalStepDataset(x, y, starts[ep_train], lengths[ep_train], args.seq_len)
        val_ds = TemporalStepDataset(x, y, starts[ep_val], lengths[ep_val], args.seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        x_train = torch.from_numpy(x[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        x_val   = torch.from_numpy(x[val_idx])
        y_val   = torch.from_numpy(y[val_idx])

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # ── 模型 ─────────────────────────────────────────────────────────────────
    device = select_device()
    c4_pair_dim = None if args.c4_pair_dim < 0 else int(args.c4_pair_dim)
    if args.temporal:
        model = BCDiffusionTemporal(
            obs_dim=x.shape[1],
            act_dim=y.shape[1],
            T=args.T,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            hidden=args.hidden,
            depth=args.depth,
            scheduler=args.scheduler,
            seq_len=args.seq_len,
            tf_layers=args.tf_layers,
            tf_heads=args.tf_heads,
            tf_dropout=args.tf_dropout,
            router_hidden=args.router_hidden,
            obs_backbone=args.obs_backbone,
            c4_pair_dim=c4_pair_dim,
        ).to(device)
    else:
        model = BCDiffusion(
            obs_dim=x.shape[1],
            act_dim=y.shape[1],
            T=args.T,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            hidden=args.hidden,
            depth=args.depth,
            scheduler=args.scheduler,
            obs_backbone=args.obs_backbone,
            c4_pair_dim=c4_pair_dim,
        ).to(device)

    # ── EMA ──────────────────────────────────────────────────────────────────
    use_ema  = args.ema_decay > 0
    ema_shadow = None
    if use_ema:
        ema_shadow = EMA(model, decay=args.ema_decay, device=device)
        print(f"EMA enabled: decay={args.ema_decay}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1 - 1 / 20) + 1 / 20

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── SWA ──────────────────────────────────────────────────────────────────
    swa_start  = args.swa_start
    swa_active = False
    swa_count  = 0
    swa_buffer = {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if p.requires_grad}

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    outdir      = ensure_dir(args.outdir)
    best        = float("inf")
    best_path   = outdir / "best.pt"
    train_curve = []
    val_curve   = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_samples  = 0
        for batch in train_loader:
            if args.temporal:
                seqb, maskb, xb, yb = batch
                seqb = seqb.to(device)
                maskb = maskb.to(device)
                xb = xb.to(device)
                yb = yb.to(device)
                loss = model(xb, yb, obs_seq=seqb, obs_mask=maskb, action_noise_std=args.action_noise_std)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                loss = model(xb, yb, action_noise_std=args.action_noise_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()
            global_step += 1
            train_loss  += loss.item() * xb.size(0)
            n_samples   += xb.size(0)

            # EMA 更新（每步）
            if ema_shadow is not None:
                ema_shadow.update()

            # SWA 累加
            if swa_active:
                for (name, p), buf in zip(model.named_parameters(), swa_buffer.values()):
                    buf.add_(p.data)
                swa_count += 1

        train_loss /= n_samples

        # ── Val ──
        model.eval()
        val_loss = 0.0
        v_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.temporal:
                    seqb, maskb, xb, yb = batch
                    seqb = seqb.to(device)
                    maskb = maskb.to(device)
                    xb = xb.to(device)
                    yb = yb.to(device)
                    batch_loss = model(xb, yb, obs_seq=seqb, obs_mask=maskb).item()
                else:
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    batch_loss = model(xb, yb).item()
                val_loss += batch_loss * xb.size(0)
                v_samples += xb.size(0)
        val_loss /= v_samples

        train_curve.append(float(train_loss))
        val_curve.append(float(val_loss))

        lr_now = scheduler.get_last_lr()[0]
        marker = " [SWA]" if swa_active else ""
        print(f"epoch={epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}{marker}")

        # SWA 激活判断
        if not swa_active and epoch >= swa_start:
            swa_active = True
            print(f"  >>> SWA started at epoch {epoch}")

        # 保存 best（用 EMA 权重）
        if val_loss < best:
            best = val_loss
            # 用 EMA shadow 权重保存
            if ema_shadow is not None:
                ema_shadow.apply_shadow()
            torch.save(
                {
                    "model":     model.state_dict(),
                    "obs_dim":   x.shape[1],
                    "act_dim":   y.shape[1],
                    "algo":      "diffusion_temporal_v1" if args.temporal else "diffusion_v2",
                    "T":         args.T,
                    "beta_min":  args.beta_min,
                    "beta_max":  args.beta_max,
                    "hidden":    args.hidden,
                    "depth":     args.depth,
                    "scheduler": args.scheduler,
                    "temporal":  args.temporal,
                    "seq_len":   args.seq_len,
                    "tf_layers": args.tf_layers,
                    "tf_heads":  args.tf_heads,
                    "tf_dropout": args.tf_dropout,
                     "router_hidden": args.router_hidden,
                     "obs_backbone": args.obs_backbone,
                     "c4_pair_dim": c4_pair_dim,
                     "obs_norm":  obs_norm.state_dict(),
                     "act_norm":  act_norm.state_dict(),
                     "ema_decay": args.ema_decay,
                 },
                best_path,
            )
            if ema_shadow is not None:
                ema_shadow.restore()

    # ── SWA 写回（平均权重替换） ─────────────────────────────────────────────
    if swa_count > 0:
        print(f"Applying SWA average ({swa_count} updates)...")
        for name, buf in swa_buffer.items():
            for pname, p in model.named_parameters():
                if name == pname:
                    p.data.copy_(buf / swa_count)
        torch.save(
            {
                "model":     model.state_dict(),
                "obs_dim":   x.shape[1],
                "act_dim":   y.shape[1],
                "algo":      "diffusion_temporal_v1_swa" if args.temporal else "diffusion_v2_swa",
                "T":         args.T,
                "beta_min":  args.beta_min,
                "beta_max":  args.beta_max,
                "hidden":    args.hidden,
                "depth":     args.depth,
                "scheduler": args.scheduler,
                "temporal":  args.temporal,
                "seq_len":   args.seq_len,
                "tf_layers": args.tf_layers,
                "tf_heads":  args.tf_heads,
                "tf_dropout": args.tf_dropout,
                 "router_hidden": args.router_hidden,
                 "obs_backbone": args.obs_backbone,
                 "c4_pair_dim": c4_pair_dim,
                 "obs_norm":  obs_norm.state_dict(),
                 "act_norm":  act_norm.state_dict(),
                 "ema_decay": args.ema_decay,
                "swa_count": swa_count,
            },
            outdir / "swa.pt",
        )

    # ── 保存训练曲线 ────────────────────────────────────────────────────────
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "best_val_loss": best,
            "train_loss": train_curve,
            "val_loss": val_curve,
            "args": vars(args),
        }, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label="train_loss")
    plt.plot(val_curve,   label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("DDPM noise prediction loss")
    plt.title("Diffusion Policy v2 Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved best={best_path}  best_val={best:.6f}")


if __name__ == "__main__":
    main()
