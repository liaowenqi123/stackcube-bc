r"""
train_bcdiffusion_consistency.py
================================
Consistency-Regularized Diffusion (CRD) 训练脚本。

核心创新：
  1. 一致性正则化：同一动作在不同噪声水平下去噪结果应一致
  2. 简化架构：depth=4, hidden=256（避免v2/v3深度模型在小数据集上过拟合）
  3. 残差FiLM：每层带残差连接，信息流更顺畅
  4. 自适应无分类器引导(CFG)：推理时动态调整引导强度

训练策略：
  - 200 epochs（用户确认200足够）
  - lr=1e-4（diffusion标准学习率）
  - EMA: decay=0.999（保留，有效且稳定）
  - 无SWA（简化，避免复杂超参）
  - 无Action Noise Augmentation（避免干扰一致性正则化）

用法示例：
    python src\train_bcdiffusion_consistency.py `
        --dataset .\data\processed\stackcube_rl_state.npz `
        --outdir .\outputs\diffusion_consistency `
        --epochs 200 `
        --batch-size 512 `
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models import BCDiffusionCRD, EMA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           required=True)
    p.add_argument("--outdir",            required=True)
    p.add_argument("--epochs",            type=int,   default=200)
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
    p.add_argument("--hidden",          type=int,   default=256,
                   help="NoisePredictor 隐层宽度（保持256简化）")
    p.add_argument("--depth",           type=int,   default=4,
                   help="NoisePredictor 层数（默认4，比v2的6更稳定）")
    p.add_argument("--scheduler",       type=str,   default="cosine",
                   choices=["linear", "cosine"],
                   help="Beta schedule：cosine 更平滑（推荐）")
    p.add_argument("--obs-backbone",    type=str,   default="mlp",
                   choices=["mlp", "c4", "c8", "se2", "harmonic"],
                   help="观测编码骨干")
    p.add_argument("--rot-pair-dim",    type=int,   default=-1,
                   help="旋转不变模式下按(x,y)成对处理的前缀维度")
    p.add_argument("--harmonic-order",  type=int,   default=4,
                   help="harmonic 骨干的最高谐波阶数 M")
    p.add_argument("--residual-film",   action="store_true", default=True,
                   help="启用残差FiLM（默认开启）")
    # ── 一致性正则化 ──────────────────────────────────────────────────────
    p.add_argument("--consistency-weight", type=float, default=0.1,
                   help="一致性正则化权重（默认0.1）")
    # ── EMA ────────────────────────────────────────────────────────────────
    p.add_argument("--ema-decay",       type=float, default=0.999,
                   help="EMA decay 系数（0.999 推荐，0=禁用 EMA）")
    # ── 无分类器引导 ──────────────────────────────────────────────────────
    p.add_argument("--cfg-strength",    type=float, default=1.0,
                   help="无分类器引导强度（1.0=无引导，>1.0=更强条件）")
    # ── SWA（可选）────────────────────────────────────────────────────────
    p.add_argument("--swa-start",       type=int,   default=0,
                   help="SWA 开始 epoch（0=禁用）")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    data  = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)

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
    rot_pair_dim = None if args.rot_pair_dim < 0 else int(args.rot_pair_dim)

    model = BCDiffusionCRD(
        obs_dim=x.shape[1],
        act_dim=y.shape[1],
        T=args.T,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        hidden=args.hidden,
        depth=args.depth,
        scheduler=args.scheduler,
        obs_backbone=args.obs_backbone,
        rot_pair_dim=rot_pair_dim,
        harmonic_order=args.harmonic_order,
        residual_film=args.residual_film,
        consistency_weight=args.consistency_weight,
        cfg_strength=args.cfg_strength,
    ).to(device)

    print(f"Model: BCDiffusionCRD (hidden={args.hidden}, depth={args.depth})")
    print(f"Consistency weight: {args.consistency_weight}")
    print(f"CFG strength: {args.cfg_strength}")

    # ── EMA ──────────────────────────────────────────────────────────────────
    use_ema = args.ema_decay > 0
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
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            loss = model(xb, yb)
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
        if not swa_active and swa_start > 0 and epoch >= swa_start:
            swa_active = True
            print(f"  >>> SWA started at epoch {epoch}")

        # 保存 best（用 EMA 权重）
        if val_loss < best:
            best = val_loss
            if ema_shadow is not None:
                ema_shadow.apply_shadow()
            torch.save(
                {
                    "model":     model.state_dict(),
                    "obs_dim":   x.shape[1],
                    "act_dim":   y.shape[1],
                    "algo":      "diffusion_consistency_v1",
                    "T":         args.T,
                    "beta_min":  args.beta_min,
                    "beta_max":  args.beta_max,
                    "hidden":    args.hidden,
                    "depth":     args.depth,
                    "scheduler": args.scheduler,
                    "obs_backbone": args.obs_backbone,
                    "rot_pair_dim": rot_pair_dim,
                    "harmonic_order": int(args.harmonic_order),
                    "residual_film": args.residual_film,
                    "consistency_weight": args.consistency_weight,
                    "cfg_strength": args.cfg_strength,
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
                "algo":      "diffusion_consistency_v1_swa",
                "T":         args.T,
                "beta_min":  args.beta_min,
                "beta_max":  args.beta_max,
                "hidden":    args.hidden,
                "depth":     args.depth,
                "scheduler": args.scheduler,
                "obs_backbone": args.obs_backbone,
                "rot_pair_dim": rot_pair_dim,
                "harmonic_order": int(args.harmonic_order),
                "residual_film": args.residual_film,
                "consistency_weight": args.consistency_weight,
                "cfg_strength": args.cfg_strength,
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
    plt.ylabel("Consistency-Regularized DDPM loss")
    plt.title("CRD Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved best={best_path}  best_val={best:.6f}")


if __name__ == "__main__":
    main()
