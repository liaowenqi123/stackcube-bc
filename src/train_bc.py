from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models import BCMLP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)
    train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=args.seed)

    # ── Normalizer: fit 仅在训练集上 ──────────────────────────────────────
    obs_norm = RunningNormalizer(const_thresh=1e-3)
    obs_norm.fit(x_raw[train_idx])

    act_norm = RunningNormalizer(const_thresh=0.0)   # act 不剔除维度，但仍归一化
    act_norm.fit(y_raw[train_idx])

    x = obs_norm.transform(x_raw)
    y = act_norm.transform(y_raw)

    print(f"obs_dim (after removing const): {x.shape[1]}  (raw: {x_raw.shape[1]})")
    print(f"act_dim: {y.shape[1]}")

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val   = torch.from_numpy(x[val_idx])
    y_val   = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,      # 避免 BatchNorm/LayerNorm 遇到 batch=1
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = select_device()
    model = BCMLP(obs_dim=x.shape[1], act_dim=y.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR schedule: 从 lr 降到 lr/20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 20
    )
    loss_fn = nn.MSELoss()

    outdir    = ensure_dir(args.outdir)
    best      = float("inf")
    best_path = outdir / "best.pt"
    train_curve = []
    val_curve   = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_idx)
        scheduler.step()

        # ── Val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_idx)

        train_curve.append(float(train_loss))
        val_curve.append(float(val_loss))

        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch={epoch:03d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  lr={lr_now:.2e}")

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model":     model.state_dict(),
                    "obs_dim":   x.shape[1],        # 归一化后的维度
                    "act_dim":   y.shape[1],
                    "algo":      "bc",
                    "obs_norm":  obs_norm.state_dict(),
                    "act_norm":  act_norm.state_dict(),
                },
                best_path,
            )

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_mse": best, "train_mse": train_curve, "val_mse": val_curve}, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label="train_mse")
    plt.plot(val_curve, label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("mse (normalized act space)")
    plt.title("BC Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "loss_curve.png", dpi=150)
    plt.close()
    print(f"saved best checkpoint: {best_path}  (best_val_mse={best:.6f})")


if __name__ == "__main__":
    main()
