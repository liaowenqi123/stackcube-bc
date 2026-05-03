"""
train_entity.py - Entity-Aware Diffusion Policy 训练脚本。

和v3唯一的区别：用BCDiffusionEntity替换BCDiffusion。

用法同 train_bcdiffusion.py:
    python run_train_patched.py --model entity ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models_entity import BCDiffusionEntity
from models import EMA


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
    p.add_argument("--T",                 type=int,   default=100)
    p.add_argument("--hidden",            type=int,   default=384)
    p.add_argument("--depth",             type=int,   default=6)
    p.add_argument("--scheduler",         type=str,   default="cosine", choices=["linear", "cosine"])
    p.add_argument("--ema-decay",         type=float, default=0.0)
    p.add_argument("--swa-start",         type=int,   default=450)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)
    train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=args.seed)

    obs_norm = RunningNormalizer(const_thresh=1e-3)
    obs_norm.fit(x_raw[train_idx])
    act_norm = RunningNormalizer(const_thresh=0.0)
    act_norm.fit(y_raw[train_idx])
    x = obs_norm.transform(x_raw)
    y = act_norm.transform(y_raw)

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])),
        batch_size=args.batch_size, shuffle=False)

    device = select_device()
    model = BCDiffusionEntity(
        obs_dim=x.shape[1], act_dim=y.shape[1],
        T=args.T, hidden=args.hidden, depth=args.depth,
        scheduler=args.scheduler,
    ).to(device)

    print(f"Model: {type(model).__name__} (hidden={args.hidden}, depth={args.depth})")
    print(f"T={args.T}, scheduler={args.scheduler}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=args.ema_decay, device=device) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # ── 在线SWA ─────────────────────────────────────────────
    swa_buffer = {}
    swa_count = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = model(xb, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema: ema.update()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = model(xb, yb)
                val_loss += loss.item() * xb.shape[0]
        val_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 在线SWA：收集最后 args.swa_start 个epoch的模型权重
        if epoch >= args.swa_start:
            sd = ema.ema_state_dict() if ema else model.state_dict()
            for k, v in sd.items():
                swa_buffer[k] = swa_buffer.get(k, 0.0) + v
            swa_count += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = ema.ema_state_dict() if ema else model.state_dict()
            torch.save({
                "model": state, "obs_dim": x.shape[1], "act_dim": y.shape[1],
                "T": args.T, "hidden": args.hidden, "depth": args.depth,
                "scheduler": args.scheduler,
                "obs_norm": obs_norm.state_dict(), "act_norm": act_norm.state_dict(),
                "train_losses": train_losses, "val_losses": val_losses,
                "epoch": epoch, "algo": "diffusion_entity",
            }, outdir / "best.pt")

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val_loss:.4f}")

    # ── 保存SWA ─────────────────────────────────────────────
    if swa_count > 0:
        for k in swa_buffer:
            swa_buffer[k] /= swa_count
        torch.save({
            "model": swa_buffer, "obs_dim": x.shape[1], "act_dim": y.shape[1],
            "T": args.T, "hidden": args.hidden, "depth": args.depth,
            "scheduler": args.scheduler,
            "obs_norm": obs_norm.state_dict(), "act_norm": act_norm.state_dict(),
            "train_losses": train_losses, "val_losses": val_losses,
            "epoch": args.epochs, "algo": "diffusion_entity",
        }, outdir / "swa.pt")
        print(f"SWA saved ({swa_count} epochs averaged)")

    # Plot
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(outdir / "loss.png", dpi=150)
    plt.close()
    json.dump({"best_val_loss": best_val_loss, "train_loss": train_losses, "val_loss": val_losses, "args": vars(args)},
              open(outdir / "metrics.json", "w"), indent=2)


if __name__ == '__main__':
    main()
