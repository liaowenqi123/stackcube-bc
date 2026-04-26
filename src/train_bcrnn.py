from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models import BCRNN


class SeqDataset(Dataset):
    def __init__(self, obs: np.ndarray, acts: np.ndarray, starts: np.ndarray, lengths: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for s, l in zip(starts.tolist(), lengths.tolist()):
            for t in range(0, max(1, l - seq_len + 1)):
                i0 = s + t
                i1 = min(s + t + seq_len, s + l)
                o = obs[i0:i1]
                a = acts[i0:i1]
                if o.shape[0] < seq_len:
                    pad = seq_len - o.shape[0]
                    o = np.pad(o, ((0, pad), (0, 0)))
                    a = np.pad(a, ((0, pad), (0, 0)))
                self.samples.append((o.astype(np.float32), a.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        o, a = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npz_dataset(args.dataset)
    obs_raw  = data["obs"].astype(np.float32)
    acts_raw = data["acts"].astype(np.float32)
    starts   = data["ep_starts"]
    lengths  = data["ep_lengths"]

    n_ep = len(starts)
    ep_train, ep_val = split_idx(n_ep, val_ratio=0.1, seed=args.seed)

    # ── Normalizer: fit on train episodes only ──────────────────────────
    train_mask = np.zeros(obs_raw.shape[0], dtype=bool)
    for s, l in zip(starts[ep_train].tolist(), lengths[ep_train].tolist()):
        train_mask[s:s + l] = True

    obs_norm = RunningNormalizer(const_thresh=1e-3)
    obs_norm.fit(obs_raw[train_mask])

    act_norm = RunningNormalizer(const_thresh=0.0)
    act_norm.fit(acts_raw[train_mask])

    obs  = obs_norm.transform(obs_raw)
    acts = act_norm.transform(acts_raw)

    print(f"obs_dim (after removing const): {obs.shape[1]}  (raw: {obs_raw.shape[1]})")
    print(f"act_dim: {acts.shape[1]}")

    train_ds = SeqDataset(obs, acts, starts[ep_train], lengths[ep_train], args.seq_len)
    val_ds   = SeqDataset(obs, acts, starts[ep_val],   lengths[ep_val],   args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = select_device()
    model  = BCRNN(obs_dim=obs.shape[1], act_dim=acts.shape[1]).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)
    loss_fn = nn.MSELoss()

    outdir    = ensure_dir(args.outdir)
    best      = float("inf")
    best_path = outdir / "best.pt"
    train_curve = []
    val_curve   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n    = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = xb.size(0)
            train_loss += loss.item() * bs
            train_n    += bs
        train_loss /= max(1, train_n)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_n    = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred, _ = model(xb)
                loss = loss_fn(pred, yb)
                bs = xb.size(0)
                val_loss += loss.item() * bs
                val_n    += bs
        val_loss /= max(1, val_n)

        train_curve.append(float(train_loss))
        val_curve.append(float(val_loss))

        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch={epoch:03d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  lr={lr_now:.2e}")

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model":    model.state_dict(),
                    "obs_dim":  obs.shape[1],
                    "act_dim":  acts.shape[1],
                    "algo":     "bcrnn",
                    "obs_norm": obs_norm.state_dict(),
                    "act_norm": act_norm.state_dict(),
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
    plt.title("BC-RNN Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "loss_curve.png", dpi=150)
    plt.close()
    print(f"saved best checkpoint: {best_path}  (best_val_mse={best:.6f})")


if __name__ == "__main__":
    main()
