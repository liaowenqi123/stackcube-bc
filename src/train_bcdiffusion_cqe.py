from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models import EMA
from models_cqe import BCQuasiEquivDiffusion


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--beta-min", type=float, default=1e-4)
    p.add_argument("--beta-max", type=float, default=2e-2)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--rot-pair-dim", type=int, default=16)
    p.add_argument("--trans-pairs", type=int, default=2)
    p.add_argument("--sym-lambda", type=float, default=0.1)
    p.add_argument("--id-lambda", type=float, default=0.05)
    p.add_argument("--theta-max-deg", type=float, default=180.0)
    p.add_argument("--trans-max", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--action-noise-std", type=float, default=0.01)
    p.add_argument("--swa-start", type=int, default=270)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)
    train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=args.seed)
    train_mask = np.zeros(len(x_raw), dtype=bool)
    train_mask[train_idx] = True
    print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")
    print(f"obs_dim={x_raw.shape[1]}  act_dim={y_raw.shape[1]}")

    obs_norm = RunningNormalizer(const_thresh=1e-3).fit(x_raw[train_mask])
    act_norm = RunningNormalizer(const_thresh=0.0).fit(y_raw[train_mask])
    x = obs_norm.transform(x_raw)
    y = act_norm.transform(y_raw)

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device()
    model = BCQuasiEquivDiffusion(
        obs_dim=x.shape[1],
        act_dim=y.shape[1],
        T=args.T,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        hidden=args.hidden,
        depth=args.depth,
        scheduler=args.scheduler,
        rot_pair_dim=args.rot_pair_dim,
        trans_pairs=args.trans_pairs,
    ).to(device)

    ema_shadow = EMA(model, decay=args.ema_decay, device=device) if args.ema_decay > 0 else None
    if ema_shadow is not None:
        print(f"EMA enabled: decay={args.ema_decay}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1 - 1 / 20) + 1 / 20

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    swa_active = False
    swa_count = 0
    swa_buffer = {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if p.requires_grad}

    outdir = ensure_dir(args.outdir)
    best = float("inf")
    best_path = outdir / "best.pt"
    train_curve: list[float] = []
    val_curve: list[float] = []
    sym_curve: list[float] = []
    id_curve: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        sym_loss_avg = 0.0
        id_loss_avg = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss, _core, sym_l, id_l = model(
                xb,
                yb,
                action_noise_std=args.action_noise_std,
                sym_lambda=args.sym_lambda,
                id_lambda=args.id_lambda,
                theta_max_deg=args.theta_max_deg,
                trans_max=args.trans_max,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()
            if ema_shadow is not None:
                ema_shadow.update()
            if swa_active:
                for (name, p), buf in zip(model.named_parameters(), swa_buffer.values()):
                    buf.add_(p.data)
                swa_count += 1
            bs = xb.size(0)
            train_loss += loss.item() * bs
            sym_loss_avg += float(sym_l.item()) * bs
            id_loss_avg += float(id_l.item()) * bs
            n_samples += bs
        train_loss /= n_samples
        sym_loss_avg /= n_samples
        id_loss_avg /= n_samples

        model.eval()
        val_loss = 0.0
        v_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                batch_loss, _, _, _ = model(
                    xb,
                    yb,
                    action_noise_std=0.0,
                    sym_lambda=args.sym_lambda,
                    id_lambda=args.id_lambda,
                    theta_max_deg=args.theta_max_deg,
                    trans_max=args.trans_max,
                )
                val_loss += batch_loss.item() * xb.size(0)
                v_samples += xb.size(0)
        val_loss /= v_samples

        train_curve.append(float(train_loss))
        val_curve.append(float(val_loss))
        sym_curve.append(float(sym_loss_avg))
        id_curve.append(float(id_loss_avg))
        marker = " [SWA]" if swa_active else ""
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} sym={sym_loss_avg:.6f} id={id_loss_avg:.6f}{marker}")

        if not swa_active and epoch >= args.swa_start:
            swa_active = True
            print(f"  >>> SWA started at epoch {epoch}")

        if val_loss < best:
            best = val_loss
            if ema_shadow is not None:
                ema_shadow.apply_shadow()
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": x.shape[1],
                    "act_dim": y.shape[1],
                    "algo": "cqe_diffusion_v1",
                    "T": args.T,
                    "beta_min": args.beta_min,
                    "beta_max": args.beta_max,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "scheduler": args.scheduler,
                    "rot_pair_dim": args.rot_pair_dim,
                    "trans_pairs": args.trans_pairs,
                    "sym_lambda": args.sym_lambda,
                    "id_lambda": args.id_lambda,
                    "theta_max_deg": args.theta_max_deg,
                    "trans_max": args.trans_max,
                    "obs_norm": obs_norm.state_dict(),
                    "act_norm": act_norm.state_dict(),
                    "ema_decay": args.ema_decay,
                },
                best_path,
            )
            if ema_shadow is not None:
                ema_shadow.restore()

    if swa_count > 0:
        print(f"Applying SWA average ({swa_count} updates)...")
        for name, buf in swa_buffer.items():
            for pname, p in model.named_parameters():
                if name == pname:
                    p.data.copy_(buf / swa_count)
        torch.save(
            {
                "model": model.state_dict(),
                "obs_dim": x.shape[1],
                "act_dim": y.shape[1],
                "algo": "cqe_diffusion_v1_swa",
                "T": args.T,
                "beta_min": args.beta_min,
                "beta_max": args.beta_max,
                "hidden": args.hidden,
                "depth": args.depth,
                "scheduler": args.scheduler,
                "rot_pair_dim": args.rot_pair_dim,
                "trans_pairs": args.trans_pairs,
                "sym_lambda": args.sym_lambda,
                "id_lambda": args.id_lambda,
                "theta_max_deg": args.theta_max_deg,
                "trans_max": args.trans_max,
                "obs_norm": obs_norm.state_dict(),
                "act_norm": act_norm.state_dict(),
                "ema_decay": args.ema_decay,
                "swa_count": swa_count,
            },
            outdir / "swa.pt",
        )

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": best,
                "train_loss": train_curve,
                "val_loss": val_curve,
                "sym_loss": sym_curve,
                "id_loss": id_curve,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(9, 5))
    plt.plot(train_curve, label="train_total")
    plt.plot(val_curve, label="val_total")
    plt.plot(sym_curve, label="train_sym")
    plt.plot(id_curve, label="train_id")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CQE Diffusion Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved best={best_path}  best_val={best:.6f}")


if __name__ == "__main__":
    main()
