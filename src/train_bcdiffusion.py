r"""
train_bcdiffusion.py - BC Diffusion Policy training script with optional Koopman DKO.
Usage:
    python src/train_bcdiffusion.py --data data/processed/stackcube_rl_state.npz --koopman
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import BCDiffusionKoopman, BCDiffusion

# ------------------------------------------------------------
# Diffusion helpers (cosine schedule + extract)
# ------------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule as proposed in Nichol & Dhariwal (2021)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """Extract values from a (timestep array) at indices t, reshape to x_shape."""
    b, *_ = t.shape
    out = a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class BCDataset(Dataset):
    def __init__(self, data_path, seq_len=4):
        dat = np.load(data_path)
        self.obs = dat["obs"]   # (N, obs_dim)
        self.acts = dat["acts"]   # (N, act_dim)
        self.seq_len = seq_len
        # obs is per-step; we form sequences by taking consecutive rows
        # For sequence starting at i, obs_seq = obs[i : i+seq_len]
        # target act = acts[i+seq_len-1]  (last step in sequence)
        self.num_samples = len(self.obs) - seq_len
        print(f"[Dataset] obs={self.obs.shape}, acts={self.acts.shape}, "
              f"seq_len={seq_len}, samples={self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # obs_seq: (seq_len, obs_dim)
        obs_seq = self.obs[idx : idx + self.seq_len]
        act = self.acts[idx + self.seq_len - 1]
        return (
            torch.from_numpy(obs_seq.astype(np.float32)),
            torch.from_numpy(act.astype(np.float32)),
        )


# ------------------------------------------------------------
# EMA (Exponential Moving Average)
# ------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_shadow(self):
        self.model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ------------------------------------------------------------
# SWA (Stochastic Weight Averaging) helper
# ------------------------------------------------------------
def swa_update(swa_model, current_model, n_swa_updates):
    """Update SWA buffered params."""
    with torch.no_grad():
        n = n_swa_updates
        for (name, p_swa), (_, p_cur) in zip(
            swa_model.named_parameters(), current_model.named_parameters()
        ):
            p_swa.data = (p_swa.data * n + p_cur.data) / (n + 1)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BC Diffusion Policy Training")
    # data
    parser.add_argument("--data", type=str, default="data/processed/stackcube_rl_state.npz")
    parser.add_argument("--seq-len", type=int, default=4)
    # model
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--koopman", action="store_true", help="Enable Koopman DKO branch")
    parser.add_argument("--koopman-h", type=int, default=4,
                        help="DKO history length h (seq_len = 2h+1 = 9)")
    parser.add_argument("--koopman-warmup-epochs", type=int, default=50,
                        help="Freeze DKO params for first N epochs, then joint train")
    parser.add_argument("--use-dko", action="store_true", default=True,
                        help="Use DKO modulation during sampling (default: True)")
    # diffusion
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--T-inf", type=int, default=20)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--swa", action="store_true", default=True)
    parser.add_argument("--swa-start", type=int, default=150)
    # logging / saving
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_koopman")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # auto-set seq_len for koopman mode: needs 2h+1 frames
    if args.koopman:
        args.seq_len = args.koopman_h * 2 + 1
        print(f"[Koopman] seq_len auto-set to {args.seq_len} (h={args.koopman_h})")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset
    dataset = BCDataset(args.data, seq_len=args.seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 归一化统计量 ────────────────────────────────────────────────
    # 从全量数据计算，并转为 torch Tensor（v3 训练也这样做的）
    obs_mean = torch.from_numpy(dataset.obs.mean(0).astype(np.float32)).to(device)
    obs_std  = torch.from_numpy(dataset.obs.std(0).clip(min=1e-6).astype(np.float32)).to(device)
    act_mean = torch.from_numpy(dataset.acts.mean(0).astype(np.float32)).to(device)
    act_std  = torch.from_numpy(dataset.acts.std(0).clip(min=1e-6).astype(np.float32)).to(device)
    print(f"[Norm] obs μ={obs_mean[:3].cpu().numpy()} σ={obs_std[:3].cpu().numpy()}")
    print(f"[Norm] act μ={act_mean.cpu().numpy()}  σ={act_std.cpu().numpy()}")
    # 转成 list 存 checkpoint（eval 需要）
    obs_norm_state = {
        "mean": dataset.obs.mean(0).tolist(),
        "std": dataset.obs.std(0).clip(min=1e-6).tolist(),
        "valid_mask": [True] * dataset.obs.shape[1],
        "eps": 1e-8,
        "const_thresh": 1e-3,
    }
    act_norm_state = {
        "mean": dataset.acts.mean(0).tolist(),
        "std": dataset.acts.std(0).clip(min=1e-6).tolist(),
        "valid_mask": [True] * dataset.acts.shape[1],
        "eps": 1e-8,
        "const_thresh": 1e-3,
    }

    # model
    obs_dim = dataset.obs.shape[1]   # 48
    act_dim = dataset.acts.shape[1]   # 8
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    if args.koopman:
        model = BCDiffusionKoopman(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=args.hidden, depth=args.depth,
            T=args.T,
        ).to(device)
    else:
        model = BCDiffusion(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=args.hidden, depth=args.depth, heads=args.heads,
            T=args.T, seq_len=args.seq_len,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    # EMA / SWA
    ema = EMA(model) if args.ema else None
    swa_model = None
    n_swa_updates = 0
    if args.swa:
        swa_model = BCDiffusionKoopman(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=args.hidden, depth=args.depth,
            T=args.T,
        ).to(device) if args.koopman else BCDiffusion(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=args.hidden, depth=args.depth, heads=args.heads,
            T=args.T, seq_len=args.seq_len,
        ).to(device)
        swa_model.load_state_dict(model.state_dict())
        for p in swa_model.parameters():
            p.requires_grad = False

    # ----------------------------------------------------------------
    # Warmup helper: freeze / unfreeze DKO params
    # ----------------------------------------------------------------
    def _set_dko_grad(model, requires_grad):
        """Freeze/unfreeze all Koopman DKO related parameters."""
        if not hasattr(model, "dko"):
            return
        for p in model.dko.parameters():
            p.requires_grad = requires_grad
        for p in model.noise_pred.f_u_modulators.parameters():
            p.requires_grad = requires_grad
        status = "trainable" if requires_grad else "FROZEN"
        print(f"  [DKO] params set to: {status}")

    # start with warmup (freeze DKO) if requested
    if args.koopman and args.koopman_warmup_epochs > 0:
        print(f"[Warmup] Freezing DKO params for first {args.koopman_warmup_epochs} epochs")
        _set_dko_grad(model, requires_grad=False)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # switch from warmup to joint training
        if args.koopman and args.koopman_warmup_epochs > 0 and epoch == args.koopman_warmup_epochs + 1:
            print(f"[JointTraining] Unfreezing DKO params at epoch {epoch}")
            _set_dko_grad(model, requires_grad=True)

        for obs_seq, act in loader:
            obs_seq = obs_seq.to(device)   # (B, seq_len, obs_dim)
            act = act.to(device)           # (B, act_dim)

            # ── 归一化（与 eval 一致） ────────────────────────────────
            obs_seq = (obs_seq - obs_mean) / obs_std
            act     = (act - act_mean) / act_std

            # forward - model returns scalar loss internally
            if args.koopman:
                disable_dko = (epoch <= args.koopman_warmup_epochs) if args.koopman_warmup_epochs > 0 else False
                loss = model(obs_seq[:, -1, :], act, obs_seq=obs_seq if not disable_dko else None, disable_dko=disable_dko)
            else:
                loss = model(obs_seq[:, -1, :], act)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            if ema:
                ema.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # SWA snapshot
        if args.swa and epoch >= args.swa_start:
            n_swa_updates += 1
            swa_update(swa_model, model, n_swa_updates)

        # logging
        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  lr={lr_now:.6f}")

        # save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.state_dict() if ema else None,
                "args": vars(args),
                # eval_policy.py needs these keys
                "koopman": args.koopman,
                "hidden": args.hidden,
                "depth": args.depth,
                "scheduler": "cosine",
                "obs_backbone": "mlp",
                "obs_norm": obs_norm_state,
                "act_norm": act_norm_state,
            }
            path = os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"  [Save] {path}")

    # final: save SWA model if used
    if args.swa and swa_model is not None:
        # apply EMA shadow to current model before SWA merge (optional)
        if ema:
            ema.apply_shadow()
        # save final
        final_path = os.path.join(args.output_dir, "ckpt_final.pt")
        torch.save({
            "epoch": args.epochs,
            "model": model.state_dict(),
            "swa_model": swa_model.state_dict() if args.swa else None,
            "args": vars(args),
            # eval_policy.py needs these keys
            "koopman": args.koopman,
            "hidden": args.hidden,
            "depth": args.depth,
            "scheduler": "cosine",
            "obs_backbone": "mlp",
            "obs_norm": obs_norm_state,
            "act_norm": act_norm_state,
        }, final_path)
        print(f"  [Final] {final_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
