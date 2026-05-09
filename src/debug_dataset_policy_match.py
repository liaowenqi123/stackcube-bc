from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_dataset_auto


def _fmt_stats(name: str, arr: np.ndarray) -> str:
    return (
        f"{name}: shape={arr.shape}, min={arr.min():.4f}, p1={np.percentile(arr, 1):.4f}, "
        f"p50={np.percentile(arr, 50):.4f}, p99={np.percentile(arr, 99):.4f}, max={arr.max():.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Check dataset / checkpoint compatibility for diffusion policy.")
    p.add_argument("--dataset", required=True, help="npz or hdf5 dataset path")
    p.add_argument("--obs-keys", nargs="+", default=None, help="required for hdf5")
    p.add_argument("--action-key", default="actions", help="hdf5 action dataset key")
    p.add_argument("--ckpt", default=None, help="optional checkpoint path to compare dims/norms")
    p.add_argument("--sample-size", type=int, default=20000, help="num transitions sampled for stats")
    args = p.parse_args()

    data = load_dataset_auto(args.dataset, obs_keys=args.obs_keys, action_key=args.action_key)
    obs = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.float32)

    n = len(obs)
    if n == 0:
        raise RuntimeError("empty dataset")

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(args.sample_size, n), replace=False)
    obs_s = obs[idx]
    act_s = acts[idx]

    print("=== Dataset Summary ===")
    print(f"path: {Path(args.dataset).resolve()}")
    print(f"num_samples: {n}")
    print(f"obs_dim: {obs.shape[1]}")
    print(f"act_dim: {acts.shape[1]}")
    print(_fmt_stats("obs", obs_s))
    print(_fmt_stats("acts", act_s))

    const_obs = int((obs.std(axis=0) < 1e-6).sum())
    const_act = int((acts.std(axis=0) < 1e-6).sum())
    print(f"constant obs dims (<1e-6 std): {const_obs}")
    print(f"constant act dims (<1e-6 std): {const_act}")

    out_of_range = float(np.mean(np.abs(act_s) > 1.5))
    print(f"fraction(|action| > 1.5): {out_of_range:.6f}")
    print("\n=== Action Dimension Diagnostics ===")
    act_std = act_s.std(axis=0)
    for d in range(act_s.shape[1]):
        col = act_s[:, d]
        frac_eq_pos1 = float(np.mean(np.isclose(col, 1.0, atol=1e-4)))
        frac_eq_neg1 = float(np.mean(np.isclose(col, -1.0, atol=1e-4)))
        frac_abs_ge_099 = float(np.mean(np.abs(col) >= 0.99))
        print(
            f"dim {d:02d}: std={act_std[d]:.6f}, "
            f"frac(==+1)={frac_eq_pos1:.4f}, frac(==-1)={frac_eq_neg1:.4f}, frac(|a|>=0.99)={frac_abs_ge_099:.4f}"
        )

    low_var_dims = np.where(act_std < 1e-3)[0].tolist()
    if low_var_dims:
        print(f"WARNING: near-constant action dims (std < 1e-3): {low_var_dims}")

    if args.ckpt is None:
        return

    print("\n=== Checkpoint Match ===")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_obs_dim = int(ckpt["obs_dim"])
    ckpt_act_dim = int(ckpt["act_dim"])
    print(f"ckpt: {Path(args.ckpt).resolve()}")
    print(f"ckpt obs_dim={ckpt_obs_dim}, act_dim={ckpt_act_dim}")

    dim_ok = (ckpt_obs_dim == obs.shape[1]) and (ckpt_act_dim == acts.shape[1])
    print(f"dim_match: {dim_ok}")
    if not dim_ok:
        print("WARNING: dimension mismatch detected. This alone can invalidate evaluation.")

    if "obs_norm" in ckpt and "act_norm" in ckpt:
        obs_mean = np.array(ckpt["obs_norm"]["mean"], dtype=np.float32)
        obs_std = np.array(ckpt["obs_norm"]["std"], dtype=np.float32)
        obs_mask = np.array(ckpt["obs_norm"]["valid_mask"], dtype=bool)
        act_mean = np.array(ckpt["act_norm"]["mean"], dtype=np.float32)
        act_std = np.array(ckpt["act_norm"]["std"], dtype=np.float32)

        if len(obs_mean) == obs.shape[1]:
            obs_train_mean = obs_s.mean(axis=0)
            obs_z = (obs_train_mean[obs_mask] - obs_mean[obs_mask]) / np.maximum(obs_std[obs_mask], 1e-8)
            print(f"obs mean shift z-score: p50={np.percentile(np.abs(obs_z), 50):.3f}, p95={np.percentile(np.abs(obs_z), 95):.3f}")
        else:
            print("WARNING: ckpt obs_norm mean length != dataset obs_dim")

        if len(act_mean) == acts.shape[1]:
            act_train_mean = act_s.mean(axis=0)
            act_z = (act_train_mean - act_mean) / np.maximum(act_std, 1e-8)
            print(f"act mean shift z-score: p50={np.percentile(np.abs(act_z), 50):.3f}, p95={np.percentile(np.abs(act_z), 95):.3f}")
        else:
            print("WARNING: ckpt act_norm mean length != dataset act_dim")
    else:
        print("checkpoint has no obs_norm/act_norm fields")

    brief = {
        "dataset_samples": int(n),
        "dataset_obs_dim": int(obs.shape[1]),
        "dataset_act_dim": int(acts.shape[1]),
        "ckpt_obs_dim": int(ckpt_obs_dim),
        "ckpt_act_dim": int(ckpt_act_dim),
        "dim_match": bool(dim_ok),
    }
    print("\n=== Brief JSON ===")
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
