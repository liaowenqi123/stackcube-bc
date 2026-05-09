from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import mani_skill.envs  # noqa: F401

from common import RunningNormalizer, select_device
from models_cqe import BCQuasiEquivDiffusion


def flatten_obs(obs) -> np.ndarray:
    parts = []

    def rec(x):
        if isinstance(x, dict):
            for k in sorted(x.keys()):
                rec(x[k])
        else:
            parts.append(np.asarray(x, dtype=np.float32).reshape(-1))

    rec(obs)
    return np.concatenate(parts, axis=0)


def flatten_state_dict(state_dict) -> np.ndarray:
    parts = []

    def rec(x):
        if isinstance(x, dict):
            for k in sorted(x.keys()):
                rec(x[k])
        else:
            a = np.asarray(x, dtype=np.float32)
            if a.ndim >= 2 and a.shape[0] == 1:
                a = a[0]
            parts.append(a.reshape(-1))

    rec(state_dict)
    return np.concatenate(parts, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-id", default="StackCube-v1")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default="./outputs/eval_cqe")
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--T-inf", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--action-clip", type=float, default=None)
    args = p.parse_args()

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    model = BCQuasiEquivDiffusion(
        obs_dim=obs_dim,
        act_dim=act_dim,
        T=int(ckpt.get("T", 100)),
        beta_min=float(ckpt.get("beta_min", 1e-4)),
        beta_max=float(ckpt.get("beta_max", 2e-2)),
        hidden=int(ckpt.get("hidden", 384)),
        depth=int(ckpt.get("depth", 6)),
        scheduler=str(ckpt.get("scheduler", "cosine")),
        rot_pair_dim=int(ckpt.get("rot_pair_dim", 16)),
        trans_pairs=int(ckpt.get("trans_pairs", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"]) if "obs_norm" in ckpt else None
    act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"]) if "act_norm" in ckpt else None
    raw_obs_dim = obs_norm.mean.shape[0] if obs_norm is not None else obs_dim

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_succ = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    def _to_bool(x) -> bool:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return bool(x.item())
            return bool(torch.any(x).item())
        return bool(x)

    with torch.no_grad():
        for ep_idx in range(args.episodes):
            obs, _ = env.reset()
            done = False
            trunc = False
            ret = 0.0
            t = 0
            frames = []
            while not done and not trunc:
                o = flatten_obs(obs)
                if o.shape[0] != raw_obs_dim:
                    o = flatten_state_dict(env.unwrapped.get_state_dict())
                if obs_norm is not None:
                    o = obs_norm.transform_single(o)
                if o.shape[0] != obs_dim:
                    raise RuntimeError(f"obs dim mismatch: got {o.shape[0]} expected {obs_dim}")

                ot = torch.from_numpy(o).to(device).unsqueeze(0)
                if args.sampler == "ddim":
                    at = model.ddim_sample(ot, T_inf=args.T_inf, eta=args.eta).squeeze(0)
                else:
                    at = model.ddpm_sample(ot, T_inf=args.T_inf).squeeze(0)

                action_norm = at.detach().cpu().numpy().astype(np.float32)
                action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean if act_norm is not None else action_norm
                if args.action_clip is not None:
                    action = np.clip(action, -args.action_clip, args.action_clip)
                obs, r, done, trunc, info = env.step(action.astype(np.float32))
                done = _to_bool(done)
                trunc = _to_bool(trunc)
                ret += float(r)
                t += 1
                if args.save_gif and ep_idx < args.gif_episodes:
                    frame = env.render()
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu().numpy()
                    if isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                        frame = frame[0]
                    if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        frames.append(frame)
                if t >= args.max_steps:
                    trunc = True
                if done and _to_bool(info.get("success", False)):
                    n_succ += 1
            ep_returns.append(ret)
            ep_lengths.append(t)
            if frames:
                imageio.mimsave(outdir / f"rollout_ep{ep_idx:03d}.gif", frames, duration=0.04)

    env.close()
    sr = n_succ / args.episodes
    metrics = {
        "success_rate": float(sr),
        "avg_return": float(np.mean(ep_returns)),
        "avg_ep_len": float(np.mean(ep_lengths)),
        "episodes": int(args.episodes),
    }
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(ep_returns, bins=20)
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title("CQE Diffusion Return Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "returns_hist.png", dpi=150)
    plt.close()

    print(f"success_rate={sr:.4f}")
    print(f"avg_return={float(np.mean(ep_returns)):.4f}")
    print(f"avg_ep_len={float(np.mean(ep_lengths)):.2f}")
    print(f"saved eval artifacts: {outdir}")


if __name__ == "__main__":
    main()
