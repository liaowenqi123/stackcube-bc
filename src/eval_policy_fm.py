r"""
eval_policy_fm.py
=================
Flow Matching Diffusion Policy 评估脚本。

用法：
    python src\eval_policy_fm.py `
        --ckpt .\outputs\flow_matching_v1\best.pt `
        --episodes 100 `
        --output-dir .\outputs\eval_flow_matching_v1 `
        --max-steps 400 `
        --solver-steps 20 `
        --solver-method euler `
        --save-gif `
        --gif-episodes 3
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import mani_skill.envs  # noqa: F401

from common import RunningNormalizer, select_device
from models_fm import BCFlowMatching, BCFlowMatchingWithNoise


def flatten_obs(obs) -> np.ndarray:
    parts = []

    def rec(x):
        if isinstance(x, dict):
            for k in sorted(x.keys()):
                rec(x[k])
        else:
            a = np.asarray(x, dtype=np.float32)
            parts.append(a.reshape(-1))

    rec(obs)
    return np.concatenate(parts, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-id", default="StackCube-v1")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default="./outputs/eval_flow_matching")
    p.add_argument("--solver-steps", type=int, default=20,
                   help="ODE求解步数（越多越精确）")
    p.add_argument("--solver-method", type=str, default="euler",
                   choices=["euler", "rk4"],
                   help="ODE求解方法")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--action-clip", type=float, default=None)
    args = p.parse_args()

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    # 加载归一化器
    obs_norm: RunningNormalizer | None = None
    act_norm: RunningNormalizer | None = None
    if "obs_norm" in ckpt:
        obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"])
    if "act_norm" in ckpt:
        act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"])

    # 创建模型
    is_noise_model = bool(ckpt.get("noise_weight", 0)) > 0

    if is_noise_model:
        model = BCFlowMatchingWithNoise(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=int(ckpt.get("T", 100)),
            hidden=int(ckpt.get("hidden", 256)),
            depth=int(ckpt.get("depth", 4)),
            obs_backbone=str(ckpt.get("obs_backbone", "mlp")),
            rot_pair_dim=int(ckpt["rot_pair_dim"]) if ckpt.get("rot_pair_dim") is not None else None,
            harmonic_order=int(ckpt.get("harmonic_order", 4)),
            residual_film=bool(ckpt.get("residual_film", True)),
            cfg_strength=float(ckpt.get("cfg_strength", 1.0)),
        ).to(device)
    else:
        model = BCFlowMatching(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=int(ckpt.get("T", 100)),
            hidden=int(ckpt.get("hidden", 256)),
            depth=int(ckpt.get("depth", 4)),
            obs_backbone=str(ckpt.get("obs_backbone", "mlp")),
            rot_pair_dim=int(ckpt["rot_pair_dim"]) if ckpt.get("rot_pair_dim") is not None else None,
            harmonic_order=int(ckpt.get("harmonic_order", 4)),
            residual_film=bool(ckpt.get("residual_film", True)),
            velocity_weight=float(ckpt.get("velocity_weight", 1.0)),
            noise_weight=float(ckpt.get("noise_weight", 0.0)),
            cfg_strength=float(ckpt.get("cfg_strength", 1.0)),
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded {ckpt['algo']}: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Solver: {args.solver_method} with {args.solver_steps} steps")

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )

    n_succ = 0
    ep_returns = []
    ep_lengths = []
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _to_bool(x) -> bool:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return bool(x.item())
            return bool(torch.any(x).item())
        return bool(x)

    def _preprocess_obs(raw_obs, raw_dim: int) -> np.ndarray:
        o = flatten_obs(raw_obs)
        if o.shape[0] != raw_dim:
            o = flatten_obs(env.unwrapped.get_state_dict())
        if obs_norm is not None:
            o = obs_norm.transform_single(o)
        return o

    raw_obs_dim = obs_norm.mean.shape[0] if obs_norm is not None else obs_dim

    with torch.no_grad():
        for ep_idx in range(args.episodes):
            obs, _ = env.reset()
            done = False
            trunc = False
            ret = 0.0
            t = 0
            frames = []

            while not done and not trunc:
                o = _preprocess_obs(obs, raw_obs_dim)
                if o.shape[0] != obs_dim:
                    raise RuntimeError(f"obs dim mismatch: got {o.shape[0]}, expected {obs_dim}")

                ot = torch.from_numpy(o).to(device).unsqueeze(0)

                # Flow Matching ODE采样
                at = model.sample(
                    obs=ot,
                    T_solve=args.solver_steps,
                    method=args.solver_method,
                ).squeeze(0)

                # 反归一化
                action_norm = at.detach().cpu().numpy().astype(np.float32)
                if act_norm is not None:
                    action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
                else:
                    action = action_norm

                if args.action_clip is not None:
                    action = np.clip(action, -args.action_clip, args.action_clip)

                action = action.astype(np.float32)

                obs, r, done, trunc, info = env.step(action)
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
    plt.title(f"Flow Matching Return Distribution")
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
