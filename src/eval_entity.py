"""
eval_entity.py - Entity-Aware Diffusion Policy 评估脚本。
用法同 eval_policy.py，支持 --algo diffusion_entity。

可评估 best.pt 或 swa.pt，用 --ckpt-type 指定。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import mani_skill.envs  # noqa: F401

from common import RunningNormalizer, select_device
from models_entity import BCDiffusionEntity


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint路径 (best.pt 或 swa.pt)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--output-dir", default="./outputs/eval_entity")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--T-inf", type=int, default=20, help="DDIM推理步数")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    args = p.parse_args()

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"])
    act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"])

    model = BCDiffusionEntity(
        obs_dim=obs_dim, act_dim=act_dim,
        T=int(ckpt.get("T", 100)),
        hidden=int(ckpt.get("hidden", 384)),
        depth=int(ckpt.get("depth", 6)),
        scheduler=str(ckpt.get("scheduler", "cosine")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = gym.make(
        "StackCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos",
        render_mode="rgb_array" if args.save_gif else None,
        max_episode_steps=args.max_steps,
    )

    n_succ = 0
    ep_returns = []
    ep_lengths = []
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=args.episodes, desc="Eval", unit="ep")
    for ep_idx in range(args.episodes):
        obs, _ = env.reset()
        ret = 0.0
        t = 0
        frames = []

        while True:
            o = obs[0].cpu().numpy()
            if obs_norm:
                o = obs_norm.transform_single(o)
            ot = torch.from_numpy(o).to(device).unsqueeze(0)

            with torch.no_grad():
                at = model.ddim_sample(ot, T_inf=args.T_inf).squeeze(0)

            action_norm = at.detach().cpu().numpy().astype(np.float32)
            action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean

            obs, r, done, trunc, info = env.step(action)
            ret += float(r)
            t += 1

            if args.save_gif and ep_idx < args.gif_episodes and t <= args.max_steps:
                frame = env.render()
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                if isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)

            if done or trunc:
                if info.get("success", False):
                    n_succ += 1
                break

        ep_returns.append(ret)
        ep_lengths.append(t)
        if frames:
            imageio.mimsave(outdir / f"rollout_ep{ep_idx:03d}.gif", frames, duration=0.04)
        pbar.update(1)
    pbar.close()
    env.close()

    sr = n_succ / args.episodes
    metrics = {
        "success_rate": float(sr),
        "avg_return": float(np.mean(ep_returns)),
        "avg_ep_len": float(np.mean(ep_lengths)),
        "episodes": int(args.episodes),
    }
    with (outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    plt.hist(ep_returns, bins=20)
    plt.xlabel("return"); plt.ylabel("count")
    plt.title(f"Entity Diffusion: SR={sr:.1%}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "returns.png", dpi=150)
    plt.close()

    print(f"success_rate={sr:.4f}")
    print(f"avg_return={float(np.mean(ep_returns)):.4f}")
    print(f"avg_ep_len={float(np.mean(ep_lengths)):.2f}")


if __name__ == "__main__":
    main()
