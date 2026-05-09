from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import load_composite_controller_config

from common import RunningNormalizer, select_device
from models import BCMLP, BCRNN, BCDiffusion


DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "object-state",
]


def flatten_obs(obs_dict: dict, obs_keys: list[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        if key not in obs_dict:
            raise KeyError(f"obs key '{key}' missing at runtime")
        arr = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0).astype(np.float32)


def make_env(env_id: str, robot: str, controller: str, horizon: int, use_camera: bool):
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )
    return suite.make(
        env_name=env_id,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=use_camera,
        use_camera_obs=False,
        reward_shaping=True,  # 启用 reward shaping
        horizon=horizon,
        control_freq=20,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["bc", "bcrnn", "diffusion"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="NutAssemblySingle")
    p.add_argument("--robot", default="Panda")
    p.add_argument("--controller", default="OSC_POSE")
    p.add_argument("--obs-keys", nargs="+", default=DEFAULT_OBS_KEYS)
    p.add_argument("--T-inf", type=int, default=None)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default="./outputs/eval_nutassembly")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--action-clip", type=float, default=None)
    args = p.parse_args()

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"]) if "obs_norm" in ckpt else None
    act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"]) if "act_norm" in ckpt else None

    if args.algo == "bc":
        model = BCMLP(obs_dim, act_dim).to(device)
    elif args.algo == "bcrnn":
        model = BCRNN(obs_dim, act_dim).to(device)
    else:
        model = BCDiffusion(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=int(ckpt.get("T", 100)),
            beta_min=float(ckpt.get("beta_min", 1e-4)),
            beta_max=float(ckpt.get("beta_max", 2e-2)),
            hidden=int(ckpt.get("hidden", 256)),
            depth=int(ckpt.get("depth", 4)),
            scheduler=str(ckpt.get("scheduler", "linear")),
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    use_cam = bool(args.save_gif)
    env = make_env(args.env_id, args.robot, args.controller, args.max_steps, use_cam)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_succ = 0
    ep_returns = []
    ep_lengths = []

    with torch.no_grad():
        for ep_idx in range(args.episodes):
            obs = env.reset()
            done = False
            ret = 0.0
            t = 0
            h = None
            frames = []

            while not done and t < args.max_steps:
                o = flatten_obs(obs, args.obs_keys)
                if obs_norm is not None:
                    o = obs_norm.transform_single(o)

                if o.shape[0] != obs_dim:
                    raise RuntimeError(f"obs dim mismatch: got {o.shape[0]} expected {obs_dim}")

                ot = torch.from_numpy(o).to(device).unsqueeze(0)

                if args.algo == "bc":
                    at = model(ot).squeeze(0)
                elif args.algo == "bcrnn":
                    at, h = model(ot.unsqueeze(1), h0=h)
                    at = at.squeeze(0).squeeze(0)
                else:
                    if args.sampler == "ddim":
                        at = model.ddim_sample(ot, T_inf=args.T_inf or 20, eta=args.eta).squeeze(0)
                    else:
                        at = model.ddpm_sample(ot, T_inf=args.T_inf).squeeze(0)

                action_norm = at.detach().cpu().numpy().astype(np.float32)
                if act_norm is not None:
                    action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
                else:
                    action = action_norm

                if args.action_clip is not None:
                    action = np.clip(action, -args.action_clip, args.action_clip)

                obs, reward, done, info = env.step(action.astype(np.float32))
                ret += float(reward)
                t += 1

                if args.save_gif and ep_idx < args.gif_episodes:
                    frame = env.sim.render(width=512, height=512, camera_name="agentview")
                    if isinstance(frame, np.ndarray):
                        frames.append(np.flipud(frame))

                if bool(info.get("success", False)):
                    n_succ += 1
                    done = True

            ep_returns.append(ret)
            ep_lengths.append(t)

            if frames:
                imageio.mimsave(outdir / f"rollout_ep{ep_idx:03d}.gif", frames, duration=0.04)

    metrics = {
        "success_rate": float(n_succ / args.episodes),
        "avg_return": float(np.mean(ep_returns)),
        "avg_ep_len": float(np.mean(ep_lengths)),
        "episodes": int(args.episodes),
        "env_id": args.env_id,
        "robot": args.robot,
    }

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(ep_returns, bins=20)
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title(f"{args.algo.upper()} Return Distribution ({args.env_id})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "returns_hist.png", dpi=150)
    plt.close()

    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"avg_return={metrics['avg_return']:.4f}")
    print(f"avg_ep_len={metrics['avg_ep_len']:.2f}")
    print(f"saved eval artifacts: {outdir}")


if __name__ == "__main__":
    main()
