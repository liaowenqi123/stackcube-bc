from __future__ import annotations

import os
os.environ["SAPIEN_NO_RENDER"] = "1"   # 必须无头渲染，使用 CPU 模拟

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
from models import BCMLP, BCRNN, BCDiffusion, BCDiffusionTemporal


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


# ── Confidence-Enhanced Diffusion sampling ─────────────────────────
def _single_diffusion_sample(model, ot, T_inf, sampler, eta,
                             temporal=False, obs_seq=None, obs_mask=None):
    """Sample one action (same as original code path)."""
    if temporal:
        if sampler == "ddim":
            return model.ddim_sample(ot, obs_seq=obs_seq, obs_mask=obs_mask,
                                     T_inf=T_inf or 20, eta=eta).squeeze(0)
        else:
            return model.ddpm_sample(ot, obs_seq=obs_seq, obs_mask=obs_mask,
                                     T_inf=T_inf).squeeze(0)
    else:
        if sampler == "ddim":
            return model.ddim_sample(ot, T_inf=T_inf or 20, eta=eta).squeeze(0)
        else:
            return model.ddpm_sample(ot, T_inf=T_inf).squeeze(0)


def compute_action_with_confidence(model, ot, T_inf, sampler, eta,
                                   K, var_threshold, conservative_scale,
                                   temporal=False, obs_seq=None, obs_mask=None):
    actions = []
    for _ in range(K):
        a = _single_diffusion_sample(model, ot, T_inf, sampler, eta,
                                     temporal=temporal,
                                     obs_seq=obs_seq,
                                     obs_mask=obs_mask)
        actions.append(a)
    actions = torch.stack(actions, dim=0)          # (K, act_dim)
    mean_action = actions.mean(dim=0)               # (act_dim,)
    var = actions.var(dim=0).sum().item()           # total variance

    if var > var_threshold:
        final_action = mean_action * conservative_scale
        is_conservative = True
    else:
        final_action = mean_action
        is_conservative = False

    return final_action, var, is_conservative
# ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["bc", "bcrnn", "diffusion"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--T-inf", type=int, default=None)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--eta", type=float, default=0.0)

    # ── New confidence arguments ───────────────────────────────────
    p.add_argument("--mc-k", type=int, default=1)
    p.add_argument("--var-threshold", type=float, default=0.01)
    p.add_argument("--conservative-scale", type=float, default=0.7)
    # ────────────────────────────────────────────────────────────────

    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-id", default="StackCube-v1")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default="./outputs/eval")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--action-clip", type=float, default=None)
    args = p.parse_args()

    use_confidence = (args.algo == "diffusion") and (args.mc_k > 1)

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    obs_norm: RunningNormalizer | None = None
    act_norm: RunningNormalizer | None = None
    if "obs_norm" in ckpt:
        obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"])
    if "act_norm" in ckpt:
        act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"])

    if args.algo == "bc":
        model = BCMLP(obs_dim, act_dim).to(device)
    elif args.algo == "bcrnn":
        model = BCRNN(obs_dim, act_dim).to(device)
    else:  # diffusion
        stored_scheduler_raw = ckpt.get("scheduler", "linear")
        stored_depth_raw = ckpt.get("depth", 4)
        stored_hidden_raw = ckpt.get("hidden", 256)
        stored_obs_backbone_raw = ckpt.get("obs_backbone", "mlp")
        stored_c4_pair_dim_raw = ckpt.get("c4_pair_dim", None)
        stored_scheduler = str(stored_scheduler_raw) if stored_scheduler_raw is not None else "linear"
        stored_depth = int(stored_depth_raw) if stored_depth_raw is not None else 4
        stored_hidden = int(stored_hidden_raw) if stored_hidden_raw is not None else 256
        stored_obs_backbone = str(stored_obs_backbone_raw) if stored_obs_backbone_raw is not None else "mlp"
        stored_c4_pair_dim = int(stored_c4_pair_dim_raw) if stored_c4_pair_dim_raw is not None else None
        is_temporal = bool(ckpt.get("temporal", False)) or str(ckpt.get("algo", "")).startswith("diffusion_temporal")

        # Common parameters for both model types
        common_kwargs = dict(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=int(ckpt.get("T", 100)),
            beta_min=float(ckpt.get("beta_min", 1e-4)),
            beta_max=float(ckpt.get("beta_max", 2e-2)),
            hidden=stored_hidden,
            depth=stored_depth,
            scheduler=stored_scheduler,
        )

        if is_temporal:
            model = BCDiffusionTemporal(
                **common_kwargs,
                seq_len=int(ckpt.get("seq_len", 8)),
                tf_layers=int(ckpt.get("tf_layers", 2)),
                tf_heads=int(ckpt.get("tf_heads", 4)),
                tf_dropout=float(ckpt.get("tf_dropout", 0.1)),
                router_hidden=int(ckpt.get("router_hidden", 128)),
                obs_backbone=stored_obs_backbone,
                c4_pair_dim=stored_c4_pair_dim,
            ).to(device)
        else:
            model = BCDiffusion(
                **common_kwargs,
                obs_backbone=stored_obs_backbone,
                # Note: c4_pair_dim is NOT supported by standard BCDiffusion
            ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        max_episode_steps=args.max_steps,
    )

    n_succ = 0
    ep_returns = []
    ep_lengths = []
    conservative_count = 0
    total_diffusion_steps = 0

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
            o = flatten_state_dict(env.unwrapped.get_state_dict())
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
            h = None
            obs_hist = deque(maxlen=getattr(model, "seq_len", 1))
            frames = []
            while not done and not trunc:
                o = _preprocess_obs(obs, raw_obs_dim)
                if o.shape[0] != obs_dim:
                    raise RuntimeError(f"obs dim mismatch after norm: got {o.shape[0]} expected {obs_dim}")
                ot = torch.from_numpy(o).to(device).unsqueeze(0)
                obs_hist.append(o)

                if args.algo == "bc":
                    at = model(ot).squeeze(0)
                elif args.algo == "bcrnn":
                    at, h = model(ot.unsqueeze(1), h0=h)
                    at = at.squeeze(0).squeeze(0)
                else:  # diffusion
                    if use_confidence:
                        temporal_flag = isinstance(model, BCDiffusionTemporal)
                        seq_t = None
                        mask_t = None
                        if temporal_flag:
                            seq_len = model.seq_len
                            seq_np = np.zeros((seq_len, obs_dim), dtype=np.float32)
                            mask_np = np.zeros((seq_len,), dtype=np.bool_)
                            hist = list(obs_hist)[-seq_len:]
                            n = len(hist)
                            seq_np[-n:] = np.stack(hist, axis=0)
                            mask_np[-n:] = True
                            seq_t = torch.from_numpy(seq_np).to(device).unsqueeze(0)
                            mask_t = torch.from_numpy(mask_np).to(device).unsqueeze(0)

                        at, var, is_cons = compute_action_with_confidence(
                            model, ot,
                            T_inf=args.T_inf,
                            sampler=args.sampler,
                            eta=args.eta,
                            K=args.mc_k,
                            var_threshold=args.var_threshold,
                            conservative_scale=args.conservative_scale,
                            temporal=temporal_flag,
                            obs_seq=seq_t,
                            obs_mask=mask_t,
                        )
                        total_diffusion_steps += 1
                        if is_cons:
                            conservative_count += 1
                    else:
                        if isinstance(model, BCDiffusionTemporal):
                            seq_len = model.seq_len
                            seq_np = np.zeros((seq_len, obs_dim), dtype=np.float32)
                            mask_np = np.zeros((seq_len,), dtype=np.bool_)
                            hist = list(obs_hist)[-seq_len:]
                            n = len(hist)
                            seq_np[-n:] = np.stack(hist, axis=0)
                            mask_np[-n:] = True
                            seq_t = torch.from_numpy(seq_np).to(device).unsqueeze(0)
                            mask_t = torch.from_numpy(mask_np).to(device).unsqueeze(0)
                            if args.sampler == "ddim":
                                at = model.ddim_sample(
                                    ot, obs_seq=seq_t, obs_mask=mask_t, T_inf=args.T_inf or 20, eta=args.eta
                                ).squeeze(0)
                            else:
                                at = model.ddpm_sample(ot, obs_seq=seq_t, obs_mask=mask_t, T_inf=args.T_inf).squeeze(0)
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
    if use_confidence and total_diffusion_steps > 0:
        con_ratio = conservative_count / total_diffusion_steps
        metrics["conservative_ratio"] = float(con_ratio)
        metrics["conservative_steps"] = int(conservative_count)
        print(f"Confidence stats: {conservative_count}/{total_diffusion_steps} "
              f"steps were conservative ({con_ratio*100:.1f}%)")

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(ep_returns, bins=20)
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title(f"{args.algo.upper()} Return Distribution")
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