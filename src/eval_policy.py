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
from tqdm import tqdm
import mani_skill.envs  # noqa: F401

from common import RunningNormalizer, select_device
from models import BCMLP, BCRNN, BCDiffusion, BCDiffusionTemporal, BCDiffusionCRD, BCDiffusionKoopman


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["bc", "bcrnn", "diffusion"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--T-inf", type=int, default=None,
                    help="Diffusion inference steps (default: same as training T)")
    p.add_argument("--sampler", type=str, default="ddpm",
                   choices=["ddpm", "ddim"],
                   help="Sampling method: ddpm (standard) or ddim (faster, often better)")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity (0=deterministic, 1=DDPM-like)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-id", default="StackCube-v1")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default="./outputs/eval")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--action-clip", type=float, default=None,
                   help="Clip action to [-x, x] after denormalization (e.g. 0.5)")
    p.add_argument("--use-dko", action="store_true",
                   help="Enable DKO modulation (only for Koopman model)")
    args = p.parse_args()

    device = select_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    # obs_dim / act_dim 可能不在 checkpoint 中，从数据文件计算
    if "obs_dim" in ckpt:
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])
    elif "args" in ckpt and isinstance(ckpt["args"], dict) and "data" in ckpt["args"]:
        import numpy as np
        dp = np.load(ckpt["args"]["data"])
        obs_dim = dp["obs"].shape[1]
        act_dim = dp["acts"].shape[1]
        print(f"[Data] obs_dim={obs_dim}, act_dim={act_dim} (from {ckpt['args']['data']})")
        dp.close()
    else:
        obs_dim, act_dim = 48, 8
        print(f"[Data] using defaults: obs_dim={obs_dim}, act_dim={act_dim}")

    # ── 加载 normalizer（向后兼容旧 checkpoint） ──────────────────────────
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
        # 从 checkpoint 自动检测模型参数
        # 先尝试顶层键，再回退到 ckpt["args"]（兼容新版训练脚本）
        args_dict = ckpt.get("args", {})
        if not isinstance(args_dict, dict):
            args_dict = {}

        def _ckpt_get(key, default=None):
            if key in ckpt:
                return ckpt[key]
            if isinstance(args_dict, dict) and key in args_dict and args_dict[key] is not None:
                return args_dict[key]
            return default

        stored_scheduler_raw    = _ckpt_get("scheduler", "cosine")
        stored_depth_raw        = _ckpt_get("depth", 4)
        stored_hidden_raw       = _ckpt_get("hidden", 256)
        stored_obs_backbone_raw = _ckpt_get("obs_backbone", "mlp")
        stored_rot_pair_dim_raw = _ckpt_get("rot_pair_dim", _ckpt_get("c4_pair_dim"))
        stored_harmonic_order_raw = _ckpt_get("harmonic_order", 4)
        stored_scheduler = str(stored_scheduler_raw) if stored_scheduler_raw is not None else "linear"
        stored_depth = int(stored_depth_raw) if stored_depth_raw is not None else 4
        stored_hidden = int(stored_hidden_raw) if stored_hidden_raw is not None else 256
        stored_obs_backbone = str(stored_obs_backbone_raw) if stored_obs_backbone_raw is not None else "mlp"
        stored_rot_pair_dim = int(stored_rot_pair_dim_raw) if stored_rot_pair_dim_raw is not None else None
        stored_harmonic_order = int(stored_harmonic_order_raw) if stored_harmonic_order_raw is not None else 4
        is_temporal = bool(_ckpt_get("temporal", False)) or str(_ckpt_get("algo", "")).startswith("diffusion_temporal")
        is_koopman = bool(_ckpt_get("koopman", False)) or str(_ckpt_get("algo", "")).startswith("diffusion_koopman")
        is_consistency = str(_ckpt_get("algo", "")).startswith("diffusion_consistency")

        if is_consistency:
            model = BCDiffusionCRD(
                obs_dim=obs_dim,
                act_dim=act_dim,
                T=int(_ckpt_get("T", 100)),
                beta_min=float(_ckpt_get("beta_min", 1e-4)),
                beta_max=float(_ckpt_get("beta_max", 2e-2)),
                hidden=stored_hidden,
                depth=stored_depth,
                scheduler=stored_scheduler,
                obs_backbone=stored_obs_backbone,
                rot_pair_dim=stored_rot_pair_dim,
                harmonic_order=stored_harmonic_order,
                residual_film=bool(_ckpt_get("residual_film", True)),
                consistency_weight=float(_ckpt_get("consistency_weight", 0.1)),
                cfg_strength=float(_ckpt_get("cfg_strength", 1.0)),
            ).to(device)
        elif is_temporal:
            model = BCDiffusionTemporal(
                obs_dim=obs_dim,
                act_dim=act_dim,
                T=int(_ckpt_get("T", 100)),
                beta_min=float(_ckpt_get("beta_min", 1e-4)),
                beta_max=float(_ckpt_get("beta_max", 2e-2)),
                hidden=stored_hidden,
                depth=stored_depth,
                scheduler=stored_scheduler,
                seq_len=int(_ckpt_get("seq_len", 8)),
                tf_layers=int(_ckpt_get("tf_layers", 2)),
                tf_heads=int(_ckpt_get("tf_heads", 4)),
                tf_dropout=float(_ckpt_get("tf_dropout", 0.1)),
                router_hidden=int(_ckpt_get("router_hidden", 128)),
                obs_backbone=stored_obs_backbone,
                rot_pair_dim=stored_rot_pair_dim,
                harmonic_order=stored_harmonic_order,
            ).to(device)
        elif is_koopman:
            model = BCDiffusionKoopman(
                obs_dim=obs_dim,
                act_dim=act_dim,
                T=int(_ckpt_get("T", 100)),
                beta_min=float(_ckpt_get("beta_min", 1e-4)),
                beta_max=float(_ckpt_get("beta_max", 2e-2)),
                hidden=stored_hidden,
                depth=stored_depth,
                scheduler=stored_scheduler,
                latent_dim=int(_ckpt_get("latent_dim", 64)),
                koopman_h=int(_ckpt_get("koopman_h", 4)),
                koopman_lambda=float(_ckpt_get("koopman_lambda", 0.1)),
                reg_lambda=float(_ckpt_get("koopman_reg", 1e-5)),
                obs_backbone=stored_obs_backbone,
                rot_pair_dim=stored_rot_pair_dim,
                harmonic_order=stored_harmonic_order,
            ).to(device)
        else:
            model = BCDiffusion(
                obs_dim=obs_dim,
                act_dim=act_dim,
                T=int(_ckpt_get("T", 100)),
                beta_min=float(_ckpt_get("beta_min", 1e-4)),
                beta_max=float(_ckpt_get("beta_max", 2e-2)),
                hidden=stored_hidden,
                depth=stored_depth,
                scheduler=stored_scheduler,
                obs_backbone=stored_obs_backbone,
                rot_pair_dim=stored_rot_pair_dim,
                harmonic_order=stored_harmonic_order,
            ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,   # 覆盖环境默认的 50 步上限
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
        """原始 obs → 归一化后的 numpy (D,)"""
        o = flatten_obs(raw_obs)
        if o.shape[0] < 48:
            # state_dict模式，展开后的维度可能不全
            o = flatten_state_dict(env.unwrapped.get_state_dict())
        # 自动补全增强特征
        if obs_norm is not None and o.shape[0] < obs_norm.mean.shape[0]:
            n_extra = obs_norm.mean.shape[0] - o.shape[0]
            if n_extra == 12:
                # 旧版：相对向量（cubeA-ee, cubeB-cubeA, goal-cubeB, ee_pos）
                geom = np.concatenate([
                    o[25:28] - o[18:21], o[32:35] - o[25:28],
                    o[39:42] - o[32:35], o[18:21],
                ])
            elif n_extra == 10:
                # SE(2)版：成对x-y距离 + 高度（旋转不变）
                pts = [o[18:21], o[25:28], o[32:35], o[39:42]]
                geom = np.array([np.linalg.norm(pts[i][:2] - pts[j][:2])
                                 for i in range(4) for j in range(i+1, 4)] +
                                [p[2] for p in pts])
            else:
                raise RuntimeError(f"Unknown augmentation: {n_extra} extra dims")
            o = np.concatenate([o, geom])
        if obs_norm is not None:
            o = obs_norm.transform_single(o)
        return o

    # 原始 obs 的维度（归一化前）
    raw_obs_dim = 48 if obs_norm is None else min(obs_norm.mean.shape[0], 48)

    with torch.no_grad():
        pbar = tqdm(total=args.episodes, desc='Eval', unit='ep')
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
                    # 支持 DDPM / DDIM 两种采样器
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
                    elif isinstance(model, BCDiffusionKoopman):
                        seq_len = getattr(model, "koopman_h", 4) * 2 + 1  # 至少需要 h+1 帧
                        hist = list(obs_hist)[-seq_len:]
                        seq_np = np.zeros((seq_len, obs_dim), dtype=np.float32)
                        n = min(len(hist), seq_len)
                        seq_np[-n:] = np.stack(hist[-n:], axis=0)
                        seq_t = torch.from_numpy(seq_np).to(device).unsqueeze(0)
                        # 是否使用 DKO 调制
                        use_dko = args.use_dko
                        if args.sampler == "ddim":
                            at = model.ddim_sample(
                                ot, obs_seq=seq_t, T_inf=args.T_inf or 20, eta=args.eta,
                                use_dko=use_dko
                            ).squeeze(0)
                        else:
                            at = model.ddpm_sample(
                                ot, obs_seq=seq_t, T_inf=args.T_inf,
                                use_dko=use_dko
                            ).squeeze(0)
                    else:
                        if args.sampler == "ddim":
                            at = model.ddim_sample(ot, T_inf=args.T_inf or 20, eta=args.eta).squeeze(0)
                        else:
                            at = model.ddpm_sample(ot, T_inf=args.T_inf).squeeze(0)

                # ── 反归一化 action ────────────────────────────────────────
                action_norm = at.detach().cpu().numpy().astype(np.float32)
                if act_norm is not None:
                    action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
                else:
                    action = action_norm

                # ── Action 安全裁剪 ────────────────────────────────────────
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
