import sys
sys.path.insert(0, "src")
print("Starting quick eval test...")

import torch
import mani_skill.envs
import gymnasium as gym
from models import BCDiffusionCRD
from common import RunningNormalizer, select_device

ckpt_path = "outputs/diffusion_consistency_v1/best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
obs_dim = int(ckpt["obs_dim"])
act_dim = int(ckpt["act_dim"])

print(f"obs_dim={obs_dim}, act_dim={act_dim}")

model = BCDiffusionCRD(
    obs_dim=obs_dim,
    act_dim=act_dim,
    T=int(ckpt.get("T", 100)),
    beta_min=float(ckpt.get("beta_min", 1e-4)),
    beta_max=float(ckpt.get("beta_max", 2e-2)),
    hidden=int(ckpt.get("hidden", 256)),
    depth=int(ckpt.get("depth", 4)),
    scheduler=str(ckpt.get("scheduler", "cosine")),
    obs_backbone=str(ckpt.get("obs_backbone", "mlp")),
    rot_pair_dim=int(ckpt["rot_pair_dim"]) if ckpt.get("rot_pair_dim") is not None else None,
    harmonic_order=int(ckpt.get("harmonic_order", 4)),
    residual_film=bool(ckpt.get("residual_film", True)),
    consistency_weight=float(ckpt.get("consistency_weight", 0.1)),
    cfg_strength=float(ckpt.get("cfg_strength", 1.0)),
)
model.load_state_dict(ckpt["model"])
model.eval()

obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"])
act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"])

device = select_device()
model = model.to(device)
print(f"Device: {device}")

env = gym.make("StackCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", max_episode_steps=400)

print("Running 3 episodes...")
n_succ = 0
for ep_idx in range(3):
    obs, _ = env.reset()
    done = False
    trunc = False
    t = 0
    while not done and not trunc:
        # flatten obs
        def flatten_obs(obs):
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

        import numpy as np
        o = flatten_obs(obs)
        if obs_norm:
            o = obs_norm.transform_single(o)
        ot = torch.from_numpy(o).to(device).unsqueeze(0)

        with torch.no_grad():
            at = model.ddim_sample(ot, T_inf=20, eta=0.0).squeeze(0)

        action_norm = at.cpu().numpy().astype(np.float32)
        if act_norm:
            action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
        else:
            action = action_norm
        action = action.astype(np.float32)

        obs, r, done, trunc, info = env.step(action)
        done = bool(done)
        trunc = bool(trunc)
        t += 1

        if done and bool(info.get("success", False)):
            n_succ += 1
            print(f"  Episode {ep_idx}: SUCCESS at step {t}")
            break
    else:
        print(f"  Episode {ep_idx}: FAILED (truncated after {t} steps)")

env.close()
print(f"\nQuick eval done: {n_succ}/3 successes")
