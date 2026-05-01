"""
gen_gif.py  — 快速生成1条成功/最长 rollout 的 GIF
用法：python gen_gif.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "src")
from models import BCDiffusion
from common import RunningNormalizer, select_device
import gymnasium as gym
import mani_skill.envs  # noqa
import imageio.v2 as imageio

CKPT = "./outputs/diffusion/best.pt"   # 用 v1（已验证 86%）
OUTDIR = "./outputs"
T_INF = 100
MAX_STEPS = 100
EPISODES = 5

device = select_device()
ckpt = torch.load(CKPT, map_location=device)

model = BCDiffusion(
    obs_dim=int(ckpt["obs_dim"]), act_dim=int(ckpt["act_dim"]),
    T=int(ckpt.get("T", 100)),
    beta_min=float(ckpt.get("beta_min", 1e-4)),
    beta_max=float(ckpt.get("beta_max", 2e-2)),
    hidden=int(ckpt.get("hidden", 256)),
    depth=int(ckpt.get("depth", 4)),
    scheduler=str(ckpt.get("scheduler", "linear")),
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

obs_norm = RunningNormalizer.from_state_dict(ckpt["obs_norm"])
act_norm = RunningNormalizer.from_state_dict(ckpt["act_norm"])
raw_dim = obs_norm.mean.shape[0]
print(f"Model loaded: depth={ckpt.get('depth',4)} hidden={ckpt.get('hidden',256)}")


def flatten_obs(obs):
    parts = []
    def rec(x):
        if isinstance(x, dict):
            for k in sorted(x): rec(x[k])
        else:
            parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
    rec(obs)
    return np.concatenate(parts, axis=0)


env = gym.make(
    "StackCube-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
    max_episode_steps=MAX_STEPS,
)

n_succ = 0
best_ep = None  # (ep_id, frames, steps, success)

for ep in range(EPISODES):
    obs, _ = env.reset(seed=ep * 17 + 42)
    done = False; t = 0; frames = []
    while not done and t < MAX_STEPS:
        o = flatten_obs(obs)
        if o.shape[0] != raw_dim:
            import importlib
            common = importlib.import_module("common")
            o = common.flatten_state_dict(env.unwrapped.get_state_dict())
        o = obs_norm.transform_single(o)
        ot = torch.from_numpy(o).float().to(device).unsqueeze(0)
        with torch.no_grad():
            at = model.ddpm_sample(ot, T_inf=T_INF).squeeze(0)
        action = at.cpu().numpy() * np.maximum(act_norm.std, 1e-8) + act_norm.mean
        obs, r, done_t, trunc_t, info = env.step(action.astype("float32"))
        done = bool(done_t.any() if hasattr(done_t, "any") else done_t)

        frame = env.render()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
            frame = frame[0]
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            frames.append(frame)
        t += 1

    succ_raw = info.get("success", False)
    success = bool(succ_raw.any() if hasattr(succ_raw, "any") else succ_raw)
    if success:
        n_succ += 1
    print(f"  Ep {ep}: steps={t}  success={success}  frames={len(frames)}")

    if best_ep is None or success or (not best_ep[3] and t > best_ep[2]):
        best_ep = (ep, frames, t, success)

env.close()

# 保存最佳 GIF
ep_id, frames, steps, success = best_ep
tag = "success" if success else "attempt"
out_path = f"{OUTDIR}/diffusion_v1_{tag}_ep{ep_id}.gif"
imageio.mimsave(out_path, frames, duration=0.05)
print(f"\nSaved GIF: {out_path}  ({len(frames)} frames)")
print(f"Final SR: {n_succ}/{EPISODES} = {n_succ/EPISODES*100:.0f}%")
