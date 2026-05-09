"""评估基线模型：100 episodes"""
import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from common import RunningNormalizer
from models import BCDiffusion

device = 'cpu'
ckpt = torch.load('outputs/diffusion_v1_harmonic_fix_e200/best.pt', map_location='cpu')
obs_norm = RunningNormalizer.from_state_dict(ckpt['obs_norm'])
act_norm = RunningNormalizer.from_state_dict(ckpt['act_norm'])

model = BCDiffusion(obs_dim=int(ckpt['obs_dim']), act_dim=int(ckpt['act_dim']),
    T=int(ckpt['T']), hidden=int(ckpt['hidden']), depth=int(ckpt['depth']),
    scheduler=str(ckpt['scheduler']), obs_backbone=str(ckpt['obs_backbone']),
    rot_pair_dim=int(ckpt.get('rot_pair_dim', -1)),
    harmonic_order=int(ckpt.get('harmonic_order', 4))).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Model loaded: obs_dim={ckpt["obs_dim"]}, hidden={ckpt["hidden"]}, depth={ckpt["depth"]}, backbone={ckpt["obs_backbone"]}')

env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=400)

for T_inf in [20, 50]:
    n_succ, returns, lengths = 0, [], []
    for ep in range(100):
        obs, _ = env.reset()
        ret = 0
        for t in range(400):
            o = obs[0].cpu().numpy()
            if obs_norm:
                o = obs_norm.transform_single(o)
            ot = torch.from_numpy(o).to(device).unsqueeze(0)
            with torch.no_grad():
                at = model.ddim_sample(ot, T_inf=T_inf).squeeze(0)
            if torch.isnan(at).any():
                break
            act = at.cpu().numpy() * np.maximum(act_norm.std, 1e-8) + act_norm.mean
            obs, r, done, trunc, info = env.step(act)
            ret += float(r)
            if done or trunc:
                if info.get('success', False):
                    n_succ += 1
                break
        returns.append(ret)
        lengths.append(t+1)
    print(f'T_inf={T_inf}: SR={n_succ}/100={n_succ}%, ret={np.mean(returns):.2f}, len={np.mean(lengths):.1f}')
    with open(f'baseline_{T_inf}steps.txt', 'w') as f:
        f.write(f'success_rate={n_succ/100:.4f}\navg_return={np.mean(returns):.4f}\navg_ep_len={np.mean(lengths):.2f}')

env.close()
print('Done')
