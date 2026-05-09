"""
eval_bcse2.py - SE(2)等变Diffusion Policy评估脚本。

支持加载所有checkpoint并比较性能，找到真正的最佳模型。
"""
import sys; sys.path.insert(0, 'src')
import os
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
import mani_skill.envs

from common import RunningNormalizer, select_device
from models_se2 import BCDiffusionSE2


def flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, torch.Tensor):
        return obs.cpu().numpy().flatten().astype(np.float32)
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


def build_obs_dict_for_eval(obs_tensor, device):
    """构建模型需要的obs_dict，直接使用obs中的ee_pos"""
    return {
        'joint_pos': obs_tensor[:, :7],
        'joint_vel': obs_tensor[:, 8:15],
        'prev_action': obs_tensor[:, 16:18],
        'ee_pos': obs_tensor[:, 18:21],  # obs中已有ee_pos
        'cubeA_pos': obs_tensor[:, 25:28],
        'cubeB_pos': obs_tensor[:, 32:35],
        'goal_pos': obs_tensor[:, 39:42],
    }


def evaluate_checkpoint(ckpt_path, device, episodes=10, max_steps=400, save_gif=False):
    """评估单个checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    obs_dim = int(ckpt['obs_dim'])
    act_dim = int(ckpt['act_dim'])
    
    obs_norm = RunningNormalizer.from_state_dict(ckpt['obs_norm'])
    act_norm = RunningNormalizer.from_state_dict(ckpt['act_norm'])
    ee_norm = RunningNormalizer.from_state_dict(ckpt['ee_norm']) if ckpt.get('ee_norm') else None
    
    model = BCDiffusionSE2(
        obs_dim=obs_dim, act_dim=act_dim,
        T=int(ckpt.get('T', 100)),
        hidden=int(ckpt.get('hidden', 256)),
        depth=int(ckpt.get('depth', 4)),
        scheduler=str(ckpt.get('scheduler', 'cosine')),
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 创建环境
    env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos',
                   render_mode='rgb_array', max_episode_steps=max_steps)
    robot = env.unwrapped.agent.robot
    
    n_succ = 0
    ep_returns = []
    ep_lengths = []
    outdir = Path(ckpt_path).parent / f"eval_{Path(ckpt_path).stem}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for ep_idx in tqdm(range(episodes), desc=f'{Path(ckpt_path).stem}', leave=False):
            obs, _ = env.reset()
            done = False
            trunc = False
            ret = 0.0
            t = 0
            frames = []
            
            while not done and not trunc:
                # 保存原始obs用于FK计算
                raw_obs = flatten_obs(obs)
                o_flat = raw_obs.copy()
                if obs_norm:
                    o_flat = obs_norm.transform_single(o_flat)
                
                ot = torch.from_numpy(o_flat).to(device).unsqueeze(0)
                
                # 直接使用obs中的ee_pos（无需FK）
                obs_dict = build_obs_dict_for_eval(ot, device)
                
                # SDDIM采样
                at = model.ddim_sample(obs_dict, T_inf=20).squeeze(0)
                
                action_norm = at.detach().cpu().numpy().astype(np.float32)
                if act_norm:
                    action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
                else:
                    action = action_norm
                
                obs, r, done, trunc, info = env.step(action)
                ret += float(r)
                t += 1
                
                if save_gif and ep_idx < 3:
                    frame = env.render()
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu().numpy()
                    if isinstance(frame, np.ndarray):
                        if frame.ndim == 4 and frame.shape[0] == 1:
                            frame = frame[0]
                        frames.append(frame)
                
                if done and info.get('success', False):
                    n_succ += 1
            
            ep_returns.append(ret)
            ep_lengths.append(t)
            
            if frames:
                imageio.mimsave(outdir / f"ep{ep_idx:03d}.gif", frames, duration=0.04)
    
    env.close()
    
    sr = n_succ / episodes if episodes > 0 else 0
    return {
        'success_rate': float(sr),
        'avg_return': float(np.mean(ep_returns)),
        'avg_ep_len': float(np.mean(ep_lengths)),
        'episodes': episodes,
        'checkpoint': str(ckpt_path),
        'epoch': ckpt.get('epoch', -1),
        'val_loss': float(ckpt.get('val_loss', ckpt.get('best_val_loss', -1))),
    }


def evaluate_all_checkpoints(output_dir, device, episodes=10):
    """评估目录中所有checkpoint"""
    outdir = Path(output_dir)
    ckpt_files = sorted(outdir.glob('ckpt_ep*.pt'))
    
    if not ckpt_files:
        # 至少评估best.pt
        ckpt_files = [outdir / 'best.pt']
    
    results = []
    for ckpt_path in tqdm(ckpt_files, desc='Checkpoints', position=0):
        if not ckpt_path.exists():
            continue
        try:
            result = evaluate_checkpoint(ckpt_path, device, episodes=episodes)
            results.append(result)
            print(f"  SR={result['success_rate']:.2%}, "
                  f"ret={result['avg_return']:.2f}, "
                  f"len={result['avg_ep_len']:.1f}, "
                  f"epoch={result['epoch']}")
        except Exception as e:
            print(f"  FAILED: {e}")
    
    # 按success_rate排序
    results.sort(key=lambda r: r['success_rate'], reverse=True)
    
    # 保存报告
    report_path = outdir / "eval_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'total_ckpts': len(results),
            'best_checkpoint': results[0],
            'all_results': results,
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Best checkpoint: {results[0]['checkpoint']}")
    print(f"Best success_rate: {results[0]['success_rate']:.2%}")
    print(f"All checkpoints evaluated: {len(results)}")
    print(f"{'='*50}")
    
    return results


if __name__ == '__main__':
    device = select_device()
    
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--ckpt', nargs='+', default=None,
                   help='只评估指定的checkpoint文件名，如 best.pt ckpt_ep0100.pt')
    p.add_argument('--episodes', type=int, default=10)
    args = p.parse_args()
    
    if args.ckpt:
        # 只评估指定ckpt
        for cname in args.ckpt:
            ckpt_path = Path(args.ckpt_dir) / cname
            if ckpt_path.exists():
                r = evaluate_checkpoint(ckpt_path, device, episodes=args.episodes)
                print(f"{cname}: SR={r['success_rate']:.2%}, ret={r['avg_return']:.2f}, len={r['avg_ep_len']:.1f}")
            else:
                print(f"Not found: {ckpt_path}")
    else:
        evaluate_all_checkpoints(args.ckpt_dir, device, episodes=args.episodes)
