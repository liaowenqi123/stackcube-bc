"""检查ManiSkill3 StackCube观测空间的详细布局"""
import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np

env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=400)

obs, _ = env.reset()
o = obs[0].cpu().numpy()

# 我们知道的前8维是关节位置
print("=== 观测布局分析 ===")
print(f"qpos (8): {[f'{x:.4f}' for x in o[0:8]]}")

# 进行几步，观察哪些维度变化
print("\n=== 观察随环境步数的变化 ===")
obs_prev = o.copy()
for step in range(3):
    action = np.zeros(8, dtype=np.float32)
    obs, r, done, trunc, info = env.step(action)
    o = obs[0].cpu().numpy()
    diff = o - obs_prev
    changed = np.where(np.abs(diff) > 1e-4)[0]
    print(f"Step {step+1}: changed dims = {changed.tolist()}")
    for d in changed:
        print(f"  dim[{d}]: {obs_prev[d]:.4f} -> {o[d]:.4f} (delta={diff[d]:.4f})")
    obs_prev = o.copy()

env.close()
