"""验证Franka FK的计算精度"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from franka_fk import forward_kinematics, fk_from_torch_joints

# 创建多个随机关节角进行验证
env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=1)

print("=== FK精度验证 ===")
print(f"{'Test':>5} | {'FK_x':>8} {'FK_y':>8} {'FK_z':>8} | {'Sim_x':>8} {'Sim_y':>8} {'Sim_z':>8} | {'Error':>8}")

errors = []
for test_idx in range(10):
    obs, _ = env.reset()
    
    # 从观测中提取关节角
    jp = obs[0, :7].cpu().numpy()  # joint positions
    
    # 计算FK位置
    fk_pos, _ = forward_kinematics(jp)
    
    # 从仿真中获取末端位置
    hand_link = [l for l in env.unwrapped.agent.robot.get_links() if 'tcp' in l.get_name().lower()]
    if hand_link:
        sim_pos = hand_link[0].pose.p
    else:
        # 尝试hand link
        hand_link = [l for l in env.unwrapped.agent.robot.get_links() if 'hand' in l.get_name().lower()]
        sim_pos = hand_link[0].pose.p if hand_link else [0, 0, 0]
    
    sim_pos_np = np.array([float(sim_pos[0]), float(sim_pos[1]), float(sim_pos[2])])
    err = np.linalg.norm(fk_pos - sim_pos_np)
    errors.append(err)
    
    print(f"{test_idx:>5} | {fk_pos[0]:>8.4f} {fk_pos[1]:>8.4f} {fk_pos[2]:>8.4f} | {sim_pos_np[0]:>8.4f} {sim_pos_np[1]:>8.4f} {sim_pos_np[2]:>8.4f} | {err:>8.6f}")

print(f"\n平均误差: {np.mean(errors):.6f}")
print(f"最大误差: {np.max(errors):.6f}")
env.close()
