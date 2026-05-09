"""
预处理数据：为每一步计算末端执行器位置和方块位置。
输出包含(obs, ee_pos, cubeA_pos, cubeB_pos, target_pos)的新数据集。
"""
import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from common import load_npz_dataset

def extract_geometry(npz_path, output_path):
    """从npz数据中提取几何特征"""
    data = load_npz_dataset(npz_path)
    
    # 加载环境
    env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=100)
    robot = env.unwrapped.agent.robot
    tcp_link = [l for l in robot.get_links() if l.get_name() == 'panda_hand_tcp'][0]
    
    obs = data["obs"]  # (N, 48)
    N = obs.shape[0]
    
    ee_pos_list = []
    cubeA_list = []
    cubeB_list = []  
    goal_list = []
    
    # qpos = obs[:, 0:8]
    # cube info is somewhere in obs[:, 18:48]
    # Let's check by running the env to get the layout
    
    # Actually, let's use the env to get positions for each timestep
    env.reset()
    
    # 通过设置qpos来批处理计算
    for i in range(N):
        qpos = obs[i, 0:8]
        
        # Set joint positions in simulation
        robot.set_qpos(torch.from_numpy(qpos))
        
        # Read TCP position
        ee_pos = tcp_link.pose.p[0].numpy()  # (3,)
        ee_pos_list.append(ee_pos)
        
        if i % 5000 == 0:
            print(f"  Processed {i}/{N} timesteps")
    
    env.close()
    
    ee_pos = np.stack(ee_pos_list, axis=0)  # (N, 3)
    
    # 方体位置 - 从obs中提取
    # 在ManiSkill3中，第0个方体在前18-24维左右
    # 第1个方体和目标在后面的维度
    # 实际布局需要查看env code
    
    print(f"ee_pos shape: {ee_pos.shape}")
    print(f"ee_pos range: [{ee_pos.min():.3f}, {ee_pos.max():.3f}]")
    
    # 保存增强后的数据
    np.savez(output_path, 
             obs=obs, 
             acts=data["acts"], 
             ee_pos=ee_pos,
             ep_starts=data.get("ep_starts"),
             ep_lengths=data.get("ep_lengths"))
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    extract_geometry(
        './data/processed/stackcube_rl_state.npz',
        './data/processed/stackcube_geom.npz'
    )
