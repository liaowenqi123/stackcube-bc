"""预处理数据：计算每一步的末端执行器位置"""
import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

def compute_ee_poses(obs_data, env_id='StackCube-v1'):
    """根据观测中的qpos设置仿真，计算TCP位置"""
    env = gym.make(env_id, obs_mode='state', control_mode='pd_joint_delta_pos', max_episode_steps=10)
    env.reset()
    robot = env.unwrapped.agent.robot
    tcp = [l for l in robot.get_links() if l.get_name() == 'panda_hand_tcp'][0]
    
    N = obs_data.shape[0]
    ee_pos = np.zeros((N, 3), dtype=np.float32)
    
    # qpos在obs的0-7维（7关节+1合并的夹爪）
    qpos_obs = obs_data[:, 0:8]
    
    for i in range(N):
        # 构造9维qpos: 7关节 + 2个夹爪（夹爪值复制）
        qpos_9d = np.zeros(9, dtype=np.float32)
        qpos_9d[:7] = qpos_obs[i, :7]
        qpos_9d[7] = qpos_obs[i, 7]  # 左手指
        qpos_9d[8] = qpos_obs[i, 7]  # 右手指（镜像）
        robot.set_qpos(torch.from_numpy(qpos_9d))
        p = tcp.pose.p
        ee_pos[i] = p[0].numpy()
        if i % 10000 == 0:
            print(f"  [{i}/{N}]")
    
    env.close()
    return ee_pos

# 加载数据
data = np.load('./data/processed/stackcube_rl_state.npz')
obs = data["obs"]
acts = data["acts"]

print(f"Computing EE positions for {len(obs)} timesteps...")
ee_pos = compute_ee_poses(obs)

# 保存
np.savez('./data/processed/stackcube_geom.npz',
         obs=obs, acts=acts, ee_pos=ee_pos,
         starts=data.get('ep_starts'), lengths=data.get('ep_lengths'))
print(f"Done! ee_pos shape={ee_pos.shape}")
print(f"ee_pos stats: min={ee_pos.min(0)}, max={ee_pos.max(0)}")
