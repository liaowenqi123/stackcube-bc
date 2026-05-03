"""构建SE(2)等变的增强观测数据"""
import sys; sys.path.insert(0, 'src')
import numpy as np
from common import load_npz_dataset

d = load_npz_dataset('data/processed/stackcube_rl_state.npz')
x = d['obs'].astype(np.float32)  # (N, 48)

# 四点位置
ee    = x[:, 18:21]  # (N, 3)
cubeA = x[:, 25:28]  # (N, 3)
cubeB = x[:, 32:35]  # (N, 3)
goal  = x[:, 39:42]  # (N, 3)

# SE(2)成对特征：pairs之间的x-y平面距离（旋转不变） + 高度差
geom = np.column_stack([
    # 成对x-y距离 (C(4,2)=6个)
    np.linalg.norm(ee[:, :2]    - cubeA[:, :2], axis=1),  # ee-cubeA
    np.linalg.norm(ee[:, :2]    - cubeB[:, :2], axis=1),  # ee-cubeB
    np.linalg.norm(ee[:, :2]    - goal[:, :2],  axis=1),  # ee-goal
    np.linalg.norm(cubeA[:, :2] - cubeB[:, :2], axis=1),  # cubeA-cubeB
    np.linalg.norm(cubeA[:, :2] - goal[:, :2],  axis=1),  # cubeA-goal
    np.linalg.norm(cubeB[:, :2] - goal[:, :2],  axis=1),  # cubeB-goal
    # 高度（姿态z值，已经旋转不变）
    ee[:, 2],    # ee_z
    cubeA[:, 2], # cubeA_z
    cubeB[:, 2], # cubeB_z
    goal[:, 2],  # goal_z
])

aug_obs = np.concatenate([x, geom], axis=1).astype(np.float32)
print(f'aug_obs: {aug_obs.shape}')  # (N, 48+10=58)

np.savez('data/processed/stackcube_se2.npz',
         obs=aug_obs, acts=d['acts'],
         ep_starts=d.get('ep_starts'), ep_lengths=d.get('ep_lengths'))
print('Saved to data/processed/stackcube_se2.npz')
