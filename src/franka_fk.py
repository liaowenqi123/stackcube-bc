"""
Franka Emika Panda 正运动学（FK）计算器。
输入7个关节角 + 可选的夹爪状态，输出末端执行器位置(x,y,z)和姿态。

基于DH参数表：
| joint | a (m) | alpha (rad) | d (m) | theta (offset) |
|-------|-------|-------------|-------|-------|
| 1     | 0     | 0           | 0.333 | q1     |
| 2     | 0     | -pi/2       | 0     | q2     |
| 3     | 0     | pi/2        | 0.316 | q3     |
| 4     | 0.0825 | pi/2       | 0     | q4     |
| 5     | -0.0825 | -pi/2     | 0.384 | q5     |
| 6     | 0     | pi/2        | 0     | q6     |
| 7     | 0.088 | pi/2        | 0     | q7     |
| tcp   | 0     | 0           | 0.107 | 0      |
"""

import numpy as np
import torch
from torch import nn


# DH参数
PANDA_DH = [
    # a(m), alpha(rad), d(m), theta_offset(rad)
    (0,      0,         0.333, 0),    # joint 1
    (0,      -np.pi/2,  0,     0),    # joint 2
    (0,      np.pi/2,   0.316, 0),    # joint 3
    (0.0825, np.pi/2,  0,     0),    # joint 4
    (-0.0825, -np.pi/2, 0.384, 0),   # joint 5
    (0,      np.pi/2,  0,     0),    # joint 6
    (0.088,  np.pi/2,  0,     0),    # joint 7
    (0,      0,         0.107, 0),    # TCP offset
]


def dh_transform(a, alpha, d, theta):
    """标准DH变换矩阵（4x4）"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [0,    sa,      ca,     d  ],
        [0,    0,       0,      1  ]
    ])
    return T


def dh_transform_torch(a, alpha, d, theta):
    """Torch版本的标准DH变换矩阵"""
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.tensor(np.cos(alpha), device=theta.device, dtype=theta.dtype)
    sa = torch.tensor(np.sin(alpha), device=theta.device, dtype=theta.dtype)
    
    # 批量构建4x4变换矩阵
    B = theta.shape[0] if theta.dim() > 0 else 1
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
        B = 1
    
    T = torch.zeros(B, 4, 4, device=theta.device, dtype=theta.dtype)
    T[:, 0, 0] = ct; T[:, 0, 1] = -st*ca; T[:, 0, 2] = st*sa; T[:, 0, 3] = a*ct
    T[:, 1, 0] = st; T[:, 1, 1] = ct*ca;  T[:, 1, 2] = -ct*sa; T[:, 1, 3] = a*st
    T[:, 2, 1] = sa; T[:, 2, 2] = ca;     T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    
    return T


def forward_kinematics(joint_angles, return_all_joints=False):
    """
    计算Franka Panda的正运动学。
    
    Args:
        joint_angles: (7,) numpy array, 7个关节角 [q1,...,q7]
        return_all_joints: 是否返回所有关节的变换矩阵
        
    Returns:
        ee_pos: (3,) 末端在基坐标系下的位置
        ee_rot: (3,3) 末端姿态旋转矩阵
        (optional) all_poses: 如果return_all_joints=True
    """
    T_accum = np.eye(4)
    all_poses = []
    
    for i, (a, alpha, d, offset) in enumerate(PANDA_DH):
        theta = joint_angles[i] + offset if i < 7 else offset
        T = dh_transform(a, alpha, d, theta)
        T_accum = T_accum @ T
        all_poses.append(T_accum.copy())
    
    ee_pos = T_accum[:3, 3]
    ee_rot = T_accum[:3, :3]
    
    if return_all_joints:
        return ee_pos, ee_rot, all_poses
    return ee_pos, ee_rot


def fk_from_torch_joints(joint_angles):
    """
    Torch版本的批量FK。
    
    Args:
        joint_angles: (B, 7) torch tensor
        
    Returns:
        ee_pos: (B, 3) 末端位置
        ee_rot: (B, 3, 3) 末端旋转矩阵
    """
    B = joint_angles.shape[0]
    device = joint_angles.device
    dtype = joint_angles.dtype
    
    T_accum = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
    
    for i, (a, alpha, d, offset) in enumerate(PANDA_DH):
        if i < 7:
            theta = joint_angles[:, i] + offset
        else:
            theta = torch.zeros(B, device=device, dtype=dtype) + offset
        
        T = dh_transform_torch(a, alpha, d, theta)
        T_accum = T_accum @ T
    
    ee_pos = T_accum[:, :3, 3]  # (B, 3)
    ee_rot = T_accum[:, :3, :3]  # (B, 3, 3)
    
    return ee_pos, ee_rot


class FKModule(nn.Module):
    """
    可微分的FK模块，可以直接在训练中使用。
    输入： (B, 7) 关节角
    输出： (B, 3) 末端位置
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_angles):
        """
        Args:
            joint_angles: (B, 7) 7个关节角
        Returns:
            ee_pos: (B, 3) 末端位置(x, y, z)
        """
        return fk_from_torch_joints(joint_angles)[0]


# 测试
if __name__ == '__main__':
    # 测试单个
    import numpy as np
    test_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    pos, rot = forward_kinematics(test_q)
    print(f"Test FK - position: {pos}")
    
    # 测试批量
    fk_mod = FKModule()
    batch_q = torch.from_numpy(test_q).float().unsqueeze(0)
    pos_batch = fk_mod(batch_q)
    print(f"Batch FK - position: {pos_batch}")
