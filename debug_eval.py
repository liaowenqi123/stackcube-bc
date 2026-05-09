"""
调试评估 - 查看模型实际输出
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from common import RunningNormalizer
from models import BCDiffusion

print("=" * 80)
print("调试评估")
print("=" * 80)

# 加载模型
ckpt_path = "./outputs/nutassemblyround_basic_v1/best.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n模型信息:")
print(f"  obs_dim: {ckpt['obs_dim']}")
print(f"  act_dim: {ckpt['act_dim']}")
print(f"  算法: {ckpt.get('algo', 'N/A')}")

# 加载归一化器
obs_norm = RunningNormalizer.from_state_dict(ckpt['obs_norm'])
act_norm = RunningNormalizer.from_state_dict(ckpt['act_norm'])

print(f"\n动作归一化参数:")
print(f"  mean: {act_norm.mean}")
print(f"  std: {act_norm.std}")

# 创建模型
device = torch.device('cpu')
model = BCDiffusion(
    obs_dim=ckpt['obs_dim'],
    act_dim=ckpt['act_dim'],
    T=int(ckpt.get('T', 100)),
    beta_min=float(ckpt.get('beta_min', 1e-4)),
    beta_max=float(ckpt.get('beta_max', 2e-2)),
    hidden=int(ckpt.get('hidden', 256)),
    depth=int(ckpt.get('depth', 4)),
    scheduler=str(ckpt.get('scheduler', 'linear')),
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# 创建环境
controller_config = load_composite_controller_config(
    controller="BASIC",
    robot="Panda",
)
env = suite.make(
    env_name="NutAssemblyRound",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=False,
    horizon=400,
    control_freq=20,
)

print(f"\n环境信息:")
print(f"  动作空间: {env.action_spec}")
print(f"  动作维度: {env.action_dim}")

# 配置的 obs_keys
obs_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "object-state"
]

def flatten_obs(obs_dict, obs_keys):
    parts = []
    for key in obs_keys:
        if key not in obs_dict:
            raise KeyError(f"obs key '{key}' missing")
        arr = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0).astype(np.float32)

# 运行一个 episode 并记录详细信息
print("\n" + "=" * 80)
print("运行测试 episode")
print("=" * 80)

obs_dict = env.reset()
obs_flat = flatten_obs(obs_dict, obs_keys)

print(f"\n初始观测:")
print(f"  维度: {obs_flat.shape}")
print(f"  范围: [{obs_flat.min():.4f}, {obs_flat.max():.4f}]")
print(f"  均值: {obs_flat.mean():.4f}")

# 归一化
obs_normalized = obs_norm.transform_single(obs_flat)
print(f"\n归一化后观测:")
print(f"  范围: [{obs_normalized.min():.4f}, {obs_normalized.max():.4f}]")
print(f"  均值: {obs_normalized.mean():.4f}")

# 模型推理
with torch.no_grad():
    obs_tensor = torch.from_numpy(obs_normalized).to(device).unsqueeze(0)
    action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).cpu().numpy()

print(f"\n模型输出（归一化）:")
print(f"  值: {action_norm}")
print(f"  范围: [{action_norm.min():.4f}, {action_norm.max():.4f}]")
print(f"  均值: {action_norm.mean():.4f}")

# 反归一化
action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
print(f"\n反归一化后动作:")
print(f"  值: {action}")
print(f"  范围: [{action.min():.4f}, {action.max():.4f}]")
print(f"  均值: {action.mean():.4f}")

# 裁剪
action_clipped = np.clip(action, -0.5, 0.5)
print(f"\n裁剪后动作 (clip=0.5):")
print(f"  值: {action_clipped}")
print(f"  范围: [{action_clipped.min():.4f}, {action_clipped.max():.4f}]")

# 检查训练数据的动作范围
data = np.load("./data/processed/nutassemblyround_basic_state.npz")
print(f"\n训练数据动作统计:")
print(f"  范围: [{data['acts'].min():.4f}, {data['acts'].max():.4f}]")
print(f"  均值: {data['acts'].mean():.4f}")
print(f"  标准差: {data['acts'].std():.4f}")

# 执行几步看看
print("\n" + "=" * 80)
print("执行前 10 步")
print("=" * 80)

obs_dict = env.reset()
total_reward = 0.0

for step in range(10):
    obs_flat = flatten_obs(obs_dict, obs_keys)
    obs_normalized = obs_norm.transform_single(obs_flat)
    
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_normalized).to(device).unsqueeze(0)
        action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).cpu().numpy()
    
    action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
    action_clipped = np.clip(action, -0.5, 0.5)
    
    obs_dict, reward, done, info = env.step(action_clipped.astype(np.float32))
    total_reward += reward
    
    print(f"Step {step}: reward={reward:.4f}, done={done}, success={info.get('success', False)}")
    
    if done:
        break

print(f"\n前 10 步总奖励: {total_reward:.4f}")

env.close()

print("\n" + "=" * 80)
print("诊断建议")
print("=" * 80)

if np.abs(action_norm).max() > 10:
    print("❌ 模型输出异常大，可能训练有问题")
elif np.abs(action).max() < 0.01:
    print("❌ 动作太小，机器人几乎不动")
elif total_reward == 0:
    print("❌ 完全没有奖励，可能:")
    print("   1. 动作空间不匹配")
    print("   2. 观测空间不匹配")
    print("   3. 任务定义不同")
    print("   4. 模型没有学到有效策略")
else:
    print("✅ 模型输出看起来正常")
