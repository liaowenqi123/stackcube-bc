"""
诊断 NutAssembly 任务 0% 成功率问题
"""
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import sys

print("=" * 80)
print("1. 检查数据集")
print("=" * 80)

# 加载数据集
data = np.load('./data/processed/nutassemblysingle_state.npz')
obs = data['obs']
acts = data['acts']
print(f"数据集 obs shape: {obs.shape}")
print(f"数据集 acts shape: {acts.shape}")
print(f"Obs 统计: min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}, std={obs.std():.4f}")
print(f"Acts 统计: min={acts.min():.4f}, max={acts.max():.4f}, mean={acts.mean():.4f}, std={acts.std():.4f}")

print("\n" + "=" * 80)
print("2. 检查模型")
print("=" * 80)

# 加载模型
ckpt = torch.load('./outputs/nutassemblysingle_diffusion_v1/best.pt', map_location='cpu')
print(f"模型 obs_dim: {ckpt['obs_dim']}")
print(f"模型 act_dim: {ckpt['act_dim']}")
print(f"模型算法: {ckpt.get('algo', 'N/A')}")

# 检查归一化参数
if 'obs_norm' in ckpt:
    obs_norm_dict = ckpt['obs_norm']
    print(f"\nObs 归一化参数:")
    print(f"  mean shape: {obs_norm_dict['mean'].shape}")
    print(f"  std shape: {obs_norm_dict['std'].shape}")
    print(f"  mean 范围: [{obs_norm_dict['mean'].min():.4f}, {obs_norm_dict['mean'].max():.4f}]")
    print(f"  std 范围: [{obs_norm_dict['std'].min():.4f}, {obs_norm_dict['std'].max():.4f}]")

if 'act_norm' in ckpt:
    act_norm_dict = ckpt['act_norm']
    print(f"\nAct 归一化参数:")
    print(f"  mean shape: {act_norm_dict['mean'].shape}")
    print(f"  std shape: {act_norm_dict['std'].shape}")
    print(f"  mean 范围: [{act_norm_dict['mean'].min():.4f}, {act_norm_dict['mean'].max():.4f}]")
    print(f"  std 范围: [{act_norm_dict['std'].min():.4f}, {act_norm_dict['std'].max():.4f}]")

print("\n" + "=" * 80)
print("3. 检查环境配置")
print("=" * 80)

# 配置的 obs_keys
config_obs_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "object-state"
]

print(f"配置的 obs_keys: {config_obs_keys}")

# 创建环境
controller_config = load_composite_controller_config(
    controller="OSC_POSE",
    robot="Panda",
)
env = suite.make(
    env_name="NutAssemblySingle",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=False,
    horizon=400,
    control_freq=20,
)

# 重置环境并获取观测
obs_dict = env.reset()
print(f"\n环境观测 keys: {list(obs_dict.keys())}")

# 检查每个 obs_key 的维度
print("\n各 obs_key 的维度:")
total_dim = 0
for key in config_obs_keys:
    if key in obs_dict:
        val = np.asarray(obs_dict[key]).reshape(-1)
        print(f"  {key}: {val.shape[0]}")
        total_dim += val.shape[0]
    else:
        print(f"  {key}: ❌ 缺失!")

print(f"\n计算的总维度: {total_dim}")
print(f"模型期望维度: {ckpt['obs_dim']}")

if total_dim != ckpt['obs_dim']:
    print(f"\n⚠️  维度不匹配! 环境={total_dim}, 模型={ckpt['obs_dim']}")
else:
    print(f"\n✅ 维度匹配!")

# 检查动作空间
print(f"\n环境动作空间: {env.action_spec}")
print(f"动作维度: {env.action_dim}")
print(f"模型动作维度: {ckpt['act_dim']}")

if env.action_dim != ckpt['act_dim']:
    print(f"\n⚠️  动作维度不匹配! 环境={env.action_dim}, 模型={ckpt['act_dim']}")
else:
    print(f"\n✅ 动作维度匹配!")

print("\n" + "=" * 80)
print("4. 测试单步推理")
print("=" * 80)

# 手动拼接观测
def flatten_obs(obs_dict, obs_keys):
    parts = []
    for key in obs_keys:
        if key not in obs_dict:
            raise KeyError(f"obs key '{key}' missing")
        arr = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0).astype(np.float32)

obs_flat = flatten_obs(obs_dict, config_obs_keys)
print(f"拼接后的观测维度: {obs_flat.shape}")
print(f"观测统计: min={obs_flat.min():.4f}, max={obs_flat.max():.4f}, mean={obs_flat.mean():.4f}")

# 归一化
from common import RunningNormalizer
obs_norm = RunningNormalizer.from_state_dict(ckpt['obs_norm'])
act_norm = RunningNormalizer.from_state_dict(ckpt['act_norm'])

obs_normalized = obs_norm.transform_single(obs_flat)
print(f"归一化后观测统计: min={obs_normalized.min():.4f}, max={obs_normalized.max():.4f}, mean={obs_normalized.mean():.4f}")

# 加载模型并推理
from models import BCDiffusion
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

with torch.no_grad():
    obs_tensor = torch.from_numpy(obs_normalized).to(device).unsqueeze(0)
    action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).cpu().numpy()

print(f"模型输出（归一化）: {action_norm}")
print(f"  统计: min={action_norm.min():.4f}, max={action_norm.max():.4f}, mean={action_norm.mean():.4f}")

# 反归一化
action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
print(f"反归一化后动作: {action}")
print(f"  统计: min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")

# 裁剪
action_clipped = np.clip(action, -0.5, 0.5)
print(f"裁剪后动作: {action_clipped}")

print("\n" + "=" * 80)
print("5. 诊断总结")
print("=" * 80)

issues = []

if total_dim != ckpt['obs_dim']:
    issues.append(f"❌ 观测维度不匹配: 环境={total_dim}, 模型={ckpt['obs_dim']}")

if env.action_dim != ckpt['act_dim']:
    issues.append(f"❌ 动作维度不匹配: 环境={env.action_dim}, 模型={ckpt['act_dim']}")

# 检查是否有缺失的 obs_key
missing_keys = [key for key in config_obs_keys if key not in obs_dict]
if missing_keys:
    issues.append(f"❌ 缺失的观测 keys: {missing_keys}")

# 检查动作范围是否合理
if np.abs(action).max() > 10.0:
    issues.append(f"⚠️  动作值异常大: max={np.abs(action).max():.4f}")

if not issues:
    print("✅ 未发现明显问题，可能需要进一步调试:")
    print("  - 检查训练数据质量")
    print("  - 检查模型是否收敛")
    print("  - 尝试更长的训练时间")
    print("  - 检查环境任务难度")
else:
    print("发现以下问题:")
    for issue in issues:
        print(f"  {issue}")

env.close()
