"""
简化诊断脚本 - 不依赖 robosuite
"""
import numpy as np
import torch
import h5py

print("=" * 80)
print("问题诊断：NutAssembly 0% 成功率")
print("=" * 80)

print("\n1. 检查原始数据文件")
print("-" * 80)

h5_path = './data/raw/robosuite/nutassemblyround/1777644504_8477166/NutAssemblyRound_demo_act_norm.hdf5'
with h5py.File(h5_path, 'r') as f:
    import json
    env_info = json.loads(f['data'].attrs['env_info'])
    print(f"原始数据环境: {env_info['env_name']}")
    print(f"机器人: {env_info['robots']}")
    print(f"控制器: {env_info['controller_configs']['type']}")
    
    demo_1 = f['data']['demo_1']
    print(f"\nDemo_1 包含的 keys: {list(demo_1.keys())}")
    print(f"States shape: {demo_1['states'].shape}")
    print(f"Actions shape: {demo_1['actions'].shape}")

print("\n2. 检查转换后的数据集")
print("-" * 80)

npz_path = './data/processed/nutassemblysingle_state.npz'
data = np.load(npz_path)
print(f"NPZ 文件: {npz_path}")
print(f"Obs shape: {data['obs'].shape}")
print(f"Acts shape: {data['acts'].shape}")
print(f"Obs 统计: min={data['obs'].min():.4f}, max={data['obs'].max():.4f}, mean={data['obs'].mean():.4f}")
print(f"Acts 统计: min={data['acts'].min():.4f}, max={data['acts'].max():.4f}, mean={data['acts'].mean():.4f}")

print("\n3. 检查训练好的模型")
print("-" * 80)

ckpt_path = './outputs/nutassemblysingle_diffusion_v1/best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"模型文件: {ckpt_path}")
print(f"Obs dim: {ckpt['obs_dim']}")
print(f"Act dim: {ckpt['act_dim']}")
print(f"算法: {ckpt.get('algo', 'N/A')}")
print(f"Scheduler: {ckpt.get('scheduler', 'N/A')}")

print("\n4. 检查配置文件")
print("-" * 80)

import json
with open('./tasks/nutassembly/config_v1.json', 'r', encoding='utf-8-sig') as f:
    config = json.load(f)

print(f"配置的任务: {config['task_name']}")
print(f"配置的机器人: {config['robot']}")
print(f"配置的控制器: {config['controller']}")
print(f"配置的 obs_keys: {config['obs_keys']}")

print("\n5. 关键问题分析")
print("=" * 80)

issues = []

# 问题1: 环境名称不匹配
if env_info['env_name'] != config['task_name']:
    issues.append({
        'level': '🔴 严重',
        'problem': '环境名称不匹配',
        'detail': f"数据来自 {env_info['env_name']}, 但评估使用 {config['task_name']}",
        'solution': f"评估时应使用 --env-id {env_info['env_name']}"
    })

# 问题2: 控制器不匹配
if env_info['controller_configs']['type'] != config['controller']:
    issues.append({
        'level': '🔴 严重',
        'problem': '控制器不匹配',
        'detail': f"数据使用 {env_info['controller_configs']['type']}, 但评估使用 {config['controller']}",
        'solution': f"评估时应使用 --controller {env_info['controller_configs']['type']}"
    })

# 问题3: 动作维度
if data['acts'].shape[1] != ckpt['act_dim']:
    issues.append({
        'level': '🔴 严重',
        'problem': '动作维度不匹配',
        'detail': f"数据动作维度={data['acts'].shape[1]}, 模型动作维度={ckpt['act_dim']}",
        'solution': '重新训练模型或检查数据转换'
    })

# 问题4: 观测维度
if data['obs'].shape[1] != ckpt['obs_dim']:
    issues.append({
        'level': '🔴 严重',
        'problem': '观测维度不匹配',
        'detail': f"数据观测维度={data['obs'].shape[1]}, 模型观测维度={ckpt['obs_dim']}",
        'solution': '重新训练模型或检查数据转换'
    })

# 问题5: 数据量
if data['obs'].shape[0] < 10000:
    issues.append({
        'level': '🟡 警告',
        'problem': '数据量可能不足',
        'detail': f"只有 {data['obs'].shape[0]} 个样本",
        'solution': '考虑收集更多演示数据'
    })

if not issues:
    print("✅ 未发现配置问题")
else:
    print(f"发现 {len(issues)} 个问题:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['level']} {issue['problem']}")
        print(f"   详情: {issue['detail']}")
        print(f"   解决: {issue['solution']}")
        print()

print("\n6. 建议的修复步骤")
print("=" * 80)

if any(issue['level'] == '🔴 严重' for issue in issues):
    print("发现严重问题，需要立即修复：\n")
    
    if env_info['env_name'] != config['task_name']:
        print("步骤 1: 修改评估脚本，使用正确的环境名称")
        print(f"  在 run_eval_v1.ps1 中，将 --env-id 改为: {env_info['env_name']}")
        print()
    
    if env_info['controller_configs']['type'] != config['controller']:
        print("步骤 2: 修改评估脚本，使用正确的控制器")
        print(f"  在 run_eval_v1.ps1 中，将 --controller 改为: {env_info['controller_configs']['type']}")
        print()
    
    print("步骤 3: 重新运行评估")
    print("  powershell -ExecutionPolicy Bypass -File .\\tasks\\nutassembly\\run_eval_v1.ps1")
else:
    print("配置看起来正常，可能需要：")
    print("  1. 检查模型训练是否收敛（查看 loss_curve.png）")
    print("  2. 尝试更长的训练时间或更多数据")
    print("  3. 调整超参数（学习率、模型大小等）")

print("\n" + "=" * 80)
