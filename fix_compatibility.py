"""
修复 robosuite 版本兼容性问题
"""
import numpy as np
import torch
import json
import h5py
from pathlib import Path

print("=" * 80)
print("修复 robosuite 兼容性问题")
print("=" * 80)

# 1. 重新转换数据，使用 BASIC 控制器
print("\n步骤 1: 重新转换数据，使用 BASIC 控制器")
print("-" * 60)

import subprocess
import sys

cmd = [
    sys.executable, "src/convert_robosuite_nutassembly.py",
    "--input-h5", "./data/raw/robosuite/nutassemblyround/1777644504_8477166/NutAssemblyRound_demo_act_norm.hdf5",
    "--output-npz", "./data/processed/nutassemblyround_basic_state.npz",
    "--obs-keys", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", 
                  "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "object-state",
    "--override-controller", "BASIC",
    "--skip-xml-replay"  # 跳过 XML 重放，避免兼容性问题
]

print(f"运行命令: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("✅ 数据转换成功!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("❌ 数据转换失败:")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    print("\n尝试不使用 skip-xml-replay...")
    
    # 尝试不跳过 XML 重放
    cmd_no_skip = cmd[:-1]  # 移除 --skip-xml-replay
    try:
        result = subprocess.run(cmd_no_skip, capture_output=True, text=True, check=True)
        print("✅ 数据转换成功!")
        print(result.stdout)
    except subprocess.CalledProcessError as e2:
        print("❌ 仍然失败，使用现有数据")
        print("STDERR:", e2.stderr)

# 2. 检查新数据
print("\n步骤 2: 检查转换后的数据")
print("-" * 60)

npz_path = "./data/processed/nutassemblyround_basic_state.npz"
if Path(npz_path).exists():
    data = np.load(npz_path)
    print(f"✅ 新数据文件存在")
    print(f"Obs shape: {data['obs'].shape}")
    print(f"Acts shape: {data['acts'].shape}")
    
    # 3. 重新训练模型（如果数据维度不同）
    old_data = np.load("./data/processed/nutassemblyround_state.npz")
    if data['obs'].shape[1] != old_data['obs'].shape[1]:
        print(f"\n⚠️  观测维度变化: {old_data['obs'].shape[1]} → {data['obs'].shape[1]}")
        print("需要重新训练模型!")
        
        print("\n步骤 3: 重新训练模型")
        print("-" * 60)
        
        train_cmd = [
            sys.executable, "src/train_bcdiffusion.py",
            "--dataset", npz_path,
            "--outdir", "./outputs/nutassemblyround_basic_diffusion_v1",
            "--epochs", "300",
            "--batch-size", "512",
            "--lr", "1e-4",
            "--hidden", "256",
            "--depth", "4",
            "--scheduler", "linear",
            "--ema-decay", "0.0",
            "--action-noise-std", "0.0"
        ]
        
        print(f"训练命令: {' '.join(train_cmd)}")
        print("请手动运行上述命令进行训练")
        
        model_path = "./outputs/nutassemblyround_basic_diffusion_v1/best.pt"
    else:
        print("✅ 观测维度相同，可以使用现有模型")
        model_path = "./outputs/nutassemblyround_diffusion_v1/best.pt"
    
    # 4. 生成评估命令
    print("\n步骤 4: 评估命令")
    print("-" * 60)
    
    eval_cmd = [
        "python", "src/eval_policy_robosuite.py",
        "--algo", "diffusion",
        "--ckpt", model_path,
        "--env-id", "NutAssemblyRound",
        "--robot", "Panda", 
        "--controller", "BASIC",
        "--sampler", "ddim",
        "--T-inf", "20",
        "--eta", "0.0",
        "--episodes", "10",  # 先测试少量 episodes
        "--max-steps", "400",
        "--action-clip", "0.5",
        "--output-dir", "./outputs/eval_basic_test"
    ]
    
    print("评估命令:")
    print(" ".join(eval_cmd))
    
else:
    print("❌ 数据转换失败，使用现有数据进行诊断")
    
    # 5. 诊断现有模型和数据的兼容性
    print("\n步骤 5: 诊断现有数据")
    print("-" * 60)
    
    # 检查模型和数据维度
    ckpt = torch.load("./outputs/nutassemblyround_diffusion_v1/best.pt", map_location='cpu')
    data = np.load("./data/processed/nutassemblyround_state.npz")
    
    print(f"模型观测维度: {ckpt['obs_dim']}")
    print(f"数据观测维度: {data['obs'].shape[1]}")
    print(f"模型动作维度: {ckpt['act_dim']}")
    print(f"数据动作维度: {data['acts'].shape[1]}")
    
    if ckpt['obs_dim'] != data['obs'].shape[1]:
        print("❌ 观测维度不匹配!")
        print("可能的解决方案:")
        print("1. 重新转换数据（推荐）")
        print("2. 重新训练模型")
        print("3. 检查 obs_keys 配置")

print("\n" + "=" * 80)
print("修复完成!")
print("=" * 80)