import sys
sys.path.insert(0, 'src')
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from common import RunningNormalizer
from models import BCDiffusion

# 加载模型
ckpt = torch.load('./outputs/nutassemblysquare_v1/best.pt', map_location='cpu')
obs_norm = RunningNormalizer.from_state_dict(ckpt['obs_norm'])
act_norm = RunningNormalizer.from_state_dict(ckpt['act_norm'])

model = BCDiffusion(
    obs_dim=ckpt['obs_dim'],
    act_dim=ckpt['act_dim'],
    T=int(ckpt.get('T', 100)),
    beta_min=float(ckpt.get('beta_min', 1e-4)),
    beta_max=float(ckpt.get('beta_max', 2e-2)),
    hidden=int(ckpt.get('hidden', 256)),
    depth=int(ckpt.get('depth', 4)),
    scheduler=str(ckpt.get('scheduler', 'linear')),
)
model.load_state_dict(ckpt['model'])
model.eval()

# 创建环境
cc = load_composite_controller_config('BASIC', 'Panda')
env = suite.make(
    'NutAssemblySquare',
    robots='Panda',
    controller_configs=cc,
    has_renderer=False,
    reward_shaping=True,
    horizon=400,
)

obs_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "object"
]

def flatten_obs(obs_dict, obs_keys):
    parts = []
    for key in obs_keys:
        arr = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0)

# 运行一个完整 episode
print("运行完整 episode...")
obs_dict = env.reset()
total_reward = 0
max_reward = 0

with torch.no_grad():
    for step in range(400):
        obs_flat = flatten_obs(obs_dict, obs_keys)
        obs_norm_val = obs_norm.transform_single(obs_flat)
        obs_tensor = torch.from_numpy(obs_norm_val).unsqueeze(0)
        
        action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).numpy()
        action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
        
        obs_dict, reward, done, info = env.step(action.astype(np.float32))
        total_reward += reward
        max_reward = max(max_reward, reward)
        
        if step % 50 == 0:
            print(f"Step {step}: reward={reward:.4f}, total={total_reward:.4f}, success={info.get('success', False)}")
        
        if info.get('success'):
            print(f"\n✅ 成功! Step {step}")
            break
        
        if done:
            print(f"\n❌ Episode 结束但未成功 (step {step})")
            break

print(f"\n最终统计:")
print(f"  总回报: {total_reward:.4f}")
print(f"  最大单步回报: {max_reward:.4f}")
print(f"  总步数: {step + 1}")
print(f"  成功: {info.get('success', False)}")

env.close()
