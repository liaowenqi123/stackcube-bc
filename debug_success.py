import sys
sys.path.insert(0, 'src')
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from common import RunningNormalizer
from models import BCDiffusion

# 加载模型
ckpt = torch.load('./outputs/nutassemblysquare_v3/best.pt', map_location='cpu')  # V3
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
    "object-state"
]

def flatten_obs(obs_dict, obs_keys):
    parts = []
    for key in obs_keys:
        arr = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0)

# 运行一个 episode
print("运行 episode 并检查成功条件...")
obs_dict = env.reset()
total_reward = 0

# 获取螺母和 peg 的 ID
nut_name = env.nuts[0].name
nut_body_id = env.obj_body_id[nut_name]
peg_body_id = env.peg1_body_id
table_offset = env.table_offset

print(f"螺母: {nut_name}")
print(f"桌面高度: {table_offset[2]:.4f}")
print(f"成功条件: 螺母 z < {table_offset[2] + 0.05:.4f}, xy 距离 < 0.03, 机械臂远离\n")

closest_to_success = {
    'step': 0,
    'xy_dist': 999,
    'z_pos': 999,
    'r_reach': 999,
    'reward': 0
}

with torch.no_grad():
    for step in range(400):
        obs_flat = flatten_obs(obs_dict, obs_keys)
        obs_norm_val = obs_norm.transform_single(obs_flat)
        obs_tensor = torch.from_numpy(obs_norm_val).unsqueeze(0)
        
        action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).numpy()
        action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
        
        obs_dict, reward, done, info = env.step(action.astype(np.float32))
        total_reward += reward
        
        # 检查成功条件
        nut_pos = env.sim.data.body_xpos[nut_body_id]
        peg_pos = env.sim.data.body_xpos[peg_body_id]
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id['right']]
        
        xy_dist = np.sqrt((nut_pos[0] - peg_pos[0])**2 + (nut_pos[1] - peg_pos[1])**2)
        z_pos = nut_pos[2]
        eef_dist = np.linalg.norm(eef_pos - nut_pos)
        r_reach = 1 - np.tanh(10.0 * eef_dist)
        
        # 记录最接近成功的时刻
        if xy_dist < closest_to_success['xy_dist']:
            closest_to_success = {
                'step': step,
                'xy_dist': xy_dist,
                'z_pos': z_pos,
                'r_reach': r_reach,
                'reward': reward
            }
        
        if step % 100 == 0:
            print(f"Step {step}:")
            print(f"  螺母位置: ({nut_pos[0]:.4f}, {nut_pos[1]:.4f}, {nut_pos[2]:.4f})")
            print(f"  Peg 位置: ({peg_pos[0]:.4f}, {peg_pos[1]:.4f}, {peg_pos[2]:.4f})")
            print(f"  XY 距离: {xy_dist:.4f} (需要 < 0.03)")
            print(f"  Z 位置: {z_pos:.4f} (需要 < {table_offset[2] + 0.05:.4f})")
            print(f"  机械臂距离: {eef_dist:.4f}, r_reach: {r_reach:.4f} (需要 < 0.6)")
            print(f"  奖励: {reward:.4f}, 总奖励: {total_reward:.4f}")
            print(f"  成功: {info.get('success', False)}\n")
        
        if info.get('success'):
            print(f"✅ 成功! Step {step}")
            break
        
        if done:
            break

print(f"\n最接近成功的时刻 (Step {closest_to_success['step']}):")
print(f"  XY 距离: {closest_to_success['xy_dist']:.4f} (需要 < 0.03)")
print(f"  Z 位置: {closest_to_success['z_pos']:.4f} (需要 < {table_offset[2] + 0.05:.4f})")
print(f"  r_reach: {closest_to_success['r_reach']:.4f} (需要 < 0.6)")

print(f"\n最终统计:")
print(f"  总回报: {total_reward:.4f}")
print(f"  成功: {info.get('success', False)}")

# 诊断
print(f"\n诊断:")
if closest_to_success['xy_dist'] > 0.03:
    print(f"  ❌ XY 对齐不够精确 (差 {closest_to_success['xy_dist'] - 0.03:.4f}m)")
if closest_to_success['z_pos'] >= table_offset[2] + 0.05:
    print(f"  ❌ 螺母没有完全掉下去 (高了 {closest_to_success['z_pos'] - (table_offset[2] + 0.05):.4f}m)")
if closest_to_success['r_reach'] >= 0.6:
    print(f"  ❌ 机械臂没有松开螺母 (r_reach={closest_to_success['r_reach']:.4f})")

env.close()
