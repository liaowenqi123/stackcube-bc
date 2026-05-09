"""
放宽成功条件的评估 - 看看模型在更宽松条件下的表现
"""
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

# 放宽的成功条件
def check_relaxed_success(env, xy_threshold=0.05, z_threshold=0.10):
    nut_name = env.nuts[0].name
    nut_body_id = env.obj_body_id[nut_name]
    peg_body_id = env.peg1_body_id
    
    nut_pos = env.sim.data.body_xpos[nut_body_id]
    peg_pos = env.sim.data.body_xpos[peg_body_id]
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id['right']]
    
    xy_dist = np.sqrt((nut_pos[0] - peg_pos[0])**2 + (nut_pos[1] - peg_pos[1])**2)
    z_pos = nut_pos[2]
    eef_dist = np.linalg.norm(eef_pos - nut_pos)
    r_reach = 1 - np.tanh(10.0 * eef_dist)
    
    xy_ok = xy_dist < xy_threshold
    z_ok = z_pos < env.table_offset[2] + z_threshold
    reach_ok = r_reach < 0.6
    
    return xy_ok and z_ok and reach_ok, xy_dist, z_pos, r_reach

# 测试不同的放宽程度
thresholds = [
    (0.03, 0.05, "官方标准"),
    (0.05, 0.10, "放宽 1 级"),
    (0.10, 0.15, "放宽 2 级"),
    (0.15, 0.20, "放宽 3 级"),
]

print("测试不同成功条件下的成功率...\n")

for xy_thresh, z_thresh, name in thresholds:
    successes = 0
    total_episodes = 10  # 减少到 10 个
    
    # 重新创建环境避免内存泄漏
    if 'env' in locals():
        env.close()
    env = suite.make(
        'NutAssemblySquare',
        robots='Panda',
        controller_configs=cc,
        has_renderer=False,
        reward_shaping=True,
        horizon=400,
    )
    
    for ep in range(total_episodes):
        obs_dict = env.reset()
        
        with torch.no_grad():
            for step in range(400):
                obs_flat = flatten_obs(obs_dict, obs_keys)
                obs_norm_val = obs_norm.transform_single(obs_flat)
                obs_tensor = torch.from_numpy(obs_norm_val).unsqueeze(0)
                
                action_norm = model.ddim_sample(obs_tensor, T_inf=20, eta=0.0).squeeze(0).numpy()
                action = action_norm * np.maximum(act_norm.std, 1e-8) + act_norm.mean
                
                obs_dict, reward, done, info = env.step(action.astype(np.float32))
                
                success, xy_dist, z_pos, r_reach = check_relaxed_success(env, xy_thresh, z_thresh)
                
                if success:
                    successes += 1
                    break
                
                if done:
                    break
    
    success_rate = successes / total_episodes
    print(f"{name} (XY<{xy_thresh}m, Z<{env.table_offset[2]+z_thresh:.2f}m): {success_rate*100:.1f}% ({successes}/{total_episodes})")

env.close()
