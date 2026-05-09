"""
StackCube PID Controller + Expert Data Collection

思路：
1. 任务分4个阶段：接近→抓取→举升→叠放
2. 每个阶段用task-space PID控制末端位置
3. 通过数值雅可比（有限差分）把末端速度映射到关节delta
4. 收集成功episode的数据 → 用于后续模仿学习
"""

import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
import imageio.v2 as imageio
import gymnasium as gym
import mani_skill.envs

# PID参数（调大！pd_joint_delta_pos需要大幅度的delta）
KP_POS = 20.0   # 位置比例增益
KP_ORI = 5.0    # 姿态比例增益
KD = 1.0        # 微分增益
MAX_VEL = 5.0   # 最大关节速度
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0
DIST_THRESH = 0.05  # 到达判定阈值（放宽）


class StackCubePID:
    """StackCube任务的PID控制器"""
    
    def __init__(self, env):
        self.env = env
        self.robot = env.unwrapped.agent.robot
        self.phase = 0
        self.phase_timer = 0
        
        # 记录前一步误差（用于微分项）
        self.prev_pos_error = None
        self.prev_ori_error = None
        
        # 雅可比计算参数
        self.delta_q = 1e-6
        self.damping = 0.1  # 阻尼最小二乘
        
        # 夹爪状态
        self.gripper_open = GRIPPER_OPEN
        self.gripper_closed = GRIPPER_CLOSE
        
        # 抓取参数
        self.grasp_offset_z = 0.0   # TCP到方块中心高度（指尖能夹住）
        self.lift_z = 0.15
        self.place_offset_z = 0.04
    
    def get_ee_pose_sim(self, joint_pos):
        """用仿真器计算末端位置（比DH参数准确）"""
        qpos_9d = np.zeros(9)
        qpos_9d[:7] = joint_pos[:7]
        qpos_9d[7] = joint_pos[7] if len(joint_pos) > 7 else 0.04
        qpos_9d[8] = qpos_9d[7]
        self.robot.set_qpos(torch.from_numpy(qpos_9d))
        tcp = [l for l in self.robot.get_links() if 'tcp' in l.get_name().lower()][0]
        return tcp.pose.p[0].numpy()
    
    def compute_jacobian(self, joint_pos):
        """数值雅可比：用仿真器算，保证方向正确"""
        n_joints = 7
        pos0 = self.get_ee_pose_sim(joint_pos)
        J = np.zeros((3, n_joints))
        
        for i in range(n_joints):
            q_plus = joint_pos.copy()
            q_plus[i] += self.delta_q
            pos_plus = self.get_ee_pose_sim(q_plus)
            J[:, i] = (pos_plus - pos0) / self.delta_q
        
        return J
    
    def damped_least_squares_ik(self, J, desired_vel):
        """阻尼最小二乘IK: q_dot = J^T (J J^T + λ²I)^(-1) x_dot"""
        JJT = J @ J.T
        JJT_reg = JJT + self.damping ** 2 * np.eye(3)
        return J.T @ np.linalg.solve(JJT_reg, desired_vel)
    
    def get_target_for_phase(self, ee_pos, cubeA_pos, cubeB_pos, goal_pos):
        """根据当前阶段返回目标末端位置"""
        if self.phase == 0:
            # Phase 0: 移动到cubeA上方（pre-grasp）
            target = cubeA_pos.copy()
            target[2] = max(cubeA_pos[2] + 0.10, 0.12)
            
            dist = np.linalg.norm(ee_pos - target)
            if dist < 0.04 or self.phase_timer > 150:
                self.phase = 1
                self.phase_timer = 0
                
        elif self.phase == 1:
            # Phase 1: 下到方块高度（指尖在方块两侧）
            target = cubeA_pos.copy()
            target[2] = cubeA_pos[2]  # TCP到方块中心
            
            dist = np.linalg.norm(ee_pos - target)
            if dist < DIST_THRESH or self.phase_timer > 80:
                self.phase = 2
                self.phase_timer = 0
                
        elif self.phase == 2:
            # Phase 2: 举起cubeA
            target = cubeA_pos.copy()
            target[2] = self.lift_z
            
            dist = np.linalg.norm(ee_pos - target)
            if dist < DIST_THRESH or self.phase_timer > 100:
                self.phase = 3
                self.phase_timer = 0
                
        elif self.phase == 3:
            # Phase 3: 移动到cubeB上方（叠放）
            target = cubeB_pos.copy()
            target[2] = self.lift_z
            
            xy_dist = np.linalg.norm(ee_pos[:2] - target[:2])
            if xy_dist < DIST_THRESH or self.phase_timer > 150:
                self.phase = 4
                self.phase_timer = 0
                
        elif self.phase == 4:
            # Phase 4: 放下cubeA到cubeB上
            target = cubeB_pos.copy()
            target[2] = cubeB_pos[2] + 0.06
            
            dist = np.linalg.norm(ee_pos - target)
            if dist < DIST_THRESH or self.phase_timer > 80:
                self.phase = 5
                self.phase_timer = 0
                
        elif self.phase == 5:
            # Phase 5: 松开夹爪后缩回
            target = cubeB_pos.copy()
            target[2] = self.lift_z
            
            if self.phase_timer > 50:
                return None  # 任务结束
        
        else:
            return None
        
        self.phase_timer += 1
        return target
    
    def get_gripper_action(self):
        """根据阶段返回夹爪动作"""
        if self.phase <= 1:
            return self.gripper_open  # 打开（准备抓取）
        elif self.phase <= 4:
            return self.gripper_closed  # 闭合（抓着）
        else:
            return self.gripper_open  # 打开（释放）
    
    def act(self, obs):
        """根据观测输出动作"""
        o = obs[0].cpu().numpy()
        joint_pos = o[:7]
        ee_pos = o[18:21]
        cubeA_pos = o[25:28]
        cubeB_pos = o[32:35]
        goal_pos = o[39:42]
        
        target = self.get_target_for_phase(ee_pos, cubeA_pos, cubeB_pos, goal_pos)
        
        if target is None:
            return np.zeros(8, dtype=np.float32)  # 任务完成
        
        # 位置误差
        pos_error = target - ee_pos
        if self.prev_pos_error is None:
            self.prev_pos_error = pos_error.copy()
        pos_error_dot = pos_error - self.prev_pos_error
        
        # 期望末端速度（PD控制）
        desired_vel = KP_POS * pos_error + KD * pos_error_dot
        desired_vel = np.clip(desired_vel, -MAX_VEL, MAX_VEL)
        
        # 雅可比 + IK → 关节速度
        J = self.compute_jacobian(joint_pos)
        joint_vel = self.damped_least_squares_ik(J, desired_vel)
        joint_vel = np.clip(joint_vel, -MAX_VEL, MAX_VEL)
        
        self.prev_pos_error = pos_error.copy()
        
        # 关节位置delta（速度×dt，dt≈0.1在仿真中）
        joint_delta = joint_vel * 0.1
        
        # 夹爪
        gripper_target = self.get_gripper_action()
        gripper_delta = (gripper_target - o[7]) * 0.5  # 柔顺抓取
        
        action = np.concatenate([joint_delta, [gripper_delta]])
        return action.astype(np.float32)


def evaluate_pid(episodes=50, collect_data=False, save_gif=False):
    """评估PID控制器，可选收集数据和保存GIF"""
    env = gym.make('StackCube-v1', obs_mode='state',
                   control_mode='pd_joint_delta_pos',
                   render_mode='rgb_array' if save_gif else None,
                   max_episode_steps=400)
    
    n_succ = 0
    ep_returns = []
    ep_lengths = []
    collected_obs = []
    collected_acts = []
    outdir = 'outputs/pid_gifs'
    if save_gif:
        import os
        os.makedirs(outdir, exist_ok=True)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        pid = StackCubePID(env)
        done = trunc = False
        ret = 0
        t = 0
        frames = []
        last_phase = -1
        
        while not done and not trunc:
            action = pid.act(obs)
            obs, r, done, trunc, info = env.step(action)
            ret += float(r)
            t += 1
            
            if collect_data and pid.phase <= 5:
                collected_obs.append(obs[0].cpu().numpy())
                collected_acts.append(action)
            
            if save_gif and ep < 3:
                frame = env.render()
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                    frame = frame[0]
                if frame is not None and frame.ndim == 3:
                    frames.append(frame)
            
            if pid.phase != last_phase:
                print(f'  Ep {ep+1}: phase {last_phase} → {pid.phase} (t={t})')
                last_phase = pid.phase
        
        ep_returns.append(ret)
        ep_lengths.append(t)
        
        if done and info.get('success', False):
            n_succ += 1
        
        if frames:
            imageio.mimsave(f'{outdir}/ep{ep+1:03d}.gif', frames, duration=0.04)
        
        print(f'Ep {ep+1}/{episodes}: final_phase={pid.phase}, ret={ret:.1f}, len={t}, success={info.get("success", False).item() if isinstance(info.get("success"), torch.Tensor) else info.get("success", False)}')
    
    env.close()
    print(f'\n=== PID Result: {n_succ}/{episodes} = {n_succ/episodes*100:.1f}% ===')
    print(f'Avg return: {np.mean(ep_returns):.2f}, Avg len: {np.mean(ep_lengths):.1f}')
    
    if collect_data and len(collected_obs) > 0:
        import os
        os.makedirs('data/processed', exist_ok=True)
        np.savez('data/processed/stackcube_pid.npz',
                 obs=np.stack(collected_obs),
                 acts=np.stack(collected_acts))
        print(f'Collected {len(collected_obs)} timesteps -> data/processed/stackcube_pid.npz')
    
    return n_succ / episodes


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--collect', action='store_true', help='收集数据用于模仿学习')
    p.add_argument('--save-gif', action='store_true', help='保存GIF到outputs/pid_gifs/')
    args = p.parse_args()
    
    evaluate_pid(episodes=args.episodes, collect_data=args.collect, save_gif=args.save_gif)
