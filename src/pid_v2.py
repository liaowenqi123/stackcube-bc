"""
StackCube PID Controller v2 - 简化版

核心思路：直接末端位置误差 → 关节增量动作
不用雅可比，靠末端位置差分来指导关节运动
"""
import sys; sys.path.insert(0, 'src')
import os, numpy as np, torch, imageio.v2 as imageio
import gymnasium as gym, mani_skill.envs

KP_XYZ = 3.0      # 末端位置增益 → 关节动作
KP_GRIP = 2.0     # 夹爪增益
DIST_OK = 0.03    # 到达判定
MAX_ACT = 0.5     # 单步最大关节动作幅度
Z_LIFT = 0.12     # 举起高度
Z_ABOVE = 0.08    # 方块上方偏移


class StackCubePID:
    def __init__(self, env):
        self.env = env
        self.phase = 0
        self.phase_timer = 0
        self.gripper_closed = False
    
    def act(self, obs):
        o = obs[0].cpu().numpy()
        ee = o[18:21]
        cA = o[25:28]
        cB = o[32:35]
        goal = o[39:42]
        joint = o[:7]
        gripper = o[7]
        
        # ===== 相位逻辑 =====
        
        if self.phase == 0:  # 移动到cubeA上方
            target = cA.copy()
            target[2] = max(cA[2] + Z_ABOVE, 0.10)
            dist = np.linalg.norm(ee - target)
            if dist < DIST_OK or self.phase_timer > 150:
                self.phase = 1; self.phase_timer = 0
                
        elif self.phase == 1:  # 下降到方块高度
            target = cA.copy()
            target[2] = cA[2]  # 降到方块顶面
            dist = np.linalg.norm(ee - target)
            if dist < 0.02:
                self.phase = 2; self.phase_timer = 0; self.gripper_closed = True
                
        elif self.phase == 2:  # 举起
            target = cA.copy()
            target[2] = Z_LIFT
            dist = np.linalg.norm(ee - target)
            if dist < DIST_OK:
                self.phase = 3; self.phase_timer = 0
                
        elif self.phase == 3:  # 移动到cubeB上方
            target = cB.copy()
            target[2] = Z_LIFT
            xy = np.linalg.norm(ee[:2] - target[:2])
            if xy < DIST_OK:
                self.phase = 4; self.phase_timer = 0
                
        elif self.phase == 4:  # 放下到cubeB
            target = cB.copy()
            target[2] = cB[2] + 0.05
            dist = np.linalg.norm(ee - target)
            if dist < 0.02 or self.phase_timer > 80:
                self.phase = 5; self.phase_timer = 0
                
        elif self.phase == 5:  # 松开缩回
            target = cB.copy()
            target[2] = Z_LIFT
            self.gripper_closed = False
            if self.phase_timer > 30:
                return None  # 结束
        else:
            return None
        
        # ===== 简化的动作生成 =====
        # 不用雅可比，直接期望末端位移 → 关节delta
        err = target - ee
        # 把三维误差映射到8维动作（前7关节+夹爪）
        action = np.zeros(8, dtype=np.float32)
        # 主要靠关节0-2（大臂）做x-y-z移动，关节3-6（小臂）微调
        action[:7] = np.clip(err[:3].repeat(7//3+1)[:7] * KP_XYZ * 0.3, -MAX_ACT, MAX_ACT)
        
        # 夹爪
        if self.gripper_closed:
            action[7] = -(gripper + 0.01) * KP_GRIP  # 闭合
        else:
            action[7] = (0.04 - gripper) * KP_GRIP * 0.3  # 打开
        
        self.phase_timer += 1
        return action


def evaluate(episodes=5, save_gif=False):
    env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos',
                   render_mode='rgb_array' if save_gif else None, max_episode_steps=400)
    outdir = 'outputs/pid_gifs'
    if save_gif:
        os.makedirs(outdir, exist_ok=True)
    
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        pid = StackCubePID(env)
        frames, ret = [], 0.0
        
        for t in range(400):
            act = pid.act(obs)
            if act is None:
                break  # 任务完成提前结束
            obs, r, done, trunc, info = env.step(act)
            ret += float(r)
            if save_gif and ep < 3:
                fr = env.render()
                if isinstance(fr, torch.Tensor): fr = fr.detach().cpu().numpy()
                if isinstance(fr, np.ndarray):
                    if fr.ndim == 4: fr = fr[0]
                    if fr.ndim == 3: frames.append(fr)
            if done or trunc:
                break
        
        ok = bool(info.get('success', False)) if isinstance(info.get('success'), (bool, torch.Tensor)) else False
        if ok: succ += 1
        if frames:
            imageio.mimsave(f'{outdir}/ep{ep+1:03d}_phase{pid.phase}.gif', frames, duration=0.04)
        print(f'Ep {ep+1}: phase={pid.phase}, len={t+1}, ret={ret:.1f}, success={ok}')
    
    env.close()
    print(f'\nResult: {succ}/{episodes} = {succ/episodes*100:.1f}%')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--save-gif', action='store_true')
    args = p.parse_args()
    evaluate(args.episodes, args.save_gif)
