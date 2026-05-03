"""
StackCube PID Controller v3

核心: 仿真器save/restore算雅可比 + 取消计时器强制跳转 + 抓取下沉
"""
import sys; sys.path.insert(0, 'src')
import os, numpy as np, torch, imageio.v2 as imageio
import gymnasium as gym, mani_skill.envs

KP = 50.0       # 比例增益
DAMPING = 0.01  # DLS阻尼（降低，提高精度）
MAX_ACT = 1.0   # 单步最大关节动作
DIST_OK = 0.03  # 到达阈值
DQ = 1e-4       # 雅可比步长
Z_LIFT = 0.15
Z_ABOVE = 0.10
Z_SINK = -0.015 # 下沉量（TCP到方块中心下方一点确保夹住）


class PID:
    def __init__(self, env):
        self.env = env
        self.robot = env.unwrapped.agent.robot
        self.scene = env.unwrapped.scene
        self.phase = 0
        self.tcp_link = [l for l in self.robot.get_links() if 'tcp' in l.get_name().lower()][0]
        self.gripper_closed = False
        self.ee_at_target = False
    
    def _set_qpos(self, jp):
        """设置关节位置并返回EE位置"""
        q9 = np.zeros(9)
        q9[:7] = jp[:7]; q9[7] = q9[8] = jp[7] if len(jp) > 7 else 0.04
        self.robot.set_qpos(torch.from_numpy(q9))
        return self.tcp_link.pose.p[0].numpy()
    
    def _jacobian(self, jp):
        """仿真器雅可比: 存状态→扰动关节→读变化→恢复状态"""
        # 先存状态
        state = self.scene.pack() if hasattr(self.scene, 'pack') else None
        
        ee0 = self._set_qpos(jp)
        J = np.zeros((3, 7))
        for i in range(7):
            jq = jp.copy()
            jq[i] += DQ
            ee1 = self._set_qpos(jq)
            J[:, i] = (ee1 - ee0) / DQ
        
        # 恢复状态
        if state is not None:
            self.scene.unpack(state)
        
        return J
    
    def act(self, obs):
        o = obs[0].cpu().numpy()
        ee = o[18:21]
        cA = o[25:28]
        cB = o[32:35]
        goal = o[39:42]
        jp = o[:7]
        grp = o[7]
        
        # ===== 阶段目标 =====
        if self.phase == 0:  # → cubeA上方
            tgt = cA.copy(); tgt[2] = max(cA[2] + Z_ABOVE, 0.10)
            if np.linalg.norm(ee - tgt) < DIST_OK: self.phase = 1
        elif self.phase == 1:  # → 下沉抓取
            tgt = cA.copy(); tgt[2] = cA[2] + Z_SINK
            if np.linalg.norm(ee - tgt) < 0.02:
                self.phase = 2; self.gripper_closed = True
        elif self.phase == 2:  # 举起
            tgt = cA.copy(); tgt[2] = Z_LIFT
            if np.linalg.norm(ee - tgt) < DIST_OK: self.phase = 3
        elif self.phase == 3:  # → cubeB上方
            tgt = cB.copy(); tgt[2] = Z_LIFT
            if np.linalg.norm(ee[:2] - tgt[:2]) < DIST_OK: self.phase = 4
        elif self.phase == 4:  # 放下
            tgt = cB.copy(); tgt[2] = cB[2] + 0.04
            if np.linalg.norm(ee - tgt) < 0.02: self.phase = 5
        elif self.phase == 5:  # 松开缩回
            tgt = cB.copy(); tgt[2] = Z_LIFT
            self.gripper_closed = False
            return None  # 任务完成
        else:
            return None
        
        # ===== IK + 动作 =====
        err = tgt - ee
        J = self._jacobian(jp)
        # DLS IK
        JJT = J @ J.T
        des_v = np.clip(KP * err, -2.0, 2.0)
        jv = J.T @ np.linalg.solve(JJT + DAMPING**2 * np.eye(3), des_v)
        jv = np.clip(jv, -MAX_ACT, MAX_ACT)
        
        act = np.zeros(8)
        act[:7] = jv * 0.1  # delta = vel * dt
        
        # 夹爪
        if self.gripper_closed:
            act[7] = -(grp + 0.01) * 3.0
        else:
            act[7] = (0.04 - grp) * 1.0
        
        return act.astype(np.float32)


def evaluate(episodes=5, save_gif=False):
    env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos',
                   render_mode='rgb_array' if save_gif else None, max_episode_steps=400)
    outdir = 'outputs/pid_gifs'
    if save_gif: os.makedirs(outdir, exist_ok=True)
    
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        pid = PID(env)
        frames, ret = [], 0.0
        for t in range(400):
            act = pid.act(obs)
            if act is None:
                break
            obs, r, done, trunc, info = env.step(act)
            ret += float(r)
            if save_gif and ep < 3:
                fr = env.render()
                if isinstance(fr, torch.Tensor): fr = fr.cpu().numpy()
                if isinstance(fr, np.ndarray):
                    if fr.ndim == 4: fr = fr[0]
                    if fr.ndim == 3: frames.append(fr)
            if done or trunc:
                break
        ok = bool(info.get('success', False))
        if ok: succ += 1
        if frames:
            imageio.mimsave(f'{outdir}/ep{ep+1:03d}_p{pid.phase}.gif', frames, duration=0.04)
        print(f'Ep {ep+1}: phase={pid.phase}, len={t+1}, ret={ret:.1f}, ok={ok}')
    
    print(f'\nResult: {succ}/{episodes} = {succ/episodes*100:.1f}%')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--save-gif', action='store_true')
    args = p.parse_args()
    evaluate(args.episodes, args.save_gif)
