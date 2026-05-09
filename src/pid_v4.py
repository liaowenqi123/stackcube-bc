"""
StackCube PID v4 - 6自由度IK + 姿态控制
"""
import sys; sys.path.insert(0, 'src')
import os, numpy as np, torch, imageio.v2 as imageio
import gymnasium as gym, mani_skill.envs

KP=50; KP_O=20; DAMP=0.01; MAX_A=1.0; DQ=1e-4
DIST=0.05; Z_L=0.15; Z_A=0.10; Z_S=-0.015; Z_PLACE=0.023

def aa(R):
    t=np.trace(R); th=np.arccos(np.clip((t-1)/2,-1,1))
    if th<1e-8: return np.zeros(3)
    return th/(2*np.sin(th))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

class PID:
    def __init__(self, env):
        self.env=env; self.robot=env.unwrapped.agent.robot
        self.phase=0; self.pt=0
        self.gr_cmd=1.0
        self.grasp_lock_t=None
        self.recover_xy=None
        self.retry_count=0
        self.max_retries=2
        self.prev_ee_z=None
        self.z_stall_count=0
        self.phase2_regrasp_used=False
        self.extra_sink=0.0
        self.phase1_xy_target=None
        self.phase1_fit_margin=0.0
        self.grasp_offset=None
        self.phase1_mode=0
        self.phase1_best_xy=None
        self.phase1_no_progress=0
        self.phase2_cube_z0=None
        self.tcp=[l for l in self.robot.get_links() if 'tcp' in l.get_name().lower()][0]
        self.left_finger=[l for l in self.robot.get_links() if 'leftfinger' in l.get_name().lower()][0]
        self.right_finger=[l for l in self.robot.get_links() if 'rightfinger' in l.get_name().lower()][0]
        chs=getattr(env.unwrapped,'cube_half_size',0.02)
        self.cube_half=float(chs[0].item() if hasattr(chs,'__len__') else chs)
    def _start_retry(self, ee):
        if self.retry_count>=self.max_retries:
            return False
        self.retry_count+=1
        self.phase=90; self.pt=0
        self.grasp_lock_t=None
        self.recover_xy=ee[:2].copy()
        self.gr_cmd=1.0
        self.z_stall_count=0
        self.phase2_regrasp_used=False
        self.extra_sink=0.0
        self.phase1_xy_target=None
        self.phase1_fit_margin=0.0
        self.grasp_offset=None
        self.phase1_mode=0
        self.phase1_best_xy=None
        self.phase1_no_progress=0
        self.phase2_cube_z0=None
        return True
    def _retry_xy_bias(self, Rc):
        signs=[0.0, 1.0, -1.0]
        s=signs[self.retry_count] if self.retry_count < len(signs) else 0.0
        j=Rc[:,0].copy(); j[2]=0.0
        n=np.linalg.norm(j)
        if n<1e-6: return np.zeros(2)
        j=j/n
        return (0.006*s*j[:2]).astype(np.float32)
    def _init_phase1_target(self, ee, cA, Rc):
        center_err,fit_margin=self._proj_grasp_metrics(cA,Rc)
        corr=np.clip(center_err[:2],-0.016,0.016)
        self.phase1_xy_target=(ee[:2]+corr+self._retry_xy_bias(Rc)).astype(np.float32)
        self.phase1_fit_margin=float(fit_margin)
    def _fk(self, jp, gr):
        q9=np.zeros(9); q9[:7]=jp[:7]; q9[7]=q9[8]=gr
        self.robot.set_qpos(torch.from_numpy(q9))
        p=self.tcp.pose.p[0].numpy()
        q=self.tcp.pose.q[0].numpy(); w,x,y,z=q
        R=np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])
        return p,R
    def _jac(self, jp, gr):
        q0=jp.copy(); p0,R0=self._fk(q0,gr); J=np.zeros((6,7))
        for i in range(7):
            jq=q0.copy(); jq[i]+=DQ; p1,R1=self._fk(jq,gr)
            J[:3,i]=(p1-p0)/DQ; J[3:,i]=aa(R1@R0.T)/DQ
        self._fk(q0,gr); return J
    def _trgR(self,R):
        z=np.array([0,0,-1]); xh=R[:,0]-np.dot(R[:,0],z)*z
        x=np.array([1,0,0]) if np.linalg.norm(xh)<1e-6 else xh/np.linalg.norm(xh)
        return np.column_stack([x,np.cross(z,x),z])
    def _quat_R(self, q):
        w,x,y,z=q
        return np.array([
            [1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
            [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
            [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]
        ])
    def _proj_grasp_metrics(self, cA, Rc):
        lp=self.left_finger.pose.p[0].cpu().numpy()
        rp=self.right_finger.pose.p[0].cpu().numpy()
        mid=0.5*(lp+rp)
        jaw_vec=lp-rp; jaw_w=max(np.linalg.norm(jaw_vec),1e-6); jaw_u=jaw_vec/jaw_w
        app=-Rc[:,2]; app=app/max(np.linalg.norm(app),1e-6)
        proj=lambda v: v-np.dot(v,app)*app
        center_err=proj(cA-mid)
        jaw_p=proj(jaw_u); jaw_p=jaw_p/max(np.linalg.norm(jaw_p),1e-6)
        cq=self.env.unwrapped.cubeA.pose.q[0].cpu().numpy()
        Rcube=self._quat_R(cq)
        h=self.cube_half
        corners=[cA + Rcube@np.array([sx*h,sy*h,0.0]) for sx in (-1,1) for sy in (-1,1)]
        rel=[float(np.dot(proj(p-mid),jaw_p)) for p in corners]
        max_corner=abs(max(rel, key=abs))
        jaw_half=0.5*jaw_w
        fit_margin=jaw_half-max_corner
        return center_err, fit_margin
    def act(self,obs):
        o=obs[0].cpu().numpy(); ee,cA,cB=o[18:21],o[25:28],o[32:35]; jp,gr=o[:7],o[7]
        self.pt+=1
        close_only=False
        phase1_xy_err=None
        phase1_orient_mode=False
        phase1_ready=False
        dz = 0.0 if self.prev_ee_z is None else float(ee[2]-self.prev_ee_z)
        _,Rc_now=self._fk(jp,gr)
        Rt_now=self._trgR(Rc_now)
        # ===== 阶段 =====
        if self.phase==90:
            t=np.array([*(self.recover_xy if self.recover_xy is not None else ee[:2]), max(Z_L,cA[2]+Z_A)])
            self.z_stall_count=0
            if abs(ee[2]-t[2])<0.02:
                self.phase=1; self.pt=0
                self.phase1_mode=0; self.phase1_xy_target=None; self.phase1_best_xy=None; self.phase1_no_progress=0
        elif self.phase==0:
            t=cA.copy(); t[2]=max(cA[2]+Z_A,0.10)
            self.z_stall_count=0
            if np.linalg.norm(ee-t)<DIST:
                self.phase=1; self.pt=0
                self.phase1_mode=0; self.phase1_xy_target=None; self.phase1_best_xy=None; self.phase1_no_progress=0
        elif self.phase==1:
            t=cA.copy()
            center_err,fit_margin=self._proj_grasp_metrics(cA,Rc_now)
            self.phase1_fit_margin=float(fit_margin)
            if self.phase1_mode==0:
                # 先用老方法：直接朝cube中心对齐，并保持末端垂直
                t[:2]=(cA[:2]+self._retry_xy_bias(Rc_now)).astype(np.float32)
                xy_err=float(np.linalg.norm(ee[:2]-t[:2]))
                if self.phase1_best_xy is None or xy_err < self.phase1_best_xy - 0.001:
                    self.phase1_best_xy=xy_err; self.phase1_no_progress=0
                else:
                    self.phase1_no_progress += 1
                # 垂直对准撇不过去 -> 切到投影模式，且目标固定
                if self.phase1_no_progress>12 or self.pt>26:
                    self.phase1_mode=1
                    self._init_phase1_target(ee,cA,Rc_now)
            else:
                if self.phase1_xy_target is None:
                    self._init_phase1_target(ee,cA,Rc_now)
                t[:2]=self.phase1_xy_target
                xy_err=float(np.linalg.norm(ee[:2]-self.phase1_xy_target))
            phase1_xy_err=xy_err
            phase1_orient_mode = xy_err < 0.009
            # xy优先：只要xy差不多就进入下抓尝试，避免卡住不抓
            if self.phase1_mode==0:
                ready=(xy_err<0.018) or (self.pt>14 and xy_err<0.028)
            else:
                ready=(xy_err<0.020) or (self.pt>20 and xy_err<0.030)
            phase1_ready=ready
            z_grasp=cA[2]+Z_S+self.extra_sink
            descend_ok = ee[2] < (z_grasp + 0.020)
            t[2]=z_grasp if ready else max(cA[2]+Z_A,0.10)
            if phase1_orient_mode:
                t[:2]=ee[:2]
            if ready and ee[2] > (z_grasp+0.004):
                self.z_stall_count = self.z_stall_count + 1 if abs(dz) < 8e-4 else 0
                if self.z_stall_count > 16 and self.pt>14 and self._start_retry(ee):
                    t=np.array([ee[0],ee[1],max(Z_L,cA[2]+Z_A)])
            else:
                self.z_stall_count=0
            if self.pt>45 and not ready and self._start_retry(ee):
                t=np.array([ee[0],ee[1],max(Z_L,cA[2]+Z_A)])
            if self.phase==1 and self.pt>120 and descend_ok and xy_err<0.030:
                self.phase=2; self.pt=0; self.grasp_lock_t=ee.copy()
                self.phase2_cube_z0=float(cA[2])
                t=self.grasp_lock_t; close_only=True
            if self.phase==1 and ready and descend_ok:
                self.phase=2; self.pt=0; self.grasp_lock_t=ee.copy()
                self.phase2_cube_z0=float(cA[2])
                t=self.grasp_lock_t; close_only=True
            if self.phase==1 and self.pt>55 and xy_err<0.035 and ee[2]<(z_grasp+0.030):
                self.phase=2; self.pt=0; self.grasp_lock_t=ee.copy()
                self.phase2_cube_z0=float(cA[2])
                t=self.grasp_lock_t; close_only=True
        elif self.phase==2:
            self.z_stall_count=0
            # 先原地闭爪，闭爪期间锁定手臂不动
            if (gr>0.01) and (self.pt<=14):
                t=self.grasp_lock_t if self.grasp_lock_t is not None else ee.copy()
                close_only=True
            else:
                t=(self.grasp_lock_t.copy() if self.grasp_lock_t is not None else ee.copy()); t[2]=Z_L
                lifted = (self.phase2_cube_z0 is not None) and (float(cA[2]) - self.phase2_cube_z0 > 0.012)
                # 抓住后先验证“能提起来”，确认抬起后立刻进入搬运
                if lifted:
                    self.grasp_offset=(ee-cA).copy()
                    self.phase=3; self.pt=0
                if self.pt>90:
                    if not self.phase2_regrasp_used:
                        self.phase2_regrasp_used=True
                        self.extra_sink=-0.006
                        self.phase=1; self.pt=0; self.grasp_lock_t=None
                        self.phase1_mode=0; self.phase1_xy_target=None; self.phase1_best_xy=None; self.phase1_no_progress=0
                        t=np.array([ee[0],ee[1],max(cA[2]+Z_A,0.10)])
                    elif self._start_retry(ee):
                        t=np.array([ee[0],ee[1],max(Z_L,cA[2]+Z_A)])
                # 兜底：即使抬升幅度小，已达到抬升位也继续，但优先用 lifted 判定
                if self.phase==2 and np.linalg.norm(ee-t)<0.045 and self.pt>25:
                    self.grasp_offset=(ee-cA).copy()
                    self.phase=3; self.pt=0
        elif self.phase==3:
            self.extra_sink=0.0
            t=ee.copy(); t[:2]=ee[:2]+(cB[:2]-cA[:2]); t[2]=Z_L
            if (self.pt>100 or gr<0.006 or (np.linalg.norm(ee-cA)>0.11 and ee[2]>cA[2]+0.05)) and self._start_retry(ee):
                t=np.array([ee[0],ee[1],max(Z_L,cA[2]+Z_A)])
            if self.phase==3 and np.linalg.norm(cA[:2]-cB[:2])<0.020: self.phase=4; self.pt=0
        elif self.phase==4:
            cubeA_target_z=cB[2]+2*self.cube_half
            oz=self.grasp_offset[2] if self.grasp_offset is not None else (ee[2]-cA[2])
            t=ee.copy(); t[:2]=ee[:2]+(cB[:2]-cA[:2]); t[2]=cubeA_target_z+oz
            if (self.pt>85 or gr<0.004) and self._start_retry(ee):
                t=np.array([ee[0],ee[1],max(Z_L,cA[2]+Z_A)])
            if self.phase==4 and ((np.linalg.norm(cA[:2]-cB[:2])<0.012 and abs(cA[2]-cubeA_target_z)<0.015) or self.pt>70):
                self.phase=5; self.pt=0
        elif self.phase==5:
            cubeA_target_z=cB[2]+2*self.cube_half
            oz=self.grasp_offset[2] if self.grasp_offset is not None else (ee[2]-cA[2])
            t=ee.copy(); t[:2]=ee[:2]+(cB[:2]-cA[:2]); t[2]=cubeA_target_z+oz
            if self.phase==5 and self.pt>8:
                self.phase=6; self.pt=0
        elif self.phase==6:
            t=cB.copy(); t[2]=Z_L
            if self.pt>14: return None
        else: return None
        # ===== IK =====
        Rc=Rc_now; Rt=Rt_now; eo=aa(Rt@Rc.T)
        if self.phase in (0,2,90):
            eo=np.zeros(3)
        if self.phase==1 and not phase1_orient_mode:
            eo=np.zeros(3)
        if self.phase==0:
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=95,30,4.5,1.8,2.0,0.20
        elif self.phase==1:
            # 先垂直对准，撇不过去再切投影（投影目标固定）
            if self.phase1_mode==0:
                kp_p,kp_o,clip_p,clip_o,max_a,step_scale=(88,14,3.6,0.7,1.6,0.16) if phase1_ready else (78,12,3.0,0.6,1.4,0.12)
            else:
                kp_p,kp_o,clip_p,clip_o,max_a,step_scale=(84,6,3.4,0.4,1.5,0.15) if phase1_ready else (72,0,2.8,0.0,1.3,0.11)
        elif self.phase==2:  # 抬起阶段更激进
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=110,34,5.0,2.0,2.2,0.24
        elif self.phase==3:  # 搬运阶段更激进
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=100,32,4.6,1.9,2.1,0.22
        elif self.phase==4:  # 下放并对中：慢一点但避免拖沓
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=78,26,3.0,1.3,1.4,0.12
        elif self.phase==5:  # 放手时保持中心位置，慢一点
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=50,22,2.0,1.0,0.9,0.07
        elif self.phase==90:  # 回退重试抬升
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=100,0,4.0,0.0,2.0,0.20
        else:  # phase 6 收手
            kp_p,kp_o,clip_p,clip_o,max_a,step_scale=95,30,4.2,1.6,1.9,0.20
        dv=np.zeros(6); dv[:3]=np.clip(kp_p*(t-ee),-clip_p,clip_p); dv[3:]=np.clip(kp_o*eo,-clip_o,clip_o)
        J=self._jac(jp,gr)
        jv=J.T@np.linalg.solve(J@J.T+DAMP**2*np.eye(6),dv)
        jv=np.clip(jv,-max_a,max_a); act=np.zeros(8); act[:7]=jv*step_scale
        if close_only:
            act[:7]=0.0
        # ManiSkill文档/控制器定义：gripper_pd_joint_pos 为绝对目标（归一化到[-1,1]）
        # +1 => upper(0.04, 打开), -1 => lower(-0.01, 闭合)
        # 夹爪：放松(phase 0/1) -> 并拢抓起移动+对中下放(phase 2/3/4) -> 放松(phase 5+)
        # 用平滑目标避免从 +1 到 -1 的瞬时冲击导致方块弹飞
        target_gr_cmd = -0.65 if self.phase in (2, 3, 4) else 1.0
        step = 0.45 if target_gr_cmd > self.gr_cmd else 0.20
        dcmd = np.clip(target_gr_cmd - self.gr_cmd, -step, step)
        self.gr_cmd += dcmd
        act[7] = self.gr_cmd
        self.prev_ee_z=float(ee[2])
        return act.astype(np.float32)

def evaluate(ep=5,gif=False,gif_episodes=3):
    env=gym.make('StackCube-v1',obs_mode='state',control_mode='pd_joint_delta_pos',
                 render_mode='rgb_array' if gif else None,max_episode_steps=400)
    d='outputs/pid_gifs'
    if gif: os.makedirs(d,exist_ok=True)
    save_all_gifs = gif_episodes < 0
    s=0
    for e in range(ep):
        obs,_=env.reset(); pid=PID(env); f,ret=[],0.0
        for t in range(400):
            a=pid.act(obs)
            if a is None: break
            obs,r,done,trunc,info=env.step(a); ret+=float(r)
            if gif and (save_all_gifs or e < gif_episodes):
                fr=env.render()
                if isinstance(fr,torch.Tensor): fr=fr.cpu().numpy()
                if isinstance(fr,np.ndarray):
                    if fr.ndim==4: fr=fr[0]
                    if fr.ndim==3: f.append(fr)
            if done or trunc: break
        ok=bool(info.get('success',False))
        if ok: s+=1
        if f: imageio.mimsave(f'{d}/ep{e+1:03d}_p{pid.phase}.gif',f,duration=0.04)
        print(f'Ep {e+1}: p{pid.phase} len={t+1} ret={ret:.1f} ok={ok}')
    print(f'\n{s}/{ep}={s/ep*100:.1f}%')

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=5); p.add_argument('--save-gif',action='store_true')
    p.add_argument('--gif-episodes',type=int,default=3,
                   help='保存前N个episode的GIF，设为-1时保存全部')
    a=p.parse_args(); evaluate(a.episodes,a.save_gif,a.gif_episodes)
