"""
train_bcse2.py - SE(2)等变Diffusion Policy训练脚本。

核心改进：
1. FK计算末端位置 + SE(2)等变特征提取
2. 多checkpoint保存（不再只看val_loss最低）
3. 更灵活的模型选择
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from models_se2 import BCDiffusionSE2, build_obs_dict
from models import EMA


def extract_geometry(obs_np, joint_dim=7):
    """
    从观测中提取几何信息。
    
    obs layout (48-dim):
    [0:8] = qpos
    [8:16] = qvel
    [16:18] = prev_action
    [18:48] = cube_info
    
    注意：cube_info中包含的方体位置需要根据ManiSkill实际布局调整。
    这里使用近似布局：
    dim[18:25] = cubeA (pos3 + quat4)
    dim[25:32] = cubeB (pos3 + quat4)
    dim[32:39] = goal (pos3 + quat4)
    dim[39:48] = other info (9 dims)
    """
    # 假设布局（需要根据实际数据验证）
    # cubeA设为第2个方体，cubeB设为第3个
    cubeA_pos = obs_np[:, 18:21]  # (B, 3) - 需要验证
    cubeB_pos = obs_np[:, 25:28]  # (B, 3) - 需要验证
    goal_pos = obs_np[:, 32:35]   # (B, 3) - 需要验证
    
    return cubeA_pos, cubeB_pos, goal_pos


def parse_obs(obs_batch, ee_pos_batch=None):
    """
    解析观测为模型需要的格式。
    
    obs: (B, 48) torch tensor
    ee_pos: (B, 3) torch tensor or None
    """
    # 提取关节信息
    joint_pos = obs_batch[:, :7]
    joint_vel = obs_batch[:, 8:15] if obs_batch.shape[-1] >= 15 else obs_batch[:, :7]
    prev_action = obs_batch[:, 16:18]
    
    # 提取方体位置（已验证的ManiSkill3布局）
    # [18:21] = ee_pos, [25:28] = cubeA_pos, [32:35] = cubeB_pos
    cubeA_pos = obs_batch[:, 25:28]
    cubeB_pos = obs_batch[:, 32:35]
    goal_pos = obs_batch[:, 39:42]
    
    # 使用obs中的ee_pos（在[18:21]），不依赖FK
    if ee_pos_batch is None:
        ee_pos_batch = obs_batch[:, 18:21]
    
    return {
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'prev_action': prev_action,
        'ee_pos': ee_pos_batch,
        'cubeA_pos': cubeA_pos,
        'cubeB_pos': cubeB_pos,
        'goal_pos': goal_pos,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           required=True)
    p.add_argument("--geom-data",         default=None,
                   help="包含ee_pos的几何数据文件")
    p.add_argument("--outdir",            required=True)
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--weight-decay",      type=float, default=1e-5)
    p.add_argument("--hidden",            type=int,   default=256)
    p.add_argument("--depth",             type=int,   default=4)
    p.add_argument("--T",                 type=int,   default=100)
    p.add_argument("--scheduler",         type=str,   default="cosine", choices=["linear", "cosine"])
    p.add_argument("--ema-decay",         type=float, default=0.999)
    
    # ── 多checkpoint ─────────────────────────────────────────────────────
    p.add_argument("--save-every",        type=int,   default=10,
                   help="每N个epoch保存一个checkpoint")
    p.add_argument("--ckpt-prefix",       type=str,   default="ckpt_ep",
                   help="checkpoint文件名前缀")
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # ── 加载数据 ─────────────────────────────────────────────
    data = load_npz_dataset(args.dataset)
    x_raw = data["obs"].astype(np.float32)
    y_raw = data["acts"].astype(np.float32)
    
    train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=args.seed)
    train_mask = np.zeros(len(x_raw), dtype=bool)
    train_mask[train_idx] = True
    print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")
    print(f"obs_dim={x_raw.shape[1]}  act_dim={y_raw.shape[1]}")
    
    # ── 归一化 ────────────────────────────────────────────────
    obs_norm = RunningNormalizer(const_thresh=1e-3)
    obs_norm.fit(x_raw[train_mask])
    act_norm = RunningNormalizer(const_thresh=0.0)
    act_norm.fit(y_raw[train_mask])
    
    x = obs_norm.transform(x_raw)
    y = act_norm.transform(y_raw)
    
    # ── 数据加载器（不需要geom数据，obs[18:21]已有ee_pos）───
    
    # ── 数据加载器 ────────────────────────────────────────────
    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val   = torch.from_numpy(x[val_idx])
    y_val   = torch.from_numpy(y[val_idx])
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset   = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ── 模型 ──────────────────────────────────────────────────
    device = select_device()
    model = BCDiffusionSE2(
        obs_dim=x_raw.shape[1],
        act_dim=y_raw.shape[1],
        T=args.T,
        hidden=args.hidden,
        depth=args.depth,
        scheduler=args.scheduler,
    ).to(device)
    
    print(f"Model: {type(model).__name__} (hidden={args.hidden}, depth={args.depth})")
    print(f"T={args.T}")
    print(f"EMA enabled: decay={args.ema_decay}")
    
    ema = EMA(model, decay=args.ema_decay, device=device) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ── 训练 ──────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    saved_ckpts = []
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        
        for batch in train_loader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            
            obs_dict = parse_obs(xb, None)
            loss = model(obs_dict, yb)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if ema is not None:
                ema.update()
            
            total_train_loss += loss.item() * xb.shape[0]
            n_train += xb.shape[0]
        
        avg_train_loss = total_train_loss / n_train
        
        # ── 验证 ────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                
                obs_dict = parse_obs(xb, None)
                loss = model(obs_dict, yb)
                total_val_loss += loss.item() * xb.shape[0]
                n_val += xb.shape[0]
        
        avg_val_loss = total_val_loss / n_val
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # ── 保存最佳 ──────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if ema is not None:
                ema.apply_shadow()
            state_dict = model.state_dict()
            if ema is not None:
                ema.restore()
            ckpt_path = outdir / "best.pt"
            torch.save({
                "model": state_dict,
                "obs_dim": x_raw.shape[1],
                "act_dim": y_raw.shape[1],
                "T": args.T,
                "hidden": args.hidden,
                "depth": args.depth,
                "scheduler": args.scheduler,
                "obs_norm": obs_norm.state_dict(),
                "act_norm": act_norm.state_dict(),
                "ee_norm": None,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "epoch": epoch,
                "algo": "diffusion_se2",
            }, ckpt_path)
        
        # ── 定期保存checkpoint ─────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            if ema is not None:
                ema.apply_shadow()
            state_dict = model.state_dict()
            if ema is not None:
                ema.restore()
            ckpt_path = outdir / f"{args.ckpt_prefix}{epoch+1:04d}.pt"
            torch.save({
                "model": state_dict,
                "ema_model": state_dict,
                "obs_dim": x_raw.shape[1],
                "act_dim": y_raw.shape[1],
                "T": args.T,
                "hidden": args.hidden,
                "depth": args.depth,
                "scheduler": args.scheduler,
                "obs_norm": obs_norm.state_dict(),
                "act_norm": act_norm.state_dict(),
                "ee_norm": None,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "epoch": epoch,
                "algo": "diffusion_se2",
            }, ckpt_path)
            saved_ckpts.append(str(ckpt_path))
            print(f"  [Saved checkpoint] {ckpt_path}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | best={best_val_loss:.6f}")
    
    # ── 训练结束 ────────────────────────────────────────────
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved: {saved_ckpts}")
    
    # 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'SE(2) Diffusion Training (hidden={args.hidden}, depth={args.depth})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'loss_curves.png', dpi=150)
    plt.close()
    
    # 保存训练配置
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
