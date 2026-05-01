# StackCube Baseline Reproduction

基于 ManiSkill 3.0.1 + StackCube-v1 的 Behavioral Cloning (BC) 基线复现。

> 目录路径建议使用英文，避免 SAPIEN 资产加载问题。

---

## 环境安装

```powershell
cd C:\robotics\stackcube_repro

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python src\runtime_check.py
```

**预期输出**：`runtime check passed`

---

## 数据下载 + 转换训练集

```powershell
python -m mani_skill.utils.download_asset StackCube-v1 -o .\data\raw
python -m mani_skill.utils.download_demo StackCube-v1 -o .\data\raw\demos
```

下载完成后，使用 ManiSkill 提供的 **rl/pd_joint_delta_pos** 演示（PPO 策略录制，932 条轨迹，40697 步）进行转换：

```powershell
python src\convert_maniskill_demo.py `
    --input-h5 .\data\raw\demos\StackCube-v1\rl\trajectory.none.pd_joint_delta_pos.physx_cuda.h5 `
    --output-npz .\data\processed\stackcube_rl_state.npz
```

**说明**：rl 演示必须配合 `set_state_dict` replay 方式才能正确执行（`motionplanning` 演示与当前 ManiSkill3 控制器不兼容）。转换后数据为 48 维观测 + 8 维动作，训练集 36627 步 / 验证集 4070 步。

---

## 数据可视化（可选）

```powershell
python src\visualize_dataset.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --out .\outputs\dataset_stats.png
```

---

## 训练 BC

```powershell
python src\train_bc.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --outdir .\outputs\bc `
    --epochs 300 `
    --batch-size 512 `
    --lr 1e-3
```

**输出**：
- `outputs\bc\best.pt` — 模型权重（含归一化参数）
- `outputs\bc\metrics.json` — 训练指标
- `outputs\bc\loss_curve.png` — 训练曲线

---

## 训练 BC-RNN（可选）

```powershell
python src\train_bcrnn.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --outdir .\outputs\bcrnn `
    --epochs 300 `
    --batch-size 64 `
    --lr 1e-3 `
    --seq-len 16
```

**输出**：
- `outputs\bcrnn\best.pt` — 模型权重
- `outputs\bcrnn\metrics.json`
- `outputs\bcrnn\loss_curve.png`

---

## 训练最新 Transformer+Diffusion（推荐）

本仓库里“Transformer + Diffusion”对应 `train_bcdiffusion.py --temporal`（`BCDiffusionTemporal`）。

### 一键训练 + 评估（输出 3 个 GIF）

```sh
sh ./run_transformer_diffusion.sh
```

默认会：
- 训练到 `outputs/diffusion_temporal_latest`
- 评估到 `outputs/eval_diffusion_temporal_latest`
- 生成 `rollout_ep000.gif`、`rollout_ep001.gif`、`rollout_ep002.gif`

可通过环境变量覆盖参数（示例）：

```sh
EPOCHS=300 EPISODES=20 T_INF=20 SAMPLER=ddim sh ./run_transformer_diffusion.sh
```

---

## 评估 + Rollout 可视化

```powershell
python src\eval_policy.py `
    --algo bc `
    --ckpt .\outputs\bc\best.pt `
    --episodes 100 `
    --output-dir .\outputs\eval `
    --max-steps 400 `
    --save-gif `
    --gif-episodes 3
```

**输出**：
- `metrics.json` — success_rate / avg_return / avg_ep_len
- `returns_hist.png` — 回报分布图
- `rollout_ep000.gif` 等 — 可直接展示的 rollout 动图

**当前基线结果**（BC，100 episodes eval）：

| 指标 | 值 |
|------|------|
| Success Rate | **76%** |
| Avg Return | 25.6 |
| Avg Episode Len | 109.5 步 |

---

## 完整最小命令（3 条出结果）

```powershell
# 1. 转换数据
python src\convert_maniskill_demo.py --input-h5 .\data\raw\demos\StackCube-v1\rl\trajectory.none.pd_joint_delta_pos.physx_cuda.h5 --output-npz .\data\processed\stackcube_rl_state.npz

# 2. 训练
python src\train_bc.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\bc --epochs 300

# 3. 评估
python src\eval_policy.py --algo bc --ckpt .\outputs\bc\best.pt --episodes 100 --output-dir .\outputs\eval --save-gif --gif-episodes 3
```

---

## 注意事项

- **必须用 rl 演示**：`motionplanning/trajectory.h5` 的 actions 与 ManiSkill3 的 `pd_joint_delta_pos` 控制器不兼容，即使 open-loop replay 也会失败。
- **eval 必须传 `--max-steps 400`**：ManiSkill 默认 max_episode_steps=50，不传会导致 episode 在第 50 步被截断。
- **CUDA 不可用时会回退到 CPU**：训练速度较慢，评估 GIF 渲染可能受限。
