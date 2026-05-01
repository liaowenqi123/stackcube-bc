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

## Diffusion 路线总览（v1 → v3 → 时序 → 等变骨干）

下面按你们当前代码里的真实实验脉络整理，方便直接复现实验与对比。

| 版本 | 核心改动 | 训练输出目录 |
|---|---|---|
| diffusion v1 | 基础 DDPM/DDIM diffusion policy | `outputs/diffusion` |
| diffusion v2 | 容量增大 + cosine + EMA + action noise + SWA | `outputs/diffusion_v2` |
| diffusion v3 | v2 进一步调参与训练策略迭代 | `outputs/diffusion_v3` |
| temporal diffusion | `--temporal` + Transformer 时序分支 | `outputs/diffusion_temporal_*` |
| v1 + C4/C8 | 非时序 v1 + 离散旋转不变骨干 | `outputs/diffusion_v1_c8`（默认脚本） |
| v1 + SE2 | 非时序 v1 + 连续参数化 SE(2) steerable 骨干 | `outputs/diffusion_v1_se2_e200` |
| v1 + harmonic | 非时序 v1 + 复数谐波 Fourier 特征骨干 | `outputs/diffusion_v1_harmonic_e200` |

### 当前一键脚本（默认）

```sh
sh ./run_transformer_diffusion.sh
```

当前脚本默认是：**非时序 diffusion(v1 路线) + `obs-backbone=c8`**，并导出 3 个 GIF。  
默认目录：
- 训练：`outputs/diffusion_v1_c8`
- 评估：`outputs/eval_diffusion_v1_c8`

可覆盖（不改变参数语义）：

```sh
EPOCHS=300 EPISODES=20 T_INF=20 SAMPLER=ddim OBS_BACKBONE=c8 ROT_PAIR_DIM=-1 sh ./run_transformer_diffusion.sh
```

### 训练入口（统一）

`src/train_bcdiffusion.py` 支持统一骨干开关：

- `--obs-backbone mlp`：普通 MLP 条件编码
- `--obs-backbone c4`：C4 离散旋转不变
- `--obs-backbone c8`：C8 离散旋转不变
- `--obs-backbone se2`：连续参数化 SE(2) steerable
- `--obs-backbone harmonic`：复数谐波 Fourier 特征

相关参数：
- `--rot-pair-dim`：前缀 `(x,y)` 成对向量维度（`-1` 自动）
- `--harmonic-order`：仅 harmonic 生效，谐波最高阶数 `M`

### 常用训练命令

```powershell
# v1 baseline
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion --epochs 300 --batch-size 512 --lr 1e-4

# v2 / v3 风格（非时序，强配置）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_v2 --epochs 500 --batch-size 512 --lr 1e-4 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 450

# temporal diffusion（Transformer）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_temporal_run1 --epochs 500 --batch-size 512 --lr 1e-4 --temporal --seq-len 8 --tf-layers 2 --tf-heads 4 --tf-dropout 0.1 --router-hidden 128 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 450

# v1 + C8（当前默认推荐）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_v1_c8 --epochs 500 --batch-size 512 --lr 1e-4 --obs-backbone c8 --rot-pair-dim -1 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 450

# v1 + SE2（连续等变骨干）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_v1_se2_e200 --epochs 200 --batch-size 512 --lr 1e-4 --obs-backbone se2 --rot-pair-dim -1 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 180

# v1 + harmonic（复数谐波骨干）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_v1_harmonic_e200 --epochs 200 --batch-size 512 --lr 1e-4 --obs-backbone harmonic --rot-pair-dim -1 --harmonic-order 4 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 180
```

### 评估命令模板（所有 diffusion 版本通用）

```powershell
python src\eval_policy.py `
  --algo diffusion `
  --ckpt .\outputs\<your_model_dir>\swa.pt `
  --episodes 100 `
  --output-dir .\outputs\<your_eval_dir> `
  --max-steps 400 `
  --sampler ddim `
  --T-inf 20 `
  --eta 0.0 `
  --save-gif `
  --gif-episodes 3
```

> 如果目录里没有 `swa.pt`，把 `--ckpt` 改成 `best.pt`。

### 当前已跑训练的 best_val_loss（便于横向看收敛）

| 训练目录 | best_val_loss |
|---|---:|
| `outputs/diffusion` | 0.0816 |
| `outputs/diffusion_v2` | 0.0409 |
| `outputs/diffusion_v3` | 0.0427 |
| `outputs/diffusion_temporal_latest` | 0.0772 |
| `outputs/diffusion_v1_c8` | 0.0507 |
| `outputs/diffusion_v1_se2_e200` | 0.0534 |
| `outputs/diffusion_v1_harmonic_e200` | 0.0869 |

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
