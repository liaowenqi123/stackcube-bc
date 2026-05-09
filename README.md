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
| diffusion v3 Koopman | D3P风格：DKO模块+双分支条件去噪+test-time loss融合 | `outputs/diffusion_v3_koopman` |
| temporal diffusion | `--temporal` + Transformer 时序分支 | `outputs/diffusion_temporal_*` |
| v1 + C4/C8 | 非时序 v1 + 离散旋转不变骨干 | `outputs/diffusion_v1_c8`（默认脚本） |
| v1 + SE2 | 非时序 v1 + 连续参数化 SE(2) steerable 骨干 | `outputs/diffusion_v1_se2_e200` |
| v1 + harmonic | 非时序 v1 + 复数谐波 Fourier 特征骨干 | `outputs/diffusion_v1_harmonic_e200` |
| CQE Diffusion（新） | 上下文条件“软等变”建模（非严格不变/等变） | `outputs/diffusion_cqe_v1` |

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

### CQE Diffusion（创新版，不改原 v1）

为了不改动原始 `diffusion v1`，新增了独立实现：
- `src/models_cqe.py`
- `src/train_bcdiffusion_cqe.py`
- `src/eval_policy_cqe.py`
- `run_cqe_diffusion.sh`

核心思路：同一状态下同时建模绝对分支（机械臂偏置相关）和相对几何分支，并学习一个条件化群作用器 `T_phi(h, g)`，用软一致性损失约束“旋转平移后仍有关联”，但不强行要求严格等变。

一键运行：

```sh
sh ./run_cqe_diffusion.sh
```

常用命令：

```powershell
# 训练
python src\train_bcdiffusion_cqe.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_cqe_v1 --epochs 300 --batch-size 512 --lr 1e-4 --rot-pair-dim 16 --trans-pairs 2 --sym-lambda 0.1 --id-lambda 0.05

# 评估
python src\eval_policy_cqe.py --ckpt .\outputs\diffusion_cqe_v1\swa.pt --episodes 100 --output-dir .\outputs\eval_diffusion_cqe_v1 --max-steps 400 --sampler ddim --T-inf 20 --eta 0.0 --save-gif --gif-episodes 3
```

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

# diffusion v3 Koopman（D3P风格，单行bat）
python src\train_bcdiffusion.py --dataset .\data\processed\stackcube_rl_state.npz --outdir .\outputs\diffusion_v3_koopman --epochs 500 --batch-size 512 --lr 1e-4 --koopman --seq-len 9 --hidden 384 --depth 6 --scheduler cosine --ema-decay 0.999 --action-noise-std 0.01 --swa-start 450 --latent-dim 64 --koopman-h 4 --koopman-lambda 0.1 --koopman-reg 1e-5
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

### diffusion v3 Koopman 评估（单行bat）

```powershell
python src\eval_policy.py --algo diffusion --ckpt .\outputs\diffusion_v3_koopman\swa.pt --episodes 100 --output-dir .\outputs\eval_diffusion_v3_koopman --max-steps 400 --sampler ddim --T-inf 20 --eta 0.0 --save-gif --gif-episodes 3
```

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

## 实验结果总览

### 全部方案对比

| # | 方案 | 核心思路 | 输入维度 | SR | Eps | 备注 |
|:--|------|---------|:---:|:---:|:---:|------|
| 1 | BC (MLP) | 基础MLP行为克隆 | 48 | **76%** | 100 | 基线 |
| 2 | Diffusion v1 | 基础DDPM+DDIM扩散策略 | 48 | **~70%** | 10 | |
| 3 | Diffusion v3 | 增大容量(hidden=384,depth=6)+cosine scheduler+EMA+SWA | 48 | **84%** | 50 | 🏆最佳DL |
| 4 | Diffusion v3 + Naive Aug | v3基础上拼接12维相对向量(cubeA-ee, cubeB-cubeA, goal-cubeB, ee_pos) | 48→60 | **76%** | 50 | 持平基线 |
| 5 | Diffusion v3 + SE2 Aug | v3基础上拼接10维SE(2)不变特征(6个成对x-y距离+4个z高度) | 48→58 | **68%** | 50 | 额外特征干扰 |
| 6 | Diffusion Aug v2 | 另一种相对向量拼接方案 | 48→60 | **54%** | 50 | |
| 7 | Diffusion v1 + C8骨干 | C8离散旋转不变编码器：将前pair_dim维按(x,y)成对做group average pooling | 48 | **73%** | 100 | |
| 8 | Diffusion v1 + SE2骨干 | SE(2) steerable编码器：向量分支用aI+bJ线性映射，标量分支取模长 | 48 | **58%** | 100 | |
| 9 | Diffusion v1 + Harmonic骨干 | 复数谐波Fourier编码器：z=x+iy的m阶谐波矩(Re/Im/\|.\|) | 48 | **6%** | 100 | 数值崩溃 |
| 10 | Diffusion v1 + Harmonic Fix | Harmonic修复版：添加pred_x0 clamp防止NaN | 48 | **64%** | 100 | |
| 11 | SE(2) Diffusion v3 | 独立SE(2)等变模型(成对距离+高度+FK末端)，v3训练配置 | 60 | **68%** | 50 | |
| 12 | Temporal Diffusion | 双分支：单步FiLM-MLP + Transformer时序编码器(obs序列)，路由门控融合 | 48 | **62%** | 100 | |
| 13 | CQE Diffusion | 双分支编码：绝对分支(全量obs)+相对分支(几何关系)，门控融合+可学习群作用器T_φ软等变约束 | 48 | **74%** | 100 | |
| 14 | Entity-Aware v1 | 双分支：标准NoisePredictor + TransformerEncoder(2层4头)实体Cross-Attention，门控融合 | 48 | **56%** | 50 | 5.8M参数过拟合 |
| 15 | Entity-Aware v2 | 轻量版：单层MultiheadAttention(64维2头)替代Transformer | 48 | **32%** | 50 | |
| 16 | Entity-Aware Small | 更轻量版(3.3M参数) | 48 | **58%** | 50 | |
| 17 | Flow Matching | 速度场预测替代噪声预测，条件流匹配+OT路径+ODE推理 | 48 | **0%** | 100 | 完全失败 |
| 18 | **PID Controller** | 6DOF IK(数值Jacobian)+姿态控制(TCP垂直向下)+分阶段抓取 | - | **82%** | - | 经典控制 |

### 关键发现

1. **最佳DL模型：Diffusion v3 = 84%**（hidden=384, depth=6, cosine scheduler, EMA, SWA）
2. **经典PID控制 = 82%**，接近最佳DL模型，说明StackCube任务结构性强，经典方法即可高效求解
3. **所有等变/不变性尝试均未超过v3**——机械臂本身不是旋转对称的，等变约束反而限制了模型表达能力
4. **输入维度增强(48→60)无收益**：无论拼接相对向量(12维)还是SE(2)不变特征(10维)，均不如原始48维
5. **Entity-Aware过参数化导致严重过拟合**：5.8M参数→56%，轻量化后仍不如基线
6. **Flow Matching完全失败**：连续归一化流在此任务上训练不稳定

### 观测空间布局（48维）

| 维度 | 内容 |
|------|------|
| [0:8] | qpos（7关节 + 1夹爪） |
| [8:16] | qvel |
| [16:18] | prev_action |
| [18:21] | TCP位置 (x, y, z) |
| [21:25] | TCP四元数 |
| [25:28] | cubeA位置 |
| [28:32] | cubeA四元数 |
| [32:35] | cubeB位置 |
| [35:39] | cubeB四元数 |
| [39:42] | goal位置 |
| [42:46] | goal四元数 |
| [46:48] | extra |

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
 
## 叶涵文的工作：置信度增强扩散策略 (Confidence-Enhanced Diffusion Policy)

- **定位**：在应用层面对扩散策略进行推理时安全增强，**无需修改模型结构或重新训练**。
- **方法**：
  1. **多采样**：对同一状态进行 K 次带 Dropout 的扩散去噪，得到多个候选动作；
  2. **方差估计**：计算候选动作的方差，作为模型不确定性的度量；
  3. **保守执行**：若方差超过阈值，将动作缩放至 0.7 倍，让机器人“谨慎行事”。
- **效果**：在 StackCube 任务上，成功率由 **67% 提升至 70.4%**，保守动作触发率仅 12.4%（精准干预，而非盲目保守）。
- **贡献**：与队友的模型结构改进互补，提供了一个轻量级、即插即用的不确定性感知模块，有效提升了生成式策略的安全性。

## 王潇桐的工作：State-only Diffusion Policy 在 NutAssembly 任务上的性能边界探索

- **定位**：探索 Diffusion Policy 在**仅用状态信息（state-only）**条件下能否完成精细操作任务，明确算法在该设置下的性能边界。
- **任务**：NutAssembly（螺母装配）——机器人抓取方形螺母并精确放置到 peg 上，成功要求 XY 对齐误差 < 3cm（毫米级精度）。
- **方法**：
  1. 使用 robotic 官方数据集（200 个演示，约 30K 样本），输入 44 维状态向量，输出 7 维关节控制；
  2. 对比三个版本：V1（基础 DDPM，256×4，300 epochs）、V2（增强版 384×6，cosine scheduler，EMA）、V3（V2 + SWA）；
  3. 系统化诊断：从环境配置、控制器匹配到观测维度，逐层排查。
- **实验结果**：
  - 官方条件（XY < 3cm）：三个版本成功率均为 **0%**
  - 放宽条件（XY < 15cm）：V1 达到 **20%** 成功率，平均回报 46.15
  - 详细诊断显示：模型能正确完成抓取和移动，但 XY 对齐误差约 9.7cm（要求 < 3cm），Z 位置差 2.3cm
- **原因分析**：
  1. **配置问题**（已解决）：环境名称错误、控制器不匹配、reward shaping 未启用
  2. **训练问题**（部分解决）：V2/V3 仅训练约 100 epochs，不如充分训练的 V1
  3. **算法局限**（根本原因）：State-only 信息不足，缺少视觉和历史观测，无法达到毫米级精度
- **对比文献**：原论文 Diffusion Policy（图像+历史+动作序列）可达 40-60% 成功率，state-only 设置存在固有精度天花板。
- **核心结论**：
  - ✅ 算法实现正确，训练收敛良好，模型学会了任务流程
  - ✅ 达到厘米级精度（10-15cm），放宽条件下 20% 成功率
  - ❌ 无法达到毫米级精度（< 3cm），state-only 信息不足
  - 💡 明确了算法边界，指明改进方向（添加历史观测或视觉输入）

## 实验结果分析

### NutAssembly 实验关键数据

| 实验 | 算法 | 训练样本 | Epochs | 成功率 (官方) | 平均回报 | 成功率 (放宽) | XY 误差 |
|------|------|---------|--------|--------------|---------|--------------|---------|
| V1 | Diffusion | 30,154 | 300 | 0% | **46.15** | **20%** | ~10cm |
| V2 | Diffusion 增强 | 30,154 | ~100 | 0% | - | 10% | - |
| V3 | Diffusion + SWA | 30,154 | ~100 | 0% | - | 10% | - |

### 最接近成功的诊断数据

```
XY 距离: 0.1267m (需要 < 0.03m) ❌ 差 9.67cm
Z 位置: 0.8933m (需要 < 0.87m) ❌ 高 2.33cm
r_reach: 0.33 (需要 < 0.6)     ✅ 满足
```

### 三层原因总结

1. **配置问题（已解决）**：环境名称、控制器匹配、reward shaping 等均已修复
2. **训练问题（部分解决）**：训练充分性比模型复杂度更重要——V1 跑满 300 epochs 优于 V2/V3 的 100 epochs
3. **算法局限（根本原因）**：State-only 设置下，仅凭当前 44 维状态向量无法提供足够的空间信息，毫米级精度需要视觉输入或历史观测辅助

### 核心发现

- **0% 不代表失败**，而是揭示了 State-only Diffusion Policy 在该任务上的性能边界
- 算法实现了正确性验证（训练收敛、任务流程学习），但精度受限于信息瓶颈
- 改进方向明确：添加历史观测（预期将 XY 误差从 10cm 降至 3-5cm）

## 实验任务描述：NutAssembly（螺母装配）

### 任务概述

**NutAssembly** 是 robosuite 中的一个精细操作基准任务，要求机器人从桌面抓取一个螺母并精确放置到 peg 上。

### 任务配置

| 参数 | 值 |
|------|------|
| 任务 | NutAssemblySquare |
| 环境 | robosuite 1.5.2 |
| 控制器 | BASIC (位置控制) |
| 观测维度 | 44 维（关节位置、速度、TCP 位姿、螺母位置、peg 位置等） |
| 动作维度 | 7 维（7 个关节位置指令） |
| 训练数据 | robomimic 官方 `low_dim_v15.hdf5`，200 个演示，约 30,154 个样本 |
| 评估指标 | 成功率（success rate）、平均回报（avg return）、平均 episode 长度 |

### 成功条件

```
1. XY 对齐：螺母中心到 peg 中心的水平距离 < 3cm（毫米级精度）
2. Z 位置：螺母高度低于桌面上方 5cm
3. 机械臂松开：r_reach < 0.6
```

### 与 StackCube 的对比

| 特性 | StackCube | NutAssembly |
|------|-----------|-------------|
| 平台 | ManiSkill 3.0.1 | robosuite 1.5.2 |
| 观测维度 | 48 维 | 44 维 |
| 动作维度 | 8 维（7 关节 + 夹爪） | 7 维（7 关节） |
| 精度要求 | 适中 | 毫米级（< 3cm） |
| 数据来源 | PPO 演示（932 条） | 人类演示（200 条） |
| 最佳模型 SR | 84%（Diffusion v3） | 0%（官方）/ 20%（放宽） |

## 余高迪的工作：Diffusion Policy 在 Door Opening 任务上的应用探索
- **定位**：探索 Diffusion Policy 在 robosuite Door Opening 铰链操作任务上的可行性与性能边界，系统诊断 state-only 设置在铰链物体操作中的信息瓶颈。
- **任务**：Door Opening（开门）——单臂 Panda 机械臂抓取门把手并旋转打开至 90° 以上，要求 300 步内完成。涉及铰链关节动力学建模，对策略的时序控制能力有较高要求。
- **方法**：
  1. 使用 robosuite 随机策略录制 75 条轨迹（13,704 步），输入 23 维状态向量，输出 7 维关节控制；
  2. 对比三个版本：V1（Diffusion Policy：MLP 512×3，DDPM 50 步扩散，200 epochs）、V2（BC + CNN + 增强数据集，100 epochs）、V3（BC + CNN + 去动作归一化，100 epochs）；
  3. 系统化诊断：从数据质量检查、replay 验证、归一化对齐到训练-评估状态一致性，逐层排查。
- **实验结果**：
  - 三个版本成功率均为 **0%**，平均回报 0.0
  - Diffusion V1 训练 loss 正常收敛（1.03 → 0.42），best val noise MSE = **0.348**
  - 放宽成功条件（−50% 阈值）后仍无有意义的开门动作
  - 数据 replay 检查：原始随机轨迹平均回报仅 **5.35**（0/20 成功）；human demo 平均回报 **1.25**（0/20 成功）
- **原因分析**：
  1. **数据质量问题（根本原因）**：75 条随机轨迹几乎无成功开门行为，模型从未见过有效的正样本动作序列；human demo 同样有效信息不足。训练数据本身的 replay 成功率即为 0%，策略不可能学出参数之外的开门模式。
  2. **训练问题（部分解决）**：V2/V3 仅训练约 100 epochs，未充分收敛。V1 跑满 200 epochs，但受限于底层数据质量，继续训练无法突破信息瓶颈。
  3. **模型与任务不匹配**：plain MLP 无法建模铰链关节动力学，固定 16 步 action chunk 在连续接触任务中导致误差链式累积。
  4. **评估偏移**：训练/评估状态提取方式不一致，策略看到偏移的输入分布。
- **对比文献**：原论文 Diffusion Policy（图像+历史+动作序列）在多种操作任务中达 60-80% 成功率。本实验仅用 state-only 23 维观测且无历史帧，在铰链操作中面临更大的信息缺失，差距比 NutAssembly 更显著。
- **核心结论**：
  - ✅ 算法实现正确——Diffusion Policy 训练流程完整，loss 持续收敛，前向推理和数据管道均正常工作
  - ✅ 为组内 Door Opening 任务建立了可复现的 baseline 和诊断框架
  - ❌ 成功率 0% 的根本制约是训练数据质量（75 条随机轨迹无法提供有效开门行为），而非算法实现
  - 💡 改进方向明确：收集 500+ scripted/teleop 高质量演示；引入视觉观测或历史帧输入；换用 Transformer/CNN 骨干替代 MLP

## 实验结果分析

### Door Opening 实验关键数据

| 实验 | 算法 | 训练样本 | Epochs | 成功率 | 平均回报 | Best Val Noise MSE |
|------|------|---------|--------|:------:|:--------:|:------------------:|
| V1 | Diffusion (MLP) | 13,704 | 200 | 0% | 0.0 | **0.348** |
| V2 | BC (CNN) | 增强数据 | ~100 | 0% | 0.0 | — |
| V3 | BC (CNN, no actnorm) | 13,704 | ~100 | 0% | 0.0 | — |

### 最接近成功的诊断数据

随机策略 replay 检查
Replay 成功率：0/20 = 0% 平均回报：5.35 最高单 episode 回报：49（ep 6，接近但不满足开门条件） 动作范围检查：ratio_within_-1_1 = 1.0 ✅ 动作归一化正确

Human demo replay 检查
Replay 成功率：0/20 = 0% 平均回报：1.25 最大 episode 长度：7,399 步（远超 max_steps=300，录制存在异常）

数据一致性
same_num_samples_obs_acts: ✅ sum_ep_lengths_equals_samples: ✅ 观测维度：23 维 动作维度：7 维 观测近恒定维度：0 维


### 三层原因总结

1. **数据限制（根本原因）**：75 条随机轨迹缺乏有效开门行为，replay 检查成功率即为 0%，问题在数据源头，无法通过改进模型架构解决
2. **训练问题（部分解决）**：V2/V3 仅约 100 epochs 训练不充分。但即使 V1 跑满 200 epochs，受限于底层数据质量，loss 收敛后无法突破信息瓶颈
3. **模型局限**：MLP 骨干缺乏铰链动力学建模能力，state-only + 无历史帧 的输入信息量不足以完成精细开门操作

### 核心发现

- **0% 不代表无意义**，而是揭示了 State-only Diffusion Policy 在 Door Opening 铰链操作任务上的性能边界
- 算法通过了正确性验证（训练收敛、数据管道正常），但最终性能受限于数据质量和输入信息维度
- 与王潇桐在 NutAssembly 上的发现一致：**state-only 设置存在固有精度天花板**，且铰链操作（开门）对信息量的需求更高
- 改进方向明确：高质量演示数据 + 视觉输入 + 历史观测

## 实验任务描述：Door Opening（开门）

### 任务概述

**Door Opening** 是 robosuite 中的一个铰链物体操作基准任务，要求 Panda 机械臂抓取门把手并将门旋转打开至 90° 以上。该任务测试策略在铰链动力学建模与连续接触控制方面的能力。

### 任务配置

| 参数 | 值 |
|------|------|
| 任务 | Door |
| 环境 | robosuite 1.5.2 |
| 控制器 | BASIC（位置控制） |
| 观测维度 | 23 维（关节位置、速度、TCP 位姿、门铰链角度等） |
| 动作维度 | 7 维（7 个关节位置指令） |
| 训练数据 | 随机策略录制 75 条轨迹，13,704 步 |
| 评估指标 | 成功率（success rate）、平均回报（avg return） |

### 成功条件

门开角 > 90°（从初始闭合状态开始）
机械臂在 300 步内完成动作
无碰撞或关节越限等异常终止

### 与 StackCube 的对比

| 特性 | StackCube | Door Opening |
|------|-----------|-------------|
| 平台 | ManiSkill 3.0.1 | robosuite 1.5.2 |
| 任务类型 | 堆叠方块（自由物体操作） | 开门（铰链物体操作） |
| 观测维度 | 48 维 | 23 维 |
| 动作维度 | 8 维（7 关节 + 夹爪） | 7 维（7 关节） |
| 精度要求 | 适中（堆叠对齐） | 较高（铰链轨迹连续控制） |
| 数据来源 | PPO 演示（932 条，含成功/失败混合） | 随机策略（75 条，几乎无成功样本） |
| 演示质量 | 高（专家 PPO 策略） | 低（随机探索，成功率为 0） |
| 最佳模型 SR | 84%（Diffusion v3） | 0%（全部版本） |
