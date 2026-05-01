# Diffusion Policy v2 改动说明

## 目标
将 StackCube-v1 叠放任务成功率从 **86% 提升到 ≥90%**（50 episodes → 100 episodes 评估）。

## 改动文件
- `src/models.py` — 模型架构升级
- `src/train_bcdiffusion.py` — 训练流程升级
- `src/eval_policy.py` — 推理流程升级
- `eval_wrapper.py` — 评估脚本

---

## 一、模型架构改进（models.py）

### 1. EMA 权重平均（Exponential Moving Average）
**动机**：Diffusion 模型的随机性使得单个 checkpoint 可能不是最优。EMA 对训练过程中权重做指数平滑，得到更稳定的推理权重。

```python
class EMA:
    decay: float = 0.999  # 每次更新：shadow = 0.999*shadow + 0.001*current
```
- 训练时：每步更新 shadow weights
- 保存时：用 EMA shadow 权重覆盖主模型再保存
- 预期提升：2~4%

### 2. NoisePredictor 增大容量
| 参数 | v1 | v2 |
|------|-----|-----|
| hidden | 256 | **384** |
| depth（层数） | 4 | **6** |
| 参数量 | ~0.5M | ~1.2M |

更大的模型能更好地拟合 ~40K 步数据中的复杂动作分布。

### 3. DDIM 采样（Denosing Diffusion Implicit Models）
DDIM 是一种比 DDPM 更高效的采样方法，可以用更少步数（T_inf=10~20）获得同等或更好的质量。

```python
# DDPM（100步，完全随机）：86%
# DDIM（20步，确定性）：  预期 87~89%
# DDIM（10步，eta=0.1）： 预期 86~88%
```

### 4. Cosine Beta Schedule
将线性 β 调度替换为 cosine 调度（Nichol & Dhariwal 2021），使噪声在中间步骤衰减更平滑。

---

## 二、训练流程改进（train_bcdiffusion.py）

### 1. Action Noise Augmentation
训练时对 action 加少量高斯噪声（std=0.01），相当于对数据做了平滑正则化，提升鲁棒性。

### 2. 更长训练（500 epochs）
- v1：300 epochs，val_loss 在 0.082 波动
- v2：**500 epochs**，val_loss 预期降至 0.065~0.075

### 3. SWA（Stochastic Weight Averaging）
在训练最后 50 epochs（第 450~500）累加权重平均，进一步提升泛化能力。

### 4. 更强正则化
- weight_decay: 1e-5 → 1e-4
- 梯度裁剪：1.0

---

## 三、推理改进（eval_policy.py）

### 1. Action Clipping
对反归一化后的动作做裁剪，防止异常动作输出：
```python
action = np.clip(action, -0.5, 0.5)
```
预期提升：1~2%（减少极端错误动作）

### 2. 可选 DDIM 采样
通过 `--sampler ddim --T-inf 20` 启用 DDIM。

---

## 四、预期提升来源汇总

| 改进 | 预期提升 | 原理 |
|------|---------|------|
| EMA 权重平均 | +2~4% | 更稳定的推理权重 |
| 模型容量增大 | +1~2% | 减少欠拟合 |
| 更长训练 | +1~2% | 充分收敛 |
| Action Clipping | +1~2% | 减少极端错误 |
| DDIM 采样 | +0~1% | 更精确的去噪路径 |
| Cosine Schedule | +0.5% | 更平滑的噪声衰减 |
| **总计** | **~86% → 90~93%** | |

---

## 五、其他优化方法对比（成功率 < 90%）

以下方法可作为 ablation study 或对比基线：

### A. 比 Diffusion Policy 更差的方法

| 方法 | 预期成功率 | 说明 |
|------|-----------|------|
| BC-MLP（基线） | ~76% | 无时序建模能力 |
| BC-RNN | ~78~82% | 有时序能力但容量有限 |
| Decision Transformer | ~72~78% | 自回归方式，需要更长序列 |
| CVAE Policy | ~73~78% | 生成式但缺乏 Diffusion 的多样性覆盖 |
| Regularized BC（KL散度） | ~75~79% | 正则化防止过拟合但效果有限 |

### B. 比当前 Diffusion v2 差一点的方法

| 方法 | 预期成功率 | 说明 |
|------|-----------|------|
| Diffusion v1 (300ep, hidden=256) | ~84~86% | 当前基线 |
| Diffusion v1 + EMA | ~86~88% | 单独加 EMA 的效果 |
| Diffusion + 减少 T（从100→20） | ~83~85% | DDPM 步数减少导致精度下降 |
| 纯 DDIM（T=10） | ~85~87% | 步数过少，去噪不充分 |

### C. 可能超越 Diffusion 的方法（需要更多工程）

| 方法 | 预期成功率 | 难度 |
|------|-----------|------|
| Diffusion + Ensemble（3个模型投票） | ~88~92% | 高（需训练多个模型） |
| Diffusion + 上位机视觉（RGB输入） | ~90~95% | 极高（需改环境/图像处理） |
| 3D Diffusion（增加位移先验） | ~89~92% | 高（需修改架构） |
| Reinforcement Learning（在线微调） | ~92~96% | 极高（需要在线交互） |

---

## 六、训练命令

```powershell
# 训练 Diffusion Policy v2
python src\train_bcdiffusion.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --outdir .\outputs\diffusion_v2 `
    --epochs 500 `
    --batch-size 512 `
    --lr 1e-4 `
    --hidden 384 `
    --depth 6 `
    --scheduler cosine `
    --ema-decay 0.999 `
    --action-noise-std 0.01 `
    --weight-decay 1e-4

# 评估（DDPM 100步）
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\diffusion_v2\best.pt `
    --sampler ddpm `
    --T-inf 100 `
    --action-clip 0.5 `
    --episodes 100 `
    --output-dir .\outputs\eval_diffusion_v2_100ep

# 评估（DDIM 20步，快2倍）
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\diffusion_v2\best.pt `
    --sampler ddim `
    --T-inf 20 `
    --eta 0.0 `
    --action-clip 0.5 `
    --episodes 100 `
    --output-dir .\outputs\eval_diffusion_v2_100ep_ddim20
```
