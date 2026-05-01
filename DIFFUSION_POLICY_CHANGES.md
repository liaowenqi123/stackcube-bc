# Diffusion Policy 改动说明

> 基于 Chi et al. *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"* RSS 2023  
> 论文链接：https://arxiv.org/abs/2303.04137

---

## 一、为什么要用 Diffusion Policy？

原始 BC-MLP 基线的核心瓶颈是**损失函数与多峰值动作分布不兼容**：

| 问题 | 表现 |
|------|------|
| MSE 损失取均值 | 面对相同 obs 下存在多种合理动作时，网络学到的是所有可能动作的"平均"，输出退化的中间值 |
| 无时序记忆 | MLP 每步独立预测，不知道任务处于哪个阶段（移动 / 抓取 / 放置） |
| 复合误差积累 | BC 推理时的误差会随步骤逐渐放大，越到后期越容易失控 |

Diffusion Policy 把动作预测问题重新表述为**条件去噪**：从高斯噪声出发，迭代恢复真实动作分布，天然支持多峰值，并且不会把多种合理动作"平均"掉。

---

## 二、新增 / 修改的文件一览

```
src/
├── models.py            ← 修改：新增 SinusoidalPosEmb / NoisePredictor / BCDiffusion
├── train_bcdiffusion.py ← 新增：Diffusion Policy 专用训练脚本
└── eval_policy.py       ← 修改：支持 --algo diffusion
```

---

## 三、`src/models.py` 改动详解

### 3.1 新增 `SinusoidalPosEmb`

```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int): ...
    def forward(self, t: torch.Tensor) -> torch.Tensor: ...
```

**作用**：将整数时间步 `t`（0 ~ T-1）编码为连续向量，供去噪网络感知当前处于扩散过程的哪一步。  
**原理**：与 Transformer 的正弦位置编码完全相同：

$$\text{emb}(t)_i = \begin{cases} \sin(t / 10000^{i/(d/2)}) & i < d/2 \\ \cos(t / 10000^{(i-d/2)/(d/2)}) & i \geq d/2 \end{cases}$$

---

### 3.2 新增 `NoisePredictor`（条件去噪网络 ε_θ）

```python
class NoisePredictor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, t_dim=64, depth=4): ...
    def forward(self, noisy_act, t, obs) -> torch.Tensor: ...
```

**结构**：MLP + **FiLM 条件注入**（Feature-wise Linear Modulation）

```
obs ──→ obs_emb (2层MLP+LN) ──┐
                               ├─→ cond = [obs_emb ‖ t_emb]
t   ──→ t_emb (Sinusoidal+MLP)┘
                                     ↓ 每层 FiLM
noisy_act ──→ [Linear+LN → γ·x+β → GELU] × depth ──→ ε_pred
```

**FiLM 的作用**：FiLM 让每一层的 scale（γ）和 shift（β）都由 `(obs, t)` 动态决定，比简单 concatenate 条件更有效地注入全局信息，是 Diffusion Policy 常见的条件注入方式。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden` | 256 | 隐层宽度 |
| `t_dim`  | 64  | 时间步编码维度 |
| `depth`  | 4   | FiLM-MLP 层数 |

---

### 3.3 新增 `BCDiffusion`（DDPM 封装）

```python
class BCDiffusion(nn.Module):
    def __init__(self, obs_dim, act_dim, T=100, beta_min=1e-4, beta_max=2e-2, hidden=256): ...
    def q_sample(self, x0, t, noise=None) -> torch.Tensor:  # 训练加噪
    def forward(self, obs, act) -> torch.Tensor:             # 训练损失
    def ddpm_sample(self, obs, T_inf=None) -> torch.Tensor:  # 推理去噪
```

#### 扩散调度

线性 β 调度（Linear Schedule）：

$$\beta_t = \text{linspace}(\beta_\text{min}, \beta_\text{max}, T), \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

所有调度参数注册为 `register_buffer`，随 `.to(device)` 自动移动，不参与梯度计算。

#### 训练前向（`forward`）

标准 DDPM **噪声预测损失**：

$$\mathcal{L} = \mathbb{E}_{t, \varepsilon \sim \mathcal{N}(0,I)} \left\| \varepsilon_\theta(\sqrt{\bar{\alpha}_t} a_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\; t,\; o) - \varepsilon \right\|^2$$

每次随机采样时间步 `t ~ Uniform[0, T)`，生成噪声动作 `x_t`，预测噪声 `ε_pred`，计算 MSE。

#### 推理去噪（`ddpm_sample`）

从 `x_T ~ N(0, I)` 出发，逐步去噪 T 步（默认 100）：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t, o)\right) + \sqrt{\beta_t}\,z, \quad z \sim \mathcal{N}(0,I)$$

最后一步（t=0）不加噪声，直接取均值。

> **注意**：推理时每步都需调用 `NoisePredictor` T=100 次，比 MLP 慢约 100 倍。如需加速，可将 `T_inf` 缩小到 10~20（DDIM 采样）——但当前实现为标准 DDPM，简单减少步数会轻微影响质量。

---

## 四、新增 `src/train_bcdiffusion.py` 详解

与 `train_bc.py` 的主要区别：

| 项目 | BC-MLP | Diffusion Policy |
|------|--------|-----------------|
| 损失函数 | `nn.MSELoss(pred_action, target_action)` | `BCDiffusion.forward(obs, action)` → DDPM 噪声预测 MSE |
| 学习率 | 1e-3 | **1e-4**（Diffusion 模型梯度更敏感，需要小 lr）|
| LR Schedule | CosineAnnealing | **Warmup(10%) + Cosine**（step-level 调度，更平滑） |
| 模型输出 | 直接动作 | 噪声预测（推理时需去噪采样） |
| 推荐 epoch | 300 | 300（相同即可，loss 含随机性会有波动）|

关键代码对比：

```python
# train_bc.py（旧）
loss = nn.MSELoss()(model(obs), target_action)

# train_bcdiffusion.py（新）
loss = model(obs, target_action)   # BCDiffusion.forward → DDPM loss
```

Checkpoint 中额外保存扩散超参，供 eval 时恢复模型结构：

```python
torch.save({
    "model":    model.state_dict(),
    "obs_dim":  ..., "act_dim":  ...,
    "algo":     "diffusion",
    "T":        args.T,          # ← 新增
    "beta_min": args.beta_min,   # ← 新增
    "beta_max": args.beta_max,   # ← 新增
    "hidden":   args.hidden,     # ← 新增
    "obs_norm": ..., "act_norm": ...,
}, best_path)
```

---

## 五、`src/eval_policy.py` 改动详解

三处改动，均为向后兼容（不影响原有 `bc` / `bcrnn` 的使用）：

### 5.1 新增 import

```python
# 改动前
from models import BCMLP, BCRNN

# 改动后
from models import BCMLP, BCRNN, BCDiffusion
```

### 5.2 `--algo` 可选项增加 `diffusion`

```python
# 改动前
p.add_argument("--algo", choices=["bc", "bcrnn"], required=True)

# 改动后
p.add_argument("--algo", choices=["bc", "bcrnn", "diffusion"], required=True)
```

### 5.3 模型加载分支

```python
# 改动前
if args.algo == "bc":
    model = BCMLP(obs_dim, act_dim).to(device)
else:
    model = BCRNN(obs_dim, act_dim).to(device)

# 改动后
if args.algo == "bc":
    model = BCMLP(obs_dim, act_dim).to(device)
elif args.algo == "bcrnn":
    model = BCRNN(obs_dim, act_dim).to(device)
else:  # diffusion
    model = BCDiffusion(
        obs_dim  = obs_dim,
        act_dim  = act_dim,
        T        = int(ckpt.get("T",        100)),
        beta_min = float(ckpt.get("beta_min", 1e-4)),
        beta_max = float(ckpt.get("beta_max", 2e-2)),
        hidden   = int(ckpt.get("hidden",   256)),
    ).to(device)
```

### 5.4 推理循环分支

```python
# 改动前
if args.algo == "bc":
    at = model(ot).squeeze(0)
else:
    at, h = model(ot.unsqueeze(1), h0=h)
    at = at.squeeze(0).squeeze(0)

# 改动后
if args.algo == "bc":
    at = model(ot).squeeze(0)
elif args.algo == "bcrnn":
    at, h = model(ot.unsqueeze(1), h0=h)
    at = at.squeeze(0).squeeze(0)
else:  # diffusion
    at = model.ddpm_sample(ot).squeeze(0)   # DDPM 去噪采样，返回归一化动作
```

反归一化逻辑（`action = at * std + mean`）完全复用，无需改动。

---

## 六、完整训练 + 评估命令

```powershell
# 1. 训练 Diffusion Policy（约 300 epoch，GPU 下 ~30 分钟）
python src\train_bcdiffusion.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --outdir .\outputs\diffusion `
    --epochs 300 `
    --batch-size 512 `
    --lr 1e-4

# 2. 评估
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\diffusion\best.pt `
    --episodes 100 `
    --output-dir .\outputs\eval_diffusion `
    --max-steps 400 `
    --save-gif `
    --gif-episodes 3
```

---

## 七、超参调优建议

| 超参 | 默认值 | 调优方向 |
|------|--------|---------|
| `--T` | 100 | 可降到 50 节省训练时间；推理慢时可试 DDIM |
| `--hidden` | 256 | GPU 内存足够时可调到 512 |
| `--lr` | 1e-4 | 如 loss 不下降可试 3e-4 |
| `--beta-max` | 2e-2 | 动作方差大时可略微调高到 3e-2 |
| `--epochs` | 300 | 观察 val_loss 曲线，收敛后可提前停止 |

---

## 八、预期效果

| 方法 | 成功率（100 ep eval） |
|------|----------------------|
| BC-MLP（基线）| **76%** |
| BC-RNN（seq_len=16）| ~80% |
| **Diffusion Policy（本实现）** | **预期 85~92%** |

---

## 九、依赖变更

**无新增 pip 依赖**，本实现完全基于 PyTorch，不需要 `diffusers` 库。
