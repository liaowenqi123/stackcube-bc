# Diffusion Policy V1 使用指南

## 📋 概述

**Diffusion Policy V1** 是一种基于扩散模型（DDPM）的行为克隆算法，用于机器人控制任务。它通过学习专家演示数据，训练一个能够生成连续动作的策略网络。

### 适用任务类型

Diffusion Policy V1 特别适合以下虚拟场景：

1. **机器人操作任务**
   - ✅ 物体抓取与放置（如 StackCube 堆叠任务）
   - ✅ 精细操作（装配、插入）
   - ✅ 轨迹跟踪任务

2. **任务特点**
   - ✅ 需要平滑、连续的动作序列
   - ✅ 有专家演示数据可用
   - ✅ 状态空间维度适中（本项目：48 维观测 + 8 维动作）
   - ✅ 对动作噪声有一定鲁棒性要求

3. **当前项目：StackCube-v1**
   - 任务：控制机械臂将一个立方体堆叠到另一个立方体上
   - 控制模式：`pd_joint_delta_pos`（关节增量位置控制）
   - 成功率：**70%**（V1 模型，10 episodes 评估）

---

## 🎯 V1 模型性能

根据已有的评估结果：

| 指标 | V1 性能 |
|------|---------|
| 成功率 | 70% |
| 平均回报 | 12.3 |
| 平均步数 | 43.3 步 |
| 训练轮数 | 300 epochs |
| 模型大小 | hidden=256, depth=4 |
| 调度器 | Linear |

**注意**：仓库中还有 V2 和 V3 版本，性能更优（使用 Cosine 调度器、EMA、更大模型等改进）。

---

## 🚀 快速开始：在虚拟场景中应用 V1

### 步骤 1：环境检查

确保已安装所有依赖：

```powershell
python src\runtime_check.py
```

预期输出：`runtime check passed`

### 步骤 2：使用预训练的 V1 模型

仓库中已包含训练好的 V1 模型：`outputs/diffusion/best.pt`

**运行评估（生成可视化）：**

```powershell
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\diffusion\best.pt `
    --episodes 10 `
    --output-dir .\outputs\my_eval_v1 `
    --max-steps 400 `
    --save-gif `
    --gif-episodes 3
```

**参数说明：**
- `--algo diffusion`：使用 Diffusion Policy
- `--ckpt`：模型权重路径
- `--episodes`：评估回合数
- `--max-steps 400`：每回合最大步数（**必须设置**，否则默认 50 步会截断）
- `--save-gif`：保存前 N 个回合的 GIF 动画
- `--gif-episodes 3`：保存前 3 个回合

**输出文件：**
- `outputs/my_eval_v1/metrics.json` — 成功率、平均回报等指标
- `outputs/my_eval_v1/returns_hist.png` — 回报分布直方图
- `outputs/my_eval_v1/rollout_ep000.gif` — 可视化动画（前 3 个回合）

---

### 步骤 3：使用 DDIM 采样器（更快、更高质量）

V1 模型支持两种采样方法：

1. **DDPM**（默认）：标准扩散采样，需要 100 步
2. **DDIM**：快速采样，只需 10-20 步，质量相当或更好

**使用 DDIM 采样器：**

```powershell
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\diffusion\best.pt `
    --sampler ddim `
    --T-inf 20 `
    --eta 0.0 `
    --episodes 10 `
    --output-dir .\outputs\my_eval_v1_ddim `
    --max-steps 400 `
    --save-gif `
    --gif-episodes 3
```

**新增参数：**
- `--sampler ddim`：使用 DDIM 采样器
- `--T-inf 20`：推理步数（20 步即可，比 DDPM 的 100 步快 5 倍）
- `--eta 0.0`：确定性采样（0=完全确定，1=接近 DDPM）

---

## 🔧 自定义训练 V1 模型

如果你想在自己的数据上训练 V1 模型：

### 1. 准备数据

确保数据已转换为 `.npz` 格式（包含 `obs` 和 `acts` 数组）：

```powershell
# 如果使用 ManiSkill 演示数据
python src\convert_maniskill_demo.py `
    --input-h5 .\data\raw\demos\StackCube-v1\rl\trajectory.none.pd_joint_delta_pos.physx_cuda.h5 `
    --output-npz .\data\processed\stackcube_rl_state.npz
```

### 2. 训练 V1 模型

```powershell
python src\train_bcdiffusion.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --outdir .\outputs\my_diffusion_v1 `
    --epochs 300 `
    --batch-size 512 `
    --lr 1e-4 `
    --hidden 256 `
    --depth 4 `
    --scheduler linear `
    --ema-decay 0.0 `
    --action-noise-std 0.0
```

**V1 关键参数：**
- `--hidden 256`：隐层宽度（V1 使用 256，V2/V3 使用 384）
- `--depth 4`：网络层数（V1 使用 4，V2/V3 使用 6）
- `--scheduler linear`：线性 beta 调度（V1 特征，V2/V3 使用 cosine）
- `--ema-decay 0.0`：禁用 EMA（V1 不使用，V2/V3 使用 0.999）
- `--action-noise-std 0.0`：禁用动作噪声增强（V1 不使用）

**训练输出：**
- `outputs/my_diffusion_v1/best.pt` — 最佳模型权重
- `outputs/my_diffusion_v1/loss_curve.png` — 训练曲线
- `outputs/my_diffusion_v1/metrics.json` — 训练指标

### 3. 评估自定义模型

```powershell
python src\eval_policy.py `
    --algo diffusion `
    --ckpt .\outputs\my_diffusion_v1\best.pt `
    --episodes 100 `
    --output-dir .\outputs\eval_my_v1 `
    --max-steps 400 `
    --save-gif `
    --gif-episodes 3
```

---

## 🎨 应用到其他虚拟场景

### 适配新环境的步骤

1. **准备演示数据**
   - 收集专家演示（人工遥操作、运动规划、RL 策略等）
   - 转换为 `.npz` 格式：`obs` (N, obs_dim), `acts` (N, act_dim)

2. **调整超参数**
   - 观测维度：根据你的环境状态空间
   - 动作维度：根据你的控制空间
   - 训练轮数：根据数据量调整（300-500 epochs）

3. **训练模型**
   ```powershell
   python src\train_bcdiffusion.py `
       --dataset .\data\your_task.npz `
       --outdir .\outputs\your_task_v1 `
       --epochs 300 `
       --batch-size 512 `
       --lr 1e-4 `
       --hidden 256 `
       --depth 4 `
       --scheduler linear
   ```

4. **评估模型**
   - 修改 `eval_policy.py` 中的环境 ID：`--env-id YourEnv-v0`
   - 确保环境支持 `obs_mode="state"` 和相应的控制模式

---

## 📊 性能对比：V1 vs V2 vs V3

| 版本 | 成功率 | 关键改进 |
|------|--------|----------|
| **V1** | 70% | 基础版本（linear scheduler, 无 EMA） |
| **V2** | ~80%+ | Cosine scheduler + EMA + 更大模型 |
| **V3** | ~85%+ | V2 + SWA + 动作噪声增强 |

**建议：**
- 如果追求简单和快速验证 → 使用 V1
- 如果追求更高性能 → 使用 V2 或 V3（模型路径：`outputs/diffusion_v2/best.pt`）

---

## 🔍 常见问题

### Q1: 为什么评估时必须设置 `--max-steps 400`？
**A:** ManiSkill 环境默认 `max_episode_steps=50`，会导致回合过早截断。StackCube 任务通常需要 100-200 步才能完成。

### Q2: DDPM 和 DDIM 采样器有什么区别？
**A:** 
- **DDPM**：标准扩散采样，需要 100 步，质量稳定
- **DDIM**：快速采样，10-20 步即可，速度快 5-10 倍，质量相当或更好

### Q3: 如何提高成功率？
**A:** 
1. 使用 V2/V3 模型（性能更优）
2. 增加训练数据量
3. 使用 DDIM 采样器（`--sampler ddim --T-inf 20`）
4. 调整动作裁剪范围（`--action-clip 0.5`）

### Q4: 模型训练需要多长时间？
**A:** 
- GPU（CUDA）：约 30-60 分钟（300 epochs）
- CPU：约 3-5 小时

### Q5: 如何可视化数据集？
**A:**
```powershell
python src\visualize_dataset.py `
    --dataset .\data\processed\stackcube_rl_state.npz `
    --out .\outputs\dataset_stats.png
```

---

## 📚 参考资料

- **论文**: Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" RSS 2023
- **arXiv**: https://arxiv.org/abs/2303.04137
- **ManiSkill 文档**: https://maniskill.readthedocs.io/

---

## 🎯 下一步建议

1. **运行预训练模型**：先用 V1 模型评估，熟悉流程
2. **尝试 DDIM 采样**：体验更快的推理速度
3. **对比 V2/V3**：查看性能提升
4. **自定义训练**：在自己的数据上训练模型
5. **应用到新场景**：适配到其他机器人任务

---

**祝你使用愉快！如有问题，请查看代码注释或提 Issue。** 🚀
