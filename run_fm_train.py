import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Write to log file
log_file = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_log.txt", "w")
out_file = "d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_train_out.txt"
err_file = "d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_train_err.txt"

import torch
import numpy as np

# Patch print to also write to log
original_print = print
def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    file = kwargs.get("file", None)
    flush = kwargs.get("flush", False)
    msg = sep.join(str(a) for a in args)
    log_file.write(msg + "\n")
    if flush:
        log_file.flush()
    original_print(*args, **kwargs, file=log_file, flush=flush)

print("=== FM Training Log ===")
print(f"Python: {sys.version}")

from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
from torch.utils.data import DataLoader, TensorDataset
from models_fm import BCFlowMatching, EMA

print("Imports OK")

# Load data
data = load_npz_dataset("data/processed/stackcube_rl_state.npz")
x_raw = data["obs"].astype(np.float32)
y_raw = data["acts"].astype(np.float32)

train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=42)
train_mask = np.zeros(len(x_raw), dtype=bool)
train_mask[train_idx] = True

obs_norm = RunningNormalizer(const_thresh=1e-3)
obs_norm.fit(x_raw[train_mask])
act_norm = RunningNormalizer(const_thresh=0.0)
act_norm.fit(y_raw[train_mask])

x = obs_norm.transform(x_raw)
y = act_norm.transform(y_raw)
x_train = torch.from_numpy(x[train_idx])
y_train = torch.from_numpy(y[train_idx])

print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val")
print(f"obs={x.shape[1]}, act={y.shape[1]}")

device = select_device()
print(f"Device: {device}")

model = BCFlowMatching(
    obs_dim=x.shape[1],
    act_dim=y.shape[1],
    T=100,
    hidden=256,
    depth=4,
    obs_backbone="mlp",
    residual_film=True,
    cfg_strength=1.0,
).to(device)

ema = EMA(model, decay=0.999, device=device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
total_steps = 200 * (len(train_idx) // 512)
warmup_steps = int(0.1 * total_steps)

import math
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 1/20) + 1/20

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512, shuffle=True, drop_last=True, num_workers=0)

print("Starting training...")

best = float("inf")
best_path = ensure_dir("outputs/flow_matching_v1") / "best.pt"

for epoch in range(1, 201):
    model.train()
    train_loss = 0.0
    n_samples = 0

    for batch in train_loader:
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        ema.update()
        train_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    train_loss /= n_samples

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch={epoch:03d}  train={train_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        log_file.flush()

    if train_loss < best:
        best = train_loss
        ema.apply_shadow()
        torch.save({
            "model": model.state_dict(),
            "obs_dim": x.shape[1],
            "act_dim": y.shape[1],
            "algo": "flow_matching_v1",
            "T": 100,
            "hidden": 256,
            "depth": 4,
            "obs_backbone": "mlp",
            "obs_norm": obs_norm.state_dict(),
            "act_norm": act_norm.state_dict(),
            "ema_decay": 0.999,
        }, best_path)
        ema.restore()

print(f"Training done! best={best:.6f}")
log_file.close()
print("Log saved to fm_log.txt")