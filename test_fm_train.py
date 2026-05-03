import sys
sys.path.insert(0, "src")

print("Loading libraries...")
from common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models_fm import BCFlowMatching, EMA

print("Libraries loaded OK")

# Load data
data = load_npz_dataset("data/processed/stackcube_rl_state.npz")
x_raw = data["obs"].astype(np.float32)
y_raw = data["acts"].astype(np.float32)
print(f"Data loaded: obs shape {x_raw.shape}, acts shape {y_raw.shape}")

# Split
train_idx, val_idx = split_idx(len(x_raw), val_ratio=0.1, seed=42)
train_mask = np.zeros(len(x_raw), dtype=bool)
train_mask[train_idx] = True
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Norm
obs_norm = RunningNormalizer(const_thresh=1e-3)
obs_norm.fit(x_raw[train_mask])
act_norm = RunningNormalizer(const_thresh=0.0)
act_norm.fit(y_raw[train_mask])
x = obs_norm.transform(x_raw)
y = act_norm.transform(y_raw)
print("Norm fitted OK")

# Model
model = BCFlowMatching(
    obs_dim=x.shape[1],
    act_dim=y.shape[1],
    T=100,
    hidden=256,
    depth=4,
    obs_backbone="mlp",
    residual_film=True,
    cfg_strength=1.0,
).to("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model created OK on {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Optimizer
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Optimizer created OK")

# One training step
x_train = torch.from_numpy(x[train_idx])
y_train = torch.from_numpy(y[train_idx])
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512, shuffle=True, drop_last=True)
batch = next(iter(loader))
xb, yb = batch[0].to(model.device if hasattr(model, 'device') else 'cpu'), batch[1].to(model.device if hasattr(model, 'device') else 'cpu')
loss = model(xb, yb)
print(f"Loss: {loss.item():.6f}")
loss.backward()
opt.step()
print("One training step OK!")

print("\nAll steps passed!")
