import sys
sys.path.insert(0, "src")
import torch
from models_fm import BCFlowMatching

# Test loading CRD checkpoint (to see if eval would work)
print("Testing model loading...")

# Load the FM model to test
ckpt_path = "outputs/diffusion_consistency_v1/best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"CRD Checkpoint keys: {list(ckpt.keys())}")
print(f"algo: {ckpt.get('algo', 'N/A')}")
print(f"obs_dim: {ckpt.get('obs_dim', 'N/A')}")
print(f"act_dim: {ckpt.get('act_dim', 'N/A')}")
print("CRD checkpoint loaded OK")

# Test FM model import
from models import BCDiffusionCRD
model = BCDiffusionCRD(
    obs_dim=47,  # from checkpoint
    act_dim=8,
    T=int(ckpt.get("T", 100)),
    hidden=int(ckpt.get("hidden", 256)),
    depth=int(ckpt.get("depth", 4)),
    scheduler=str(ckpt.get("scheduler", "cosine")),
    obs_backbone=str(ckpt.get("obs_backbone", "mlp")),
    rot_pair_dim=int(ckpt["rot_pair_dim"]) if ckpt.get("rot_pair_dim") is not None else None,
    harmonic_order=int(ckpt.get("harmonic_order", 4)),
    residual_film=bool(ckpt.get("residual_film", True)),
    consistency_weight=float(ckpt.get("consistency_weight", 0.1)),
    cfg_strength=float(ckpt.get("cfg_strength", 1.0)),
)
model.load_state_dict(ckpt["model"])
model.eval()
print("BCDiffusionCRD model loaded and eval-ok")

# Quick forward pass test
x = torch.randn(2, 47)
t = torch.randint(0, 100, (2,))
a = torch.randn(2, 8)
with torch.no_grad():
    out = model(x, a)
print(f"Forward pass output shape: {out.shape}, value: {out.mean().item():.4f}")

print("\nAll tests passed!")
