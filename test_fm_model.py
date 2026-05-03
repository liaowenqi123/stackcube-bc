import sys
sys.path.insert(0, "src")
import torch
print("Importing models_fm...")

from models_fm import BCFlowMatching, FlowVelocityPredictor
print("BCFlowMatching imported OK")

# Quick forward test
model = BCFlowMatching(
    obs_dim=48,
    act_dim=8,
    T=100,
    hidden=256,
    depth=4,
)
print("Model created OK")

x = torch.randn(4, 48)
y = torch.randn(4, 8)
loss = model(x, y)
print(f"Forward pass loss: {loss.item():.6f}")

# Sample test
with torch.no_grad():
    act = model.sample(x, T_solve=20)
print(f"Sample output shape: {act.shape}")

print("\nAll tests passed!")
