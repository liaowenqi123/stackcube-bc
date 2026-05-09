import torch
ckpt=torch.load('outputs/diffusion_consistency_v1/best.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('algo:', ckpt.get('algo','N/A'))
print('obs_dim:', ckpt.get('obs_dim','N/A'))
print('act_dim:', ckpt.get('act_dim','N/A'))