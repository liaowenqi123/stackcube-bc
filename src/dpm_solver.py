"""
DPM-Solver implementation for Diffusion Policy.

DPM-Solver: A unified framework for fast ODE-based sampling in diffusion models.
It achieves better results with fewer steps compared to DDIM.

Reference: 
- DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling
  (https://arxiv.org/abs/2206.00927)
- DPM-Solver++: Fast Sampling of Diffusion Models with SDE->ODE Improvement
  (https://arxiv.org/abs/2211.01095)
"""

import torch
import numpy as np


def dpm_solver_sample(
    model,
    obs: torch.Tensor,
    T_inf: int = 20,
    order: int = 2,
    return_intermediate: bool = False,
):
    """
    DPM-Solver sampling for Diffusion Policy.
    
    Args:
        model: BCDiffusion model with noise_predictor
        obs: (B, obs_dim) or (B, seq_len, obs_dim) for temporal model
        T_inf: number of inference steps
        order: solver order (1, 2, or 3)
        return_intermediate: whether to return intermediate states
        
    Returns:
        action: (B, act_dim)
    """
    device = obs.device
    B = obs.shape[0]
    act_dim = model.act_dim
    
    # Get diffusion parameters
    T = model.T
    beta_min = model.beta_min
    beta_max = model.beta_max
    
    # Compute alpha bars
    betas = torch.linspace(beta_min, beta_max, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Time discretization for DPM-Solver
    # Use log-linear spacing (better for DPM-Solver)
    t_start = 1.0 / T
    t_end = 1.0
    ts = torch.linspace(t_start, t_end, T_inf + 1, device=device)
    
    # Initialize with noise
    x = torch.randn(B, act_dim, device=device)
    
    # DPM-Solver sampling
    intermediate = []
    
    for i in range(T_inf):
        t_cur = ts[i]
        t_next = ts[i + 1]
        
        # Convert to continuous time
        t_cur_cont = t_cur * T
        t_next_cont = t_next * T
        
        # Get model prediction
        if hasattr(model, 'temporal') and model.temporal:
            t_input = torch.full((B,), t_cur_cont, device=device)
            noise_pred = model.noise_predictor(obs, x, t_input)
        else:
            t_input = torch.full((B,), t_cur_cont, device=device)
            noise_pred = model.noise_predictor(x, t_input, obs)
        
        # DPM-Solver update
        if order == 1:
            # 1st order (Euler)
            x = dpm_solver_step_1st(
                model, x, noise_pred, t_cur, t_next, alpha_bars, T
            )
        elif order == 2:
            # 2nd order
            x = dpm_solver_step_2nd(
                model, x, noise_pred, t_cur, t_next, alpha_bars, T, obs
            )
        else:
            # 3rd order (not fully implemented, fallback to 2nd)
            x = dpm_solver_step_2nd(
                model, x, noise_pred, t_cur, t_next, alpha_bars, T, obs
            )
        
        if return_intermediate:
            intermediate.append(x.clone())
    
    if return_intermediate:
        return x, intermediate
    return x


def dpm_solver_step_1st(x, noise_pred, t_cur, t_next, alpha_bars, T):
    """1st order DPM-Solver step (Euler)."""
    # Simplified 1st order step
    alpha_bar_cur = get_alpha_bar(alpha_bars, t_cur, T)
    alpha_bar_next = get_alpha_bar(alpha_bars, t_next, T)
    
    # Predict x0 from noise prediction
    x0_pred = (x - torch.sqrt(1.0 - alpha_bar_cur) * noise_pred) / torch.sqrt(alpha_bar_cur)
    
    # Update
    x_next = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * noise_pred
    
    return x_next


def dpm_solver_step_2nd(model, x, noise_pred, t_cur, t_next, alpha_bars, T, obs):
    """2nd order DPM-Solver step."""
    B = x.shape[0]
    device = x.device
    
    # Get alpha bars
    alpha_bar_cur = get_alpha_bar(alpha_bars, t_cur, T)
    alpha_bar_next = get_alpha_bar(alpha_bars, t_next, T)
    alpha_bar_mid = get_alpha_bar(alpha_bars, (t_cur + t_next) / 2.0, T)
    
    # Predict x0
    x0_pred = (x - torch.sqrt(1.0 - alpha_bar_cur) * noise_pred) / torch.sqrt(alpha_bar_cur)
    
    # 2nd order correction
    t_mid = (t_cur + t_next) / 2.0
    t_mid_cont = t_mid * T
    
    # Get prediction at midpoint
    x_mid = torch.sqrt(alpha_bar_mid) * x0_pred + torch.sqrt(1.0 - alpha_bar_mid) * noise_pred
    
    if hasattr(model, 'temporal') and model.temporal:
        t_input = torch.full((B,), t_mid_cont, device=device)
        noise_pred_mid = model.noise_predictor(obs, x_mid, t_input)
    else:
        t_input = torch.full((B,), t_mid_cont, device=device)
        noise_pred_mid = model.noise_predictor(x_mid, t_input, obs)
    
    # 2nd order update
    x0_pred_mid = (x_mid - torch.sqrt(1.0 - alpha_bar_mid) * noise_pred_mid) / torch.sqrt(alpha_bar_mid)
    x_next = torch.sqrt(alpha_bar_next) * x0_pred_mid + torch.sqrt(1.0 - alpha_bar_next) * noise_pred_mid
    
    return x_next


def get_alpha_bar(alpha_bars, t, T):
    """Get alpha_bar at continuous time t."""
    idx = torch.clamp((t * T - 1).long(), 0, T - 1)
    return alpha_bars[idx]
