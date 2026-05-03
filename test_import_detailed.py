import sys
print("Step 1: Basic imports...", flush=True)

import argparse
import json
from collections import deque
from pathlib import Path
print("  Basic imports OK", flush=True)

print("Step 2: NumPy...", flush=True)
import numpy as np
print("  NumPy OK", flush=True)

print("Step 3: PyTorch...", flush=True)
import torch
print("  PyTorch OK", flush=True)

print("Step 4: Gymnasium...", flush=True)
import gymnasium as gym
print("  Gymnasium OK", flush=True)

print("Step 5: Matplotlib...", flush=True)
import matplotlib.pyplot as plt
print("  Matplotlib OK", flush=True)

print("Step 6: imageio...", flush=True)
import imageio.v2 as imageio
print("  imageio OK", flush=True)

print("Step 7: mani_skill...", flush=True)
import mani_skill.envs
print("  mani_skill OK", flush=True)

print("Step 8: common...", flush=True)
sys.path.insert(0, 'src')
from common import RunningNormalizer, select_device
print("  common OK", flush=True)

print("Step 9: models_fm...", flush=True)
from models_fm import BCFlowMatching, BCFlowMatchingWithNoise
print("  models_fm OK", flush=True)

print("All imports successful!", flush=True)
