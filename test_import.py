import sys, os
log = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/test_log.txt", "w")
sys.stdout = log
sys.stderr = log
print("Step 1: imports")
import torch
print(f"Step 2: torch={torch.__version__}")
import numpy as np
print("Step 3: numpy imported")
import math
print("Step 4: math imported")
from torch.utils.data import DataLoader, TensorDataset
print("Step 5: DataLoader imported")
from src.common import RunningNormalizer, ensure_dir, load_npz_dataset, select_device, split_idx
print("Step 6: common imported")
from src.models_fm import BCFlowMatching, EMA
print("Step 7: models_fm imported")
print("All imports OK!")
log.close()
