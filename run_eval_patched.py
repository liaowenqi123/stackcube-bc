import os
import sys

# 打补丁：Windows兼容性
if not hasattr(os, 'getuid'):
    os.getuid = lambda: 1000

# 执行评估
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 现在导入torch（安全了）
import torch
print(f'Torch {torch.__version__} loaded')

# 运行评估
exec(open(os.path.join(os.path.dirname(__file__), 'eval_baseline.py'), encoding='utf-8').read())
