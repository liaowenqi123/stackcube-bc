"""
训练包装脚本：在导入torch前打Windows兼容性补丁
"""
import os
import sys

# 补丁1：os.getuid（Windows没有这个函数）
if not hasattr(os, 'getuid'):
    os.getuid = lambda: 1000

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 现在导入torch（会自动加载pwd mock模块）
import torch
print(f"torch {torch.__version__} loaded successfully")

# 导入并运行训练
from train_bcse2 import main

if __name__ == '__main__':
    main()
