"""打补丁版的Entity模型训练包装"""
import os
if not hasattr(os, 'getuid'):
    os.getuid = lambda: 1000

import sys
sys.path.insert(0, 'src')
import torch  # 先加载torch（触发pwd补丁），避免后续崩溃
print(f'Torch {torch.__version__} loaded')

from train_entity import main

if __name__ == '__main__':
    main()
