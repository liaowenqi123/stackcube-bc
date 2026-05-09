import sys
import os

# Monkey-patch os.getuid for Windows compatibility
if not hasattr(os, 'getuid'):
    os.getuid = lambda: 1000  # Dummy UID

# Disable torch dynamo/inductor
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Now import torch
import torch
print(f"PyTorch {torch.__version__} loaded successfully!")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run training
from train_bcdiffusion import main

if __name__ == '__main__':
    main()
