import os
import sys

# Disable torch dynamo/inductor to avoid Windows compatibility issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run training
from train_bcflowmatching import main

if __name__ == '__main__':
    main()
