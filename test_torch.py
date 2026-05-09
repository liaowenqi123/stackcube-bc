import sys
print("Testing torch import...", flush=True)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print("Torch import OK", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
