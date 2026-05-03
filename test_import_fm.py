import sys
sys.path.insert(0, 'src')
print("Testing import...", flush=True)
try:
    from eval_policy_fm import main
    print("Import OK", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
