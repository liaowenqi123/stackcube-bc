import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Redirect stdout/stderr to files
sys.stdout = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_eval_out.txt", "w")
sys.stderr = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_eval_err.txt", "w")

print("Starting Flow Matching evaluation...", flush=True)

# Import the evaluation module
from eval_policy_fm import main

# Set up command line arguments
sys.argv = [
    "eval_policy_fm.py",
    "--ckpt", "./outputs/flow_matching_v1/best.pt",
    "--episodes", "100",
    "--output-dir", "./outputs/eval_flow_matching_v1",
    "--max-steps", "400",
    "--solver-steps", "20",
    "--solver-method", "euler",
    "--save-gif",
    "--gif-episodes", "3"
]

print(f"Running with args: {sys.argv[1:]}", flush=True)

try:
    main()
    print("Evaluation completed successfully!", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.close()
    sys.stderr.close()
