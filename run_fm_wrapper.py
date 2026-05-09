import sys
sys.path.insert(0, "src")
import os

# Redirect stdout/stderr to files
sys.stdout = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_train_out.txt", "w")
sys.stderr = open("d:/文件/lwq临时文件夹/机器人学/stackcube_repro/fm_train_err.txt", "w")

print("Starting training script...")
sys.stdout.flush()

from train_bcflowmatching import main
print("Imports OK, calling main()...")
sys.stdout.flush()

try:
    main()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.close()
    sys.stderr.close()
