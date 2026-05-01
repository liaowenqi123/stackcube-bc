"""
eval_wrapper.py — 带进度输出的 Diffusion Policy 评估包装器
轮询输出目录，检测已完成 episodes 数，直到完成所有 episodes。
"""
import subprocess, sys, time, json, os
from pathlib import Path

SCRIPT = "src/eval_policy.py"
CKPT   = "outputs/diffusion_v3/swa.pt"
OUTDIR = "outputs/eval_diffusion_v3_swa_50ep"
EPISODES = 50
MAX_STEPS = 400
T_INF = 100
SAMPLER = "ddpm"
ETA = 0.0

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

print(f"Starting {EPISODES}-episode evaluation | sampler={SAMPLER} T_inf={T_INF}...")
proc = subprocess.Popen(
    [sys.executable, SCRIPT,
     "--algo", "diffusion",
     "--ckpt", CKPT,
     "--sampler", SAMPLER,
     "--T-inf", str(T_INF),
     "--eta", str(ETA),
     "--episodes", str(EPISODES),
     "--output-dir", OUTDIR,
     "--max-steps", str(MAX_STEPS),
     "--save-gif",
     "--gif-episodes", "3"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# 等待 metrics.json 出现（完成标志）
start = time.time()
last_logged = -1
while proc.poll() is None:
    metrics_path = outdir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            elapsed = time.time() - start
            sr = m.get("success_rate", 0.0)
            avg_ret = m.get("avg_return", 0.0)
            avg_len = m.get("avg_ep_len", 0.0)
            eps_done = m.get("episodes", "?")
            print(f"\n[INTERRUPTED - partial results]")
            print(f"  Success Rate: {sr*100:.1f}%")
            print(f"  Avg Return:   {avg_ret:.3f}")
            print(f"  Avg Ep Len:   {avg_len:.1f}")
            print(f"  Episodes:     {eps_done}")
            print(f"  Elapsed:      {elapsed:.0f}s")
            break
        except Exception:
            pass
    # 每 60s 打印一次进度
    elapsed = time.time() - start
    completed = len(list(outdir.glob("rollout_*.gif")))
    if completed != last_logged:
        print(f"  [{elapsed:.0f}s] {completed} GIFs generated, waiting for metrics.json...")
        last_logged = completed
    time.sleep(15)
else:
    stdout, _ = proc.communicate()
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s (RC={proc.returncode})")
    for line in stdout.split("\n"):
        s = line.strip()
        if s and "pinocchio" not in s.lower() and "warning" not in s.lower():
            print(s)

    metrics_path = outdir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        print("\n=== FINAL METRICS ===")
        print(f"  Success Rate: {m['success_rate']*100:.1f}%")
        print(f"  Avg Return:   {m['avg_return']:.3f}")
        print(f"  Avg Ep Len:   {m['avg_ep_len']:.1f}")
        print(f"  Episodes:     {m['episodes']}")
