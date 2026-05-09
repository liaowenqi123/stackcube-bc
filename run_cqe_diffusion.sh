#!/usr/bin/env sh
set -eu

# One-click: train + eval Contextual Quasi-Equivariant (CQE) Diffusion.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-./data/processed/stackcube_rl_state.npz}"
OUTDIR="${OUTDIR:-./outputs/diffusion_cqe_v1}"
EVAL_DIR="${EVAL_DIR:-./outputs/eval_diffusion_cqe_v1}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-4}"

ROT_PAIR_DIM="${ROT_PAIR_DIM:-16}"
TRANS_PAIRS="${TRANS_PAIRS:-2}"
SYM_LAMBDA="${SYM_LAMBDA:-0.1}"
ID_LAMBDA="${ID_LAMBDA:-0.05}"
THETA_MAX_DEG="${THETA_MAX_DEG:-180}"
TRANS_MAX="${TRANS_MAX:-0.05}"

EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-400}"
T_INF="${T_INF:-20}"
SAMPLER="${SAMPLER:-ddim}"
ETA="${ETA:-0.0}"

if [ ! -f "$DATASET" ]; then
  echo "Dataset not found: $DATASET"
  exit 1
fi

echo "[1/2] Training CQE Diffusion..."
"$PYTHON_BIN" src/train_bcdiffusion_cqe.py \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --rot-pair-dim "$ROT_PAIR_DIM" \
  --trans-pairs "$TRANS_PAIRS" \
  --sym-lambda "$SYM_LAMBDA" \
  --id-lambda "$ID_LAMBDA" \
  --theta-max-deg "$THETA_MAX_DEG" \
  --trans-max "$TRANS_MAX"

CKPT="$OUTDIR/swa.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="$OUTDIR/best.pt"
fi

echo "[2/2] Evaluating and exporting 3 GIFs..."
"$PYTHON_BIN" src/eval_policy_cqe.py \
  --ckpt "$CKPT" \
  --episodes "$EPISODES" \
  --output-dir "$EVAL_DIR" \
  --max-steps "$MAX_STEPS" \
  --sampler "$SAMPLER" \
  --T-inf "$T_INF" \
  --eta "$ETA" \
  --save-gif \
  --gif-episodes 3

echo "Done."
echo "Model dir: $OUTDIR"
echo "Eval dir : $EVAL_DIR"
