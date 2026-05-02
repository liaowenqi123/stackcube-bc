#!/usr/bin/env sh
set -eu

# One-click: train basic Diffusion(v1 path) + C8 invariant backbone and run eval with 3 GIFs.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-./data/processed/stackcube_rl_state.npz}"
OUTDIR="${OUTDIR:-./outputs/diffusion_v1_c8}"
EVAL_DIR="${EVAL_DIR:-./outputs/eval_diffusion_v1_c8}"

EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-4}"

EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-400}"
T_INF="${T_INF:-20}"
SAMPLER="${SAMPLER:-ddim}"
ETA="${ETA:-0.0}"
OBS_BACKBONE="${OBS_BACKBONE:-c8}"
ROT_PAIR_DIM="${ROT_PAIR_DIM:--1}"

if [ ! -f "$DATASET" ]; then
  echo "Dataset not found: $DATASET"
  echo "Generate it first, e.g.:"
  echo "  $PYTHON_BIN src/convert_maniskill_demo.py --input-h5 ./data/raw/demos/StackCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 --output-npz $DATASET"
  exit 1
fi

echo "[1/2] Training Diffusion (non-temporal) with rotational invariant backbone..."
"$PYTHON_BIN" src/train_bcdiffusion.py \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --obs-backbone "$OBS_BACKBONE" \
  --rot-pair-dim "$ROT_PAIR_DIM" \
  --T 100 \
  --hidden 384 \
  --depth 6 \
  --scheduler cosine \
  --ema-decay 0.999 \
  --action-noise-std 0.01 \
  --swa-start 450

CKPT="$OUTDIR/swa.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="$OUTDIR/best.pt"
fi

echo "[2/2] Evaluating and exporting 3 GIFs..."
"$PYTHON_BIN" src/eval_policy.py \
  --algo diffusion \
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
