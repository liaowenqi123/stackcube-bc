python run_train_entity.py --dataset ./data/processed/stackcube_rl_state.npz --outdir ./outputs/entity_small --epochs 500 --batch-size 128 --lr 1e-4 --hidden 196 --depth 4 --swa-start 450


python src/eval_entity.py --ckpt outputs/entity_small/best.pt --episodes 50 --output-dir outputs/eval_entity_small --T-inf 20 --max-steps 200 --save-gif --gif-episodes 3