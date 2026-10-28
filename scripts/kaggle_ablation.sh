#!/usr/bin/env bash
# Paper group C on Kaggle: 4 MathWriting fine-tunes from pretrain_best, one improvement
# added at a time, each scored on the same first 500 test images. Two runs at a time,
# one per GPU (Kaggle T4 x2). Resumes from *_last.pth if present.
# Results -> /kaggle/working/ablation_results.txt, logs -> /kaggle/working/abl_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-10}
OUT=/kaggle/working/ablation_results.txt
pip install -q datasets nltk gdown
python scripts/kaggle_fetch_checkpoints.py
[ -s data/mathwriting/test.jsonl ] || python scripts/download_mathwriting.py --train-cap 40000 --val-cap 4000
{ echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"; echo "EPOCHS=$EPOCHS";
  wc -l data/mathwriting/*.jsonl; } | tee "$OUT"

run() {  # tag gpu flags...
    local tag=$1 gpu=$2; shift 2
    local resume=""
    [ -f "checkpoints/${tag}_last.pth" ] && resume="--resume checkpoints/${tag}_last.pth"
    CUDA_VISIBLE_DEVICES=$gpu python -m model.train --stage finetune \
        --init-from checkpoints/pretrain_best.pth \
        --train data/mathwriting/train.jsonl --val data/mathwriting/val.jsonl \
        --epochs "$EPOCHS" --tag "$tag" "$@" $resume > "/kaggle/working/${tag}.log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu python -m model.evaluate --manifest data/mathwriting/test.jsonl \
        --ckpt "checkpoints/${tag}_best.pth" --limit 500 2>>"/kaggle/working/${tag}.log" \
        | sed "s/^/${tag} /" >> "$OUT"
}
gpus=$(nvidia-smi -L | wc -l)
g1=$(( gpus > 1 ? 1 : 0 ))
( run abl_base 0 --no-fine-grid --no-coverage --no-photo-aug; run abl_coverage 0 --no-photo-aug ) &
( run abl_grid "$g1" --no-coverage --no-photo-aug; run abl_photo "$g1" ) &
# progress: last epoch line of each run every 5 min, so the notebook output never goes silent
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
    sleep 300
    for f in /kaggle/working/abl_*.log; do
        [ -f "$f" ] && echo "$(basename "$f"): $(grep -E '^\[abl_' "$f" | tail -1)"
    done
done
wait
echo "== results"; cat "$OUT"
