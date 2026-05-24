#!/usr/bin/env bash
# Reference InfoF1 + CiteF1 on WikiVideo outputs.
#
# Activate the MIRAGE conda env first (default name: video_rag_eval):
#   conda activate video_rag_eval
#
# Then either set the four path env vars below before invoking, or override
# them on the command line:
#   PRED=/path/to/submission.jsonl REF=... VIDEO_DIR=... OUT=... bash run_wikivideo.sh
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PY:-python}"
MODEL="${MODEL:-qwen_7b}"

PRED="${PRED:-/path/to/CRAFT/outputs/craft_wikivideo_main/submission.jsonl}"
REF="${REF:-/path/to/wikivideo/annotations/train_ground_truth.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/wikivideo/en}"
OUT="${OUT:-./metrics/wikivideo}"

mkdir -p "$OUT"

# 1. Reference InfoF1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PY" "$SCRIPTS/infof1.py" \
    --eval_type reference --prediction "$PRED" --reference "$REF" \
    --video_dir "$VIDEO_DIR" --output_dir "$OUT" --model_name "$MODEL"

# 2. Reference CiteF1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PY" "$SCRIPTS/citef1.py" \
    --eval_type reference --prediction "$PRED" --reference "$REF" \
    --video_dir "$VIDEO_DIR" --output_dir "$OUT" --model_name "$MODEL"

# 3. Collection InfoF1 (VLM-grounded, expensive; uncomment to enable)
# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PY" "$SCRIPTS/infof1.py" \
#     --eval_type collection --prediction "$PRED" --reference "$REF" \
#     --video_dir "$VIDEO_DIR" --output_dir "$OUT" --model_name "$MODEL"
