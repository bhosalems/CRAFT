#!/usr/bin/env bash
# Reference InfoF1 + CiteF1 on MAGMaR-2026 outputs.
#
# Activate the MIRAGE conda env first (default name: video_rag_eval):
#   conda activate video_rag_eval
#
# Then either set the four path env vars below before invoking, or override
# them on the command line:
#   PRED=/path/to/submission.jsonl REF=... VIDEO_DIR=... OUT=... bash run_magmar.sh
set -euo pipefail

# Resolve to this script's own directory so the script works no matter where
# it's invoked from.
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python binary. Defaults to whatever 'python' resolves to in the active conda
# env; override with PY=/abs/path/to/python if you need a specific interpreter.
PY="${PY:-python}"

# MIRAGE judge model.
MODEL="${MODEL:-qwen_7b}"

# Required inputs (override on the command line; placeholders below).
PRED="${PRED:-/path/to/CRAFT/outputs/craft_magmar_main/submission.jsonl}"
REF="${REF:-/path/to/MAGMaR2026_test/ground_truth.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/MAGMaR2026_test}"
OUT="${OUT:-./metrics/magmar}"

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
