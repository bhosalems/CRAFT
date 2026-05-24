#!/usr/bin/env bash
set -euo pipefail

# Parallel CLIP feature extraction across N GPUs.
# Each shard handles pairs[i::N] and writes scores.shard_i_of_N.json etc.
# A merge step then concatenates them back into scores.json / frames.json / meta.json.

NUM_GPUS=${NUM_GPUS:-8}
DATASET_PATH=${DATASET_PATH:-/a2il/data/mbhosale/MAGMaR2026_test/}
TOPIC_MAPPING=${TOPIC_MAPPING:-/home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/data/topic_video_mapping_v2.json}
QUERIES=${QUERIES:-/a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries_visual.jsonl}
MODEL=${MODEL:-mclip}
OUTPUT_FILE=${OUTPUT_FILE:-/a2il/data/mbhosale/MAGMaR2026_test/outscores}
DATASET_NAME=${DATASET_NAME:-magmar2026}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/extract_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "[launcher] spawning ${NUM_GPUS} shards; logs in ${LOG_DIR}"

pids=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 python -u "${SCRIPT_DIR}/feature_extract_folder.py" \
        --dataset_path "${DATASET_PATH}" \
        --topic_mapping "${TOPIC_MAPPING}" \
        --queries "${QUERIES}" \
        --extract_feature_model "${MODEL}" \
        --output_file "${OUTPUT_FILE}" \
        --device cuda \
        --dataset_name "${DATASET_NAME}" \
        --num_shards "${NUM_GPUS}" \
        --shard_id "${i}" \
        > "${LOG_DIR}/shard_${i}.log" 2>&1 &
    pids+=($!)
    echo "[launcher] shard ${i} -> GPU ${i} -> PID ${pids[-1]}"
done

# Wait for every shard; surface failures clearly.
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[launcher] shard ${i} (PID ${pids[$i]}) FAILED — see ${LOG_DIR}/shard_${i}.log"
        fail=1
    else
        echo "[launcher] shard ${i} done"
    fi
done

if [[ $fail -ne 0 ]]; then
    echo "[launcher] one or more shards failed; not merging."
    exit 1
fi

echo "[launcher] all shards done; merging..."
python "${SCRIPT_DIR}/merge_shards.py" \
    --output_file "${OUTPUT_FILE}" \
    --dataset_name "${DATASET_NAME}" \
    --extract_feature_model "${MODEL}" \
    --num_shards "${NUM_GPUS}"
echo "[launcher] complete"
