#!/usr/bin/env bash
set -euo pipefail

# Run from repository root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate environment if available.
if [[ -f ".venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Video decode backend. torchvision leaks threads/mmaps per decode and pegs
# vm.max_map_count on long critic runs; decord uses ffmpeg C API directly and
# stays flat. Requires `pip install decord`.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
# Cap BLAS/OpenMP thread pools. The pipeline is per-claim serial; wide
# parallelism here just multiplies torchvision's per-call thread spawns.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Move Triton's GPU-kernel compile cache to /tmp (local disk). The default
# ~/.triton/cache is on NFS-mounted /home, which races between parallel
# workers — when one worker exits and another launches on a recycled slot,
# the second sees "Stale file handle" on cached .so files compiled by the
# first. /tmp is local fs and handles concurrent reads/writes safely.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"

# Log helper: append a timestamped banner to a log file so successive runs
# against the same OUT_DIR don't clobber each other. `tee -a` keeps prior
# content and appends the new output.
log_banner() {
	local log_path="$1"
	mkdir -p "$(dirname "$log_path")"
	{
		echo
		echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $(basename "$log_path") run begin ====="
	} >> "$log_path"
}

# ---- GPU slot pool ---------------------------------------------------------
# Bounded-concurrency launcher. Each "slot" maps 1:1 to a GPU index, so at
# any moment at most $POOL_SIZE workers are running, each pinned to a
# distinct GPU via CUDA_VISIBLE_DEVICES (set by the caller after acquiring
# a slot). Without this, the previous round-robin (`GPU = i % POOL_SIZE`)
# launched ALL N>POOL_SIZE workers in parallel; on the test set (130 queries,
# 8 GPUs) that put 2-3 vLLM instances on each card simultaneously and they
# OOM'd each other at startup.
#
# Caller protocol:
#   pool_init <pool_size>              # zero out POOL_GPU_PID, POOL_FAIL=0
#   for QID in ...; do
#       pool_acquire <pool_size>       # blocks until slot free; sets ACQUIRED_GPU
#       GPU=$ACQUIRED_GPU
#       CUDA_VISIBLE_DEVICES="$GPU" cmd ... &
#       POOL_GPU_PID[$GPU]="$!"
#   done
#   pool_drain                         # wait for remaining; updates POOL_FAIL
pool_init() {
	local _pn="$1"
	local g
	POOL_GPU_PID=()
	for ((g=0; g<_pn; g++)); do POOL_GPU_PID[$g]=""; done
	POOL_FAIL=0
}

# Wait until GPU $1 has at most $POOL_GPU_FREE_MIB_THRESHOLD MiB used,
# capped at POOL_GPU_FREE_TIMEOUT seconds. CUDA context teardown is async
# after a vLLM worker exits — its child processes can still hold 15+ GiB
# of weights/KV cache for several seconds after parent exit. Without
# this gate, the next worker we launch on the same slot races with the
# dying vLLM and OOMs mid-runtime when the GPU has too little headroom.
POOL_GPU_FREE_MIB_THRESHOLD="${POOL_GPU_FREE_MIB_THRESHOLD:-1500}"
POOL_GPU_FREE_TIMEOUT="${POOL_GPU_FREE_TIMEOUT:-60}"
_pool_wait_gpu_free() {
	# Wait for the slot's physical GPU(s) to drain. The slot index $1 maps to
	# physical GPUs based on $GPUS_PER_WORKER (set later when the parallel
	# launcher runs): {slot} for 1 GPU/slot, {slot*2, slot*2+1} for 2 GPU/slot.
	# Without checking BOTH physical GPUs of a pair, a freshly-released slot
	# can launch a new worker whose UNLI-on-cuda:1 races a dying UNLI on the
	# same physical GPU, leading to KV-cache shortage errors on the new vLLM.
	local slot="$1"
	local gpus_per="${GPUS_PER_WORKER:-1}"
	local phys_list
	if [[ "$gpus_per" == "2" ]]; then
		phys_list="$(( slot * 2 )) $(( slot * 2 + 1 ))"
	else
		phys_list="$slot"
	fi

	# nvidia-smi may not be on PATH inside some envs (containers, slurmd).
	# If unavailable, fall back to a fixed grace sleep so we still get
	# *some* protection against the race.
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		sleep 5
		return
	fi
	local i used phys all_ok
	for ((i=0; i<POOL_GPU_FREE_TIMEOUT; i++)); do
		all_ok=1
		for phys in $phys_list; do
			# Trailing `|| true` keeps a non-existent-GPU error (nvidia-smi exit 6)
			# from killing the script under `set -e + pipefail` — without it, the
			# whole pipeline (including this assignment) inherits the non-zero
			# status and bash exits silently mid-pool. Hit when GPUS_PER_WORKER
			# leaks across stages and a slot index points past the last GPU.
			used=$( { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$phys" 2>/dev/null || true; } | tr -d ' ')
			if [[ -z "$used" || "$used" == "[N/A]" ]]; then
				# Unexpected nvidia-smi output; bail out with a small grace.
				sleep 5
				return
			fi
			if (( used >= POOL_GPU_FREE_MIB_THRESHOLD )); then
				all_ok=0
				break
			fi
		done
		(( all_ok )) && return
		sleep 1
	done
	# Hard-timeout reached. The next launcher will try anyway; if it OOMs
	# we surface that to the caller via POOL_FAIL.
}

pool_acquire() {
	local _pn="$1"
	local g pid
	while :; do
		for ((g=0; g<_pn; g++)); do
			pid="${POOL_GPU_PID[$g]}"
			if [[ -z "$pid" ]]; then
				ACQUIRED_GPU=$g
				return 0
			fi
			# Reap if the worker on this slot has exited.
			if ! kill -0 "$pid" 2>/dev/null; then
				if ! wait "$pid"; then POOL_FAIL=1; fi
				POOL_GPU_PID[$g]=""
				# CRITICAL: wait for the dying vLLM's CUDA context to
				# release GPU memory before the next worker grabs the slot.
				_pool_wait_gpu_free "$g"
				ACQUIRED_GPU=$g
				return 0
			fi
		done
		# All slots busy: yield CPU until at least one background job exits.
		# `wait -n` requires bash >= 4.3; fall back to a brief sleep otherwise.
		wait -n 2>/dev/null || sleep 1
	done
}

pool_drain() {
	local pid
	for pid in "${POOL_GPU_PID[@]}"; do
		if [[ -n "$pid" ]]; then
			if ! wait "$pid"; then POOL_FAIL=1; fi
		fi
	done
}

PYTHON_BIN="/home/csgrad/mbhosale/phd/SCALE/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "[error] Missing venv interpreter at $PYTHON_BIN"
	exit 1
fi

# Inputs (override via env or CLI args if needed).
# QUERIES_JSONL_DEFAULT="/a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries.jsonl"
# MAPPING_JSON_DEFAULT="data/topic_video_mapping.json"

# # Allow env overrides.
# QUERIES_JSONL="${QUERIES_JSONL:-$QUERIES_JSONL_DEFAULT}"
# MAPPING_JSON="${MAPPING_JSON:-$MAPPING_JSON_DEFAULT}"

# WikiVideo MultiVENT subset: 52 events, 428 chunked video IDs, 52 hand-curated
# queries from akhilvssg/magmar-2026-test-asr-embeddings:training/train_*.
# Step 0 chunking has already been run; data/topic_video_mapping_wikivideo_v2.json
# contains chunked IDs and /a2il/data/mbhosale/wikivideo/en/ already has chunk
# MP4s. Recommended: launch with SKIP_CHUNK=1 to skip re-chunking.
#
# Note: we pass the v1 (pre-chunk) mapping here because run_query.sh derives the
# v2 path from the basename (data/<base>_v2.json). The v1 file just needs to
# exist for that derivation; the v2 file is what actually gets used downstream.
QUERIES_JSONL="/home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/data/wikivideo_queries.jsonl"
MAPPING_JSON="data/topic_video_mapping_wikivideo.json"

# Video root where <video_id>.mp4 files live (also where chunks are written).
# Wikivideo videos are under /a2il/data/mbhosale/wikivideo/en/ — the `en/`
# subdir is how the original wikivideo tarball was structured. Chunks live
# in the same dir alongside originals (<video_id>__chunk000.mp4 etc.).
VIDEO_ROOT="/a2il/data/mbhosale/wikivideo/en"

# Max chunk duration in seconds (videos longer than this get split).
MAX_CHUNK_SECONDS="${MAX_CHUNK_SECONDS:-120}"

# Step 5 (higher-level inference) model. Defaults to Qwen3.5-9B per the user's
# explicit preference for a lightweight Step 5 — Step 5 is text-only over
# already-extracted claim packets, so the bigger 30B model adds little value
# and (more importantly) needs ~24 GB KV cache at 262K context which OOMs
# alongside its own weights on a 47 GB A6000. The 9B fits comfortably.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"

# Chunk map (chunk_id -> {video_id, start, end}) — written by Step 0 and
# read by format_submission to remap chunk IDs back to originals when
# producing the final JSONL.
CHUNK_MAP_JSON="${CHUNK_MAP_JSON:-data/video_chunk_map_wikivideo.json}"

# Branch output directory (all artifacts live under this).
# Usage:
#   ./run_query_wikivideo.sh <OUT_DIR> [QUERIES_JSONL] [MAPPING_JSON] [CLAIMS_JSONL]
# Examples:
#   ./run_query_wikivideo.sh outputs/outputs_query_wikivideo_v1
#   SKIP_STEP1=1 ./run_query_wikivideo.sh outputs/outputs_query_wikivideo_v1
OUT_DIR="${1:-outputs/outputs_query_wikivideo_v1}"

# Optional CLI overrides for inputs.
if [[ -n "${2:-}" ]]; then
	QUERIES_JSONL="$2"
fi
if [[ -n "${3:-}" ]]; then
	MAPPING_JSON="$3"
fi

# Optional: provide an existing claims JSONL to reuse and force-skip Step 1.
CLAIMS_JSONL_INPUT="${4:-}"

mkdir -p "$OUT_DIR"

# Chunking outputs: v2 mapping (with chunk IDs) and chunk map (chunk->original).
# The v2 mapping suffix is derived from the original mapping filename.
MAPPING_BASE="$(basename "$MAPPING_JSON" .json)"
MAPPING_V2_JSON="data/${MAPPING_BASE}_v2.json"
CHUNK_MAP_JSON="${CHUNK_MAP_JSON:-data/video_chunk_map.json}"

# Step 1/1.5 intermediate outputs for query-claims branch.
QUERY_CLAIMS_DIR="$OUT_DIR/query_claims_single"
UNLI_QUERY_CLAIMS_DIR="$OUT_DIR/unli_query_claims_single_lora"
QUERY_CLAIMS_JSONL="$QUERY_CLAIMS_DIR/query_conditioned_claims.jsonl"
UNLI_PREDICTIONS_JSONL="$UNLI_QUERY_CLAIMS_DIR/unli_predictions.jsonl"
CALIBRATED_QUERY_CLAIMS="$UNLI_QUERY_CLAIMS_DIR/query_conditioned_claims_calibrated.jsonl"

# Downstream outputs.
CLAIM_PACKETS_DIR="$OUT_DIR/claim_packets"
INFERENCES_QUERY_DIR="$OUT_DIR/inferences_query"
REPORTS_QUERY_DIR="$OUT_DIR/reports_query_based"

# Ensure per-stage output dirs exist BEFORE tee writes logs.
# (tee opens the output file immediately; it won't wait for Python to create dirs.)
mkdir -p \
	"$QUERY_CLAIMS_DIR" \
	"$UNLI_QUERY_CLAIMS_DIR" \
	"$CLAIM_PACKETS_DIR" \
	"$INFERENCES_QUERY_DIR" \
	"$REPORTS_QUERY_DIR"

# Set to 1 to apply the optional vLLM patch helper before inference.
APPLY_VLLM_PATCH="${APPLY_VLLM_PATCH:-0}"

# Step 1 skip policy:
# - SKIP_STEP1=0  -> always (re)run Step 1
# - SKIP_STEP1=1  -> always skip Step 1 (requires existing claims jsonl)
# - SKIP_STEP1=auto (default) -> skip Step 1 only if claims jsonl exists
SKIP_STEP1="${SKIP_STEP1:-auto}"

# Step 1.5 chunked mode: how many videos per subprocess in predict_unli.
# 0 = run single-process (old behavior).
# > 0 = run predict_unli_chunked.py, spawning a fresh predict_unli.py subprocess
#       per chunk of N videos. Use this to avoid pyav/libav mmap leaks that
#       exhaust vm.max_map_count on long runs. Start with 40.
STEP15_CHUNK_SIZE="${STEP15_CHUNK_SIZE:-0}"

# Step 1.5 audio mode: set NO_AUDIO=1 to skip audio decoding in the UNLI
# scorer. Recommended when any video is longer than ~2-3 minutes — the audio
# buffer (float32 stereo at source sample rate, e.g. ~220 MB for a 9.5 min
# 48 kHz stereo clip) can trigger libav EAGAIN cascades that destroy throughput.
# Audio adds little signal for video-text NLI scoring.
STEP15_NO_AUDIO="${NO_AUDIO:-0}"

# Critic loop: max feedback rounds per video in Step 1 (0 = disabled).
MAX_CRITIC_ROUNDS="${MAX_CRITIC_ROUNDS:-0}"
# Optional: override critic UNLI loading.
# By default (conf/step1_query_claims.yaml), the critic matches Step 1.5:
#   base_model=AdoptedIrelia/UNLI + lora_path=AdoptedIrelia/UNLI/lora
# Set these env vars to override the config at runtime.
CRITIC_UNLI_BASE_MODEL="${CRITIC_UNLI_BASE_MODEL:-}"
CRITIC_UNLI_LORA_PATH="${CRITIC_UNLI_LORA_PATH:-}"
# CUDA device index for the UNLI critic. vLLM always uses cuda:0 for tp=1,
# so default critic to cuda:1 when 2+ GPUs are allocated. Set CRITIC_GPU=0
# to co-locate on a single GPU.
CRITIC_GPU="${CRITIC_GPU:-}"
# Text critic sub-checks. Contradictions use an MNLI cross-encoder; coverage
# uses a small instruction-tuned LLM. Leave blank to keep YAML defaults.
CRITIC_NLI_MODEL="${CRITIC_NLI_MODEL:-}"
CRITIC_NLI_DEVICE="${CRITIC_NLI_DEVICE:-}"
CRITIC_NLI_ENABLED="${CRITIC_NLI_ENABLED:-}"
CRITIC_NLI_SCREEN_THRESHOLD="${CRITIC_NLI_SCREEN_THRESHOLD:-}"
CRITIC_NLI_MAX_CANDIDATES="${CRITIC_NLI_MAX_CANDIDATES:-}"
CRITIC_COVERAGE_MODEL="${CRITIC_COVERAGE_MODEL:-}"
CRITIC_COVERAGE_GPU="${CRITIC_COVERAGE_GPU:-}"
CRITIC_COVERAGE_ENABLED="${CRITIC_COVERAGE_ENABLED:-}"
COVERAGE_FOLLOWUP_ROUNDS="${COVERAGE_FOLLOWUP_ROUNDS:-}"

# Optional: override vLLM GPU memory utilization for Step 1.
# When critic is enabled, extract_query_claims.py auto-defaults to 0.6 to leave
# room for UNLI on the same GPU. Set GPU_MEM_UTIL=0.5 (or lower) if you still
# hit CUDA OOM; set to 0.9 if the critic runs on a different GPU than vLLM.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"

# ---- ASR cache ----
# This pipeline does NOT generate ASR; it only consumes a pre-built cache
# directory. Generate it separately with extract_asr.py — see that script's
# docstring for the two-env workflow (Qwen3-ASR in main env, omniASR in a
# fairseq2-compatible env). Set ASR_DIR= (empty) to disable ASR entirely.
# Wikivideo ASR cache lives at /a2il/data/mbhosale/wikivideo/asr/, which is
# the PARENT of VIDEO_ROOT (videos live under .../wikivideo/en/, ASR is one
# directory up so it isn't co-mingled with chunk MP4s).
# `${VAR-default}` (no colon) only falls back when ASR_DIR is *unset* —
# `${VAR:-default}` would overwrite an explicit empty assignment with the
# default, which contradicts the "Set ASR_DIR= (empty) to disable ASR" note
# above and silently re-enables the ASR cache. Keep the no-colon form.
ASR_DIR="${ASR_DIR-/a2il/data/mbhosale/wikivideo/asr}"

# Step 1b query-level parallelism. When >1, the script shards queries
# across that many GPUs (round-robin via CUDA_VISIBLE_DEVICES). Each
# worker loads its own VLM into a single visible GPU (TP=1 inside the
# worker), so the model must fit on one card — use the FP8 build of
# Qwen3-VL-30B-A3B-Instruct on A6000s, or the bf16 9B on smaller cards.
# Set PARALLEL_QUERIES=1 (default) to keep the existing single-process flow.
PARALLEL_QUERIES="${PARALLEL_QUERIES:-1}"

# Step 1.5 query-level parallelism. When >1, shard the combined claims JSONL
# by query_id and launch one predict_unli worker per query, each pinned to
# its own GPU via CUDA_VISIBLE_DEVICES. Each worker writes its own predictions
# JSONL into a per-query subdir; the script concatenates them at the end into
# the canonical $UNLI_PREDICTIONS_JSONL path. STEP15_CHUNK_SIZE still applies
# inside each worker for libav mmap-leak hygiene. UNLI is small (~1 GB), so
# this scales cleanly across GPUs even on contended cards.
# Set PARALLEL_STEP15=1 (default) to keep the existing single-process flow.
PARALLEL_STEP15="${PARALLEL_STEP15:-1}"

# Step 5 (higher-level inference) query-level parallelism. When >1, launch one
# infer_higher_level.py worker per query, each pinned to its own GPU and
# filtered via --only-query-ids. Each worker writes its inferences into a
# per-query subdir; the script concatenates them into the canonical
# $INFERENCES_QUERY_DIR/inferences.jsonl that Step 6 expects. Step 5 loads its
# own VLM (same model as Step 1b via $MODEL_NAME), so the model must fit on
# one GPU at TP=1.
# Set PARALLEL_STEP5=1 (default) to keep the existing single-process flow.
PARALLEL_STEP5="${PARALLEL_STEP5:-1}"

if [[ "$APPLY_VLLM_PATCH" == "1" && -f "sbatch/patch_vllm_qwen3_next.py" ]]; then
	echo "[setup] Applying optional vLLM patch"
	log_banner "$OUT_DIR/setup_patch_vllm.log"
	"$PYTHON_BIN" sbatch/patch_vllm_qwen3_next.py 2>&1 | tee -a "$OUT_DIR/setup_patch_vllm.log"
fi

echo "[query-based] Running explicit 7-stage downstream query pipeline"

# ---- Step 0: Chunk long videos ----
# Splits any video longer than MAX_CHUNK_SECONDS into <=MAX_CHUNK_SECONDS MP4s
# in the same directory, produces a v2 mapping with chunk IDs, and a chunk map
# for mapping chunk IDs back to originals at submission time.
SKIP_CHUNK="${SKIP_CHUNK:-auto}"
AUTO_SKIP_CHUNK=0
if [[ "$SKIP_CHUNK" == "1" ]]; then
	AUTO_SKIP_CHUNK=1
elif [[ "$SKIP_CHUNK" == "auto" && -f "$MAPPING_V2_JSON" && -f "$CHUNK_MAP_JSON" ]]; then
	AUTO_SKIP_CHUNK=1
fi

if [[ "$AUTO_SKIP_CHUNK" == "1" ]]; then
	echo "[0/7] SKIP: Reusing existing v2 mapping at $MAPPING_V2_JSON"
else
	echo "[0/7] Chunk long videos (> ${MAX_CHUNK_SECONDS}s)"
	CHUNK_ARGS=(
		--video-root "$VIDEO_ROOT"
		--max-seconds "$MAX_CHUNK_SECONDS"
		--mapping-in "$MAPPING_JSON"
		--mapping-out "$MAPPING_V2_JSON"
		--chunk-map-out "$CHUNK_MAP_JSON"
	)
	log_banner "$OUT_DIR/step0_chunk_videos.log"
	"$PYTHON_BIN" chunk_videos.py "${CHUNK_ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/step0_chunk_videos.log"
fi

# From here on, use the v2 mapping (with chunk IDs) for all pipeline steps.
MAPPING_JSON="$MAPPING_V2_JSON"

# ---- ASR cache check (no ASR generation here) ----
# The Step 1b VLM prompt picks up per-video transcripts from $ASR_DIR if
# it exists. Generate them with extract_asr.py before running this script.
if [[ -n "$ASR_DIR" && -d "$ASR_DIR" ]]; then
	asr_count=$(find "$ASR_DIR" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
	echo "[asr] using cache at $ASR_DIR ($asr_count transcript files)"
elif [[ -n "$ASR_DIR" ]]; then
	echo "[asr] WARNING: ASR_DIR=$ASR_DIR does not exist — claims will be visual-only.
       Run extract_asr.py first to populate it, or set ASR_DIR= to silence this."
else
	echo "[asr] disabled (ASR_DIR is empty)"
fi

if [[ -n "$CLAIMS_JSONL_INPUT" ]]; then
	mkdir -p "$QUERY_CLAIMS_DIR"
	if [[ ! -f "$CLAIMS_JSONL_INPUT" ]]; then
		echo "[error] Provided CLAIMS_JSONL does not exist: $CLAIMS_JSONL_INPUT"
		exit 1
	fi
	cp -f "$CLAIMS_JSONL_INPUT" "$QUERY_CLAIMS_JSONL"
	SKIP_STEP1=1
	echo "[setup] Reusing provided claims JSONL -> $QUERY_CLAIMS_JSONL"
fi

# ---- Step 1: Extract query-conditioned claims ----

AUTO_SKIP_STEP1=0
if [[ "$SKIP_STEP1" == "1" ]]; then
	AUTO_SKIP_STEP1=1
elif [[ "$SKIP_STEP1" == "auto" && -f "$QUERY_CLAIMS_JSONL" ]]; then
	AUTO_SKIP_STEP1=1
fi

if [[ "$AUTO_SKIP_STEP1" == "1" ]]; then
	if [[ ! -f "$QUERY_CLAIMS_JSONL" ]]; then
		echo "[error] SKIP_STEP1 requested but claims JSONL not found: $QUERY_CLAIMS_JSONL"
		exit 1
	fi
	echo "[1/7] SKIP: Reusing existing query claims at $QUERY_CLAIMS_JSONL"
else
	echo "[1/7] Extract query-conditioned claims (Step 1b)"
	CRITIC_OVERRIDES=()
	if [[ -n "${CRITIC_UNLI_BASE_MODEL}" ]]; then
		CRITIC_OVERRIDES+=("critic.unli_base_model=${CRITIC_UNLI_BASE_MODEL}")
	fi
	if [[ -n "${CRITIC_UNLI_LORA_PATH}" ]]; then
		CRITIC_OVERRIDES+=("critic.unli_lora_path=${CRITIC_UNLI_LORA_PATH}")
	fi
	if [[ -n "${CRITIC_GPU}" ]]; then
		CRITIC_OVERRIDES+=("critic.gpu=${CRITIC_GPU}")
	fi
	if [[ -n "${CRITIC_NLI_MODEL}" ]]; then
		CRITIC_OVERRIDES+=("critic.nli_model=${CRITIC_NLI_MODEL}")
	fi
	if [[ -n "${CRITIC_NLI_DEVICE}" ]]; then
		CRITIC_OVERRIDES+=("critic.nli_device=${CRITIC_NLI_DEVICE}")
	fi
	if [[ -n "${CRITIC_NLI_ENABLED}" ]]; then
		CRITIC_OVERRIDES+=("critic.nli_enabled=${CRITIC_NLI_ENABLED}")
	fi
	if [[ -n "${CRITIC_COVERAGE_MODEL}" ]]; then
		CRITIC_OVERRIDES+=("critic.coverage_model=${CRITIC_COVERAGE_MODEL}")
	fi
	if [[ -n "${CRITIC_COVERAGE_GPU}" ]]; then
		CRITIC_OVERRIDES+=("critic.coverage_gpu=${CRITIC_COVERAGE_GPU}")
	fi
	if [[ -n "${CRITIC_COVERAGE_ENABLED}" ]]; then
		CRITIC_OVERRIDES+=("critic.coverage_enabled=${CRITIC_COVERAGE_ENABLED}")
	fi
	if [[ -n "${CRITIC_NLI_SCREEN_THRESHOLD}" ]]; then
		CRITIC_OVERRIDES+=("critic.nli_screen_threshold=${CRITIC_NLI_SCREEN_THRESHOLD}")
	fi
	if [[ -n "${CRITIC_NLI_MAX_CANDIDATES}" ]]; then
		CRITIC_OVERRIDES+=("critic.nli_max_candidates=${CRITIC_NLI_MAX_CANDIDATES}")
	fi
	if [[ -n "${COVERAGE_FOLLOWUP_ROUNDS}" ]]; then
		CRITIC_OVERRIDES+=("critic.coverage_followup_rounds=${COVERAGE_FOLLOWUP_ROUNDS}")
	fi
	if [[ -n "${GPU_MEM_UTIL}" ]]; then
		CRITIC_OVERRIDES+=("runtime.gpu_memory_utilization=${GPU_MEM_UTIL}")
	fi
	if [[ -n "$ASR_DIR" ]]; then
		CRITIC_OVERRIDES+=("asr_dir=${ASR_DIR}")
	fi
	log_banner "$QUERY_CLAIMS_DIR/step1b_query_claims.log"

	if [[ "$PARALLEL_QUERIES" -gt 1 ]]; then
		# Read query_ids from the queries JSONL and shard them across
		# $PARALLEL_QUERIES GPUs round-robin. Each worker is pinned to one
		# GPU via CUDA_VISIBLE_DEVICES; from its perspective that device
		# is cuda:0, so we force critic.gpu=0 inside each worker.
		QIDS=$("$PYTHON_BIN" - "$QUERIES_JSONL" <<-'PY'
		import json, sys
		ids = []
		with open(sys.argv[1]) as f:
		    for line in f:
		        line = line.strip()
		        if line:
		            ids.append(str(json.loads(line)["query_id"]))
		print(" ".join(ids))
		PY
		)
		if [[ -z "$QIDS" ]]; then
			echo "[error] no query_ids found in $QUERIES_JSONL"
			exit 1
		fi
		IFS=' ' read -r -a QID_ARR <<< "$QIDS"
		N_QIDS_TOTAL=${#QID_ARR[@]}

		# Resume prefilter: skip queries whose per-query JSONL already exists.
		# extract_query_claims.py also has a Python-side resume gate (default
		# on), but that fires only AFTER the worker has loaded vLLM (~30s and
		# ~30 GiB GPU memory), which is wasteful and adds GPU contention. By
		# pruning at the bash layer we never launch workers we don't need.
		# Set RESUME_STEP1=0 to disable and force re-extraction of all queries.
		RESUME_STEP1="${RESUME_STEP1:-1}"
		PENDING_QIDS=()
		SKIPPED_QIDS=()
		for _q in "${QID_ARR[@]}"; do
			if [[ "$RESUME_STEP1" == "1" && -s "$QUERY_CLAIMS_DIR/query_${_q}.jsonl" ]]; then
				SKIPPED_QIDS+=("$_q")
			else
				PENDING_QIDS+=("$_q")
			fi
		done
		if (( ${#SKIPPED_QIDS[@]} > 0 )); then
			echo "[1/7] resume: ${#SKIPPED_QIDS[@]}/${N_QIDS_TOTAL} per-query JSONLs already exist; skipping queries: ${SKIPPED_QIDS[*]}"
		fi
		QID_ARR=("${PENDING_QIDS[@]}")
		N_QIDS=${#QID_ARR[@]}
		if (( N_QIDS == 0 )); then
			echo "[1/7] resume: nothing to extract — all per-query JSONLs already present; proceeding to recombine"
		else
			echo "[1/7] Parallel step 1b: $N_QIDS queries across $PARALLEL_QUERIES GPUs"
		fi

		# Strip critic.gpu / critic.coverage_gpu / runtime.tensor_parallel_size
		# from CRITIC_OVERRIDES so we can re-set them per worker below.
		WORKER_OVERRIDES=()
		for o in "${CRITIC_OVERRIDES[@]}"; do
			case "$o" in
				critic.gpu=*|critic.coverage_gpu=*|runtime.tensor_parallel_size=*) ;;
				*) WORKER_OVERRIDES+=("$o") ;;
			esac
		done

		# Per-worker GPU allocation policy.
		#
		# CRITIC_GPU_ALONE=0 (default): each worker gets ONE GPU. vLLM + UNLI critic
		#   share that GPU. Cheapest for cards big enough to hold both (e.g. 9B + 3B
		#   UNLI on A6000 fits comfortably).
		# CRITIC_GPU_ALONE=1: each worker gets TWO consecutive GPUs. vLLM uses the
		#   first, UNLI critic uses the second. Required when vLLM and UNLI together
		#   exceed a single card — e.g. Qwen3-VL-30B FP8 (~30 GB) + UNLI Omni-3B
		#   (~15 GB) = ~45 GB which on A6000 (47 GB) leaves no headroom for KV cache.
		#   Halves effective concurrency: PARALLEL_QUERIES * 2 <= total visible GPUs.
		CRITIC_GPU_ALONE="${CRITIC_GPU_ALONE:-0}"
		GPUS_PER_WORKER=1
		if [[ "$CRITIC_GPU_ALONE" == "1" ]]; then
			GPUS_PER_WORKER=2
			WORKER_OVERRIDES+=("critic.gpu=1" "critic.coverage_gpu=1" "runtime.tensor_parallel_size=1")
		else
			WORKER_OVERRIDES+=("critic.gpu=0" "critic.coverage_gpu=0" "runtime.tensor_parallel_size=1")
		fi

		# Bounded-concurrency launcher: at most $PARALLEL_QUERIES workers running
		# at once, each on its own slot of $GPUS_PER_WORKER GPUs. Without this cap,
		# N > PARALLEL_QUERIES queries (e.g. 130 on the test set) all start in
		# parallel and OOM each other on the same physical card.
		pool_init "$PARALLEL_QUERIES"
		for QID in "${QID_ARR[@]}"; do
			pool_acquire "$PARALLEL_QUERIES"
			GPU="$ACQUIRED_GPU"
			if [[ "$GPUS_PER_WORKER" == "2" ]]; then
				PHYS_GPU0=$(( GPU * 2 ))
				PHYS_GPU1=$(( GPU * 2 + 1 ))
				CUDA_LIST="${PHYS_GPU0},${PHYS_GPU1}"
				WORKER_LOG="$QUERY_CLAIMS_DIR/step1b_query_${QID}_gpu${PHYS_GPU0}-${PHYS_GPU1}.log"
			else
				CUDA_LIST="$GPU"
				WORKER_LOG="$QUERY_CLAIMS_DIR/step1b_query_${QID}_gpu${GPU}.log"
			fi
			log_banner "$WORKER_LOG"
			echo "[1/7] launching worker for query=$QID on GPUs $CUDA_LIST -> $WORKER_LOG"
			CUDA_VISIBLE_DEVICES="$CUDA_LIST" "$PYTHON_BIN" run_step1_query_claims.py \
				data.queries_jsonl="$QUERIES_JSONL" \
				data.mapping="$MAPPING_JSON" \
				data.video_root="$VIDEO_ROOT" \
				max_critic_rounds="$MAX_CRITIC_ROUNDS" \
				"${WORKER_OVERRIDES[@]}" \
				"only_query_ids=$QID" \
				output.out_dir="$QUERY_CLAIMS_DIR" \
				>> "$WORKER_LOG" 2>&1 &
			POOL_GPU_PID[$GPU]="$!"
		done
		pool_drain
		FAIL="$POOL_FAIL"
		if [[ "$FAIL" -ne 0 ]]; then
			echo "[error] one or more step 1b workers failed; see $QUERY_CLAIMS_DIR/step1b_query_*_gpu*.log"
			exit 1
		fi

		# Recombine: no-filter invocation. Every per-query JSONL exists,
		# so n_run=0 and the script just concatenates into the combined file.
		echo "[1/7] all workers done, recombining into $QUERY_CLAIMS_JSONL"
		"$PYTHON_BIN" run_step1_query_claims.py \
			data.queries_jsonl="$QUERIES_JSONL" \
			data.mapping="$MAPPING_JSON" \
			data.video_root="$VIDEO_ROOT" \
			max_critic_rounds=0 \
			output.out_dir="$QUERY_CLAIMS_DIR" 2>&1 | tee -a "$QUERY_CLAIMS_DIR/step1b_query_claims.log"
	else
		"$PYTHON_BIN" run_step1_query_claims.py \
			data.queries_jsonl="$QUERIES_JSONL" \
			data.mapping="$MAPPING_JSON" \
			data.video_root="$VIDEO_ROOT" \
			max_critic_rounds="$MAX_CRITIC_ROUNDS" \
			"${CRITIC_OVERRIDES[@]}" \
			output.out_dir="$QUERY_CLAIMS_DIR" 2>&1 | tee -a "$QUERY_CLAIMS_DIR/step1b_query_claims.log"
	fi
fi

# ---- Step 1.5: UNLI predict (with gc.collect fix) ----
echo "[2/7] Predict UNLI support for query claims (Step 1.5)"
STEP15_OVERRIDES=()
if [[ "$STEP15_NO_AUDIO" == "1" ]]; then
	STEP15_OVERRIDES+=("no_audio=true")
fi

if [[ "$PARALLEL_STEP15" -gt 1 ]]; then
	# Shard the combined claims JSONL by query_id, launch one predict_unli
	# worker per query (round-robin GPUs), then concatenate their per-query
	# predictions JSONLs into $UNLI_PREDICTIONS_JSONL.
	SHARD_DIR="$UNLI_QUERY_CLAIMS_DIR/_shards"
	WORKERS_DIR="$UNLI_QUERY_CLAIMS_DIR/_workers"
	mkdir -p "$SHARD_DIR" "$WORKERS_DIR"

	S15_QIDS=$("$PYTHON_BIN" - "$QUERY_CLAIMS_JSONL" "$SHARD_DIR" <<-'PY'
	import json, os, sys
	from collections import defaultdict
	src, out_dir = sys.argv[1], sys.argv[2]
	by_qid = defaultdict(list)
	with open(src) as f:
	    for line in f:
	        line = line.rstrip("\n")
	        if not line.strip():
	            continue
	        rec = json.loads(line)
	        qid = str(rec.get("query_id", "")).strip()
	        if qid:
	            by_qid[qid].append(line)
	# Sort numerically when possible, otherwise lexicographically.
	def _key(q):
	    try:
	        return (0, int(q))
	    except ValueError:
	        return (1, q)
	qids = sorted(by_qid.keys(), key=_key)
	for qid in qids:
	    with open(os.path.join(out_dir, f"claims_q{qid}.jsonl"), "w") as f:
	        for ln in by_qid[qid]:
	            f.write(ln + "\n")
	print(" ".join(qids))
	PY
	)
	if [[ -z "$S15_QIDS" ]]; then
		echo "[error] no query_ids found in $QUERY_CLAIMS_JSONL"
		exit 1
	fi
	IFS=' ' read -r -a S15_QID_ARR <<< "$S15_QIDS"
	S15_N_QIDS=${#S15_QID_ARR[@]}
	echo "[2/7] Parallel step 1.5: $S15_N_QIDS queries across $PARALLEL_STEP15 GPUs"

	# Step 1.5 always uses 1 GPU per worker. Reset here because Step 1 may
	# have left GPUS_PER_WORKER=2 (via CRITIC_GPU_ALONE=1), which would make
	# _pool_wait_gpu_free probe phys GPUs 2*slot, 2*slot+1 — past the last
	# real GPU for slots >= n_gpus/2 — and silently kill the script.
	GPUS_PER_WORKER=1
	pool_init "$PARALLEL_STEP15"
	for QID in "${S15_QID_ARR[@]}"; do
		pool_acquire "$PARALLEL_STEP15"
		GPU="$ACQUIRED_GPU"
		SHARD_IN="$SHARD_DIR/claims_q${QID}.jsonl"
		WORKER_OUT_DIR="$WORKERS_DIR/q${QID}"
		WORKER_LOG="$UNLI_QUERY_CLAIMS_DIR/step1_5_q${QID}_gpu${GPU}.log"
		mkdir -p "$WORKER_OUT_DIR"
		log_banner "$WORKER_LOG"
		echo "[2/7] launching step1.5 worker for query=$QID on GPU $GPU -> $WORKER_LOG"
		CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_step1_5_predict_unli.py \
			artifact_type=query-claims \
			data.artifacts_jsonl="$SHARD_IN" \
			data.video_root="$VIDEO_ROOT" \
			chunk_size="$STEP15_CHUNK_SIZE" \
			"${STEP15_OVERRIDES[@]}" \
			output.out_dir="$WORKER_OUT_DIR" \
			>> "$WORKER_LOG" 2>&1 &
		POOL_GPU_PID[$GPU]="$!"
	done
	pool_drain
	S15_FAIL="$POOL_FAIL"
	if [[ "$S15_FAIL" -ne 0 ]]; then
		echo "[error] one or more step 1.5 workers failed; see $UNLI_QUERY_CLAIMS_DIR/step1_5_q*_gpu*.log"
		exit 1
	fi

	# Concatenate per-worker predictions into the canonical output path.
	: > "$UNLI_PREDICTIONS_JSONL"
	for QID in "${S15_QID_ARR[@]}"; do
		WORKER_PRED="$WORKERS_DIR/q${QID}/unli_predictions.jsonl"
		if [[ -s "$WORKER_PRED" ]]; then
			cat "$WORKER_PRED" >> "$UNLI_PREDICTIONS_JSONL"
		fi
	done
	echo "[2/7] step 1.5 done; combined predictions -> $UNLI_PREDICTIONS_JSONL"
else
	"$PYTHON_BIN" run_step1_5_predict_unli.py \
		artifact_type=query-claims \
		data.artifacts_jsonl="$QUERY_CLAIMS_JSONL" \
		data.video_root="$VIDEO_ROOT" \
		chunk_size="$STEP15_CHUNK_SIZE" \
		"${STEP15_OVERRIDES[@]}" \
		output.out_dir="$UNLI_QUERY_CLAIMS_DIR" 2>&1 | tee "$UNLI_QUERY_CLAIMS_DIR/step1_5_query_predict.log"
fi

# ---- Step 1.5: Calibrate ----
echo "[3/7] Calibrate UNLI scores into claims (Step 1.5)"
"$PYTHON_BIN" run_step1_5_calibrate_unli.py \
	artifact_type=query-claims \
	data.claims_jsonl="$QUERY_CLAIMS_JSONL" \
	data.unli_jsonl="$UNLI_PREDICTIONS_JSONL" \
	output.out_dir="$UNLI_QUERY_CLAIMS_DIR" 2>&1 | tee "$UNLI_QUERY_CLAIMS_DIR/step1_5_query_calibrate.log"

# ---- Step 2a: Assemble claim packets ----
echo "[4/7] Assemble claim packets"
"$PYTHON_BIN" assemble_packets.py \
	--stream query-based \
	--claims "$CALIBRATED_QUERY_CLAIMS" \
	--queries-jsonl "$QUERIES_JSONL" \
	--mapping "$MAPPING_JSON" \
	--out-dir "$CLAIM_PACKETS_DIR" \
	--verbose 2>&1 | tee "$CLAIM_PACKETS_DIR/step2_query_assemble.log"

# ---- Step 2b: Higher-level inference ----
echo "[5/7] Infer higher-level query inferences"

if [[ "$PARALLEL_STEP5" -gt 1 ]]; then
	# Launch one infer_higher_level worker per query_id, each pinned to a
	# single GPU via CUDA_VISIBLE_DEVICES and filtered with --only-query-ids.
	# Each worker writes into a per-query subdir; we then concatenate the
	# combined files into the canonical $INFERENCES_QUERY_DIR layout that
	# Step 6 (generate_report.py) expects.
	S5_QIDS=$("$PYTHON_BIN" - "$QUERIES_JSONL" <<-'PY'
	import json, sys
	ids = []
	with open(sys.argv[1]) as f:
	    for line in f:
	        line = line.strip()
	        if line:
	            ids.append(str(json.loads(line)["query_id"]))
	print(" ".join(ids))
	PY
	)
	if [[ -z "$S5_QIDS" ]]; then
		echo "[error] no query_ids found in $QUERIES_JSONL"
		exit 1
	fi
	IFS=' ' read -r -a S5_QID_ARR <<< "$S5_QIDS"
	S5_N_QIDS=${#S5_QID_ARR[@]}
	S5_WORKERS_DIR="$INFERENCES_QUERY_DIR/_workers"
	mkdir -p "$S5_WORKERS_DIR"
	echo "[5/7] Parallel step 5: $S5_N_QIDS queries across $PARALLEL_STEP5 GPUs"

	# Step 5 always uses 1 GPU per worker; reset in case Step 1 left it at 2.
	GPUS_PER_WORKER=1
	pool_init "$PARALLEL_STEP5"
	for QID in "${S5_QID_ARR[@]}"; do
		pool_acquire "$PARALLEL_STEP5"
		GPU="$ACQUIRED_GPU"
		WORKER_OUT_DIR="$S5_WORKERS_DIR/q${QID}"
		WORKER_LOG="$INFERENCES_QUERY_DIR/step2_query_infer_q${QID}_gpu${GPU}.log"
		mkdir -p "$WORKER_OUT_DIR"
		log_banner "$WORKER_LOG"
		echo "[5/7] launching step5 worker for query=$QID on GPU $GPU -> $WORKER_LOG"
		CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" infer_higher_level.py \
			--stream query-based \
			--packets-dir "$CLAIM_PACKETS_DIR" \
			--claims "$CALIBRATED_QUERY_CLAIMS" \
			--queries-jsonl "$QUERIES_JSONL" \
			--model "$MODEL_NAME" \
			--out-dir "$WORKER_OUT_DIR" \
			--only-query-ids "$QID" \
			--verbose >> "$WORKER_LOG" 2>&1 &
		POOL_GPU_PID[$GPU]="$!"
	done
	pool_drain
	S5_FAIL="$POOL_FAIL"
	if [[ "$S5_FAIL" -ne 0 ]]; then
		echo "[error] one or more step 5 workers failed; see $INFERENCES_QUERY_DIR/step2_query_infer_q*_gpu*.log"
		exit 1
	fi

	# Merge per-worker outputs into the canonical $INFERENCES_QUERY_DIR.
	# Step 6 reads $INFERENCES_QUERY_DIR/inferences.jsonl (combined file).
	: > "$INFERENCES_QUERY_DIR/inferences.jsonl"
	: > "$INFERENCES_QUERY_DIR/skipped_packets.jsonl"
	for QID in "${S5_QID_ARR[@]}"; do
		WD="$S5_WORKERS_DIR/q${QID}"
		# Combined inferences (the file Step 6 actually reads).
		if [[ -s "$WD/inferences.jsonl" ]]; then
			cat "$WD/inferences.jsonl" >> "$INFERENCES_QUERY_DIR/inferences.jsonl"
		fi
		# Skipped-packets log (concatenated for completeness).
		if [[ -s "$WD/skipped_packets.jsonl" ]]; then
			cat "$WD/skipped_packets.jsonl" >> "$INFERENCES_QUERY_DIR/skipped_packets.jsonl"
		fi
		# Per-query files (preserved in canonical dir for parity with single-process mode).
		if [[ -f "$WD/query_${QID}.jsonl" ]]; then
			cp -f "$WD/query_${QID}.jsonl" "$INFERENCES_QUERY_DIR/query_${QID}.jsonl"
		fi
	done
	echo "[5/7] step 5 done; combined inferences -> $INFERENCES_QUERY_DIR/inferences.jsonl"
else
	"$PYTHON_BIN" infer_higher_level.py \
		--stream query-based \
		--packets-dir "$CLAIM_PACKETS_DIR" \
		--claims "$CALIBRATED_QUERY_CLAIMS" \
		--queries-jsonl "$QUERIES_JSONL" \
		--model "$MODEL_NAME" \
		--out-dir "$INFERENCES_QUERY_DIR" \
		--verbose 2>&1 | tee "$INFERENCES_QUERY_DIR/step2_query_infer.log"
fi

# ---- Step 3: Generate reports ----
echo "[6/7] Generate query-based reports"
"$PYTHON_BIN" generate_report.py \
	--stream query-based \
	--packets-dir "$CLAIM_PACKETS_DIR" \
	--claims "$CALIBRATED_QUERY_CLAIMS" \
	--inferences "$INFERENCES_QUERY_DIR" \
	--queries-jsonl "$QUERIES_JSONL" \
	--out-dir "$REPORTS_QUERY_DIR" \
	--verbose 2>&1 | tee "$REPORTS_QUERY_DIR/step3_query_report.log"

# ---- Format submission ----
TEAM_ID="${TEAM_ID:-ub_a2il}"
RUN_ID="${RUN_ID:-magmar_query-v5}"
TASK="${TASK:-oracle}"
SUBMISSION_JSONL="$OUT_DIR/submission.jsonl"

echo "[post] Format submission JSONL"
FORMAT_ARGS=(
	--reports "$REPORTS_QUERY_DIR/all_reports.json"
	--team-id "$TEAM_ID"
	--run-id "$RUN_ID"
	--task "$TASK"
	--out "$SUBMISSION_JSONL"
)
if [[ -f "$CHUNK_MAP_JSON" ]]; then
	FORMAT_ARGS+=(--chunk-map "$CHUNK_MAP_JSON")
fi

"$PYTHON_BIN" format_submission.py "${FORMAT_ARGS[@]}" 2>&1 | tee "$OUT_DIR/format_submission.log"

echo ""
echo "Done. Key outputs:"

echo "  - $QUERY_CLAIMS_DIR"
echo "  - $CLAIM_PACKETS_DIR"
echo "  - $INFERENCES_QUERY_DIR"
echo "  - $REPORTS_QUERY_DIR"
echo "  - $SUBMISSION_JSONL"

echo ""
echo "Logs saved in their respective output directories above."
