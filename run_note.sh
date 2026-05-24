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

PYTHON_BIN="./.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "[error] Missing venv interpreter at $PYTHON_BIN"
	exit 1
fi

# Inputs (edit these if your paths differ).
QUERIES_JSONL="/a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries.jsonl"
MAPPING_JSON="data/topic_video_mapping.json"
MODEL_NAME="Qwen/Qwen3.5-9B"

# Step 1 intermediate outputs for note-based branch.
GENERAL_NOTES_DIR="outputs_notes_branchv1/general_notes_single"
UNLI_GENERAL_NOTES_DIR="outputs_notes_branchv1/unli_general_notes_single_lora"
GENERAL_NOTES_JSONL="$GENERAL_NOTES_DIR/general_notes.jsonl"
UNLI_PREDICTIONS_JSONL="$UNLI_GENERAL_NOTES_DIR/unli_predictions.jsonl"
CALIBRATED_GENERAL_NOTES="$UNLI_GENERAL_NOTES_DIR/general_notes_calibrated.jsonl"

# outputs_notes_branchv1.
NOTE_PACKETS_DIR="outputs_notes_branchv1v1/note_packets"
INFERENCES_NOTE_DIR="outputs_notes_branchv1/inferences_note"
REPORTS_NOTE_DIR="outputs_notes_branchv1/reports_note_based"
EVAL_NOTE_DIR="outputs_notes_branchv1/evaluation_note_based"

# Set to 1 to apply the optional vLLM patch helper before inference.
APPLY_VLLM_PATCH="${APPLY_VLLM_PATCH:-0}"

# Logs.
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ "$APPLY_VLLM_PATCH" == "1" && -f "sbatch/patch_vllm_qwen3_next.py" ]]; then
	echo "[setup] Applying optional vLLM patch"
	"$PYTHON_BIN" sbatch/patch_vllm_qwen3_next.py 2>&1 | tee "$LOG_DIR/setup_patch_vllm.log"
fi

echo "[note-based] Running explicit 4-stage downstream note pipeline"

echo "[1/7] Extract general notes (Step 1a)"
"$PYTHON_BIN" run_step1_general_notes.py \
	data.mapping="$MAPPING_JSON" \
	output.out_dir="$GENERAL_NOTES_DIR" 2>&1 | tee "$LOG_DIR/step1a_general_notes.log"

echo "[2/7] Predict UNLI support for general notes (Step 1.5)"
"$PYTHON_BIN" run_step1_5_predict_unli.py \
	artifact_type=general-notes \
	data.artifacts_jsonl="$GENERAL_NOTES_JSONL" \
	output.out_dir="$UNLI_GENERAL_NOTES_DIR" 2>&1 | tee "$LOG_DIR/step1_5_note_predict.log"

echo "[3/7] Calibrate UNLI scores into notes (Step 1.5)"
"$PYTHON_BIN" run_step1_5_calibrate_unli.py \
	artifact_type=general-notes \
	data.claims_jsonl="$GENERAL_NOTES_JSONL" \
	data.unli_jsonl="$UNLI_PREDICTIONS_JSONL" \
	output.out_dir="$UNLI_GENERAL_NOTES_DIR" 2>&1 | tee "$LOG_DIR/step1_5_note_calibrate.log"

echo "[4/7] Assemble note packets"
"$PYTHON_BIN" assemble_packets.py \
	--stream note-based \
	--claims "$CALIBRATED_GENERAL_NOTES" \
	--queries-jsonl "$QUERIES_JSONL" \
	--mapping "$MAPPING_JSON" \
	--out-dir "$NOTE_PACKETS_DIR" \
	--verbose 2>&1 | tee "$LOG_DIR/step2_note_assemble.log"

echo "[5/7] Infer higher-level note inferences"
"$PYTHON_BIN" infer_higher_level.py \
	--stream note-based \
	--packets-dir "$NOTE_PACKETS_DIR" \
	--claims "$CALIBRATED_GENERAL_NOTES" \
	--queries-jsonl "$QUERIES_JSONL" \
	--model "$MODEL_NAME" \
	--out-dir "$INFERENCES_NOTE_DIR" \
	--verbose 2>&1 | tee "$LOG_DIR/step2_note_infer.log"

echo "[6/7] Generate note-based reports"
"$PYTHON_BIN" generate_report.py \
	--stream note-based \
	--packets-dir "$NOTE_PACKETS_DIR" \
	--claims "$CALIBRATED_GENERAL_NOTES" \
	--inferences "$INFERENCES_NOTE_DIR" \
	--queries-jsonl "$QUERIES_JSONL" \
	--out-dir "$REPORTS_NOTE_DIR" \
	--verbose 2>&1 | tee "$LOG_DIR/step3_note_report.log"

echo "[7/7] Evaluate note-based outputs"
"$PYTHON_BIN" evaluate.py \
	--queries-jsonl "$QUERIES_JSONL" \
	--mapping "$MAPPING_JSON" \
	--general-notes "$CALIBRATED_GENERAL_NOTES" \
	--note-packets "$NOTE_PACKETS_DIR" \
	--inferences-note "$INFERENCES_NOTE_DIR" \
	--reports-note-based "$REPORTS_NOTE_DIR" \
	--out-dir "$EVAL_NOTE_DIR" \
	--verbose 2>&1 | tee "$LOG_DIR/step4_note_eval.log"

echo ""
echo "Done. Key outputs:"
echo "  - $NOTE_PACKETS_DIR"
echo "  - $INFERENCES_NOTE_DIR"
echo "  - $REPORTS_NOTE_DIR"
echo "  - $EVAL_NOTE_DIR"

echo ""
echo "  - $LOG_DIR"
