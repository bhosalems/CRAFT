# 🚀 CRAFT Critic-Refined Adaptive Key-Frame Targeting** for multimodal video question answering [ACL 2026 MAGMaR Workshop]

[Mahesh Bhosale*](https://bhosalems.github.io/)<sup>1</sup>, [Abdul Wasi*](https://scholar.google.com/citations?user=_2friTYAAAAJ&hl=en)<sup>1</sup>, [Vishvesh Trivedi*](https://github.com/NerdyVisky)<sup>2</sup>, [Pengyu Yan](https://scholar.google.com/citations?user=q2QMx5gAAAAJ&hl=en)<sup>1</sup>, [David Doermann](https://scholar.google.com/citations?user=RoGOW9AAAAAJ&hl=en)<sup>1</sup>.

**<sup>1</sup>University at Buffalo  |  <sup>2</sup>New York University **

[Paper](https://arxiv.org/abs/2605.19075v1)

## Overview

CRAFT is a query-conditioned multi-video question answering pipeline for real-world news events. It retrieves query-relevant evidence across heterogeneous video collections, performs per-video ASR with multilingual fallback, selects adaptive keyframes, and uses a hybrid critic loop to verify, repair, and consolidate claims. By combining UNLI temporal entailment, DeBERTa-v3 cross-claim screening, and Llama-3.2-3B adjudication, CRAFT produces grounded reports where each atomic fact is emitted once and linked to all supporting source videos. The system achieves strong performance on MAGMaR 2026 and a MAGMaR-style WikiVideo benchmark, demonstrating robust claim-centric evidence aggregation beyond a single dataset.

<p align="center">
  <img src="figures/teaser_final_MAGMAR.png" alt="CRAFT pipeline overview" width="80%"/>
</p>

See [PIPELINE.md](PIPELINE.md) for the per-stage methods, models, and design rationale, and [ASR_SETUP.md](ASR_SETUP.md) for setting up the two-environment ASR pre-pass.

---

## 🚀 Quick Start

### ⚙️ Environment setup

```bash
git clone https://github.com/bhosalems/CRAFT.git
cd CRAFT
# create and activate a conda virtual environment
conda create -n craft python=3.13 -y
conda activate craft
# install the main pipeline dependencies
pip install -r requirements.txt
# video decode backend used by Stage 1b / 1.5 (avoids torchvision mmap leaks)
pip install decord
```

`requirements.txt` covers the main `.venv` (Stage 1b, 1.5, 2b, 3). The omniASR fallback (Burmese / Nepali / 1600+) needs an isolated env - run `bash setup_asr_omni.sh` once, then see [§3.2](#32-asr-cache-stage-05) for the two-step ASR workflow.

### ▶️ Reproducing the main MAGMaR-2026 result

Once you have prepared the data (see [📦 Datasets](#-datasets) below) and exported `VIDEO_ROOT` to the directory holding the MAGMaR mp4s, the reference configuration used to produce the headline numbers in the paper is:

```bash
export VIDEO_ROOT=/path/to/MAGMaR2026_test          # set once

MODEL_NAME=Qwen/Qwen3.6-27B-FP8 \
PARALLEL_QUERIES=8 PARALLEL_STEP15=8 PARALLEL_STEP5=8 \
MAX_CRITIC_ROUNDS=4 COVERAGE_FOLLOWUP_ROUNDS=1 \
CRITIC_NLI_ENABLED=true CRITIC_COVERAGE_ENABLED=true \
STEP15_CHUNK_SIZE=10 GPU_MEM_UTIL=0.85 \
ASR_DIR="$VIDEO_ROOT/asr" \
bash run_query.sh outputs/craft_magmar_main
```

The final per-query reports land in `outputs/craft_magmar_main/reports_query_based/all_reports.json`, plus a MAGMaR-format JSONL at `outputs/craft_magmar_main/submission.jsonl` that MIRAGE can score directly (see [Evaluation](#evaluation) below).

> **Note-taking branch.** A parallel **general-note** stream — [run_step1_general_notes.py](run_step1_general_notes.py), [run_note.sh](run_note.sh), [run_note_branch_pipeline.py](run_note_branch_pipeline.py) — produces query-agnostic per-video notes intended as an alternative grounding substrate. It is **present but not verified end-to-end on the current branch**, and this README documents only the query branch.

---

## 📦 Datasets

CRAFT runs on two benchmarks. All commands below are written against environment variables (`$VIDEO_ROOT`, `$ASR_DIR`) so the same scripts work on any machine — set them once, point at your local layout, and the rest is path-free.

### MAGMaR-2026

> Hand-curated, persona-augmented queries (8 dev + 19 test) over a topic-organised video collection. Topics → video lists ship inside this repo; you only need the raw videos and the official queries file.

**1. Get the videos and queries.** The MAGMaR-2026 test release (videos, queries, and pre-computed ASR embeddings) is distributed on the Hugging Face Hub at [akhilvssg/magmar-2026-test-asr-embeddings](https://huggingface.co/datasets/akhilvssg/magmar-2026-test-asr-embeddings). Download it and point `VIDEO_ROOT` at the result:

```bash
export VIDEO_ROOT=/path/to/MAGMaR2026_test
huggingface-cli download akhilvssg/magmar-2026-test-asr-embeddings \
    --repo-type dataset --local-dir "$VIDEO_ROOT"
# expected layout after download:
#   $VIDEO_ROOT/*.mp4
#   $VIDEO_ROOT/MAGMaR2026_queries_dev.jsonl
#   $VIDEO_ROOT/MAGMaR2026_queries.jsonl
```

The topic→video mapping (`data/topic_video_mapping{_dev,_test,}.json`) is committed in this repo — no extra download needed.

**2. Chunk long videos.** Stage 1b's VLM has a fixed frame budget, so videos > 120 s are split into ≤ 120 s MP4s alongside the originals (`<video_id>__chunk000.mp4`, `__chunk001.mp4`, …). A chunk map records `chunk_id → {video_id, start, end}` so the report-formatting step can map the chunked IDs back to their parent videos. Original MP4s are never modified or deleted; chunks are written with `stream copy` when possible (re-encode otherwise), so chunking is fast.

`run_query.sh` invokes this automatically (`SKIP_CHUNK=auto` — runs if `*_v2.json` and the chunk map are missing, skips otherwise), or you can run it manually:

```bash
python chunk_videos.py \
    --video-root "$VIDEO_ROOT" \
    --mapping-in   data/topic_video_mapping_dev.json \
    --mapping-out  data/topic_video_mapping_dev_v2.json \
    --chunk-map-out data/video_chunk_map.json \
    --max-seconds 120
```

The script needs **ffmpeg + ffprobe on `PATH`** (it falls back to PyAV if either is missing, but ffmpeg is much faster). It runs idempotently — re-running is a no-op when all expected chunk MP4s already exist; pass `--force` to recreate them. Pass `--mapping-in` / `--mapping-out` multiple times to chunk against multiple mappings in a single sweep (the chunk MP4s are shared).

**3. ASR cache.** ASR is a **pre-pass** that the pipeline only *reads from*. Two backends are needed because their Python deps conflict: Qwen3-ASR runs in the main venv (30 langs); **omniASR** runs in an isolated `asr_omni` conda env (1600+ langs, required for Burmese / Nepali). Full reasoning in [ASR_SETUP.md](ASR_SETUP.md).

**The easy path:** the MAGMaR-2026 HF dataset above already ships the JSON ASR cache the pipeline consumes, so once `huggingface-cli download` finishes you should already have `$VIDEO_ROOT/asr/<video_id>.json` for every video. Skip to Stage 1b — no further ASR work required.

If you do not see the `asr/` subdirectory in the download, build it from scratch:

```bash
# Step 1 — Qwen3-ASR in the main venv (30 languages):
python extract_asr.py --mode qwen \
    --video-root "$VIDEO_ROOT" \
    --mapping    data/topic_video_mapping_dev_v2.json \
    --out-dir    "$VIDEO_ROOT/asr" \
    --device cuda:0 --verbose

# Step 2 — omniASR fallback for videos flagged `needs_fallback: true`
#          (Burmese Q3, Nepali Q4, etc.):
bash setup_asr_omni.sh           # one-shot installer for the asr_omni conda env
conda activate asr_omni
python extract_asr.py --mode omni \
    --video-root "$VIDEO_ROOT" \
    --mapping    data/topic_video_mapping_dev_v2.json \
    --out-dir    "$VIDEO_ROOT/asr" \
    --verbose
```

Stage 1b reads transcripts from `$ASR_DIR` automatically (defaults to `$VIDEO_ROOT/asr`); set `ASR_DIR=` to disable ASR for an A/B run.

---

### WikiVideo (MultiVENT 2.0)

> 52 news events from the MultiVENT 2.0 subset of WikiVideo, scored MAGMaR-style. Persona/background queries are synthesised from each event's Wikipedia article using a few-shot prompt seeded from the MAGMaR dev set.

**1. Get the videos and annotations.** WikiVideo / MultiVENT 2.0 is distributed by HLTCOE on the Hugging Face Hub at [hltcoe/wikivideo](https://huggingface.co/datasets/hltcoe/wikivideo):

```bash
export WIKIVIDEO_ROOT=/path/to/wikivideo
huggingface-cli download hltcoe/wikivideo \
    --repo-type dataset --local-dir "$WIKIVIDEO_ROOT"
# expected layout after download:
#   $WIKIVIDEO_ROOT/en/*.mp4                                  # videos
#   $WIKIVIDEO_ROOT/annotations/final_data_2015-2025.json     # per-event claims + gold article
#   $WIKIVIDEO_ROOT/annotations/multivent1_matched_queries_videos.json
```

**2. Generate persona-augmented queries** (one-time — output already shipped in this repo at [data/wikivideo_queries.jsonl](data/wikivideo_queries.jsonl), so this step is optional unless you change the few-shot seed):

```bash
python generate_wikivideo_queries.py \
    --out-queries  data/wikivideo_queries.jsonl \
    --out-mapping  data/topic_video_mapping_wikivideo.json
```

The script reads the MultiVENT matched-queries + annotation JSONs from `WIKIVIDEO_ROOT` and the MAGMaR dev queries from `DEV_QUERIES`, both currently set as module constants near the top of [generate_wikivideo_queries.py](generate_wikivideo_queries.py#L36-L40). Update those two paths to match your local layout before running; the script fabricates a `(persona_title, background, query)` per event with a local Qwen vLLM model.

**3. Chunk the videos** (same procedure as MAGMaR):

```bash
python chunk_videos.py \
    --video-root "$WIKIVIDEO_ROOT/en" \
    --mapping-in   data/topic_video_mapping_wikivideo.json \
    --mapping-out  data/topic_video_mapping_wikivideo_v2.json \
    --chunk-map-out data/video_chunk_map_wikivideo.json \
    --max-seconds 120
```

**4. ASR cache** — same two-step Qwen + omniASR workflow as MAGMaR, just with WikiVideo paths:

```bash
python extract_asr.py --mode qwen \
    --video-root "$WIKIVIDEO_ROOT/en" \
    --mapping    data/topic_video_mapping_wikivideo_v2.json \
    --out-dir    "$WIKIVIDEO_ROOT/asr" \
    --device cuda:0 --verbose

# Then the omniASR fallback for videos flagged `needs_fallback: true`:
conda activate asr_omni
python extract_asr.py --mode omni \
    --video-root "$WIKIVIDEO_ROOT/en" \
    --mapping    data/topic_video_mapping_wikivideo_v2.json \
    --out-dir    "$WIKIVIDEO_ROOT/asr" \
    --verbose
```

---

### Data files reference

What ships in [data/](data/) (so you can tell which file the orchestrator wants):

| File | Purpose |
|---|---|
| [data/topic_video_mapping.json](data/topic_video_mapping.json) | MAGMaR test split — 10 topics → video IDs (pre-chunk) |
| [data/topic_video_mapping_dev.json](data/topic_video_mapping_dev.json) | MAGMaR dev split — 8 topics → video IDs (pre-chunk) |
| [data/topic_video_mapping_test.json](data/topic_video_mapping_test.json) | MAGMaR official test split alias |
| `data/topic_video_mapping*_v2.json` | post-chunking mapping; same topics but video IDs include `__chunkNNN` suffixes. **`run_query.sh` derives this name from the input and reads/writes it automatically.** |
| [data/video_chunk_map.json](data/video_chunk_map.json) | chunk-id → `{video_id, start, end}` — used by the report-formatting step to remap chunk IDs back to parent video IDs |
| [data/topic_video_mapping_wikivideo.json](data/topic_video_mapping_wikivideo.json) | WikiVideo — 52 events → video IDs (pre-chunk) |
| [data/topic_video_mapping_wikivideo_dev.json](data/topic_video_mapping_wikivideo_dev.json) | WikiVideo dev fragment (4 events overlapping the MAGMaR dev set) |
| [data/video_chunk_map_wikivideo.json](data/video_chunk_map_wikivideo.json) | chunk-id → parent map for WikiVideo |
| [data/wikivideo_queries.jsonl](data/wikivideo_queries.jsonl) | 52 synthesised persona queries (one line per query) |
| [data/wikivideo_queries_dev.jsonl](data/wikivideo_queries_dev.jsonl) | dev subset of the WikiVideo queries |
| [data/dev/](data/dev/) | small fixture dev split + sample queries used by smoke tests |

Each line of a `queries*.jsonl` file is one query record with: `query_id`, `query_type` (biased / unbiased), `language`, `title` (event / topic), `persona_title`, `background`, `query`. Each value of a `topic_video_mapping*.json` is a list of video IDs whose extension is `.mp4` under `$VIDEO_ROOT`.

---

## Running on WikiVideo

Same orchestrator as MAGMaR, configured for the WikiVideo paths and queries. For WikiVideo we extract with **`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`** — the 30B-A3B mixture-of-experts VL model, FP8-quantised — rather than the dense 27B used on MAGMaR. WikiVideo has ~8× more videos per topic, so the larger sparse expert set both fits the per-card budget at FP8 and is more sample-efficient across the longer-tail topic mix:

```bash
export WIKIVIDEO_ROOT=/path/to/wikivideo

SKIP_CHUNK=1 \
MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
PARALLEL_QUERIES=8 PARALLEL_STEP15=8 PARALLEL_STEP5=8 \
MAX_CRITIC_ROUNDS=2 COVERAGE_FOLLOWUP_ROUNDS=1 \
CRITIC_NLI_ENABLED=true CRITIC_COVERAGE_ENABLED=true \
STEP15_CHUNK_SIZE=10 GPU_MEM_UTIL=0.95 \
VIDEO_ROOT="$WIKIVIDEO_ROOT/en" \
ASR_DIR="$WIKIVIDEO_ROOT/asr" \
bash run_query_wikivideo.sh outputs/craft_wikivideo_main
```

Two values differ from the MAGMaR command above: `MAX_CRITIC_ROUNDS=2` (the 30B-A3B model already produces tighter atomic claims, so two repair rounds suffice) and `GPU_MEM_UTIL=0.95` (the FP8 weights plus the UNLI critic still leave KV-cache headroom at this fraction). Critic config, calibration, packet assembly, inference, and report assembly are otherwise identical to the MAGMaR run — only the data inputs and the extractor model differ.

---

## Evaluation

We score CRAFT with the **MIRAGE** judge (Qwen2.5-7B-Instruct), which evaluates per-claim entailment against each query's gold reference and reports `info_f1` (information precision/recall) and `cite_f1` (citation precision/recall). MIRAGE lives in a separate repository — point its `run.sh` at the JSONL emitted at the end of `run_query.sh` / `run_query_wikivideo.sh`. The orchestrators stop after writing that file and do not invoke any in-tree evaluator.
