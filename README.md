# 🚀 CRAFT Critic-Refined Adaptive Key-Frame Targeting** for multimodal video question answering [ACL 2026 MAGMaR Workshop]

[Mahesh Bhosale*](https://bhosalems.github.io/)<sup>1</sup>, [Abdul Wasi*](https://scholar.google.com/citations?user=_2friTYAAAAJ&hl=en)<sup>1</sup>, [Vishvesh Trivedi*](https://github.com/NerdyVisky)<sup>2</sup>, [Pengyu Yan](https://scholar.google.com/citations?user=q2QMx5gAAAAJ&hl=en)<sup>1</sup>, [David Doermann](https://scholar.google.com/citations?user=RoGOW9AAAAAJ&hl=en)<sup>1</sup>.

**<sup>1</sup>University at Buffalo  |  <sup>2</sup>New York University **

[Paper](https://arxiv.org/abs/2605.19075v1)

## Overview

CRAFT is a query-conditioned multi-video QA pipeline for real-world news events. It performs multilingual ASR, adaptive keyframe selection, and a hybrid critic loop (UNLI temporal entailment + DeBERTa-v3 cross-claim screen + Llama-3.2-3B adjudicator) to verify and consolidate atomic claims into grounded, citation-backed reports. We evaluate on MAGMaR 2026 and a MAGMaR-style WikiVideo benchmark.

<p align="center">
  <img src="figures/teaser_final_MAGMAR.png" alt="CRAFT pipeline overview" width="80%"/>
</p>

See [PIPELINE.md](PIPELINE.md) for per-stage methods and [ASR_SETUP.md](ASR_SETUP.md) for the two-environment ASR pre-pass.

---

## 🚀 Quick Start

```bash
git clone https://github.com/bhosalems/CRAFT.git && cd CRAFT
conda create -n craft python=3.13 -y && conda activate craft
pip install -r requirements.txt
```

Prepare data ([📦 Datasets](#-datasets)), then run MAGMaR:

```bash
export VIDEO_ROOT=/path/to/MAGMaR2026_test
PARALLEL_QUERIES=8 PARALLEL_STEP15=8 PARALLEL_STEP5=8 \
    bash run_query.sh outputs/craft_magmar_main
```

Outputs: per-query reports at `outputs/craft_magmar_main/reports_query_based/all_reports.json` and a MAGMaR-format JSONL at `outputs/craft_magmar_main/submission.jsonl` for MIRAGE. All other knobs ship at reference values — see [Configuration defaults](#configuration-defaults).

> A separate **general-note** branch ([run_step1_general_notes.py](run_step1_general_notes.py), [run_note.sh](run_note.sh), [run_note_branch_pipeline.py](run_note_branch_pipeline.py)) is present but not end-to-end verified; this README covers only the query branch.

---

## 📦 Datasets

### MAGMaR-2026

```bash
export VIDEO_ROOT=/path/to/MAGMaR2026_test
huggingface-cli download akhilvssg/magmar-2026-test-asr-embeddings \
    --repo-type dataset --local-dir "$VIDEO_ROOT"
```

The release includes videos, `MAGMaR2026_queries{_dev,}.jsonl`, and a pre-built ASR cache at `$VIDEO_ROOT/asr/<video_id>.json`. Topic→video mappings ship in this repo.

### WikiVideo (MultiVENT 2.0)

```bash
export WIKIVIDEO_ROOT=/path/to/wikivideo
huggingface-cli download hltcoe/wikivideo \
    --repo-type dataset --local-dir "$WIKIVIDEO_ROOT"
```

Expected layout: `$WIKIVIDEO_ROOT/en/*.mp4` and `$WIKIVIDEO_ROOT/annotations/{final_data_2015-2025,multivent1_matched_queries_videos}.json`. Synthesised persona queries are shipped in [data/wikivideo_queries.jsonl](data/wikivideo_queries.jsonl); to regenerate, edit the path constants at [generate_wikivideo_queries.py:36-40](generate_wikivideo_queries.py#L36-L40) and run it.

### Chunking

Videos > 120 s are split into `<video_id>__chunk000.mp4`, … (originals untouched). The orchestrators do this automatically; manual invocation:

```bash
python chunk_videos.py \
    --video-root  "$VIDEO_ROOT" \
    --mapping-in  data/topic_video_mapping_dev.json \
    --mapping-out data/topic_video_mapping_dev_v2.json \
    --chunk-map-out data/video_chunk_map.json \
    --max-seconds 120
```

Needs `ffmpeg`/`ffprobe` on `PATH` (falls back to PyAV). Idempotent; `--force` to recreate. For WikiVideo, swap in `--video-root "$WIKIVIDEO_ROOT/en"` and the `*_wikivideo*` paths.

### ASR cache

Pre-pass; the pipeline only reads from `$ASR_DIR` (default `$VIDEO_ROOT/asr`, set empty to disable). MAGMaR's HF release ships it. To rebuild:

```bash
# Step 1 — Qwen3-ASR (30 langs):
python extract_asr.py --mode qwen \
    --video-root "$VIDEO_ROOT" --mapping data/topic_video_mapping_dev_v2.json \
    --out-dir "$VIDEO_ROOT/asr" --device cuda:0 --verbose

# Step 2 — omniASR for `needs_fallback: true` videos (1600+ langs, incl. Burmese/Nepali):
bash setup_asr_omni.sh && conda activate asr_omni
python extract_asr.py --mode omni \
    --video-root "$VIDEO_ROOT" --mapping data/topic_video_mapping_dev_v2.json \
    --out-dir "$VIDEO_ROOT/asr" --verbose
```

Two envs are required because `omnilingual-asr` pins `fairseq2` ≤ 0.6 (torch ≤ 2.9.1) while the main env runs torch 2.10+cu128. See [ASR_SETUP.md](ASR_SETUP.md).

### Data files

| File | Purpose |
|---|---|
| `data/topic_video_mapping{_dev,_test,}.json` | MAGMaR topics → video IDs (pre-chunk) |
| `data/topic_video_mapping*_v2.json` | post-chunk variant; auto-derived by `run_query.sh` |
| `data/video_chunk_map{_wikivideo,}.json` | chunk-id → `{video_id, start, end}` (parent-id remap at output time) |
| `data/topic_video_mapping_wikivideo{,_dev}.json` | WikiVideo events → video IDs |
| `data/wikivideo_queries{,_dev}.jsonl` | synthesised persona queries (`query_id`, `query_type`, `language`, `title`, `persona_title`, `background`, `query`) |
| `data/dev/` | smoke-test fixtures |

---

## Running on WikiVideo

Same orchestrator; Stage 1b defaults to `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`, Stage 2b stays on `Qwen/Qwen3.5-9B`.

```bash
export WIKIVIDEO_ROOT=/path/to/wikivideo
SKIP_CHUNK=1 \
VIDEO_ROOT="$WIKIVIDEO_ROOT/en" ASR_DIR="$WIKIVIDEO_ROOT/asr" \
PARALLEL_QUERIES=8 PARALLEL_STEP15=8 PARALLEL_STEP5=8 \
    bash run_query_wikivideo.sh outputs/craft_wikivideo_main
```

### Configuration defaults

| Env var | MAGMaR | WikiVideo | Controls |
|---|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | Stage 1b extractor (Hydra `model.model=…`) |
| `STAGE2_MODEL_NAME` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B` | Stage 2b consolidator |
| `MAX_CRITIC_ROUNDS` | `4` | `4` | per-video critic re-extraction rounds |
| `COVERAGE_FOLLOWUP_ROUNDS` | `1` | `1` | re-prompts when coverage gaps are flagged |
| `CRITIC_NLI_ENABLED` / `CRITIC_COVERAGE_ENABLED` | `true` | `true` | DeBERTa-v3 contradiction screen / Llama-3.2-3B coverage audit |
| `STEP15_CHUNK_SIZE` | `10` | `10` | Stage 1.5 sub-batch size |
| `GPU_MEM_UTIL` | `0.85` | `0.85` | vLLM `gpu_memory_utilization` |
| `PARALLEL_QUERIES` / `_STEP15` / `_STEP5` | `1` | `1` | parallel workers — set to GPU count |
| `ASR_DIR` | `$VIDEO_ROOT/asr` | same | empty disables ASR |
| `SKIP_CHUNK` / `SKIP_STEP1` | `auto` | `auto` | `1` = skip, `auto` = skip if outputs exist |

Atomic claims are produced by the prompts ([prompts.py](prompts.py)); the post-hoc `format_submission.py --atomize` splitter is off by default. The Hydra config ([conf/step1_query_claims.yaml](conf/step1_query_claims.yaml)) also defaults to MAGMaR (`override /model: qwen3_5_vl`) for direct invocations of `run_step1_query_claims.py`; orchestrators override at runtime via `model.model=$MODEL_NAME`.

---

## Evaluation

Scored with the **MIRAGE** judge (Qwen2.5-7B-Instruct) on `info_f1` and `cite_f1`. Point MIRAGE's `run.sh` at the JSONL produced by the orchestrator.
