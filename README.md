# CRAFT

**C**laim **R**etrieval, **A**tomization, and **F**aithful **T**ranscription — the query-conditioned VLM pipeline behind our MAGMaR-2026 submissions and our WikiVideo experiments.

Given a set of persona-augmented queries and the videos associated with each query's topic, CRAFT extracts grounded atomic claims from the video (and its ASR transcript), calibrates per-claim confidence with a video-text NLI model, consolidates the surviving claims into higher-level inferences, and emits a citation-backed report in the MAGMaR-2026 submission schema.

See [PIPELINE.md](PIPELINE.md) for the per-stage methods, models, and design rationale, and [ASR_SETUP.md](ASR_SETUP.md) for setting up the two-environment ASR pre-pass.

---

## 1. Pipeline at a glance

```
Stage 0    chunk_videos.py            split videos > 120s into MP4 chunks
Stage 0.5  extract_asr.py             per-video transcripts (Qwen3-ASR + omni/whisper fallback)
Stage 1b   extract_query_claims.py    VLM claim extraction + critic loop + coverage audit
Stage 1.5  predict_unli.py            video-grounded NLI scoring
           calibrate_unli.py          isotonic calibration -> per-claim probability
Stage 2a   assemble_packets.py        top-K calibrated claims per query
Stage 2b   infer_higher_level.py      atomic consolidation, dedup via citation merging
Stage 3    generate_report.py         section text + source_citations
Submit     format_submission.py       chunk-id -> parent-id remap, JSONL submission
```

All stages are content-addressed and idempotent: re-running with the same inputs is a no-op for any stage whose cache is already populated. Parallelism is applied at the three GPU-bound stages (`PARALLEL_QUERIES`, `PARALLEL_STEP15`, `PARALLEL_STEP5`).

---

## 2. Project structure

### Orchestrators
| File | Purpose |
|---|---|
| [run_query.sh](run_query.sh) | end-to-end MAGMaR pipeline (Stage 0 → submission) |
| [run_query_wikivideo.sh](run_query_wikivideo.sh) | same orchestrator, configured for WikiVideo paths and queries |
| [run_query_ablation.sh](run_query_ablation.sh) | critic-rounds / no-ASR / no-AKS ablations |
| [run_ablation_critic_rounds.sh](run_ablation_critic_rounds.sh) | sweeps `MAX_CRITIC_ROUNDS ∈ {0..N}` |

### Stage entry points
| File | Stage |
|---|---|
| [chunk_videos.py](chunk_videos.py) | 0 — PyAV cutpoint splitting at 120 s boundaries |
| [extract_asr.py](extract_asr.py) | 0.5 — Qwen3-ASR / Whisper / omniASR backends, loop detection, translation |
| [run_step1_query_claims.py](run_step1_query_claims.py) → [extract_query_claims.py](extract_query_claims.py) | 1b — VLM extraction with critic loop, coverage audit, topic fallback |
| [run_step1_5_predict_unli.py](run_step1_5_predict_unli.py) → [predict_unli.py](predict_unli.py) / [predict_unli_chunked.py](predict_unli_chunked.py) | 1.5 — UNLI scoring |
| [run_step1_5_calibrate_unli.py](run_step1_5_calibrate_unli.py) → [calibrate_unli.py](calibrate_unli.py) | 1.5 — calibration attachment |
| [assemble_packets.py](assemble_packets.py) | 2a — top-K packet assembly |
| [infer_higher_level.py](infer_higher_level.py) | 2b — atomic consolidation |
| [generate_report.py](generate_report.py) | 3 — report assembly |
| [format_submission.py](format_submission.py) | submission — chunk-id remap |

### Configuration
| Path | Purpose |
|---|---|
| [conf/](conf/) | Hydra config tree (`step1_query_claims.yaml`, `step1_5_*.yaml`, `runtime/`, `model/`) |
| [prompts.py](prompts.py) | all VLM / critic / coverage / inference prompts |
| [contracts.py](contracts.py) | JSON schemas + `resolve_video_path()` (AKS-first, chunk-fallback) |

### Auxiliary
| File | Purpose |
|---|---|
| [generate_wikivideo_queries.py](generate_wikivideo_queries.py) | builds `data/wikivideo_queries.jsonl` + mapping from MultiVENT annotations |
| [setup_asr_omni.sh](setup_asr_omni.sh) | one-shot installer for the `asr_omni` conda env (see §3.2) |
| [data/](data/) | topic→video mappings (`topic_video_mapping*.json`), chunk maps, queries |
| [sbatch/](sbatch/) | SLURM wrappers + optional vLLM patches |

> **Note-taking branch.** A parallel **general-note** stream — [extract_general_notes.py](extract_general_notes.py), [run_step1_general_notes.py](run_step1_general_notes.py), [run_note.sh](run_note.sh), [run_note_branch_pipeline.py](run_note_branch_pipeline.py) — produces query-agnostic per-video notes intended as an alternative grounding substrate. It is **present but not verified end-to-end on the current branch**, and this README documents only the query branch.

---

## 3. One-time setup

### 3.1 Python environment

The pipeline uses the shared SCALE venv at [/home/csgrad/mbhosale/phd/SCALE/.venv](/home/csgrad/mbhosale/phd/SCALE/.venv). If you are starting from scratch:

```bash
cd /home/csgrad/mbhosale/phd/SCALE
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r CRAFT/requirements.txt
# Video decode backend used by Stage 1b / 1.5 (avoids torchvision mmap leaks):
pip install decord
```

`run_query.sh` auto-activates `./.venv` if present and otherwise expects the interpreter at the shared path above (see `PYTHON_BIN` at the top of [run_query.sh](run_query.sh)).

### 3.2 ASR cache (Stage 0.5)

ASR is a **pre-pass**. The query pipeline only **consumes** a cache directory of per-video transcripts — it never generates one. This is by design: `omnilingual-asr` is pinned to `fairseq2 ∈ [0.5.2, 0.6.0]` whose wheels only exist for torch ≤ 2.9.1, while the main pipeline's `.venv` runs torch 2.10+cu128. The two cannot coexist in one Python env, so they communicate only via the on-disk cache.

The two backends together give us the language coverage we need:

| Backend | Coverage | Where it runs |
|---|---|---|
| **Qwen3-ASR-1.7B** | 30 languages incl. EN / ZH / YUE / TH | main `.venv` |
| **omniASR-LLM-7B** | 1600+ languages — required for **Burmese** (Q3) and **Nepali** (Q4) on the MAGMaR dev set | isolated `asr_omni` conda env |
| Whisper-large-v3 | fallback + non-English → English translation | main `.venv` |

The full workflow lives in [ASR_SETUP.md](ASR_SETUP.md). The two key steps:

**Step 1 — Qwen3-ASR in the main venv** (covers 30 languages):

```bash
.venv/bin/python extract_asr.py \
    --mode qwen \
    --video-root /a2il/data/mbhosale/MAGMaR2026_test \
    --mapping   data/topic_video_mapping_dev_v2.json \
    --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \
    --device cuda:0 --verbose
```

Videos whose detected language is outside Qwen3-ASR's 30-language set get an empty transcript and `"needs_fallback": true`. Step 2 fills those in.

**Step 2 — omniASR in an isolated conda env** (covers everything else). Use the one-shot installer in this repo, which automates env creation, pins torch 2.8.0 + cu128, installs fairseq2 0.6 from Meta's wheel index, and verifies the imports:

```bash
bash setup_asr_omni.sh                     # creates conda env 'asr_omni'
conda activate asr_omni
python extract_asr.py \
    --mode omni \
    --video-root /a2il/data/mbhosale/MAGMaR2026_test \
    --mapping   data/topic_video_mapping_dev_v2.json \
    --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \
    --verbose
```

omniASR only touches videos that need it (missing cache OR `needs_fallback: true`), so re-running is cheap. See [ASR_SETUP.md §Step 2](ASR_SETUP.md) for the manual install path, troubleshooting (`libcudart.so.13` / `requires PyTorch 2.8.0`), and the venv-instead-of-conda variant.

**Step 3 — done.** The Stage 1b VLM prompt picks up transcripts from `$ASR_DIR` automatically (default `$VIDEO_ROOT/asr`). Set `ASR_DIR=` (empty) to disable ASR for an A/B run.

Cache schema (one file per video, content-addressed by `video_id`):

```jsonc
{
  "video_id": "...",
  "asr_model": "Qwen/Qwen3-ASR-1.7B" | "facebook/omniASR-LLM-7B" | "openai/whisper-large-v3",
  "language": "English" | "Burmese" | ...,
  "text": "...",
  "text_en": "...",              // only for non-English (Whisper translate pass)
  "asr_loop_detected": false,    // true => text is suppressed
  "no_audio": false,
  "needs_fallback": false        // true => Qwen3-ASR couldn't handle it; run omniASR
}
```

### 3.3 Data layout

| What | Where (default) |
|---|---|
| MAGMaR-2026 videos | `/a2il/data/mbhosale/MAGMaR2026_test/*.mp4` |
| MAGMaR-2026 ASR cache | `/a2il/data/mbhosale/MAGMaR2026_test/asr/` |
| MAGMaR queries (dev / test) | `/a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries{_dev,}.jsonl` |
| MAGMaR topic→video mapping | [data/topic_video_mapping_dev.json](data/topic_video_mapping_dev.json), [data/topic_video_mapping.json](data/topic_video_mapping.json) |
| WikiVideo videos | `/a2il/data/mbhosale/wikivideo/en/*.mp4` |
| WikiVideo ASR cache | `/a2il/data/mbhosale/wikivideo/asr/` |
| WikiVideo queries | [data/wikivideo_queries.jsonl](data/wikivideo_queries.jsonl) |
| WikiVideo topic→video mapping | [data/topic_video_mapping_wikivideo.json](data/topic_video_mapping_wikivideo.json) |

---

## 4. Running on MAGMaR-2026 (V12 parameters)

V12 was the best-performing dev-set configuration: 8 queries, 8 GPUs, one worker per query, Qwen3.6-27B-FP8 as the extractor, full critic loop with coverage audit, and ASR enabled.

### 4.1 Prerequisites

- ASR cache populated at `/a2il/data/mbhosale/MAGMaR2026_test/asr/` (see [ASR_SETUP.md](ASR_SETUP.md)).
- 8 visible GPUs (A6000-class or better — Qwen3.6-27B-FP8 + UNLI critic co-locate on one card).
- The pre-chunked mapping `data/topic_video_mapping_dev_v2.json` already exists; otherwise Stage 0 will produce it.

### 4.2 Command

```bash
cd /home/csgrad/mbhosale/phd/SCALE/CRAFT

TEAM_ID=cite_chasers \
RUN_ID=magmar_query_v12 \
TASK=oracle \
MODEL_NAME=Qwen/Qwen3.6-27B-FP8 \
PARALLEL_QUERIES=8 \
PARALLEL_STEP15=8 \
PARALLEL_STEP5=8 \
MAX_CRITIC_ROUNDS=4 \
COVERAGE_FOLLOWUP_ROUNDS=1 \
CRITIC_NLI_ENABLED=true \
CRITIC_COVERAGE_ENABLED=true \
STEP15_CHUNK_SIZE=10 \
GPU_MEM_UTIL=0.85 \
ASR_DIR=/a2il/data/mbhosale/MAGMaR2026_test/asr \
bash run_query.sh outputs/outputs_query_branchv12
```

### 4.3 What each V12 knob does

| Knob | V12 value | Effect |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.6-27B-FP8` | Stage 1b extractor and Stage 2b consolidator |
| `MAX_CRITIC_ROUNDS` | `4` | per-video critic re-extraction rounds (UNLI + MNLI + LLM-adjudicator) |
| `COVERAGE_FOLLOWUP_ROUNDS` | `1` | one targeted re-prompt sweep per query when the coverage auditor flags gaps |
| `CRITIC_NLI_ENABLED` | `true` | DeBERTa-v3 MNLI contradiction screen between claim pairs |
| `CRITIC_COVERAGE_ENABLED` | `true` | Llama-3.2-3B coverage audit + adjudicator |
| `STEP15_CHUNK_SIZE` | `10` | Stage 1.5 runs `predict_unli_chunked.py`, restarting every 10 videos to release libav mmaps |
| `GPU_MEM_UTIL` | `0.85` | vLLM `gpu_memory_utilization`; co-locates with UNLI on the same card |
| `PARALLEL_QUERIES` | `8` | one worker per query, each pinned to its own GPU via `CUDA_VISIBLE_DEVICES` |
| `ASR_DIR` | populated | enables persona+ASR+visual prompting; required to recover Burmese/Nepali content |
| `TEAM_ID` / `RUN_ID` / `TASK` | submission metadata | written into the final `submission.jsonl` |

Defaults set inside `extract_query_claims.py` and `conf/step1_query_claims.yaml` already match V12 for everything else (UNLI critic `AdoptedIrelia/UNLI` + LoRA, MNLI screener `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, coverage LLM `meta-llama/Llama-3.2-3B-Instruct`, `temperature=0.3`, `top_p=0.8`, `top_k=20`, `max_tokens=2048`, `fps=1.0`, `max_frames=128`).

### 4.4 Outputs

```
outputs/outputs_query_branchv12/
  query_claims_single/query_conditioned_claims.jsonl
  unli_query_claims_single_lora/query_conditioned_claims_calibrated.jsonl
  claim_packets/all_packets.json
  inferences_query/inferences.jsonl
  reports_query_based/all_reports.json
  submission.jsonl                          # final MAGMaR-2026 submission — feed this to MIRAGE
```

### 4.5 Test split

Swap the queries / mapping at the top of [run_query.sh](run_query.sh) (or pass them as CLI args 2/3) to point at the 19-query test set:

```bash
... bash run_query.sh outputs/outputs_query_test_v12 \
    /a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries.jsonl \
    data/topic_video_mapping.json
```

---

## 5. Running on WikiVideo

WikiVideo uses the same orchestrator with WikiVideo-specific paths and a pre-built persona/query file generated from MultiVENT annotations.

### 5.1 Build the WikiVideo queries (one-time)

```bash
python generate_wikivideo_queries.py \
    --out-queries data/wikivideo_queries.jsonl \
    --out-mapping data/topic_video_mapping_wikivideo.json
```

This reads the MultiVENT matched-queries + annotations files (see the docstring at the top of [generate_wikivideo_queries.py](generate_wikivideo_queries.py#L1-L25)) and fabricates a `(persona_title, background, query)` for each event using a few-shot prompt seeded from the MAGMaR dev set.

### 5.2 Prerequisites

- Videos already chunked under `/a2il/data/mbhosale/wikivideo/en/` (chunk MP4s live alongside originals).
- WikiVideo ASR cache at `/a2il/data/mbhosale/wikivideo/asr/`.
- `data/topic_video_mapping_wikivideo_v2.json` exists (chunked mapping). Launch with `SKIP_CHUNK=1` to skip re-chunking.

### 5.3 Command

```bash
SKIP_CHUNK=1 \
TEAM_ID=cite_chasers \
RUN_ID=wikivideo_query_v1 \
MODEL_NAME=Qwen/Qwen3.6-27B-FP8 \
PARALLEL_QUERIES=8 \
PARALLEL_STEP15=8 \
PARALLEL_STEP5=8 \
MAX_CRITIC_ROUNDS=4 \
COVERAGE_FOLLOWUP_ROUNDS=1 \
CRITIC_NLI_ENABLED=true \
CRITIC_COVERAGE_ENABLED=true \
STEP15_CHUNK_SIZE=10 \
GPU_MEM_UTIL=0.85 \
bash run_query_wikivideo.sh outputs/outputs_query_wikivideo_v1
```

The WikiVideo orchestrator already points at the WikiVideo video root, ASR dir, queries, and mapping — see lines [187-326 of run_query_wikivideo.sh](run_query_wikivideo.sh#L187-L326). Everything else (critic config, calibration, packet assembly, inference, report assembly) is identical to the MAGMaR run.

### 5.4 Notes

- WikiVideo has 52 events × ~8 videos each = ~428 chunked video IDs, so Stage 1b is the dominant cost. With 8-way `PARALLEL_QUERIES` on A6000-class cards it finishes in 4–6 h.
- Lighter extractor: drop to `MODEL_NAME=Qwen/Qwen3.5-9B` and `GPU_MEM_UTIL=0.6` if you only have one card per worker and can't fit the 27B FP8 build alongside UNLI.
- Disable ASR with `ASR_DIR=` (empty) for an ablation run.

---

## 6. Common knobs

| Env var | Where it acts | Notes |
|---|---|---|
| `SKIP_CHUNK` | Stage 0 | `1` = skip chunking, `auto` (default) = skip if v2 mapping + chunk map exist |
| `SKIP_STEP1` | Stage 1b | `1` = reuse existing claims JSONL, `auto` = skip if file exists |
| `PARALLEL_QUERIES` | Stage 1b | bounded GPU slot pool; `>1` shards queries across `$N` GPUs |
| `PARALLEL_STEP15` | Stage 1.5 | shards by `query_id`; UNLI is small so this scales linearly |
| `PARALLEL_STEP5` | Stage 2b | shards inference across GPUs; output is concatenated for Stage 3 |
| `STEP15_CHUNK_SIZE` | Stage 1.5 | restart subprocess every N videos to release libav mmaps |
| `STEP15_NO_AUDIO` / `NO_AUDIO` | Stage 1.5 | `1` = skip audio decode (recommended when any clip > 2–3 min) |
| `MAX_CRITIC_ROUNDS` | Stage 1b | `0` disables the critic loop entirely |
| `CRITIC_GPU_ALONE` | Stage 1b | `1` = put UNLI on its own GPU (needs 2 GPUs per worker; for 30B-FP8 + UNLI > 47 GB) |
| `APPLY_VLLM_PATCH` | one-shot | applies [sbatch/patch_vllm_qwen3_next.py](sbatch/patch_vllm_qwen3_next.py) before Stage 1b |
| `AKS_VIDEO_ROOT` | resolver | when set, the contract resolver tries `<AKS_VIDEO_ROOT>/q<QID>/<vid>.mp4` first |

---

## 7. Evaluation

Evaluation lives in a separate repository at [/home/csgrad/mbhosale/phd/SCALE/mirage/](/home/csgrad/mbhosale/phd/SCALE/mirage/). Point its `run.sh` at the `submission.jsonl` produced at the end of `run_query.sh` / `run_query_wikivideo.sh` — the orchestrators stop after the submission step and do not invoke any in-tree evaluator.
