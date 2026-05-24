# MAGMaR2026 query-branch pipeline (`run_query.sh`)

This document describes every component in the query-branch pipeline, in
the order it executes, with enough detail to build a paper-quality
methods figure. Each component lists:

- **Role** - what it does in one sentence.
- **Inputs / Outputs** - files and formats it consumes / emits.
- **Models or algorithms** - the concrete system used.
- **Failure mode addressed** - which MIRAGE axis (\textsc{InfoP-Ref},
  \textsc{InfoR}, \textsc{CiteP}, \textsc{CiteR}) it targets, and the
  failure pattern observed during development that motivated it.
- **Configuration knobs** - environment variables / Hydra overrides
  that change behavior.

---

## 0. Pipeline architecture at a glance

```
┌──────────────────────────────────────────────────────────────────────┐
│  Inputs                                                              │
│    queries.jsonl       - 8 (dev) / 19 (test) personas + queries      │
│    topic_video_mapping  - topic → [video_id, ...]                    │
│    raw videos           - *.mp4 under <video_root>                   │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼  Stage 0 - chunking (chunk_videos.py)
        │     produces v2 mapping with __chunkNNN ids, chunk map JSON
        ▼  Stage 0.5 - ASR (extract_asr.py)
        │     produces <video_root>/asr/<vid>.json
        ▼  Stage 0.7 - DKS (optional, in-repo at AKS/)
        │     produces <aks_root>/q<QID>/<vid>.mp4
        ▼  Stage 1b - query-conditioned extraction (extract_query_claims.py)
        │     critic-loop + topic-fallback over each (query, video) pair
        │     produces query_<N>.jsonl (per query) + combined JSONL
        ▼  Stage 1.5 - UNLI scoring + calibration
        │     predict_unli.py / predict_unli_chunked.py
        │     calibrate_unli.py
        │     produces query_conditioned_claims_calibrated.jsonl
        ▼  Stage 2a - claim packets (assemble_packets.py)
        │     top-K calibrated claims per query
        ▼  Stage 2b - higher-level inference (infer_higher_level.py)
        │     atomic consolidation, dedup via citation merging
        ▼  Stage 3 - report assembly (generate_report.py)
        │     section text + source_citations
        ▼  Output formatter (format_submission.py)
              chunk-id → parent-id remap. final JSONL in MAGMaR format
```

All stages are **content-addressed and idempotent**: re-running with
the same inputs is a no-op for any stage whose cache is already
populated. Caches live next to the source data
(`<video_root>/asr`, `<aks_root>`) so they are reusable across
branches and across runs.

---

## 1. Stage 0 - video chunking

| Property | Value |
|---|---|
| Script | `chunk_videos.py` |
| Inputs | base topic-video mapping JSON, raw `*.mp4` |
| Outputs | `topic_video_mapping_v2.json` (with `__chunkNNN` ids), `data/video_chunk_map.json` |
| Algorithm | PyAV (libav) cutpoint splitting at 120-second boundaries |
| Failure mode addressed | **Frame-budget exhaustion** - when a video exceeds the VLM's effective frame budget (`max_frames=128` × `fps=1.0` ⇒ 128 s at reference settings), the sampler silently truncates everything past that horizon. Chunking trades one long underrepresented video for several short fully-covered ones. |
| Knobs | `MAX_CHUNK_SECONDS` (env, default 120), `SKIP_CHUNK=1` to skip when v2 mapping already exists |

The chunk map is round-tripped to **submission formatting**: chunked
ids `<vid>__chunk000` are mapped back to the parent `<vid>` so the
final submission's `citations` field uses the parent ids the evaluator
expects.

---

## 2. Stage 0.5 - per-video ASR

| Property | Value |
|---|---|
| Script | `extract_asr.py` |
| Inputs | post-chunking topic mapping, raw `*.mp4` |
| Outputs | `<video_root>/asr/<video_id>.json` |
| Backends | **Qwen3-ASR-1.7B** (primary, 30 langs incl. EN/ZH/YUE/TH). **Whisper-large-v3** (fallback for Burmese/Nepali/etc.). optional `omniASR-LLM-7B` (requires fairseq2) |
| Translation | Whisper `task=translate` over every non-English entry → `text_en` field |
| Loop detector | TTR < 0.18 over ≥20 tokens, OR longest run ≥ 8, OR top-3-gram ≥ 40% - any flag clears the `text` field |
| Failure mode addressed | **Visual-only extraction misses spoken evidence**. For queries whose video collection is dominated by a low-resource spoken language (e.g. Burmese, Nepali), a persona-strict visual filter can reject every otherwise-relevant clip. ASR provides an alternative grounding modality. The translation pass provides English anchors so the English MIRAGE judge can score non-English audio claims. |
| Knobs | `ASR_DIR` (env, default `<video_root>/asr`). `--mode {qwen,whisper,omni,translate,clean-loops,auto}` |

Cache schema:
```json
{
  "video_id": "...",
  "asr_model": "Qwen/Qwen3-ASR-1.7B" | "openai/whisper-large-v3" | ...,
  "language": "English" | "Burmese" | ...,
  "text": "...",
  "text_en": "..." (only for non-English),
  "translation_model": "openai/whisper-large-v3 (translate)",
  "asr_loop_detected": true|false,
  "no_audio": true|false,
  "needs_fallback": true|false
}
```

---

## 3. Stage 0.7 - Dynamic Keyframe Selection (DKS)

| Property | Value |
|---|---|
| Scripts | `AKS/feature_extract_folder.py` (scoring), `AKS/frame_select.py` (top-N selection), `AKS/cut_aks_clips.py` (clip cutting). Directory `AKS/` keeps the upstream import path. the algorithm we run is CRAFT's DKS variant. |
| Inputs | post-chunking topic mapping, queries JSONL, chunked `*.mp4` |
| Outputs | per-(query, video) score JSONs at `<output>/<dataset>/<model>/scores.json`, selected-frame indices at `<output>/selected_frames/<dataset>/<model>/`, and curated clips at `<aks_root>/q<QID>/<video_id>.mp4` |
| Scoring backbone | **CLIP** (`openai/clip-vit-base-patch32`) by default. `mclip` / `blip` / `sevila` available via `--extract_feature_model` (BLIP / SeViLA need the upstream SeViLA env) |
| Selector | recursive split-and-judge top-`N` over per-frame scores, `N = AKS/frame_select.py --max_num_frames` (default 64) |
| Resolver | `resolve_video_path()` in `contracts.py` tries `<AKS_VIDEO_ROOT>/q<QID>/<video_id>.mp4` first. on miss, prints a one-line fallback notice and uses the chunked source instead |
| Failure mode addressed | **Frame-content dilution** - a 120-s chunk contains many frames irrelevant to the query, which the VLM sometimes uses to generate topic-but-not-query relevant claims that fail \textsc{InfoP-Ref}. DKS hands the VLM only the query-relevant frames, lifting both \textsc{InfoP-Ref} and \textsc{CiteP}. Gains are visible only at constrained frame budgets — we evaluate at `MAX_FRAMES=64` (MAGMaR) and `MAX_FRAMES=32` (WikiVideo) against the 128-frame reference. |
| Knobs | `AKS_VIDEO_ROOT` (env, enables DKS at run time). `--max_num_frames` (selector top-N). `MAX_FRAMES` orchestrator env var (matched to top-N at run time). When `AKS_VIDEO_ROOT` is unset, the resolver always uses the chunked source. |

DKS is **opt-in** and **non-blocking**: missing per-query subdirectories
fall back to the chunked source with a single-line notice, so an
incomplete DKS run does not interrupt extraction. Built on AKS (Tang
et al., CVPR 2025). CRAFT's CLIP variant uses HF Transformers directly
and does not require the upstream SeViLA stack.

---

## 4. Stage 1b - query-conditioned VLM extraction

| Property | Value |
|---|---|
| Script | `extract_query_claims.py` (Hydra wrapper: `run_step1_query_claims.py`) |
| Inputs | `query_<QID>` from queries JSONL + that query's topic videos + ASR cache + (optionally) DKS clips |
| Outputs | `query_<N>.jsonl` (per query) + combined `query_conditioned_claims.jsonl` |
| VLM | **Qwen3.5-9B** (MAGMaR default) or **Qwen3-VL-30B-A3B-Instruct-FP8** (WikiVideo default), served by **vLLM**. Set via the `MODEL_NAME` env var, passed through as `model.model=...` to Hydra. |
| Critic models | UNLI (Qwen2.5-Omni-3B + LoRA, `AdoptedIrelia/UNLI`). DeBERTa-v3 MNLI cross-encoder. Llama-3.2-3B-Instruct adjudicator |
| Knobs | `MODEL_NAME`, `MAX_FRAMES`, `MAX_CRITIC_ROUNDS`, `PARALLEL_QUERIES`, `GPU_MEM_UTIL`, `CRITIC_GPU`, `ASR_DIR`, `AKS_VIDEO_ROOT` |

This is the **largest single component**. It contains:

### 4.1 Per-(query, video) prompt

For each (query, video) pair we issue **one VLM call**. The prompt
contains:

1. **Persona block** - `persona_title`, `background`, `query` (verbatim
   from queries JSONL).
2. **Video** - handed via vLLM's chat-completion video slot. Path is
   resolved through DKS-first (when `AKS_VIDEO_ROOT` is set), chunk-fallback.
3. **Speech transcript block** - pasted from the ASR cache when
   present. Includes `detected_language`, `asr_model`, `text`, and
   (when non-English) `english_translation`. Loop-flagged entries are
   suppressed entirely.
4. **Output schema** - strict JSON shape per claim:
   `{claim, confidence, evidence, source, timestamp}`.
5. **Rules block** - three design choices target specific MIRAGE axes:

   - **Coverage-friendly relevance** (\textsc{CiteR} fix). The
     original "return empty if not directly answering the query" rule
     caused 30-50% of long-tail-topic videos to contribute zero claims,
     making them invisible to all citation downstream. We replace it
     with: *prefer direct answers but emit 1-2 atomic facts the
     persona would still find useful as background. only return empty
     when the clip shows nothing identifiable about the topic.*
   - **Atomic claims** (\textsc{InfoP-Ref} fix). Each claim must be a
     single declarative sentence judgeable as one yes/no entailment.
     Conjunctive joins ("X with Y", "X causing Y", "X including Y")
     are forbidden because compound claims fail entailment whenever
     any sub-clause is unsupported.
   - **Source attribution** (modality lineage). Each claim is tagged
     `video_visual` / `video_text` / `transcript` / `asr` so
     downstream stages can preserve the modality each claim came from.

### 4.2 Hybrid per-video critic loop

Up to `MAX_CRITIC_ROUNDS` rounds per video. Each round runs three
checks in parallel and merges results into a structured critic
report:

| Check | Model | Catches |
|---|---|---|
| Temporal grounding | UNLI `score_clip(video, claim, t0, t1)` | claim hallucinated at the timestamp it was assigned to |
| Cross-claim contradiction (screen) | DeBERTa-v3 MNLI cross-encoder | candidate pairs above `nli_screen_threshold=0.5` |
| Contradiction adjudication | Llama-3.2-3B-Instruct | confirms screen candidates and emits NL repair feedback |

Issues with severity `error` (UNLI < 0.3 or confirmed contradiction)
trigger a **re-extraction** call: the VLM gets a new prompt embedding
its own previous claims plus the issues list. The loop exits early
when the claim-set hash is unchanged after a re-extraction
(stagnation).

| Failure mode addressed | |
|---|---|
| VLM hallucinated claim with a fabricated timestamp | UNLI temporal-grounding flag |
| Numeric/entity contradiction across two extracted claims | MNLI screen + LLM adjudicator |
| Critic feedback ignored / model hits a fixed point | stagnation exit |

### 4.3 Per-query coverage audit + targeted follow-up

After all videos for a query are processed once, the coverage LLM
(Llama-3.2-3B) inspects the aggregated claim set and judges whether
the query's scope is sufficiently addressed. If gaps exist, a single
follow-up sweep over each video re-prompts the VLM with the gap list
and merges any new grounded claims.

| Failure mode addressed | persona-narrow extraction missing aspects of the query that were available in the videos but never surfaced (\textsc{InfoR}). |

### 4.4 Topic fallback (zero-claim safety net)

When a query yields zero claims after the strict pass + coverage
follow-up, a **topic fallback** re-issues each video's prompt with
persona, background, and query elided - leaving only the topic. The
fallback is gated on the failure condition: under normal operation it
never affects extraction quality.

| Failure mode addressed | Empty-output for a query whose persona-strict pass rejected every clip. The fallback recovers a non-empty set of atomic claims by dropping the persona constraint and re-prompting on topic-level relevance only. |

### 4.5 Query-level parallelism

When `PARALLEL_QUERIES > 1`, `run_query.sh` shards queries across GPUs
via an 8-slot pool. Each worker pins to a single visible device via
`CUDA_VISIBLE_DEVICES`. A **barrier** waits for the dying vLLM
worker's KV cache to release before the next worker reuses the slot.
without this, naïve round-robin packs multiple vLLM instances on the
same card and they OOM-cascade at startup.

The **combined `query_conditioned_claims.jsonl` is intentionally NOT
written when workers run with `--only-query-ids`** - a final no-filter
recombine pass produces it after all workers finish.

---

## 5. Stage 1.5 - UNLI scoring and calibration

| Property | Value |
|---|---|
| Scripts | `predict_unli.py` (or `predict_unli_chunked.py` for long runs) + `calibrate_unli.py` |
| Inputs | `query_conditioned_claims.jsonl`, raw `*.mp4` |
| Outputs | `unli_predictions.jsonl`, `query_conditioned_claims_calibrated.jsonl` |
| Model | Qwen2.5-Omni-3B + LoRA (`AdoptedIrelia/UNLI`) - same model as the Stage 1b critic |
| Inner loop | for each video: load frames once (mm cache) → for each claim cited to that video, call `score(video, claim) → [0,1]` |
| Failure mode addressed | **Per-claim confidence not yet a probability**. The Stage 2a packet ranker needs a calibrated score, not a raw classifier output, to make relative-quality decisions across claims from different videos and modalities. |
| Knobs | `STEP15_CHUNK_SIZE` (videos per subprocess to bound mmap leaks). `STEP15_NO_AUDIO=1` to skip audio decode |

The `chunked` variant of `predict_unli` exists because long runs leak
mmaps via libav and eventually exhaust `vm.max_map_count`. sharding
into N-video subprocesses reclaims those mmaps between chunks.

---

## 6. Stage 2a - confidence-ranked claim packets

| Property | Value |
|---|---|
| Script | `assemble_packets.py` |
| Inputs | `query_conditioned_claims_calibrated.jsonl`, queries JSONL, mapping |
| Outputs | `claim_packets/query_<N>.json` (one per query) + `all_packets.json` |
| Algorithm | sort claims by `calibration.unli.prob` descending. keep top-K (`unli_threshold=null`, ranking only) |
| Failure mode addressed | **Information overload at inference**. Without packet selection, the inference LLM receives 90+ raw claims for some queries. the prompt blows past `max_tokens` and the JSON output truncates, causing silent parser failure. Confidence ranking preserves long-tail evidence that hard thresholds would discard. |
| Knobs | `unli_threshold` (config), packet size cap |

---

## 7. Stage 2b - higher-level inference

| Property | Value |
|---|---|
| Script | `infer_higher_level.py` |
| Inputs | per-query packets, calibrated claims |
| Outputs | `inferences_query/query_<N>.jsonl` + combined `inferences.jsonl` |
| Model | Qwen3.5-9B (text-only, reused from Stage 1b) |
| Failure mode addressed | **\textsc{InfoP-Ref} drag from compound prose** + **redundant claim explosion**. |
| Knobs | model id (`MODEL_NAME`), packet size cap upstream |

The prompt has three rules that bake in the precision-recall design:

- **One atomic fact per inference.** Same atomicity rule as Stage 1b,
  applied to consolidation output.
- **Deduplicate by merging citations, not paraphrasing.** When two
  evidence items state the same fact, the inference is emitted *once*
  with `source_ids = [evidence_id1, evidence_id2, ...]`. This
  preserves \textsc{CiteR} (every supporting source remains attached)
  while collapsing redundant claims that would inflate
  \textsc{InfoP}'s denominator.
- **No elaboration.** The model may not introduce numbers, names,
  dates, or causal links not present in the packet. Targets the
  dominant \textsc{InfoP-Ref} failure mode where the LLM glues atomic
  facts into compound prose by inventing connective tissue.

**Known failure mode**: when a packet has more than ~50 evidence
items, the inference output JSON can truncate past `max_tokens=2048`
and the parser rejects the whole call. The report builder then
**silently falls back to one section per raw claim**, which inflates
\textsc{CiteR} and \textsc{InfoR} at the cost of \textsc{InfoP-Ref}.
Mitigate by lowering the packet size cap or raising `max_tokens`.

---

## 8. Stage 3 - report assembly

| Property | Value |
|---|---|
| Script | `generate_report.py` |
| Inputs | claim packets, calibrated claims, inferences (per-query JSONLs) |
| Outputs | `reports_query_based/all_reports.json` (one report object per query) |
| Algorithm | for each query: if inferences exist, emit one section per inference with `source_citations` populated from the inference's `source_ids`. otherwise fall back to one section per top-K calibrated claim |
| Failure mode addressed | **End-to-end provenance**. The text the evaluator scores is the text the inference (or fallback) emitted. the citations the evaluator scores are the ones Stage 1b assigned. No rewriting happens here. |

The fallback path matters for metric stability: when Stage 2b
silently fails for a query, this stage emits one section per raw
claim, which inflates \textsc{CiteR} but widens variance in
\textsc{InfoP}. The Stage 2a packet-size cap and Stage 2b
`max_tokens` are the two knobs that keep this fallback rare.

---

## 9. Output formatting

| Property | Value |
|---|---|
| Script | `format_submission.py` |
| Inputs | `all_reports.json`, chunk-id ↔ parent-id map |
| Outputs | `submission.jsonl` (one line per query, MAGMaR-2026 schema) |
| Knobs | `--no-atomize` (default. the post-hoc splitter is OFF because Stages 1b and 2b already enforce atomic claims). `--chunk-map` |
| Failure mode addressed | **Chunked-id leakage to evaluator**. The MIRAGE judge expects parent video ids. without remapping, every cited `<vid>__chunk000` would be a non-match. |

---

## 10. Evaluation against ground truth

| Property | Value |
|---|---|
| External tool | MIRAGE scorer (sibling repo at [`mirage/`](mirage/), env `video_rag_eval`) |
| Variants we run | `info_f1` reference, `cite_f1` reference |
| Skipped | `info_f1` / `cite_f1` collection (VLM-grounded. expensive) |

---

## 11. Miscellaneous

**Caching, idempotency, parallelism.** Every stage is content-addressed: Stage 0 chunks are written once (cache hit on file presence). Stage 0.5 ASR is per-video JSON (force re-run with `--force`). Stage 0.7 DKS produces per-(query, video) clips and the resolver prefers them when `AKS_VIDEO_ROOT` is set. Stage 1b writes per-query JSONLs with resume support, with a final no-filter recombine pass. Stage 1.5 caches predictions per claim and calibration is stateless. Stages 2a, 3, and output formatting are stateless. Stage 2b writes per-query JSONLs with a silent fallback when JSON parsing fails. `run_query.sh` is the single orchestrator - it detects partial state (`SKIP_CHUNK=auto`, `SKIP_STEP1=auto`) and only re-runs what's missing. Parallelism is gated by `PARALLEL_QUERIES` (Stage 1b), `PARALLEL_STEP15` (Stage 1.5), and `PARALLEL_STEP5` (Stage 2b).
