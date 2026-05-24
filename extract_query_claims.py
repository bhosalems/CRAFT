#!/usr/bin/env python3
"""
Step 1b: Query-conditioned VLM claim extraction from raw video.

For each official query, extracts claims from all videos mapped to that query's topic.
Uses VLM with raw video (not LLM with precomputed captions).

Usage:
    # single-query
    python note_taking/extract_query_claims.py \
        --query-mode single-query \
        --model Qwen/Qwen3.5-9B \
        --out-dir note_taking/outputs/query_claims_single

    # expanded-query
    python note_taking/extract_query_claims.py \
        --query-mode expanded-query \
        --model Qwen/Qwen3.5-9B \
        --out-dir note_taking/outputs/query_claims_expanded
"""

import os

# Must be set before qwen_omni_utils is imported (transitively via models.vlm).
# torchvision's video backend leaks threads and mmaps per decode; decord uses
# ffmpeg C API directly and stays flat. Without this, long critic-loop runs
# hit vm.max_map_count and die with libav EAGAIN.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
# Cap thread pools so torch/numpy/OpenMP don't each spawn one worker per core.
# The critic pipeline serializes per-claim, so wide parallelism only leaks.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

try:
    import cv2 as _cv2  # noqa: F401

    _cv2.setNumThreads(1)
except Exception:
    pass

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import logging

from contracts import (
    DEFAULT_EXPANDED_QUERIES,
    DEFAULT_QUERIES_JSONL,
    DEFAULT_TOPIC_MAPPING,
    DEFAULT_UNLI_MODEL,
    DEFAULT_VIDEO_ROOT,
    DEFAULT_VLM_MODEL,
    build_query_topic_map,
    load_expanded_queries,
    load_queries,
    load_topic_mapping,
    resolve_video_path,
    validate_critic_report,
    validate_query_conditioned_claim,
)
from prompts import (
    call_llm_with_retry,
    prompt_query_claims,
    prompt_query_claims_expanded,
    prompt_query_claims_expanded_retry,
    prompt_query_claims_retry,
    prompt_query_claims_topic_fallback,
    prompt_query_claims_topic_fallback_retry,
    prompt_query_claims_with_coverage_feedback,
    prompt_query_claims_with_coverage_feedback_retry,
    prompt_query_claims_with_feedback,
    prompt_query_claims_with_feedback_retry,
)
from run_metadata import build_run_manifest, write_resolved_config, write_run_manifest

from models.vlm import Qwen3_5_VL, TextCoverageLLM, TextNLI, UNLI

_logger = logging.getLogger(__name__)


def _load_asr_transcript(asr_dir: Optional[str], video_id: str) -> Optional[dict]:
    """Load a per-video ASR cache JSON written by extract_asr.py, or None if
    the cache directory is unset, the file is missing, or it has no usable
    transcript text.
    """
    if not asr_dir:
        return None
    path = os.path.join(asr_dir, f"{video_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception as exc:
        _logger.warning("Failed to read ASR cache %s: %s", path, exc)
        return None
    if rec.get("no_audio"):
        return None
    if not (rec.get("text") or "").strip():
        return None
    return rec


# UNLI score thresholds for the temporal grounding critic. Three-tier policy:
#
#   score <  TEMPORAL_DROP_THRESHOLD   →  emit `temporal_drop` (error severity).
#       Critic drops the claim outright without a re-extraction call. UNLI
#       strongly disagrees with this claim's support at its claimed clip;
#       asking the VLM to fix it usually returns the same claim or fewer.
#       Save the compute.
#
#   DROP <= score < TEMPORAL_FAIL_THRESHOLD  →  emit `temporal_grounding` (error).
#       Critic triggers a feedback round so the VLM can revise the timestamp
#       or rephrase. This is the only tier that gets a re-extraction call.
#
#   FAIL <= score < TEMPORAL_WARN_THRESHOLD  →  emit `temporal_grounding` (warning).
#       Informational only — never triggers re-extraction (see fix #1 in
#       _critic_has_actionable_issues).
#
#   score >= WARN  →  no issue emitted, claim passes unchanged.
TEMPORAL_DROP_THRESHOLD = 0.05
TEMPORAL_FAIL_THRESHOLD = 0.30
TEMPORAL_WARN_THRESHOLD = 0.50


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_unli_temporal_checks(
    unli_scorer: "UNLI",
    video_path: str,
    claims: List[dict],
    verbose: bool = False,
    score_cache: Optional[dict] = None,
) -> List[dict]:
    """Checks 1+4: score each claim against its claimed timestamp clip via UNLI.

    Returns a list of issue dicts for claims that fail temporal grounding.

    ``score_cache`` keyed by ``(claim_text, start, end, video_path)`` skips
    redundant UNLI calls across critic rounds. When the model re-extracts and
    re-emits the same claim with the same timestamp (common case under
    selective re-extraction), we'd otherwise re-score it identically every
    round — pure waste of GPU time. Caller is expected to initialize a fresh
    cache per video (per critic-loop instance).
    """
    issues = []
    for i, c in enumerate(claims):
        ts = c.get("timestamp")
        claim_text = c.get("claim", "")
        if not ts or not isinstance(ts, list) or len(ts) != 2 or not claim_text:
            continue

        start, end = float(ts[0]), float(ts[1])
        if end <= start:
            issues.append({
                "claim_index": i,
                "check": "timestamp_window",
                "severity": "error",
                "message": f"Invalid timestamp window: end ({end}) <= start ({start}).",
                "suggested_timestamp": None,
                "conflicting_claim_index": None,
            })
            continue

        cache_key = (claim_text, start, end, video_path)
        cached_score = score_cache.get(cache_key) if score_cache is not None else None
        if cached_score is not None:
            clip_score = cached_score
            if verbose:
                print(f"      [unli] claim {i} [{start:.1f}-{end:.1f}] score={clip_score:.3f} (cached)")
        else:
            try:
                clip_score = unli_scorer.score_clip(video_path, claim_text, start, end)
            except Exception as exc:
                _logger.warning("UNLI score_clip failed for claim %d: %s", i, exc)
                continue
            if score_cache is not None:
                score_cache[cache_key] = clip_score
            if verbose:
                print(f"      [unli] claim {i} [{start:.1f}-{end:.1f}] score={clip_score:.3f}")

        if clip_score < TEMPORAL_DROP_THRESHOLD:
            # Bottom tier: UNLI strongly rejects this claim. Don't waste a
            # re-extraction call asking the VLM to fix something the
            # temporal-grounding scorer is essentially certain is wrong.
            # The critic loop drops these directly without sending feedback
            # to the generator.
            issues.append({
                "claim_index": i,
                "check": "temporal_drop",
                "severity": "error",
                "message": (
                    f"UNLI clip score {clip_score:.3f} < {TEMPORAL_DROP_THRESHOLD} — "
                    f"claim has essentially no support at timestamp [{start}, {end}]; "
                    f"dropping without re-extraction."
                ),
                "suggested_timestamp": None,
                "conflicting_claim_index": None,
            })
        elif clip_score < TEMPORAL_FAIL_THRESHOLD:
            # Middle tier: low score but not catastrophic. Worth asking the
            # VLM to re-examine the video and either correct the timestamp
            # or drop the claim itself.
            issues.append({
                "claim_index": i,
                "check": "temporal_grounding",
                "severity": "error",
                "message": (
                    f"UNLI clip score {clip_score:.3f} < {TEMPORAL_FAIL_THRESHOLD} — "
                    f"claim is likely NOT supported at timestamp [{start}, {end}]. "
                    f"Re-examine the video and assign the correct timestamp or remove this claim."
                ),
                "suggested_timestamp": None,
                "conflicting_claim_index": None,
            })
        elif clip_score < TEMPORAL_WARN_THRESHOLD:
            issues.append({
                "claim_index": i,
                "check": "temporal_grounding",
                "severity": "warning",
                "message": (
                    f"UNLI clip score {clip_score:.3f} < {TEMPORAL_WARN_THRESHOLD} — "
                    f"weak support at timestamp [{start}, {end}]. "
                    f"Consider adjusting the timestamp window."
                ),
                "suggested_timestamp": None,
                "conflicting_claim_index": None,
            })

    return issues


def _run_text_critic(
    nli_scorer: Optional["TextNLI"],
    coverage_llm: Optional["TextCoverageLLM"],
    claims: List[dict],
    query_id: str,
    query_text: str,
    persona_title: str,
    background: str,
    topic: str,
    verbose: bool = False,
    nli_screen_threshold: float = 0.5,
    nli_max_candidates: int = 5,
) -> Optional[dict]:
    """Check 3 (contradictions only): two-stage cascade.

    Stage A: MNLI cross-encoder screens all claim pairs with a low
    ``nli_screen_threshold`` (recall-heavy). MNLI is fast but has known
    false positives on entity-substitution pairs (e.g. three candidates'
    vote shares in the same riding).

    Stage B: the coverage LLM (already loaded) adjudicates each candidate.
    Only pairs it confirms become actionable errors, and its explanation +
    resolution hint are carried into the re-extraction feedback so the VLM
    knows *what* to fix, not just *that* something is wrong.

    Coverage is handled separately (per-query, not per-video) in
    ``_run_query_coverage_and_followup``; this function's returned report
    always has ``query_coverage_sufficient=True`` so the existing merge in
    ``_run_hybrid_critic`` keeps working without a schema change.
    """
    issues: List[dict] = []
    if nli_scorer is None or len(claims) < 2:
        return {
            "issues": issues,
            "query_coverage_sufficient": True,
            "coverage_gaps": [],
        }

    claim_texts = [str(c.get("claim", "")) for c in claims]
    try:
        candidates = nli_scorer.find_contradictions(
            claim_texts, threshold=nli_screen_threshold,
        )
    except Exception as exc:
        _logger.warning("TextNLI screen failed: %s", exc)
        candidates = []

    # Highest-contradiction pairs first, bounded so Llama cost stays predictable.
    candidates.sort(key=lambda t: t[2], reverse=True)
    candidates = candidates[:nli_max_candidates]

    if verbose and candidates:
        print(
            f"      [nli-screen] {len(candidates)} candidate pair(s) above "
            f"threshold {nli_screen_threshold}"
        )

    # Stage B: LLM adjudication — or fall back to a strict MNLI threshold
    # when the coverage LLM isn't loaded.
    for i, j, c_prob, e_prob in candidates:
        if coverage_llm is not None:
            try:
                adj = coverage_llm.adjudicate_contradiction(
                    claim_texts[i], claim_texts[j], c_prob,
                )
            except Exception as exc:
                _logger.warning("LLM adjudication failed for pair (%d,%d): %s", i, j, exc)
                continue
            if not adj.get("is_contradiction", False):
                if verbose:
                    print(
                        f"      [adjudicate] claim {i} vs {j} "
                        f"c={c_prob:.3f} → NOT a contradiction (rejected)"
                    )
                continue
            explanation = adj.get("explanation", "") or ""
            hint = adj.get("resolution_hint", "") or ""
            if verbose:
                print(
                    f"      [adjudicate] claim {i} vs {j} c={c_prob:.3f} → "
                    f"CONTRADICTION: {explanation}"
                )
            message = (
                f"Claim {i} contradicts claim {j}: {explanation} "
                f"Resolution hint: {hint}"
            ).strip()
        else:
            # No LLM judge available — fall back to a strict MNLI-only threshold.
            strict_threshold = 0.9
            if c_prob < strict_threshold or e_prob > 0.15:
                continue
            if verbose:
                print(
                    f"      [nli-strict] claim {i} vs {j} c={c_prob:.3f} "
                    f"e={e_prob:.3f} (no LLM judge, using strict threshold)"
                )
            message = (
                f"MNLI flags claim {i} as contradicting claim {j} "
                f"(contradiction probability {c_prob:.3f}, entailment {e_prob:.3f})."
            )

        issues.append({
            "claim_index": i,
            "check": "contradiction",
            "severity": "error",
            "message": message,
            "suggested_timestamp": None,
            "conflicting_claim_index": j,
        })

    return {
        "issues": issues,
        "query_coverage_sufficient": True,
        "coverage_gaps": [],
    }


def _run_hybrid_critic(
    generator_model: "Qwen3_5_VL",
    unli_scorer: "UNLI",
    nli_scorer: Optional["TextNLI"],
    coverage_llm: Optional["TextCoverageLLM"],
    video_path: str,
    claims: List[dict],
    query_id: str,
    query_text: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    verbose: bool = False,
    nli_screen_threshold: float = 0.5,
    nli_max_candidates: int = 5,
    unli_score_cache: Optional[dict] = None,
) -> Optional[dict]:
    """Per-video critic: UNLI temporal grounding + MNLI-screened/LLM-adjudicated
    contradictions.

    Coverage is intentionally NOT checked here — it needs to see claims from
    all videos for the query, so it is handled once per query after every
    video has been processed (see ``_run_query_coverage_and_followup``).

    The ``coverage_llm`` arg is passed in so the contradiction cascade can
    reuse it as the second-stage adjudicator.
    """
    # --- Checks 1+4: UNLI temporal grounding via clip scoring ---
    unli_issues = _run_unli_temporal_checks(
        unli_scorer, video_path, claims, verbose=verbose,
        score_cache=unli_score_cache,
    )

    # --- Check 3: MNLI-screened + LLM-adjudicated contradictions ---
    text_report = _run_text_critic(
        nli_scorer, coverage_llm, claims,
        query_id, query_text, persona_title, background, topic,
        verbose=verbose,
        nli_screen_threshold=nli_screen_threshold,
        nli_max_candidates=nli_max_candidates,
    )

    if text_report is None:
        if verbose:
            print(f"    [critic] text critic returned invalid report for video_id={video_id}")
        # Still return UNLI issues if we have them
        if unli_issues:
            return {
                "issues": unli_issues,
                "query_coverage_sufficient": True,
                "coverage_gaps": [],
            }
        return None

    # --- Merge ---
    merged = {
        "issues": unli_issues + text_report.get("issues", []),
        "query_coverage_sufficient": text_report.get("query_coverage_sufficient", True),
        "coverage_gaps": text_report.get("coverage_gaps") or [],
    }

    errs = validate_critic_report(merged)
    if errs:
        if verbose:
            print(f"    [critic] merged report validation errors: {errs}")
        # Return what we have even if some fields are off
        return merged

    return merged


def _critic_has_actionable_issues(report: dict) -> bool:
    """Return True if the critic report contains ERROR-severity issues or
    insufficient coverage. Warning-severity issues are informational only
    and do NOT trigger re-extraction — the previous behavior of treating
    every issue (including warnings) as actionable wasted compute by
    re-running the VLM whose prompt explicitly says to "fix every error-
    severity issue" (i.e., it won't change anything for warnings)."""
    for iss in report.get("issues") or []:
        if iss.get("severity") == "error":
            return True
    if not report.get("query_coverage_sufficient", True):
        return True
    return False


def _extract_single_video(
    model,
    video_path: str,
    query_id: str,
    query_text: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    query_mode: str,
    sub_queries: Optional[List[str]],
    asr_transcript: Optional[dict] = None,
) -> List[dict]:
    """Initial claim extraction for a single video (no critic loop)."""
    if query_mode == "single-query":
        prompt = prompt_query_claims(
            query_id=query_id, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            asr_transcript=asr_transcript,
        )
        retry = prompt_query_claims_retry(
            query_id=query_id, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            asr_transcript=asr_transcript,
        )
    else:
        # Expanded variant doesn't accept asr_transcript yet; safe to skip.
        prompt = prompt_query_claims_expanded(
            query_id=query_id, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            sub_queries=sub_queries or [],
        )
        retry = prompt_query_claims_expanded_retry(
            query_id=query_id, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            sub_queries=sub_queries or [],
        )

    result = call_llm_with_retry(model, prompt, retry, "claims", video_path=video_path)
    return result.get("claims", [])


def _run_topic_fallback_pass(
    *,
    model,
    query: dict,
    topic: str,
    video_ids: List[str],
    video_root: str,
    run_id: str,
    asr_dir: Optional[str] = None,
    verbose: bool = False,
) -> List[dict]:
    """Relaxed last-resort sweep used when the strict per-query pass yields
    zero claims for an entire query. Re-runs the same persona+query context
    against each video with a topic-anchored prompt that drops the strict
    "return empty if not answered" rule. Claims here are flagged
    ``is_fallback=True`` so downstream stages can treat them differently if
    desired. Crucially, every claim produced is still grounded in *this*
    query's video pool — no cross-query borrowing.
    """
    qid = query["query_id"]
    query_text = query.get("query", "")
    persona_title = query.get("persona_title", "")
    background = query.get("background", "")

    fallback_claims: List[dict] = []
    for vid in video_ids:
        video_id = str(vid)
        video_path = resolve_video_path(video_root, video_id, query_id=qid)
        if video_path is None:
            continue

        if verbose:
            print(
                f"  [fallback] query_id={qid} topic={topic} video_id={video_id} "
                f"source_path={video_path}",
                flush=True,
            )

        asr_transcript = _load_asr_transcript(asr_dir, video_id)
        prompt = prompt_query_claims_topic_fallback(
            query_id=qid, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            asr_transcript=asr_transcript,
        )
        retry = prompt_query_claims_topic_fallback_retry(
            query_id=qid, query=query_text, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            asr_transcript=asr_transcript,
        )
        try:
            result = call_llm_with_retry(
                model, prompt, retry, "claims", video_path=video_path,
            )
        except Exception as exc:
            _logger.warning(
                "Topic-fallback VLM call failed (qid=%s, vid=%s): %s",
                qid, video_id, exc,
            )
            continue

        raw = result.get("claims", []) or []
        for idx, claim_data in enumerate(raw):
            claim_id = f"qc-{qid}-{video_id}-fb{idx:03d}"
            fallback_claims.append({
                "claim_id": claim_id,
                "query_id": qid,
                "video_id": video_id,
                "topic": topic,
                "claim": claim_data.get("claim", ""),
                "confidence": claim_data.get("confidence"),
                "evidence": claim_data.get("evidence"),
                "source": claim_data.get("source"),
                "timestamp": claim_data.get("timestamp"),
                "source_path": video_path,
                "run_id": run_id,
                "is_post_grounded": False,
                "is_fallback": True,
            })

    if verbose:
        print(f"  [fallback] query {qid} ({topic}): added {len(fallback_claims)} fallback claim(s)")
    return fallback_claims


def _reextract_with_feedback(
    model,
    video_path: str,
    query_id: str,
    query_text: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    previous_claims: List[dict],
    critic_report: dict,
    asr_dir: Optional[str] = None,
) -> List[dict]:
    """Re-extract claims using critic feedback. Loads the same ASR transcript
    used in the initial pass so the model has the full context (visual +
    speech) when revising."""
    asr_transcript = _load_asr_transcript(asr_dir, video_id)
    prompt = prompt_query_claims_with_feedback(
        query_id=query_id, query=query_text, persona_title=persona_title,
        background=background, topic=topic, video_id=video_id,
        previous_claims=previous_claims, critic_report=critic_report,
        asr_transcript=asr_transcript,
    )
    retry = prompt_query_claims_with_feedback_retry(
        query_id=query_id, query=query_text, persona_title=persona_title,
        background=background, topic=topic, video_id=video_id,
        previous_claims=previous_claims, critic_report=critic_report,
        asr_transcript=asr_transcript,
    )
    result = call_llm_with_retry(model, prompt, retry, "claims", video_path=video_path)
    return result.get("claims", [])


def extract_claims_vlm(
    model,
    query: dict,
    video_ids: List[str],
    video_root: str,
    query_mode: str,
    run_id: str,
    *,
    sub_queries: Optional[List[str]] = None,
    topic: str = "",
    verbose: bool = False,
    max_critic_rounds: int = 0,
    critic_scorer: Optional["UNLI"] = None,
    nli_scorer: Optional["TextNLI"] = None,
    coverage_llm: Optional["TextCoverageLLM"] = None,
    nli_screen_threshold: float = 0.5,
    nli_max_candidates: int = 5,
    asr_dir: Optional[str] = None,
) -> List[dict]:
    """Extract query-relevant claims from raw video via VLM, one video at a time.

    When *max_critic_rounds* > 0 and *critic_scorer* is provided, each video's
    claims go through a hybrid critic loop:
      - UNLI ``score_clip()`` verifies temporal grounding (checks 1+4)
      - ``nli_scorer`` (TextNLI / MNLI) detects pairwise contradictions (check 3)
      - ``coverage_llm`` (small instruction-tuned LLM) judges coverage (check 2)
    The generator then re-extracts with the critic's feedback. The loop runs for
    at most *max_critic_rounds* iterations or until no actionable issues remain.
    """
    all_claims = []

    qid = query["query_id"]
    query_text = query.get("query", "")
    persona_title = query.get("persona_title", "")
    background = query.get("background", "")

    for vid in video_ids:
        video_id = str(vid)
        video_path = resolve_video_path(video_root, video_id, query_id=qid)

        if video_path is None:
            if verbose:
                print(f"  [skip] query_id={qid} topic={topic} video_id={video_id} (missing video file)")
            continue

        if verbose:
            print(
                f"  [video] query_id={qid} topic={topic} video_id={video_id} source_path={video_path}",
                flush=True,
            )

        # --- Initial extraction ---
        asr_transcript = _load_asr_transcript(asr_dir, video_id)
        raw_claims = _extract_single_video(
            model, video_path, qid, query_text, persona_title,
            background, topic, video_id, query_mode, sub_queries,
            asr_transcript=asr_transcript,
        )

        # --- Critic loop (only if UNLI scorer is available) ---
        if critic_scorer is not None:
            # Per-video UNLI score cache. Keyed by (claim_text, start, end,
            # video_path) → clip_score. Across critic rounds, claims that
            # survive untouched (non-flagged) will be re-checked by the next
            # round's UNLI pass; without the cache, each round re-runs UNLI
            # forward over the same clip identically. The cache makes round-N
            # essentially free for the non-flagged subset.
            unli_score_cache: dict = {}
            for critic_round in range(max_critic_rounds):
                if not raw_claims:
                    if verbose:
                        print(f"    [critic] round {critic_round + 1}: no claims to review, skipping")
                    break

                if verbose:
                    print(
                        f"    [critic] round {critic_round + 1}/{max_critic_rounds}: "
                        f"reviewing {len(raw_claims)} claims",
                        flush=True,
                    )

                report = _run_hybrid_critic(
                    model, critic_scorer, nli_scorer, coverage_llm,
                    video_path, raw_claims,
                    qid, query_text, persona_title, background, topic, video_id,
                    verbose=verbose,
                    nli_screen_threshold=nli_screen_threshold,
                    nli_max_candidates=nli_max_candidates,
                    unli_score_cache=unli_score_cache,
                )

                if report is None:
                    if verbose:
                        print(f"    [critic] round {critic_round + 1}: critic returned no report, stopping")
                    break

                if not _critic_has_actionable_issues(report):
                    if verbose:
                        n_issues = len(report.get("issues", []))
                        n_warnings = sum(1 for iss in report.get("issues", []) if iss.get("severity") == "warning")
                        print(
                            f"    [critic] round {critic_round + 1}: no actionable issues "
                            f"({n_issues} total, {n_warnings} warning-only), done"
                        )
                    break

                # Partition flagged claims into two tiers based on the check type:
                #   `temporal_drop` (score < DROP_THRESHOLD): drop directly, no
                #       VLM call. UNLI strongly rejects these; asking the model
                #       to fix is unlikely to help and wastes compute.
                #   `temporal_grounding` (DROP <= score < FAIL) or `contradiction`:
                #       re-extract with feedback; the model can revise.
                drop_indices = set()
                reextract_indices = set()
                for iss in report.get("issues") or []:
                    if iss.get("severity") != "error":
                        continue
                    idx = iss.get("claim_index")
                    if not isinstance(idx, int) or not (0 <= idx < len(raw_claims)):
                        continue
                    if iss.get("check") == "temporal_drop":
                        drop_indices.add(idx)
                    else:
                        reextract_indices.add(idx)

                # A claim flagged with both tiers takes the more-severe action (drop).
                reextract_indices -= drop_indices

                n_drop = len(drop_indices)
                n_reextract = len(reextract_indices)
                n_warnings = sum(1 for iss in report.get("issues", []) if iss.get("severity") == "warning")

                if verbose:
                    print(
                        f"    [critic] round {critic_round + 1}: "
                        f"{n_drop} drop (idx={sorted(drop_indices)}), "
                        f"{n_reextract} re-extract (idx={sorted(reextract_indices)}), "
                        f"{n_warnings} warnings",
                        flush=True,
                    )

                # Preserve non-flagged claims verbatim. The whole-set re-extraction
                # in earlier versions could shrink the claim set (e.g. 5→4) when
                # the model came back with fewer items, which silently lost good
                # claims and hurt downstream CITE recall. Selective replacement
                # keeps verified-OK claims and only asks the model to revise the
                # specific ones the critic flagged.
                all_flagged = drop_indices | reextract_indices
                non_flagged = [c for i, c in enumerate(raw_claims) if i not in all_flagged]

                # Only call the VLM if there's something to re-extract. If all
                # errors are drop-tier, skip the model call entirely.
                if n_reextract > 0:
                    new_claims = _reextract_with_feedback(
                        model, video_path, qid, query_text, persona_title,
                        background, topic, video_id, raw_claims, report,
                        asr_dir=asr_dir,
                    )
                else:
                    if verbose:
                        print(
                            f"    [critic] round {critic_round + 1}: all errors are drop-tier "
                            f"(no re-extraction needed)"
                        )
                    new_claims = []

                # Merge: keep non-flagged + add any new claims not already
                # present (deduped by lowercased claim text). If the model's
                # re-extraction happens to re-emit a non-flagged claim verbatim,
                # the dedup ignores it; if it emits a fresh replacement for a
                # flagged claim, we accept it.
                seen_texts = {
                    str(c.get("claim", "")).strip().lower() for c in non_flagged
                }
                merged: List[dict] = list(non_flagged)
                added = 0
                for nc in new_claims:
                    text = str(nc.get("claim", "")).strip().lower()
                    if text and text not in seen_texts:
                        merged.append(nc)
                        seen_texts.add(text)
                        added += 1
                if verbose:
                    print(
                        f"    [critic] round {critic_round + 1}: kept "
                        f"{len(non_flagged)} non-flagged, dropped {n_drop} (tier-1) + "
                        f"{n_reextract} (tier-2 re-extract candidates), "
                        f"added {added} replacements → {len(merged)} total"
                    )

                # Stagnation: if the merged set matches the previous set, the
                # model produced nothing useful for the flagged claims AND we
                # already dropped them — no further rounds will help.
                prev_signature = frozenset(
                    str(c.get("claim", "")).strip().lower() for c in raw_claims
                )
                new_signature = frozenset(
                    str(c.get("claim", "")).strip().lower() for c in merged
                )
                raw_claims = merged
                if new_signature == prev_signature:
                    if verbose:
                        print(
                            f"    [critic] round {critic_round + 1}: claim set "
                            f"unchanged after merge — stopping (stagnation)"
                        )
                    break

        # --- Build final records ---
        for idx, claim_data in enumerate(raw_claims):
            claim_id = f"qc-{qid}-{video_id}-{idx:03d}"
            record = {
                "claim_id": claim_id,
                "query_id": qid,
                "video_id": video_id,
                "topic": topic,
                "claim": claim_data.get("claim", ""),
                "confidence": claim_data.get("confidence"),
                "evidence": claim_data.get("evidence"),
                "source": claim_data.get("source"),
                "timestamp": claim_data.get("timestamp"),
                "source_path": video_path,
                "run_id": run_id,
                "is_post_grounded": False,
            }
            all_claims.append(record)

    return all_claims


def _run_query_coverage_and_followup(
    *,
    model,
    coverage_llm: Optional["TextCoverageLLM"],
    query: dict,
    topic: str,
    video_ids: List[str],
    video_root: str,
    aggregated_claims: List[dict],
    run_id: str,
    followup_rounds: int,
    out_dir: str,
    asr_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[dict], Optional[dict]]:
    """Query-level coverage audit + bounded, targeted re-extraction.

    After all videos for a query have been processed once, look at the
    aggregated claim set and ask the coverage LLM whether the query's scope
    is sufficiently addressed. If gaps exist, run a bounded number of
    follow-up passes — one sweep of every video, asking the VLM to find
    evidence for those specific gap aspects. New claims are merged in.

    The coverage report is persisted to ``<out_dir>/coverage_report_{qid}.json``
    for auditability regardless of whether re-extraction ran.
    """
    if coverage_llm is None:
        return aggregated_claims, None

    qid = query["query_id"]
    query_text = query.get("query", "")
    persona_title = query.get("persona_title", "")
    background = query.get("background", "")

    def _judge(claims: List[dict]) -> dict:
        try:
            return coverage_llm.check_coverage(
                query=query_text,
                persona_title=persona_title,
                background=background,
                topic=topic,
                claims=claims,
            )
        except Exception as exc:
            _logger.warning("Query coverage check failed for qid=%s: %s", qid, exc)
            return {"query_coverage_sufficient": True, "coverage_gaps": []}

    history: List[dict] = []
    current_claims = list(aggregated_claims)

    initial = _judge(current_claims)
    history.append({"round": 0, **initial, "claim_count": len(current_claims)})
    if verbose:
        print(
            f"  [coverage] query {qid}: initial sufficient="
            f"{initial.get('query_coverage_sufficient')} "
            f"gaps={len(initial.get('coverage_gaps', []))}"
        )

    added_claims: List[dict] = []
    rounds_used = 0
    for round_idx in range(1, followup_rounds + 1):
        if initial.get("query_coverage_sufficient", True):
            break
        gaps = list(initial.get("coverage_gaps") or [])
        if not gaps:
            break

        if verbose:
            print(
                f"  [coverage] query {qid} round {round_idx}/{followup_rounds}: "
                f"{len(gaps)} gap(s), running targeted follow-up over {len(video_ids)} video(s)"
            )

        # Group existing claims by video so each follow-up prompt can show
        # what's already been said about that video (to avoid duplicates).
        by_video: Dict[str, List[dict]] = {}
        for c in current_claims:
            by_video.setdefault(str(c.get("video_id")), []).append(c)

        new_this_round: List[dict] = []
        for vid in video_ids:
            video_id = str(vid)
            video_path = resolve_video_path(video_root, video_id, query_id=qid)
            if video_path is None:
                continue

            existing_here = by_video.get(video_id, [])
            asr_transcript = _load_asr_transcript(asr_dir, video_id)
            prompt = prompt_query_claims_with_coverage_feedback(
                query_id=qid, query=query_text, persona_title=persona_title,
                background=background, topic=topic, video_id=video_id,
                coverage_gaps=gaps, existing_claims=existing_here,
                asr_transcript=asr_transcript,
            )
            retry = prompt_query_claims_with_coverage_feedback_retry(
                query_id=qid, query=query_text, persona_title=persona_title,
                background=background, topic=topic, video_id=video_id,
                coverage_gaps=gaps, existing_claims=existing_here,
                asr_transcript=asr_transcript,
            )
            try:
                result = call_llm_with_retry(
                    model, prompt, retry, "claims", video_path=video_path,
                )
            except Exception as exc:
                _logger.warning(
                    "Coverage follow-up VLM call failed (qid=%s, vid=%s): %s",
                    qid, video_id, exc,
                )
                continue
            raw = result.get("claims", []) or []

            # Dedup against both already-aggregated and already-added claims.
            seen = {str(c.get("claim", "")).strip() for c in current_claims}
            seen.update(str(c.get("claim", "")).strip() for c in new_this_round)

            existing_count = len(existing_here)
            for offset, claim_data in enumerate(raw):
                text = str(claim_data.get("claim", "")).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                claim_id = f"qc-{qid}-{video_id}-cov{round_idx:02d}-{existing_count + offset:03d}"
                new_this_round.append({
                    "claim_id": claim_id,
                    "query_id": qid,
                    "video_id": video_id,
                    "topic": topic,
                    "claim": claim_data.get("claim", ""),
                    "confidence": claim_data.get("confidence"),
                    "evidence": claim_data.get("evidence"),
                    "source": claim_data.get("source"),
                    "timestamp": claim_data.get("timestamp"),
                    "source_path": video_path,
                    "run_id": run_id,
                    "is_post_grounded": False,
                })

        rounds_used = round_idx
        if verbose:
            print(
                f"  [coverage] query {qid} round {round_idx}: added {len(new_this_round)} new claim(s)"
            )

        if not new_this_round:
            # Nothing new surfaced — further rounds over the same videos with
            # the same gaps won't help either.
            break

        current_claims.extend(new_this_round)
        added_claims.extend(new_this_round)

        # Re-judge to see if the new claims closed the gaps.
        initial = _judge(current_claims)
        history.append({
            "round": round_idx,
            **initial,
            "claim_count": len(current_claims),
            "added_this_round": len(new_this_round),
        })

    # Always persist the coverage report, even when no follow-up ran, so
    # the evaluator and future debugging have a paper trail.
    report = {
        "query_id": qid,
        "topic": topic,
        "followup_rounds_requested": followup_rounds,
        "followup_rounds_used": rounds_used,
        "total_claims_before": len(aggregated_claims),
        "total_claims_after": len(current_claims),
        "claims_added": len(added_claims),
        "history": history,
    }
    try:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, f"coverage_report_{qid}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        _logger.warning("Failed to write coverage report for qid=%s: %s", qid, exc)

    return current_claims, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1b: Query-conditioned VLM claim extraction")
    ap.add_argument("--query-mode", choices=["single-query", "expanded-query"], required=True)
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--mapping", default=DEFAULT_TOPIC_MAPPING)
    ap.add_argument("--expanded-queries", default=DEFAULT_EXPANDED_QUERIES)
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--model", default=DEFAULT_VLM_MODEL)
    ap.add_argument("--download-dir", default="")
    ap.add_argument("--quantization", default="",
                    help=("vLLM quantization mode (e.g. awq, gptq, fp8). "
                          "Leave empty to disable."))
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--max-critic-rounds", type=int, default=0,
                    help="Max critic-generator feedback rounds per video (0 = disabled)")
    ap.add_argument("--critic-unli-model", default=DEFAULT_UNLI_MODEL,
                    help="UNLI model for critic temporal grounding (only used when --max-critic-rounds > 0)")
    ap.add_argument("--critic-unli-base-model", default=None,
                    help="Base model for UNLI LoRA mode")
    ap.add_argument("--critic-unli-lora-path", default=None,
                    help="LoRA adapter path for UNLI")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help=("vLLM GPU memory fraction. If unset, defaults to 0.9 when "
                          "critic is disabled or lives on a different GPU, and 0.45 "
                          "when critic shares vLLM's GPU."))
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help=("vLLM tensor_parallel_size. Set >1 to shard the VLM across "
                          "multiple GPUs (cuda:0..cuda:TP-1). Required for 30B-class "
                          "models on 24 GiB cards. Critic UNLI must move to a GPU "
                          "outside the TP range (use --critic-gpu)."))
    ap.add_argument("--max-model-len", type=int, default=None,
                    help=("vLLM max context length. Qwen3-VL-30B advertises 262K which "
                          "demands 24 GB KV cache and refuses to start on a 47 GB card "
                          "after weights occupy 40 GB. Set to 32768 or 65536 for typical "
                          "video extraction prompts. Default None = use vLLM/model default."))
    ap.add_argument("--critic-gpu", type=int, default=1,
                    help=("CUDA device index for the UNLI critic. vLLM always uses "
                          "cuda:0 for tp=1, so setting --critic-gpu=1 puts the two "
                          "models on separate GPUs. Set to 0 to co-locate."))
    # --- Text critic: contradictions (MNLI screen + LLM adjudicate) +
    # --- coverage (small instruct LLM, per-query with targeted follow-up) ---
    ap.add_argument("--critic-nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    help="HF cross-encoder NLI checkpoint for pairwise contradiction screening.")
    ap.add_argument("--critic-nli-device", default="cpu",
                    help=("Device for the MNLI model ('cpu' or 'cuda:<idx>'). Default cpu "
                          "keeps it off the GPUs; the model is small and per-pair inference "
                          "is sub-100ms."))
    ap.add_argument("--critic-nli-screen-threshold", type=float, default=0.5,
                    help=("MNLI screen threshold (recall-heavy). Pairs above this are sent "
                          "to the LLM adjudicator. Lower = catch more candidates."))
    ap.add_argument("--critic-nli-max-candidates", type=int, default=5,
                    help=("Hard cap on how many candidate pairs per video per critic round "
                          "go to the LLM adjudicator (cost bound)."))
    ap.add_argument("--critic-nli-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable MNLI screening for contradictions.")
    ap.add_argument("--critic-coverage-model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help=("HF causal LM used both as contradiction adjudicator AND as the "
                          "per-query coverage auditor."))
    ap.add_argument("--critic-coverage-gpu", type=int, default=-1,
                    help=("CUDA device index for the coverage/adjudicator LLM. -1 = reuse "
                          "--critic-gpu (shares the UNLI GPU)."))
    ap.add_argument("--critic-coverage-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help=("Enable the LLM as both contradiction adjudicator and per-query "
                          "coverage auditor. When disabled, MNLI falls back to a strict "
                          "threshold; coverage is skipped."))
    ap.add_argument("--coverage-followup-rounds", type=int, default=1,
                    help=("Max targeted re-extraction passes after the per-query coverage "
                          "audit flags gaps. 0 = audit-and-log only."))
    ap.add_argument("--topic-fallback-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help=("If a query yields zero claims after the strict per-video pass and "
                          "coverage follow-up, sweep its videos once more with a relaxed, "
                          "topic-anchored prompt so the submission isn't blank. The fallback "
                          "stays within this query's videos — it never copies another query's "
                          "answers."))
    ap.add_argument("--asr-dir", default="",
                    help=("Directory of per-video ASR transcript JSONs produced by "
                          "extract_asr.py. When set, each transcript is rendered into "
                          "the VLM prompt as a 'Speech transcript' block so claims can "
                          "be grounded in audio as well as visual content. Empty = disabled."))
    ap.add_argument("--only-query-ids", default="",
                    help=("Comma-separated query_ids to process. When set, only those "
                          "queries run extraction; others are silently skipped. Use this "
                          "for Slurm array jobs / multi-GPU parallelism (one worker per "
                          "query). The combined query_conditioned_claims.jsonl is NOT "
                          "written when this filter is active — run a final no-filter "
                          "invocation after all workers finish to recombine."))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resolved-config-out", default="")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help=("Skip queries whose per-query JSONL already exists in "
                          "--out-dir and load those claims into the combined file. "
                          "Use --no-resume to force re-extraction."))
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.out_dir is None:
        suffix = args.query_mode.replace("-", "_")
        args.out_dir = f"outputs/query_claims_{suffix}"

    manifest = build_run_manifest(
        script_name="extract_query_claims.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "model": args.model,
            "download_dir": args.download_dir,
            "quantization": args.quantization or None,
            "query_mode": args.query_mode,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "allowed_local_media_path": args.video_root,
            "max_critic_rounds": args.max_critic_rounds,
            "critic_unli_model": args.critic_unli_model if args.max_critic_rounds > 0 else None,
            "critic_unli_base_model": args.critic_unli_base_model if args.max_critic_rounds > 0 else None,
            "critic_unli_lora_path": args.critic_unli_lora_path if args.max_critic_rounds > 0 else None,
            "critic_gpu": args.critic_gpu if args.max_critic_rounds > 0 else None,
            "critic_nli_enabled": args.critic_nli_enabled if args.max_critic_rounds > 0 else None,
            "critic_nli_model": args.critic_nli_model if (args.max_critic_rounds > 0 and args.critic_nli_enabled) else None,
            "critic_nli_device": args.critic_nli_device if (args.max_critic_rounds > 0 and args.critic_nli_enabled) else None,
            "critic_nli_screen_threshold": args.critic_nli_screen_threshold if (args.max_critic_rounds > 0 and args.critic_nli_enabled) else None,
            "critic_nli_max_candidates": args.critic_nli_max_candidates if (args.max_critic_rounds > 0 and args.critic_nli_enabled) else None,
            "critic_coverage_enabled": args.critic_coverage_enabled,
            "critic_coverage_model": args.critic_coverage_model if args.critic_coverage_enabled else None,
            "critic_coverage_gpu": args.critic_coverage_gpu if args.critic_coverage_enabled else None,
            "coverage_followup_rounds": args.coverage_followup_rounds if args.critic_coverage_enabled else None,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)
    resolved_config_path = None
    if args.resolved_config_out:
        resolved_config_path = write_resolved_config(
            args.out_dir,
            vars(args),
            filename=args.resolved_config_out,
        )

    # Load data
    queries = load_queries(args.queries_jsonl)
    topic_map = load_topic_mapping(args.mapping)
    qtm = build_query_topic_map(queries, list(topic_map.keys()))

    expanded = {}
    if args.query_mode == "expanded-query":
        expanded = load_expanded_queries(args.expanded_queries)

    print(f"Query mode: {args.query_mode}")
    print(f"Model: {args.model}")
    print(f"Queries: {len(queries)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the ordered list of queries to process; if --resume is on, mark
    # any query whose per-query JSONL already exists in --out-dir as "reuse".
    # A partial run that crashed mid-loop can then pick up where it left off
    # without re-extracting claims that were already written to disk.
    only_qids = (
        {s.strip() for s in args.only_query_ids.split(",") if s.strip()}
        if args.only_query_ids else None
    )
    planned = []  # list of (topic, q, query_path, reuse: bool)
    for topic, topic_queries in sorted(qtm.items()):
        for q in topic_queries:
            qid = q["query_id"]
            if only_qids and str(qid) not in only_qids:
                continue
            query_path = os.path.join(args.out_dir, f"query_{qid}.jsonl")
            reuse = bool(args.resume) and os.path.exists(query_path)
            planned.append((topic, q, query_path, reuse))
    if only_qids:
        print(f"Filtering to {len(planned)} of the queries (--only-query-ids={sorted(only_qids)})")

    n_reuse = sum(1 for _, _, _, reuse in planned if reuse)
    n_run = len(planned) - n_reuse
    if args.resume:
        print(f"Resume: reusing {n_reuse} existing per-query JSONLs; {n_run} queries to extract")
    if n_run == 0:
        print("Resume: nothing to extract — will only recombine existing per-query JSONLs")

    # Initialize UNLI critic scorer if critic loop is enabled and there is
    # actual extraction work to do; pure-recombine runs don't need it.
    critic_scorer = None
    if args.max_critic_rounds > 0 and n_run > 0:
        print(f"Critic loop: enabled (max {args.max_critic_rounds} rounds per video)")
        unli_kwargs = dict(
            model=args.critic_unli_model,
            download_dir=args.download_dir,
            device_map={"": f"cuda:{args.critic_gpu}"},
        )
        if args.critic_unli_base_model:
            unli_kwargs["base_model"] = args.critic_unli_base_model
        if args.critic_unli_lora_path:
            unli_kwargs["lora_path"] = args.critic_unli_lora_path
        print(f"Critic GPU: cuda:{args.critic_gpu}")
        if args.critic_unli_base_model or args.critic_unli_lora_path:
            print(
                "Initializing UNLI critic (base+LoRA): "
                f"model={args.critic_unli_model} "
                f"base_model={args.critic_unli_base_model or '∅'} "
                f"lora_path={args.critic_unli_lora_path or '∅'}"
            )
        else:
            print(f"Initializing UNLI critic (merged): model={args.critic_unli_model}")
        # NOTE: UNLI is loaded to GPU BEFORE vLLM so vLLM's memory probe sees the
        # reduced free memory. We also explicitly lower vLLM's gpu_memory_utilization
        # below so both models can coexist on the same GPU.
        critic_scorer = UNLI(**unli_kwargs)

    # --- Text critic: MNLI-screened + LLM-adjudicated contradictions,
    # plus per-query coverage (same LLM) with bounded targeted follow-up. ---
    nli_scorer = None
    coverage_llm = None
    # max_critic_rounds is the master switch for the entire critic system
    # (UNLI + NLI + coverage auditor). When it's 0, skip the coverage LLM too.
    want_coverage_llm = (
        args.critic_coverage_enabled
        and args.max_critic_rounds > 0
        and n_run > 0
    )
    if args.max_critic_rounds > 0 and n_run > 0:
        if args.critic_nli_enabled:
            print(
                f"Initializing TextNLI (contradiction screener): "
                f"model={args.critic_nli_model} device={args.critic_nli_device} "
                f"screen_threshold={args.critic_nli_screen_threshold} "
                f"max_candidates={args.critic_nli_max_candidates}"
            )
            nli_scorer = TextNLI(
                model=args.critic_nli_model,
                device=args.critic_nli_device,
                download_dir=args.download_dir,
            )
    # Coverage LLM is shared: used both as contradiction adjudicator by the
    # per-video critic AND as the per-query coverage auditor. Load it if
    # either use case is on.
    if want_coverage_llm:
        cov_gpu = args.critic_coverage_gpu if args.critic_coverage_gpu >= 0 else args.critic_gpu
        print(
            f"Initializing TextCoverageLLM (contradiction judge + coverage auditor): "
            f"model={args.critic_coverage_model} device=cuda:{cov_gpu}"
        )
        coverage_llm = TextCoverageLLM(
            model=args.critic_coverage_model,
            device_map=f"cuda:{cov_gpu}",
            download_dir=args.download_dir,
        )

    # Pick vLLM GPU memory utilization.
    # - Critic disabled: use vLLM default (0.9) unless user overrode.
    # - Critic enabled, co-located on cuda:0: UNLI takes ~15 GiB and needs
    #   headroom for per-claim video-decode activation spikes (~2.5 GiB).
    #   On a ~48 GiB GPU, 0.45 leaves ~10 GiB free after both are resident.
    # - Critic enabled on a separate GPU: vLLM owns cuda:0 alone, use its
    #   default (0.9).
    if args.gpu_memory_utilization is not None:
        gpu_mem_util = args.gpu_memory_utilization
    elif args.max_critic_rounds > 0 and args.critic_gpu == 0:
        gpu_mem_util = 0.45
    else:
        gpu_mem_util = None  # use vLLM's default (0.9)

    if gpu_mem_util is not None:
        print(f"vLLM gpu_memory_utilization: {gpu_mem_util}")

    model = None
    if n_run > 0:
        print(f"\nInitializing model: {args.model}")
        if args.tensor_parallel_size and args.tensor_parallel_size > 1:
            print(f"vLLM tensor_parallel_size: {args.tensor_parallel_size}")
        if args.quantization:
            print(f"vLLM quantization: {args.quantization}")

        # vLLM only accepts a single ``allowed_local_media_path`` string and
        # rejects any video path that isn't a subpath of it. When AKS is on,
        # paths come from a sibling directory (``$AKS_VIDEO_ROOT``), so we
        # expand the allowed prefix to the common parent of both roots.
        allowed_media_path = args.video_root
        aks_root = os.environ.get("AKS_VIDEO_ROOT", "").strip()
        if aks_root:
            try:
                allowed_media_path = os.path.commonpath(
                    [os.path.abspath(args.video_root), os.path.abspath(aks_root)]
                )
            except ValueError:
                # commonpath raises on cross-drive paths (Windows) — fall back
                # to the original video root and accept that AKS clips on a
                # different mount won't be reachable.
                pass
            print(f"vLLM allowed_local_media_path expanded to {allowed_media_path} (AKS_VIDEO_ROOT={aks_root})")

        model = Qwen3_5_VL(
            model=args.model,
            download_dir=args.download_dir,
            quantization=args.quantization or None,
            fps=args.fps,
            enable_thinking=args.enable_thinking,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            max_frames=args.max_frames,
            allowed_local_media_path=allowed_media_path,
            gpu_memory_utilization=gpu_mem_util,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )

    all_claims = []

    pbar = tqdm(total=len(planned), desc="Step 1b queries", unit="query")

    for topic, q, query_path, reuse in planned:
        qid = q["query_id"]
        pbar.set_postfix(topic=topic, query_id=qid)

        if reuse:
            existing = []
            with open(query_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
            all_claims.extend(existing)
            if args.verbose:
                print(f"  [resume] query {qid} ({topic}): reused {len(existing)} claims from {query_path}")
            pbar.update(1)
            continue

        sub_queries = None
        if args.query_mode == "expanded-query":
            exp = expanded.get(qid, {})
            sub_queries = exp.get("sub_queries", [])

        video_ids = topic_map.get(topic, [])
        claims = extract_claims_vlm(
            model, q, video_ids, args.video_root,
            args.query_mode, manifest["run_id"],
            sub_queries=sub_queries, topic=topic, verbose=args.verbose,
            max_critic_rounds=args.max_critic_rounds,
            critic_scorer=critic_scorer,
            nli_scorer=nli_scorer,
            coverage_llm=coverage_llm,
            nli_screen_threshold=args.critic_nli_screen_threshold,
            nli_max_candidates=args.critic_nli_max_candidates,
            asr_dir=args.asr_dir or None,
        )

        # Per-query coverage audit + bounded, targeted follow-up. Runs once
        # over the aggregated claim set (all videos for this query). Gaps
        # the audit flags are fed back to the VLM as a focused prompt on
        # each video; any new grounded claims are merged in.
        if coverage_llm is not None and args.coverage_followup_rounds >= 0 and claims:
            claims, _cov_report = _run_query_coverage_and_followup(
                model=model,
                coverage_llm=coverage_llm,
                query=q,
                topic=topic,
                video_ids=video_ids,
                video_root=args.video_root,
                aggregated_claims=claims,
                run_id=manifest["run_id"],
                followup_rounds=max(0, args.coverage_followup_rounds),
                out_dir=args.out_dir,
                asr_dir=args.asr_dir or None,
                verbose=args.verbose,
            )

        # Topic-fallback: if the strict per-query pass + coverage follow-up
        # yielded nothing, sweep the same query's videos once more with a
        # relaxed prompt that asks for topic-relevant background facts the
        # persona would still find useful. This stays within this query's
        # video pool — it does not reuse another query's claims.
        if args.topic_fallback_enabled and not claims and model is not None and video_ids:
            if args.verbose:
                print(f"  [fallback] query {qid} ({topic}): strict pass returned 0 claims, running topic fallback")
            claims = _run_topic_fallback_pass(
                model=model,
                query=q,
                topic=topic,
                video_ids=video_ids,
                video_root=args.video_root,
                run_id=manifest["run_id"],
                asr_dir=args.asr_dir or None,
                verbose=args.verbose,
            )

        # Validate
        for claim in claims:
            errs = validate_query_conditioned_claim(claim)
            if errs and args.verbose:
                print(f"  WARN claim {claim.get('claim_id')}: {errs}")

        all_claims.extend(claims)

        # Write per-query JSONL so a later crash can resume here.
        _write_jsonl(query_path, claims)

        if args.verbose:
            sq_info = f" ({len(sub_queries)} subqueries)" if sub_queries else ""
            print(f"  query {qid} ({topic}): {len(claims)} claims{sq_info}")

        pbar.update(1)

    pbar.close()

    # Write combined flat file. When --only-query-ids was used we have only
    # a partial view of the queries, so writing this would clobber a
    # complete combined file from a sibling worker. Skip it; a final
    # no-filter invocation will recombine after all workers finish.
    combined_path = os.path.join(args.out_dir, "query_conditioned_claims.jsonl")
    if only_qids:
        print(f"\n[ok] {len(all_claims)} claims for {sorted(only_qids)} written to per-query JSONLs")
        print(f"[ok] -> {args.out_dir}/query_{{N}}.jsonl")
        print(f"[skip] combined file (run extract_query_claims.py with no --only-query-ids to recombine)")
    else:
        _write_jsonl(combined_path, all_claims)
        print(f"\n[ok] {len(all_claims)} query-conditioned claims across {len(queries)} queries")
        print(f"[ok] -> {args.out_dir}/query_{{N}}.jsonl")
        print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")
    if resolved_config_path:
        print(f"[ok] -> {resolved_config_path}")


if __name__ == "__main__":
    main()
