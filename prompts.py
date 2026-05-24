#!/usr/bin/env python3
"""
Prompt builders and parsing helpers for the v1 note-taking/query-based pipelines.

This module freezes the prompt contracts before they are wired into the
extraction pipeline so later implementation can depend on stable interfaces.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, Sequence


# Toggle for the Stage 1b atomicity instructions (see _ATOMIC_CLAIM_RULES below).
# Default ON to preserve historical behaviour; export FORCE_ATOMIC_CLAIMS=0 in
# the shell to let the VLM emit sentence-style compound claims instead.
_FORCE_ATOMIC_CLAIMS = os.environ.get("FORCE_ATOMIC_CLAIMS", "1").strip().lower() not in ("0", "false", "no", "off", "")

_logger = logging.getLogger(__name__)
_SPECULATIVE_RE = re.compile(
    r"\b(maybe|might|possibly|probably|appears|seems|suggests|likely|unclear)\b",
    re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


def _json_dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def _clean_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _format_timestamp(timestamp: Optional[Sequence[float]]) -> str:
    if timestamp is None:
        return "null"
    if len(timestamp) != 2:
        raise ValueError("timestamp must have exactly two values")
    return _json_dump([float(timestamp[0]), float(timestamp[1])])


def _format_optional_json_block(label: str, value: object) -> str:
    return f"{label}:\n{_json_dump(value)}"


def _strict_json_tail(expected_shape: object) -> str:
    return (
        "\nOutput strict JSON only.\n"
        "No markdown, no code fences, no explanation, no extra keys outside the schema.\n"
        f"Expected shape:\n{_json_dump(expected_shape)}"
    )


def _retry_wrapper(base_prompt: str, expected_shape: object) -> str:
    return (
        base_prompt
        + "\n\nYour previous answer was invalid."
        + _strict_json_tail(expected_shape)
    )


QWEN_SCORE_INSTRUCTION = """
To help you make more accurate and consistent judgments, here is an expanded explanation of how to interpret and assign support percentages. These examples are designed to cover a range of real-world cases you may encounter in the annotation task.
100% - /Full and unambiguous support:
The video clearly shows the exact event described in the claim. There is no need for guessing or interpretation.
80-100% - Almost complete support:
The main content in the claim is shown, but there may be minor ambiguity in location, identity, or completeness. The overall claims are supported by the video.
60-80% - Strong partial support:
The video strongly suggests the claim is true, but some critical details may be missing, obscured, or ambiguous, limiting the ability to confirm the claim with certainty. The video gives strong but not definitive support.
40-60% - Moderate partial support:
There is some alignment with the claim, but large portions are either missing, unclear, or open to interpretation. While the footage may point in the same general direction as the claim, it lacks the clarity or completeness needed for confident verification.
20-40% - Minimal weak support:
There are small visual or audio cues that could hint at the claim, but they are insufficient to be confident.
0-20% - Very weak or speculative support:
There may be the slightest indirect reference, such as a related object or setting, but nothing concrete happens.
0% - No support or contradiction:
The video does not relate to the claim at all, or it directly shows something opposite.
""".strip()

QWEN_SCORE_PROMPT = """
Based on the provided video and text, evaluate the probability that the text is true.
Your answer must be a floating point number between 0 and 1, and you must strictly follow the format below:
<answer>probability_value</answer>
Where probability_value is the result you calculate.
The text to evaluate is:

{text}
""".strip()


def prompt_observation_notes(
    *,
    topic: str,
    video_id: str,
    caption_text: str = "",
    ocr_text: str = "",
    transcript_text: str = "",
    timestamp: Optional[Sequence[float]] = None,
    chunk_metadata: Optional[object] = None,
) -> str:
    expected_shape = {
        "observations": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes from evidence for a single video.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
    ]
    if chunk_metadata is not None:
        parts.extend(["", _format_optional_json_block("Chunk metadata", chunk_metadata)])
    parts.extend(
        [
            "",
            "Evidence:",
            f"Caption text:\n{caption_text or ''}",
            f"OCR text:\n{ocr_text or ''}",
            f"Transcript text:\n{transcript_text or ''}",
            "",
            "Rules:",
            "- Record only directly supported observations.",
            "- No inference, speculation, causality, identity guessing, or cross-video synthesis.",
            "- One observation per atomic visible, audible, or textual fact.",
            "- Preserve uncertainty explicitly when the evidence is ambiguous.",
            "- Use modality `visual` for scene content, `ocr` for on-screen text, and `audio` for transcript or speech.",
            "- Use the provided timestamp span for each observation when no narrower timestamp is available.",
            "- If no evidence is present, return an empty observations list.",
        ]
    )
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_qwen_score(text: str) -> str:
    return f"{QWEN_SCORE_INSTRUCTION}\n\n{QWEN_SCORE_PROMPT.format(text=text)}"


def prompt_qwen_score_retry(text: str) -> str:
    return (
        prompt_qwen_score(text)
        + "\n\nYour previous answer was invalid. Reply with only one decimal in the exact form "
        + "<answer>0.73</answer>."
    )


def parse_qwen_score_answer(raw_text: str) -> Optional[float]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    match = _ANSWER_TAG_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def prompt_observation_notes_retry(
    *,
    topic: str,
    video_id: str,
    caption_text: str = "",
    ocr_text: str = "",
    transcript_text: str = "",
    timestamp: Optional[Sequence[float]] = None,
    chunk_metadata: Optional[object] = None,
) -> str:
    return _retry_wrapper(
        prompt_observation_notes(
            topic=topic,
            video_id=video_id,
            caption_text=caption_text,
            ocr_text=ocr_text,
            transcript_text=transcript_text,
            timestamp=timestamp,
            chunk_metadata=chunk_metadata,
        ),
        {
            "observations": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_grounded_notes(
    *,
    topic: str,
    video_id: str,
    observation_notes: Sequence[dict],
    video_path: Optional[str] = None,
) -> str:
    expected_shape = {
        "grounded_notes": [
            {
                "claim": "...",
                "source_observation_ids": ["obs-1"],
                "timestamp_union": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are deriving grounded notes from observation notes for a single video.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- video_path: {video_path or ''}",
        "",
        _format_optional_json_block("Observation notes", list(observation_notes)),
        "",
        "Rules:",
        "- Every grounded note must be fully supported by one or more supplied observation notes.",
        "- No grounded note may reference unstated evidence.",
        "- No cross-video aggregation.",
        "- Claims must be atomic, specific, and non-duplicative.",
        "- If an observation is too weak to support a claim, do not emit a grounded note from it.",
        "- source_observation_ids must reference the supporting observation note ids.",
        "- timestamp_union must be the minimal span covering the linked observations when timestamps exist; otherwise use null.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_grounded_notes_retry(
    *,
    topic: str,
    video_id: str,
    observation_notes: Sequence[dict],
    video_path: Optional[str] = None,
) -> str:
    return _retry_wrapper(
        prompt_grounded_notes(
            topic=topic,
            video_id=video_id,
            observation_notes=observation_notes,
            video_path=video_path,
        ),
        {
            "grounded_notes": [
                {
                    "claim": "...",
                    "source_observation_ids": ["obs-1"],
                    "timestamp_union": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_conditioned_single(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    evidence: object,
    topic: Optional[str] = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    parts = [
        "You are extracting candidate facts from ONE video's evidence for a report.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic or ''}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Evidence", evidence),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} candidate facts from this video if possible.",
        "- Extract only information relevant to the query.",
        "- Facts must be evidence-grounded and citation-ready.",
        "- Avoid generic scene summary unless it directly serves the query.",
        "- Avoid duplicates and paraphrases.",
        "- Prefer concrete, report-usable facts over broad descriptions.",
        "- If the evidence does not answer the query, return an empty facts list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
        "- video_id and video_path must match the evidence provided.",
        "- caption field should contain the supporting caption text if source is video_visual.",
        "- ocr field should contain the supporting OCR text if source is video_text.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_conditioned_single_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    evidence: object,
    topic: Optional[str] = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    return _retry_wrapper(
        prompt_query_conditioned_single(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            evidence=evidence,
            topic=topic,
            per_video_target=per_video_target,
        ),
        expected_shape,
    )


def prompt_query_conditioned_expanded(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    sub_queries: Sequence[str],
    evidence: object,
    topic: Optional[str] = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    parts = [
        "You are extracting candidate facts from ONE video's evidence for a report.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic or ''}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Coverage guidance subqueries", list(sub_queries)),
        "",
        _format_optional_json_block("Evidence", evidence),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} candidate facts from this video if possible.",
        "- Use subqueries only as coverage guidance, not as evidence.",
        "- Do not mention subqueries in the output.",
        "- Extract only information relevant to the official query.",
        "- Facts must be evidence-grounded and citation-ready.",
        "- Do not emit unsupported facts even if a subquery suggests them.",
        "- Avoid duplicates and paraphrases.",
        "- If the evidence does not answer the query, return an empty facts list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
        "- video_id and video_path must match the evidence provided.",
        "- caption field should contain the supporting caption text if source is video_visual.",
        "- ocr field should contain the supporting OCR text if source is video_text.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_conditioned_expanded_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    sub_queries: Sequence[str],
    evidence: object,
    topic: Optional[str] = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    return _retry_wrapper(
        prompt_query_conditioned_expanded(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            sub_queries=sub_queries,
            evidence=evidence,
            topic=topic,
            per_video_target=per_video_target,
        ),
        expected_shape,
    )


def prompt_observation_video(
    *,
    topic: str,
    video_id: str,
    transcript_text: str = "",
    timestamp: Optional[Sequence[float]] = None,
    perception_query: str = "",
) -> str:
    expected_shape = {
        "observations": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes directly from a raw video or sampled video chunk.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
        "",
        f"Transcript text:\n{transcript_text or ''}",
        "",
        "Rules:",
        "- Output observations only, not claims.",
        "- Record only directly supported observations from the sampled video evidence.",
        "- No inference, speculation, causality, identity guessing, or cross-video synthesis.",
        "- If transcript is used, keep audio-derived observations separate from visual observations via modality.",
        "- Use the provided timestamp span when no narrower timestamp is available.",
        "- If there is no usable evidence, return an empty observations list.",
    ]
    if perception_query:
        parts.append(f"- Focus on details relevant to: {perception_query}")
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_observation_video_retry(
    *,
    topic: str,
    video_id: str,
    transcript_text: str = "",
    timestamp: Optional[Sequence[float]] = None,
    perception_query: str = "",
) -> str:
    return _retry_wrapper(
        prompt_observation_video(
            topic=topic,
            video_id=video_id,
            transcript_text=transcript_text,
            timestamp=timestamp,
            perception_query=perception_query,
        ),
        {
            "observations": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str:
    text = strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("no complete JSON object found")


def parse_json_with_expected_key(text: str, expected_key: str) -> dict:
    blob = extract_first_json_object(text)
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    if expected_key not in obj:
        raise ValueError(f"expected top-level key missing: {expected_key}")
    return obj


def call_llm_with_retry(model, prompt, retry_prompt, expected_key, *, video_path=None):
    """Call model.infer(), parse JSON, retry once on failure. Returns dict."""
    raw = model.infer(video_path, prompt)
    try:
        return parse_json_with_expected_key(raw, expected_key)
    except (ValueError, json.JSONDecodeError):
        pass
    raw = model.infer(video_path, retry_prompt)
    try:
        return parse_json_with_expected_key(raw, expected_key)
    except (ValueError, json.JSONDecodeError) as exc:
        _logger.warning("LLM JSON parse failed after retry: %s", exc)
        return {expected_key: []}


def contains_speculative_language(text: str) -> bool:
    return bool(_SPECULATIVE_RE.search(str(text or "")))


# ---------------------------------------------------------------------------
# PR2: General VLM note extraction prompts
# ---------------------------------------------------------------------------


def prompt_general_notes(
    *,
    topic: Optional[str] = None,
    video_id: str,
    include_topic: bool = True,
    timestamp: Optional[Sequence[float]] = None,
) -> str:
    expected_shape = {
        "notes": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes directly from a raw video.",
        "",
        "Video context:",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
        "",
        "Rules:",
        "- Record only directly observable content.",
        "- No inference, speculation, causality, or cross-video synthesis.",
        "- Capture OCR (on-screen text), events, and visible scene details.",
        "- One note per atomic visible, audible, or textual fact.",
        "- Use modality `visual` for scene content, `ocr` for on-screen text, and `audio` for transcript or speech.",
        "- Use the provided timestamp span for each note when no narrower timestamp is available.",
        "- If there is no usable evidence, return an empty notes list.",
    ]
    if include_topic:
        parts.insert(4, f"- topic: {topic or ''}")
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_general_notes_retry(
    *,
    topic: Optional[str] = None,
    video_id: str,
    include_topic: bool = True,
    timestamp: Optional[Sequence[float]] = None,
) -> str:
    return _retry_wrapper(
        prompt_general_notes(
            topic=topic,
            video_id=video_id,
            include_topic=include_topic,
            timestamp=timestamp,
        ),
        {
            "notes": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# PR3: Query-conditioned VLM claim extraction prompts
# ---------------------------------------------------------------------------


# Shared atomicity rules used by every claim-producing prompt in step 1b.
# The MIRAGE judge scores each predicted claim string as one yes/no
# entailment, so a compound claim with one unsupported clause fails the
# whole entailment. Producing atomic claims at extraction time avoids
# the post-hoc splitting in format_submission.py.
#
# Gated by FORCE_ATOMIC_CLAIMS (default on). When the env var is set to
# "0"/"false"/"no"/"off", this list becomes empty so the splices in each
# Stage 1b prompt builder turn into no-ops without touching call sites.
_ATOMIC_CLAIM_RULES = [
    "- One atomic fact per claim. Do NOT join independent facts with",
    "  'and', 'with', 'while', 'including', 'resulting in', 'causing',",
    "  'prompting', 'followed by', 'leading to', etc. Emit each as its",
    "  own claim instead.",
    "- Where one observation states multiple independent facts (who, what,",
    "  when, where, how-much), split them into separate atomic claims.",
    "- A claim should be a single declarative sentence judgeable as one",
    "  yes/no statement, not a compound of multiple judgements.",
    "- Still consolidate duplicates: if two observations state the same",
    "  fact, emit it ONCE — do not paraphrase the same fact twice.",
] if _FORCE_ATOMIC_CLAIMS else []


# Step 2b atomicity fragments. Same FORCE_ATOMIC_CLAIMS toggle as Stage 1b.
# When the env var is unset or "1"/"true"/etc., these fragments reproduce
# the historical Step 2b prompt byte-for-byte; with FORCE_ATOMIC_CLAIMS=0
# the intro qualifier/tail collapse to empty strings and the atomic rules
# splice contributes nothing.
_STEP2B_ATOMIC_INTRO_QUALIFIER = "atomic, " if _FORCE_ATOMIC_CLAIMS else ""
_STEP2B_ATOMIC_INTRO_TAIL = ", so each inference must be a SINGLE atomic fact" if _FORCE_ATOMIC_CLAIMS else ""
_STEP2B_ATOMIC_RULES = [
    "- Emit ONE atomic fact per inference. Do NOT join independent facts with",
    "  'and', 'with', 'while', 'including', 'resulting in', 'causing',",
    "  'prompting', 'followed by', 'leading to', etc.",
    "- A single declarative sentence judgeable as one yes/no statement.",
] if _FORCE_ATOMIC_CLAIMS else []


# Common output-field rules shared by every Step 1 claim-extraction variant.
# Kept in one place so the source enum, timestamp shape, and confidence range
# can't drift between the strict pass, the topic fallback, the critic feedback
# pass, and the coverage-followup pass. If you need to change any of these,
# change them here and every variant inherits the update.
_CLAIM_OUTPUT_RULES = [
    "- source must be one of `video_visual`, `video_text`, `transcript`, or `asr`.",
    "  Use `asr` when the claim is supported primarily by the speech transcript.",
    "- timestamp must be [start, end] in seconds.",
    "- confidence must be a float between 0 and 1.",
]


# Permission for ASR-only claims. Shared by every extraction variant that runs
# with an ASR transcript available. The strict-pass and feedback-pass and
# coverage-followup all encouraged this in slightly different wording before
# the refactor — a single canonical version prevents the model from getting
# inconsistent guidance across passes for the same video.
_ASR_CLAIMS_VALID_RULE = [
    "- ASR-only claims are valid: if the speech transcript states substantive",
    "  factual content about the topic (death tolls, locations, dates, named",
    "  entities, etc.), emit those claims with source=`asr` even when the visual",
    "  is abstract or generic (b-roll, satellite imagery, maps, graphics, news",
    "  anchor footage).",
]


def _is_token_looped(
    text: str,
    *,
    min_tokens: int = 30,
    dominance_ratio: float = 0.4,
    min_repeat: int = 15,
) -> bool:
    """Detect Whisper-style token loops where a single token *dominates* the
    transcript (e.g. ``F**k! F**k! F**k! ...`` from a translation hop on
    low-resource languages).

    A loop is flagged when the most-frequent token both:
      - repeats at least ``min_repeat`` times, AND
      - accounts for at least ``dominance_ratio`` of all tokens.

    Plain absolute-count thresholds false-positive on long legitimate
    passages (common words like "the" / "and" easily exceed any small
    repeat count). Real prose follows a Zipfian distribution where the
    top token sits around ~5-7% of total tokens; a loop blows past that.

    Why this exists: ``asr_loop_detected`` only catches the source-language
    pass; the translation pass (``openai/whisper-large-v3 (translate)``)
    can independently degenerate even when the source transcript is fine.
    """
    tokens = text.split()
    if len(tokens) < min_tokens:
        return False
    counts: dict = {}
    top_count = 0
    for tok in tokens:
        c = counts.get(tok, 0) + 1
        counts[tok] = c
        if c > top_count:
            top_count = c
    if top_count < min_repeat:
        return False
    return (top_count / len(tokens)) >= dominance_ratio


def _format_asr_block(asr: Optional[dict]) -> Optional[list]:
    """Render an ASR transcript dict (from extract_asr.py's cache) into
    prompt lines. Returns None when no transcript is available so the
    caller can skip the block entirely.

    When the transcript is non-English and a ``text_en`` field is present
    (produced by ``extract_asr.py --mode translate``), both the original
    transcript and the English translation are rendered so the VLM can
    ground claims in either.

    Cache entries flagged with ``asr_loop_detected`` (Whisper hallucination
    on low-resource languages) are skipped entirely so the VLM never sees
    a corrupt transcript. The translation hop is checked separately because
    it can degenerate independently of the source-language pass.
    """
    if not asr:
        return None
    if asr.get("asr_loop_detected"):
        # Loop hallucination — defensive: don't pass it to the VLM at all.
        return None
    text = (asr.get("text") or "").strip()
    if not text:
        return None
    # Source-language token loop (catches degenerate transcripts not flagged
    # at extract time, e.g. after model upgrades).
    if _is_token_looped(text):
        return None
    lang = asr.get("language") or "unknown"
    model = asr.get("asr_model") or "unknown"
    text_en = (asr.get("text_en") or "").strip()

    lines = [
        "Speech transcript (auto-generated; may contain ASR errors):",
        f"- detected_language: {lang}",
        f"- asr_model: {model}",
        f"- text: {text}",
    ]
    # Only include the English translation when it is non-empty, distinct
    # from the source text, and not itself a token loop. The translation
    # pipeline (``openai/whisper-large-v3 (translate)``) has been observed
    # to degenerate into repeated profanity tokens on Thai/Burmese inputs,
    # which then triggers safety-aligned refusal in the downstream VLM.
    if text_en and text_en != text and not _is_token_looped(text_en):
        lines.append(f"- english_translation: {text_en}")
    # Defense-in-depth note: even after cache-side filtering, ASR can be
    # noisy on songs / overlapping speech. Tell the model to disregard
    # transcripts that don't match the visual/audio content.
    lines.append(
        "- If the transcript above looks repetitive, looped, or otherwise"
        " inconsistent with the video content, treat it as unreliable and"
        " ground claims in the visual evidence alone."
    )
    return lines


def prompt_query_claims(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting query-relevant claims directly from a raw video.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
    ]
    asr_block = _format_asr_block(asr_transcript)
    if asr_block:
        parts.append("")
        parts.extend(asr_block)
    parts.extend([
        "",
        "Rules:",
        f"- Extract up to {per_video_target} claims from this video.",
        "- Claims must be directly supported by observable video content,",
        "  the speech transcript above, or both. NEVER fabricate.",
        *_ASR_CLAIMS_VALID_RULE,
        "- Avoid generic scene summary unless it directly serves the query.",
        "- Avoid duplicates and paraphrases.",
        # Coverage-friendly relevance: keep at least 1-2 atomic facts even when
        # the video does not directly answer the query. Empty extraction was the
        # main driver of low CITE F1 in v19+ runs (videos that returned 0 claims
        # could not be cited downstream, even when GT lists them as supporting
        # evidence). The fix is to broaden what qualifies — claims about the
        # topic that any informed reader would find relevant context still count.
        # This rule was added in v11 and is what made v12 the best run so far.
        "- Prefer claims that directly answer the query, but do NOT return an",
        "  empty list just because the video doesn't address the query head-on.",
        "  When the video is on-topic but oblique, emit 1-2 atomic facts about",
        "  the topic that this persona would still find useful as background.",
        "- Only return an empty list when *neither* the visual content nor the",
        "  speech transcript mentions the topic — e.g. unrelated stock footage",
        "  with no on-topic narration. If either modality is on-topic, extract.",
        *_ATOMIC_CLAIM_RULES,
        *_CLAIM_OUTPUT_RULES,
    ])
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    return _retry_wrapper(
        prompt_query_claims(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            per_video_target=per_video_target,
            asr_transcript=asr_transcript,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_claims_topic_fallback(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    """Relaxed fallback: only used when the strict per-query pass produced
    zero claims for the entire query. Drops persona/background/query from
    the prompt body so the model does not gate on persona-specific
    relevance (which is exactly what made the strict pass return empty).
    The function still accepts those args so the call site need not
    branch — they are intentionally ignored.
    """
    del query, persona_title, background  # intentionally unused — see docstring

    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting topic-relevant claims directly from a raw video.",
        "Extract any factual observations the video supports about the given",
        "topic. Do NOT filter for any specific persona or sub-question — that",
        "filtering already happened upstream and returned nothing.",
        "",
        "Topic context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
    ]
    asr_block = _format_asr_block(asr_transcript)
    if asr_block:
        parts.append("")
        parts.extend(asr_block)
    parts.extend([
        "",
        "Rules:",
        f"- Extract up to {per_video_target} claims about the topic from this video.",
        "- Claims must be directly supported by observable video content,",
        "  the speech transcript above, or both. NEVER fabricate.",
        *_ASR_CLAIMS_VALID_RULE,
        "- Prefer concrete, factual observations (who, what, when, where, how-much)",
        "  over interpretation or commentary.",
        "- If the video genuinely contains no observations about this topic,",
        "  return an empty claims list — but a video selected for this topic",
        "  almost always shows something topic-relevant.",
        "- Avoid duplicates and paraphrases.",
        *_ATOMIC_CLAIM_RULES,
        *_CLAIM_OUTPUT_RULES,
    ])
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_topic_fallback_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    return _retry_wrapper(
        prompt_query_claims_topic_fallback(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            per_video_target=per_video_target,
            asr_transcript=asr_transcript,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_claims_expanded(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    sub_queries: Sequence[str],
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting query-relevant claims directly from a raw video.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
        "",
        _format_optional_json_block("Coverage guidance subqueries", list(sub_queries)),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} claims from this video.",
        "- Use subqueries only as coverage guidance, not as evidence.",
        "- Do not mention subqueries in the output.",
        "- Claims must be directly supported by observable video content,",
        "  the speech transcript, or both. NEVER fabricate.",
        *_ASR_CLAIMS_VALID_RULE,
        "- Do not emit unsupported claims even if a subquery suggests them.",
        "- Avoid duplicates and paraphrases.",
        "- If neither the visual content nor the speech transcript supports a",
        "  query-relevant claim, return an empty claims list. The pipeline has a",
        "  dedicated topic-fallback pass that runs when ALL videos for a query",
        "  return empty — do not pre-emptively emit generic topic facts here.",
        *_ATOMIC_CLAIM_RULES,
        *_CLAIM_OUTPUT_RULES,
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_expanded_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    sub_queries: Sequence[str],
    per_video_target: int = 5,
) -> str:
    return _retry_wrapper(
        prompt_query_claims_expanded(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            sub_queries=sub_queries, per_video_target=per_video_target,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# PR3b: Critic-in-the-loop prompts for claim enrichment
# ---------------------------------------------------------------------------


def prompt_critic_text_review(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    claims: Sequence[dict],
) -> str:
    """Text-only critic prompt for checks 2 (query coverage) and 3 (contradictions).

    This prompt does NOT require video input — it operates purely on the claim
    text and query context, making it much cheaper than a full VLM call.
    Temporal grounding (check 1) and timestamp window (check 4) are handled
    separately by the UNLI clip scorer.
    """
    expected_shape = {
        "issues": [
            {
                "claim_index": 0,
                "check": "contradiction",
                "severity": "error",
                "message": "...",
                "conflicting_claim_index": 1,
            }
        ],
        "query_coverage_sufficient": True,
        "coverage_gaps": ["..."],
    }

    claims_block = _json_dump([
        {
            "index": i,
            "claim": c.get("claim", ""),
            "confidence": c.get("confidence"),
            "evidence": c.get("evidence", ""),
            "source": c.get("source", ""),
            "timestamp": c.get("timestamp"),
        }
        for i, c in enumerate(claims)
    ])

    parts = [
        "You are a critic reviewing claims that were extracted from a video.",
        "You do NOT have access to the video. Your job is to review the claims",
        "based on their text content alone.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        "",
        f"Claims to review:\n{claims_block}",
        "",
        "Run the following checks:",
        "",
        "CHECK 1 — Query coverage:",
        "Consider whether the claims collectively provide sufficient evidence to answer",
        "the query. Set query_coverage_sufficient=true if the claims adequately address",
        "the query's key aspects given the topic. If not, set it to false and list the",
        "missing aspects in coverage_gaps as short phrases.",
        "Be realistic — a single short video may not cover every aspect of the query.",
        "Only list gaps for major aspects that a video on this topic would likely contain.",
        "",
        "CHECK 2 — Cross-claim contradictions:",
        "Check if any two claims contradict each other (e.g. conflicting numbers,",
        "opposite assertions about the same fact, inconsistent statistics).",
        "Report with check='contradiction', severity='error', and include",
        "conflicting_claim_index pointing to the other claim.",
        "",
        "Rules:",
        "- Only report genuine issues. Do not flag claims that are consistent.",
        "- If there are no issues at all, return an empty issues list.",
        "- severity must be 'error' or 'warning'.",
        "- check must be one of: query_coverage, contradiction.",
        "- conflicting_claim_index is only used for contradiction checks.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_critic_text_review_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    claims: Sequence[dict],
) -> str:
    return _retry_wrapper(
        prompt_critic_text_review(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, claims=claims,
        ),
        {
            "issues": [],
            "query_coverage_sufficient": True,
            "coverage_gaps": [],
        },
    )


def prompt_query_claims_with_feedback(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    previous_claims: Sequence[dict],
    critic_report: dict,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    """Build a re-extraction prompt that feeds the critic's errors to the generator.

    The generator sees its own previous output and the specific issues found,
    so it can make targeted corrections instead of starting from scratch.

    The ASR transcript is included when available so the generator has the
    same context as the initial extraction pass — without it, critic-driven
    revisions effectively run visual-only and lose access to spoken evidence.
    """
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }

    prev_claims_block = _json_dump([
        {
            "index": i,
            "claim": c.get("claim", ""),
            "confidence": c.get("confidence"),
            "evidence": c.get("evidence", ""),
            "source": c.get("source", ""),
            "timestamp": c.get("timestamp"),
        }
        for i, c in enumerate(previous_claims)
    ])

    issues_block = _json_dump(critic_report.get("issues", []))
    coverage_gaps = critic_report.get("coverage_gaps") or []
    coverage_sufficient = critic_report.get("query_coverage_sufficient", True)

    parts = [
        "You are re-extracting query-relevant claims from a raw video.",
        "A critic has reviewed your previous extraction and found issues.",
        "Use the critic's feedback to produce a corrected and improved set of claims.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
    ]
    asr_block = _format_asr_block(asr_transcript)
    if asr_block:
        parts.append("")
        parts.extend(asr_block)
    parts.extend([
        "",
        f"Your previous claims:\n{prev_claims_block}",
        "",
        f"Critic issues found:\n{issues_block}",
    ])

    if not coverage_sufficient and coverage_gaps:
        parts.extend([
            "",
            "The critic found the following aspects of the query are NOT covered:",
            _json_dump(coverage_gaps),
            "If the video or transcript contains evidence for these gaps, extract additional claims.",
            "If neither the video nor the transcript supports them, do not fabricate claims.",
        ])

    parts.extend([
        "",
        "Instructions:",
        "- Fix every error-severity issue from the critic report. The critic emits",
        "  these issue types: temporal_grounding, timestamp_window, contradiction,",
        "  and coverage_gap (via coverage_gaps list, not in issues[]).",
        "- For temporal_grounding errors: re-examine the video and assign the correct timestamp,",
        "  or remove the claim if unsupported anywhere in the video.",
        "- For timestamp_window issues: adjust the timestamp span to match the content duration.",
        "- For contradiction errors: re-examine the conflicting claims and keep only the one",
        "  actually supported by the video, or correct the wrong one.",
        "- For coverage gaps: look harder at the video AND speech transcript for evidence",
        "  addressing the missing aspects. If neither modality supports a gap, do not",
        "  fabricate a claim for it.",
        "- Keep claims that had no issues unchanged.",
        f"- Extract up to {per_video_target} claims total.",
        "- Claims must be directly supported by observable video content, the speech transcript",
        "  above, or both. NEVER fabricate.",
        *_ASR_CLAIMS_VALID_RULE,
        *_ATOMIC_CLAIM_RULES,
        *_CLAIM_OUTPUT_RULES,
    ])
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_with_feedback_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    previous_claims: Sequence[dict],
    critic_report: dict,
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    return _retry_wrapper(
        prompt_query_claims_with_feedback(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            previous_claims=previous_claims, critic_report=critic_report,
            per_video_target=per_video_target,
            asr_transcript=asr_transcript,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_claims_with_coverage_feedback(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    coverage_gaps: Sequence[str],
    existing_claims: Sequence[dict],
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    """Targeted follow-up pass: ask the VLM to look for evidence of specific
    missing aspects identified by a query-level coverage audit.

    Used after all videos for a query have been processed once and the
    coverage LLM flagged aspects that the aggregated claim set does not
    address. The VLM is shown the gap list and asked to find support for
    those aspects in THIS specific video, without fabricating.

    The ASR transcript is included when available so this follow-up can
    surface ASR-only evidence (e.g. spoken death tolls, dates, named
    entities) that the initial visual-leaning extraction may have missed.
    """
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    existing_titles = [str(c.get("claim", "")) for c in existing_claims]
    parts = [
        "You are running a targeted follow-up extraction pass on a raw video.",
        "A separate coverage audit across all videos for this query found "
        "aspects that the current claim set does not address. Look at THIS "
        "video (and its speech transcript, if any) for evidence of any of "
        "those aspects.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
    ]
    asr_block = _format_asr_block(asr_transcript)
    if asr_block:
        parts.append("")
        parts.extend(asr_block)
    parts.extend([
        "",
        _format_optional_json_block("Missing aspects to look for", list(coverage_gaps)),
        "",
        _format_optional_json_block(
            "Existing claims from this video (do not duplicate)", existing_titles
        ),
        "",
        "Rules:",
        f"- Emit up to {per_video_target} NEW claims that fill one of the missing aspects.",
        "- If neither the video nor the speech transcript contains evidence for any",
        "  missing aspect, return an empty claims list.",
        "- NEVER fabricate; a claim must be directly supported by observable video content",
        "  or the speech transcript above.",
        *_ASR_CLAIMS_VALID_RULE,
        "- Do not repeat or paraphrase the existing claims listed above.",
        "- Each claim must name which missing aspect it addresses in the evidence field.",
        *_ATOMIC_CLAIM_RULES,
        *_CLAIM_OUTPUT_RULES,
    ])
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_with_coverage_feedback_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    coverage_gaps: Sequence[str],
    existing_claims: Sequence[dict],
    per_video_target: int = 5,
    asr_transcript: Optional[dict] = None,
) -> str:
    return _retry_wrapper(
        prompt_query_claims_with_coverage_feedback(
            query_id=query_id, query=query, persona_title=persona_title,
            background=background, topic=topic, video_id=video_id,
            coverage_gaps=coverage_gaps, existing_claims=existing_claims,
            per_video_target=per_video_target,
            asr_transcript=asr_transcript,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# PR6: Higher-level inference prompts
# ---------------------------------------------------------------------------


def prompt_higher_level_inference(
    *,
    query_id: str,
    query: str,
    topic: str,
    evidence_items: Sequence[dict],
) -> str:
    expected_shape = {
        "inferences": [
            {
                "claim": "...",
                "source_ids": ["id-1", "id-2"],
            }
        ]
    }
    parts = [
        f"You are consolidating evidence items into {_STEP2B_ATOMIC_INTRO_QUALIFIER}judge-ready claims for a query.",
        "Your output is the final report content — each inference becomes one judged",
        f"yes/no entailment in the evaluator{_STEP2B_ATOMIC_INTRO_TAIL}.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Evidence items", list(evidence_items)),
        "",
        "Rules:",
        "- Consolidate duplicates FIRST: scan the evidence items for any that state the SAME",
        "  fact (even with different wording). For each such group, emit ONE inference with",
        "  source_ids listing EVERY evidence item id in the group. Deduplication is the",
        "  PRIMARY action of this step — do NOT emit a separate inference per evidence item.",
        "- Avoid duplicating evidence items verbatim as inferences.",
        *_STEP2B_ATOMIC_RULES,
        "- Every inference must cite its supporting evidence via source_ids.",
        "- source_ids must reference the IDs of evidence items used.",
        "- Do NOT elaborate beyond the evidence: no extra numbers, names, dates,",
        "  locations, or causal claims that are not literally in the evidence items.",
        "- If the evidence is insufficient for any inference, return an empty inferences list.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_higher_level_inference_retry(
    *,
    query_id: str,
    query: str,
    topic: str,
    evidence_items: Sequence[dict],
) -> str:
    return _retry_wrapper(
        prompt_higher_level_inference(
            query_id=query_id, query=query, topic=topic,
            evidence_items=evidence_items,
        ),
        {
            "inferences": [
                {
                    "claim": "...",
                    "source_ids": ["id-1", "id-2"],
                }
            ]
        },
    )
