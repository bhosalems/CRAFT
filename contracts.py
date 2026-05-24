#!/usr/bin/env python3
"""
PR1: Frozen data contracts and dataset interfaces.

This module is the single source of truth for:
  - loading the 19 official MAGMaR2026 queries
  - loading the topic-video mapping
  - deterministic query-title -> topic-key normalization
  - artifact schemas for general notes, query-conditioned claims, note packets,
    claim packets, higher-level inferences, and claims (compat)
  - legacy schemas retained for observation_notes, grounded_notes, and query_packet
  - schema validation helpers

No later module may redefine query-topic matching or artifact shapes.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths (defaults; overridable via function args or Hydra config)
# Environment variable overrides: set MAGMAR_VIDEO_ROOT, MAGMAR_QUERIES_JSONL,
# or MAGMAR_CAPTIONS_JSONL to change defaults without editing code.
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_ROOT = os.environ.get(
    "MAGMAR_VIDEO_ROOT", "/exp/scale26/data/magmar26"
)
DEFAULT_QUERIES_JSONL = os.environ.get(
    "MAGMAR_QUERIES_JSONL",
    os.path.join(DEFAULT_VIDEO_ROOT, "MAGMaR2026_queries.jsonl"),
)
DEFAULT_TOPIC_MAPPING = "data/topic_video_mapping.json"
TEST_TOPIC_MAPPING = "data/topic_video_mapping_test.json"
DEFAULT_EXPANDED_QUERIES = "retrieval/expanded_queries.json"
DEFAULT_CAPTIONS_JSONL = os.environ.get(
    "MAGMAR_CAPTIONS_JSONL",
    os.path.join(DEFAULT_VIDEO_ROOT, "captions/Qwen3-Omni-30B-A3B-Instruct/captions.jsonl"),
)

# Model identifiers (single source of truth for argparse defaults)
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct" #"Qwen/Qwen3.5-9B" Initial deafult was Qwen 3.5 we probably should use this instead.
DEFAULT_UNLI_MODEL = "AdoptedIrelia/UNLI"
DEFAULT_UNLI_LORA_PATH = "AdoptedIrelia/UNLI/lora"

# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

QUERY_FIELDS = (
    "query_id", "query_type", "language", "title",
    "persona_title", "background", "query",
)


def load_queries(path: str = DEFAULT_QUERIES_JSONL) -> List[dict]:
    """Load the 19 official MAGMaR2026 queries from JSONL."""
    queries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in QUERY_FIELDS:
                if field not in rec:
                    raise ValueError(f"Query record missing field '{field}': {rec}")
            queries.append(rec)
    return queries


def load_expanded_queries(path: str = DEFAULT_EXPANDED_QUERIES) -> Dict[str, dict]:
    """Load expanded sub-queries keyed by query_id."""
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Topic-video mapping
# ---------------------------------------------------------------------------


def load_topic_mapping(path: str = DEFAULT_TOPIC_MAPPING) -> Dict[str, List[str]]:
    """Load topic -> [video_id, ...] mapping."""
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for topic, vids in data.items():
        if isinstance(vids, list):
            out[str(topic)] = [str(v) for v in vids]
    return out

# ---------------------------------------------------------------------------
# Query-title -> topic-key normalization
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]")


def _stem_token(t: str) -> str:
    """Minimal suffix stripping for matching (fires->fire, floods->flood)."""
    if t.endswith("es") and len(t) > 3:
        return t[:-1]  # fires -> fire, etc.
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _normalize_to_tokens(s: str) -> frozenset:
    """Lowercase, strip punctuation/underscores/hyphens, stem, return token set.

    Substitutes non-alphanumeric characters with a SPACE (not empty string) so
    Unicode dashes like en-dash (U+2013, used in titles like "2020–2021") split
    into separate tokens instead of being mashed together (e.g., "20202021").
    Wikipedia titles routinely use en-dashes for date ranges and compounds; the
    earlier collapse-to-empty behavior caused queries like
    "2020–2021 China–India skirmishes" to fail Jaccard matching against the
    underscore-separated mapping key "2020_2021 China_India skirmishes".
    """
    s = s.lower().replace("-", " ").replace("_", " ")
    s = _NORMALIZE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return frozenset(_stem_token(t) for t in s.split())


def build_query_topic_map(
    queries: List[dict],
    topic_keys: Sequence[str],
) -> Dict[str, List[dict]]:
    """
    Map each query to a topic key via normalized token-set matching.

    Returns {topic_key: [query_dict, ...]}.
    Every query must map to exactly one topic. Raises ValueError otherwise.
    """
    topic_tokens = {k: _normalize_to_tokens(k) for k in topic_keys}

    result: Dict[str, List[dict]] = {k: [] for k in topic_keys}
    unmatched = []

    for q in queries:
        q_tokens = _normalize_to_tokens(q["title"])
        best_key = None
        best_score = 0.0

        for t_key, t_tokens in topic_tokens.items():
            if not t_tokens or not q_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            union = len(q_tokens | t_tokens)
            score = overlap / union  # Jaccard
            if score > best_score:
                best_score = score
                best_key = t_key

        if best_key is None or best_score < 0.5:
            unmatched.append(q)
        else:
            result[best_key].append(q)

    if unmatched:
        titles = [q["title"] for q in unmatched]
        raise ValueError(
            f"Could not match {len(unmatched)} query title(s) to topics: {titles}"
        )

    return result

# ---------------------------------------------------------------------------
# Caption loading
# ---------------------------------------------------------------------------


def load_captions_index(
    path: str = DEFAULT_CAPTIONS_JSONL,
) -> Dict[str, dict]:
    """Build {normalized_video_id: caption_record} from captions JSONL."""
    idx: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            vid = meta.get("video_id") or rec.get("video_id") or rec.get("doc_id")
            vid_norm = normalize_video_id(str(vid or ""))
            if vid_norm and vid_norm not in idx:
                idx[vid_norm] = rec
    return idx


def normalize_video_id(video_id: str) -> str:
    """Strip known prefixes (youtube-, etc.) from video IDs."""
    s = str(video_id or "").strip()
    if s.startswith("youtube-"):
        return s[len("youtube-"):]
    return s

# ---------------------------------------------------------------------------
# Video file resolution
# ---------------------------------------------------------------------------


def resolve_video_path(
    video_root: str,
    video_id: str,
    query_id: Optional[str] = None,
) -> Optional[str]:
    """Find the .mp4 file for a video ID, handling common mismatches.

    When AKS_VIDEO_ROOT env var is set AND query_id is provided, this first
    tries <AKS_VIDEO_ROOT>/q<query_id>/<video_id>.mp4 (the per-query AKS-curated
    clip). If that clip is missing, prints a one-line fallback notice and
    falls through to the standard <video_root>/<video_id>.mp4 lookup.

    When AKS_VIDEO_ROOT is unset OR query_id is None, behavior is unchanged.
    Toggle AKS on/off by setting/unsetting AKS_VIDEO_ROOT.
    """
    aks_root = os.environ.get("AKS_VIDEO_ROOT")
    if aks_root and query_id is not None:
        aks_path = _resolve_in_dir(os.path.join(aks_root, f"q{query_id}"), video_id)
        if aks_path is not None:
            return aks_path
        print(
            f"[aks-fallback] q{query_id}/{video_id}: AKS clip not found, using source video",
            flush=True,
        )
    return _resolve_in_dir(video_root, video_id)


def _resolve_in_dir(dir_path: str, video_id: str) -> Optional[str]:
    """Find <dir_path>/<video_id>.mp4, with the existing prefix-match fallback."""
    direct = os.path.join(dir_path, f"{video_id}.mp4")
    if os.path.exists(direct):
        return direct
    try:
        for fname in os.listdir(dir_path):
            if fname.endswith(".mp4") and fname.startswith(video_id[:11]):
                candidate = os.path.join(dir_path, fname)
                if os.path.exists(candidate):
                    return candidate
    except OSError:
        pass
    return None

# ---------------------------------------------------------------------------
# Artifact schemas
# ---------------------------------------------------------------------------

OBSERVATION_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "modality": str,  # visual | ocr | audio
    "text": str,
}

OBSERVATION_NOTE_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "extractor": str,
    "confidence": (float, int, type(None)),
    "source_path": (str, type(None)),
    "is_post_grounded": bool,
}

GROUNDED_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "claim": str,
    "source_observation_ids": list,
}

GROUNDED_NOTE_OPTIONAL_FIELDS = {
    "timestamp_union": (list, type(None)),
    "calibration": (dict, type(None)),
    "extractor": (str, type(None)),
    "is_post_grounded": bool,
}

COMPAT_CLAIM_REQUIRED_FIELDS = {
    "claim": str,
}

COMPAT_CLAIM_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "calibration": (dict, type(None)),
}

FACT_REQUIRED_FIELDS = {
    "fact": str,
    "video_id": str,
}

FACT_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "video_path": (str, type(None)),
    "caption": (str, type(None)),
    "ocr": (str, type(None)),
}

GENERAL_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "text": str,
    "modality": str,       # visual | ocr | audio
}
GENERAL_NOTE_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "extractor": (str, type(None)),
    "confidence": (float, int, type(None)),
    "source_path": (str, type(None)),
    "run_id": (str, type(None)),
    "is_post_grounded": bool,   # always false for Step 1a
}

QUERY_CONDITIONED_CLAIM_REQUIRED_FIELDS = {
    "claim_id": str,
    "query_id": str,
    "video_id": str,
    "topic": str,
    "claim": str,
}
QUERY_CONDITIONED_CLAIM_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "source_path": (str, type(None)),
    "run_id": (str, type(None)),
    "is_post_grounded": bool,   # always false for Step 1b
}

NOTE_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "stream": str,         # "general_note"
}
NOTE_PACKET_OPTIONAL_FIELDS = {
    "note_ids": list,
    "scores": list,
    "provenance": dict,
}

CLAIM_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "stream": str,         # "query_based"
}
CLAIM_PACKET_OPTIONAL_FIELDS = {
    "claim_ids": list,
    "scores": list,
    "provenance": dict,
}

HIGHER_LEVEL_INFERENCE_REQUIRED_FIELDS = {
    "inference_id": str,
    "query_id": str,
    "topic": str,
    "claim": str,
    "source_ids": list,    # note_ids or claim_ids from the packet
}
HIGHER_LEVEL_INFERENCE_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "is_post_grounded": bool,   # always true for Step 2
    "calibration": (dict, type(None)),
    "stream": (str, type(None)),
    "run_id": (str, type(None)),
}

QUERY_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "pipeline": str,  # "note_taking" | "single_query" | "expanded_query"
}

QUERY_PACKET_OPTIONAL_FIELDS = {
    "retrieved_note_ids": list,
    "retrieved_fact_ids": list,
    "scores": list,
    "provenance": dict,
}

REPORT_CITATION_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
}

REPORT_CITATION_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "claim": (str, type(None)),
}

# ---------------------------------------------------------------------------
# Critic error report schema (claim-level issues found by the critic VLM)
# ---------------------------------------------------------------------------

CRITIC_ISSUE_REQUIRED_FIELDS = {
    "claim_index": int,         # 0-based index into the claims list
    "check": str,               # one of: temporal_grounding, query_coverage,
                                #         contradiction, timestamp_window
    "severity": str,            # "error" or "warning"
    "message": str,             # human-readable explanation of the issue
}
CRITIC_ISSUE_OPTIONAL_FIELDS = {
    "suggested_timestamp": (list, type(None)),   # corrected [start, end]
    "conflicting_claim_index": (int, type(None)),  # for contradiction check
}

CRITIC_REPORT_REQUIRED_FIELDS = {
    "issues": list,             # list of issue dicts
    "query_coverage_sufficient": bool,  # True if claims adequately answer query
}
CRITIC_REPORT_OPTIONAL_FIELDS = {
    "coverage_gaps": (list, type(None)),   # missing aspects of the query
}

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _check_fields(
    record: dict,
    required: dict,
    optional: dict,
    label: str,
) -> List[str]:
    """Validate a single record against required/optional field specs. Returns errors."""
    errors = []
    for field, expected_type in required.items():
        if field not in record:
            errors.append(f"{label}: missing required field '{field}'")
        elif not isinstance(record[field], expected_type):
            errors.append(
                f"{label}: field '{field}' expected {expected_type.__name__}, "
                f"got {type(record[field]).__name__}"
            )
    for field, expected_types in optional.items():
        if field in record:
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types,)
            if not isinstance(record[field], expected_types):
                errors.append(
                    f"{label}: field '{field}' expected {expected_types}, "
                    f"got {type(record[field]).__name__}"
                )
    return errors


def validate_observation_note(note: dict) -> List[str]:
    return _check_fields(
        note, OBSERVATION_NOTE_REQUIRED_FIELDS,
        OBSERVATION_NOTE_OPTIONAL_FIELDS, "observation_note",
    )


def validate_grounded_note(note: dict) -> List[str]:
    return _check_fields(
        note, GROUNDED_NOTE_REQUIRED_FIELDS,
        GROUNDED_NOTE_OPTIONAL_FIELDS, "grounded_note",
    )


def validate_compat_claim(claim: dict) -> List[str]:
    return _check_fields(
        claim, COMPAT_CLAIM_REQUIRED_FIELDS,
        COMPAT_CLAIM_OPTIONAL_FIELDS, "compat_claim",
    )


def validate_query_packet(packet: dict) -> List[str]:
    return _check_fields(
        packet, QUERY_PACKET_REQUIRED_FIELDS,
        QUERY_PACKET_OPTIONAL_FIELDS, "query_packet",
    )


def validate_fact(fact: dict) -> List[str]:
    return _check_fields(
        fact, FACT_REQUIRED_FIELDS,
        FACT_OPTIONAL_FIELDS, "fact",
    )


def validate_report_citation(citation: dict) -> List[str]:
    return _check_fields(
        citation, REPORT_CITATION_REQUIRED_FIELDS,
        REPORT_CITATION_OPTIONAL_FIELDS, "report_citation",
    )


def validate_general_note(note: dict) -> List[str]:
    return _check_fields(
        note, GENERAL_NOTE_REQUIRED_FIELDS,
        GENERAL_NOTE_OPTIONAL_FIELDS, "general_note",
    )


def validate_query_conditioned_claim(claim: dict) -> List[str]:
    return _check_fields(
        claim, QUERY_CONDITIONED_CLAIM_REQUIRED_FIELDS,
        QUERY_CONDITIONED_CLAIM_OPTIONAL_FIELDS, "query_conditioned_claim",
    )


def validate_note_packet(packet: dict) -> List[str]:
    return _check_fields(
        packet, NOTE_PACKET_REQUIRED_FIELDS,
        NOTE_PACKET_OPTIONAL_FIELDS, "note_packet",
    )


def validate_claim_packet(packet: dict) -> List[str]:
    return _check_fields(
        packet, CLAIM_PACKET_REQUIRED_FIELDS,
        CLAIM_PACKET_OPTIONAL_FIELDS, "claim_packet",
    )


def validate_higher_level_inference(inference: dict) -> List[str]:
    return _check_fields(
        inference, HIGHER_LEVEL_INFERENCE_REQUIRED_FIELDS,
        HIGHER_LEVEL_INFERENCE_OPTIONAL_FIELDS, "higher_level_inference",
    )


def validate_critic_issue(issue: dict) -> List[str]:
    return _check_fields(
        issue, CRITIC_ISSUE_REQUIRED_FIELDS,
        CRITIC_ISSUE_OPTIONAL_FIELDS, "critic_issue",
    )


def validate_critic_report(report: dict) -> List[str]:
    errors = _check_fields(
        report, CRITIC_REPORT_REQUIRED_FIELDS,
        CRITIC_REPORT_OPTIONAL_FIELDS, "critic_report",
    )
    for issue in report.get("issues", []):
        errors.extend(validate_critic_issue(issue))
    return errors

# ---------------------------------------------------------------------------
# Convenience: load everything needed for a pipeline run
# ---------------------------------------------------------------------------


def load_all(
    *,
    queries_jsonl: str = DEFAULT_QUERIES_JSONL,
    topic_mapping: str = DEFAULT_TOPIC_MAPPING,
    expanded_queries: str = DEFAULT_EXPANDED_QUERIES,
    captions_jsonl: Optional[str] = None,
) -> dict:
    """
    Load all frozen data contracts in one call.

    Returns dict with keys: queries, topic_map, query_topic_map, expanded.
    Includes captions_idx only when captions_jsonl is provided for legacy flows.
    """
    queries = load_queries(queries_jsonl)
    topic_map = load_topic_mapping(topic_mapping)
    query_topic_map = build_query_topic_map(queries, list(topic_map.keys()))
    expanded = load_expanded_queries(expanded_queries)
    bundle = {
        "queries": queries,
        "topic_map": topic_map,
        "query_topic_map": query_topic_map,
        "expanded": expanded,
    }
    if captions_jsonl:
        bundle["captions_idx"] = load_captions_index(captions_jsonl)
    return bundle
