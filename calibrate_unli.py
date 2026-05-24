#!/usr/bin/env python3
"""
Step 1.5 / 2.5: Calibrate extracted artifacts with UNLI-style support probabilities.

Supports multiple artifact types:
  - claims (default): per-video claims.jsonl from extract_grounded_notes.py
  - general-notes: general_notes.jsonl from extract_general_notes.py
  - query-claims: query_conditioned_claims.jsonl from extract_query_claims.py
  - inferences: higher-level inference JSONL from infer_higher_level.py

Expected UNLI preds input: JSONL where each line contains enough to derive:
  - video_id (or query_id for inferences)
  - claim/text
  - probability/support score in [0,1]

Common accepted shapes:
  - {"video_id": "...", "claim": "...", "prob": 0.73, ...}
  - {"meta": {"video_id": "...", "claim": "..."}, "text": "<answer>0.73</answer>"}
  - {"meta": {"video_id": "...", "claim": "..."}, "outputs": ["<answer>0.73</answer>"]}
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Optional, Tuple
from run_metadata import build_run_manifest, write_resolved_config, write_run_manifest


_ANSWER_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_prob(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    m = _ANSWER_RE.search(s)
    if m:
        try:
            val = float(m.group(1))
        except Exception:
            return None
        if 0.0 <= val <= 1.0:
            return val
    # Fallback: raw float in the string.
    try:
        val = float(s)
    except Exception:
        return None
    if 0.0 <= val <= 1.0:
        return val
    return None


def _unli_key(id_a: str, text: str) -> Tuple[str, str]:
    return (str(id_a or "").strip(), str(text or "").strip())


def _extract_unli_record(rec: dict) -> Optional[Tuple[Tuple[str, str], Optional[str], dict]]:
    """Parse a single UNLI prediction record.

    Returns (tuple_key, stable_id_or_None, payload) on success, None on failure.
    """
    if not isinstance(rec, dict):
        return None

    # 1) Direct fields
    video_id = rec.get("video_id")
    query_id = rec.get("query_id")
    claim = rec.get("claim")
    prob = rec.get("prob")

    # 2) meta envelope
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    if video_id is None:
        video_id = meta.get("video_id")
    if query_id is None:
        query_id = meta.get("query_id")
    if claim is None:
        claim = meta.get("claim")

    # Also try "text" field as claim fallback
    if claim is None:
        claim = rec.get("text") if isinstance(rec.get("text"), str) and not _ANSWER_RE.search(str(rec.get("text", ""))) else None
    if claim is None:
        claim = meta.get("text")

    # Probability from direct prob, or from text/outputs.
    prob_val = None
    if prob is not None:
        try:
            prob_val = float(prob)
        except Exception:
            prob_val = None
    if prob_val is None:
        if isinstance(rec.get("text"), str):
            prob_val = _parse_prob(rec.get("text"))
        elif isinstance(rec.get("outputs"), list) and rec.get("outputs"):
            prob_val = _parse_prob(rec.get("outputs")[-1])

    if claim is None or prob_val is None:
        return None

    if not (0.0 <= prob_val <= 1.0):
        return None

    # Use video_id or query_id as the primary key component
    primary_id = video_id if video_id is not None else query_id
    if primary_id is None:
        return None

    # Extract stable artifact ID (direct fields, then meta envelope)
    stable_id = (
        rec.get("note_id")
        or rec.get("claim_id")
        or rec.get("inference_id")
        or meta.get("note_id")
        or meta.get("claim_id")
        or meta.get("inference_id")
    )

    key = _unli_key(primary_id, claim)
    payload = {
        "prob": prob_val,
        "raw": rec,
    }
    # Preserve label if present (some UNLI formats include it).
    if isinstance(meta.get("label"), (int, float, str)):
        payload["label"] = meta.get("label")
    return key, stable_id, payload


def _build_unli_index(
    unli_jsonl: str,
) -> Tuple[Dict[Tuple[str, str], dict], Dict[str, dict]]:
    """Build UNLI lookup indices from predictions JSONL.

    Returns (key_index, id_index):
      - key_index: {(primary_id, text): payload} — existing tuple-key index
      - id_index:  {stable_artifact_id: payload} — new index keyed by stable ID
    """
    key_index: Dict[Tuple[str, str], dict] = {}
    id_index: Dict[str, dict] = {}
    for rec in _iter_jsonl(unli_jsonl):
        parsed = _extract_unli_record(rec)
        if parsed is None:
            continue
        key, stable_id, payload = parsed
        if key not in key_index:
            key_index[key] = payload
        if stable_id is not None and stable_id not in id_index:
            id_index[stable_id] = payload
    return key_index, id_index


def _calibrate_claims(claims_jsonl: str, key_index: Dict, id_index: Dict, out: str) -> Tuple[int, int]:
    """Calibrate per-video claims.jsonl (legacy format — no stable IDs)."""
    total_claims = 0
    matched = 0
    with open(out, "w") as outf:
        for video_rec in _iter_jsonl(claims_jsonl):
            if not isinstance(video_rec, dict):
                continue
            video_id = str(video_rec.get("video_id") or "").strip()
            claims = video_rec.get("claims") if isinstance(video_rec.get("claims"), list) else []
            new_claims = []
            for c in claims:
                total_claims += 1
                if not isinstance(c, dict):
                    new_claims.append(c)
                    continue
                claim_text = str(c.get("claim") or "").strip()
                key = _unli_key(video_id, claim_text)
                item = dict(c)
                if key in key_index:
                    item["calibration"] = {"unli": key_index[key]}
                    item["confidence"] = key_index[key].get("prob")
                    matched += 1
                new_claims.append(item)
            new_video_rec = dict(video_rec)
            new_video_rec["claims"] = new_claims
            outf.write(json.dumps(new_video_rec, ensure_ascii=False) + "\n")
    return total_claims, matched


def _calibrate_general_notes(notes_jsonl: str, key_index: Dict, id_index: Dict, out: str) -> Tuple[int, int]:
    """Calibrate general_notes.jsonl — prefer note_id match, fall back to (video_id, text)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(notes_jsonl):
        total += 1
        item = dict(rec)
        note_id = rec.get("note_id")
        payload = id_index.get(note_id) if note_id else None
        if payload is None:
            video_id = str(rec.get("video_id") or "").strip()
            text = str(rec.get("text") or "").strip()
            key = _unli_key(video_id, text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def _calibrate_query_claims(claims_jsonl: str, key_index: Dict, id_index: Dict, out: str) -> Tuple[int, int]:
    """Calibrate query_conditioned_claims.jsonl — prefer claim_id match, fall back to (video_id, claim)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(claims_jsonl):
        total += 1
        item = dict(rec)
        claim_id = rec.get("claim_id")
        payload = id_index.get(claim_id) if claim_id else None
        if payload is None:
            video_id = str(rec.get("video_id") or "").strip()
            claim_text = str(rec.get("claim") or "").strip()
            key = _unli_key(video_id, claim_text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def _calibrate_inferences(inferences_jsonl: str, key_index: Dict, id_index: Dict, out: str) -> Tuple[int, int]:
    """Calibrate inference JSONL — prefer inference_id match, fall back to (query_id, claim)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(inferences_jsonl):
        total += 1
        item = dict(rec)
        inference_id = rec.get("inference_id")
        payload = id_index.get(inference_id) if inference_id else None
        if payload is None:
            query_id = str(rec.get("query_id") or "").strip()
            claim_text = str(rec.get("claim") or "").strip()
            key = _unli_key(query_id, claim_text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-jsonl", required=True, help="Input artifact JSONL path")
    ap.add_argument("--unli-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--artifact-type",
        choices=["claims", "general-notes", "query-claims", "inferences"],
        default="claims",
        help="Type of artifact to calibrate (default: claims for backward compat)",
    )
    ap.add_argument("--resolved-config-out", default="")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    manifest = build_run_manifest(
        script_name="calibrate_unli.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "artifact_type": args.artifact_type,
            "claims_jsonl": args.claims_jsonl,
            "unli_jsonl": args.unli_jsonl,
            "out": args.out,
        },
        extra={"prediction_manifest": "predict_run_manifest.json"},
    )
    manifest_path = write_run_manifest(out_dir, manifest, filename="calibrate_run_manifest.json")
    resolved_config_path = None
    if args.resolved_config_out:
        resolved_config_path = write_resolved_config(
            out_dir,
            vars(args),
            filename=args.resolved_config_out,
        )

    key_index, id_index = _build_unli_index(args.unli_jsonl)
    os.makedirs(out_dir, exist_ok=True)

    if args.artifact_type == "claims":
        total, matched = _calibrate_claims(args.claims_jsonl, key_index, id_index, args.out)
    elif args.artifact_type == "general-notes":
        total, matched = _calibrate_general_notes(args.claims_jsonl, key_index, id_index, args.out)
    elif args.artifact_type == "query-claims":
        total, matched = _calibrate_query_claims(args.claims_jsonl, key_index, id_index, args.out)
    elif args.artifact_type == "inferences":
        total, matched = _calibrate_inferences(args.claims_jsonl, key_index, id_index, args.out)

    print(f"[ok] UNLI records indexed: {len(key_index)} by key, {len(id_index)} by ID")
    print(f"[ok] Artifacts processed: {total}, matched: {matched}")
    print(f"[ok] Wrote calibrated output -> {args.out}")
    print(f"[ok] -> {manifest_path}")
    if resolved_config_path:
        print(f"[ok] -> {resolved_config_path}")


if __name__ == "__main__":
    main()
