#!/usr/bin/env python3
"""
Step 3: Report generation for both streams.

Supports three modes:
  - Legacy mode (no --stream): note-taking or query-based via --grounded
  - general-note stream: reports from note packets + optional inferences
  - query-based stream: reports from claim packets + optional inferences

Usage:
    # Legacy: note-taking packets
    python note_taking/generate_report.py \
        --packets-dir note_taking/outputs/query_packets_note_taking \
        --grounded note_taking/outputs/grounded_heuristic/grounded_notes.jsonl \
        --out-dir note_taking/outputs/reports_note_taking

    # Stream: general-note
    python note_taking/generate_report.py \
        --stream general-note \
        --packets-dir note_taking/outputs/note_packets \
        --notes note_taking/outputs/general_notes/general_notes.jsonl \
        --inferences note_taking/outputs/inferences_note \
        --out-dir note_taking/outputs/reports_note_based

    # Stream: query-based
    python note_taking/generate_report.py \
        --stream query-based \
        --packets-dir note_taking/outputs/claim_packets \
        --claims note_taking/outputs/query_claims_single/query_conditioned_claims.jsonl \
        --inferences note_taking/outputs/inferences_query \
        --out-dir note_taking/outputs/reports_query_based
"""

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

from contracts import (
    DEFAULT_QUERIES_JSONL,
    load_queries,
    validate_report_citation,
)
from run_metadata import build_run_manifest, write_run_manifest


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_grounded_index(path: Optional[str]) -> Dict[str, dict]:
    """Index grounded notes by note_id for citation resolution."""
    if not path or not os.path.exists(path):
        return {}
    idx = {}
    for note in _iter_jsonl(path):
        nid = note.get("note_id", "")
        if nid:
            idx[nid] = note
    return idx


def _load_note_index(path: Optional[str]) -> Dict[str, dict]:
    """Index general notes by note_id."""
    if not path or not os.path.exists(path):
        return {}
    idx = {}
    for note in _iter_jsonl(path):
        nid = note.get("note_id", "")
        if nid:
            idx[nid] = note
    return idx


def _load_claim_index(path: Optional[str]) -> Dict[str, dict]:
    """Index query-conditioned claims by claim_id."""
    if not path or not os.path.exists(path):
        return {}
    idx = {}
    for claim in _iter_jsonl(path):
        cid = claim.get("claim_id", "")
        if cid:
            idx[cid] = claim
    return idx


def _load_inference_index(inferences_dir: Optional[str]) -> Dict[str, List[dict]]:
    """Load inferences grouped by query_id."""
    if not inferences_dir or not os.path.isdir(inferences_dir):
        return {}
    combined = os.path.join(inferences_dir, "inferences.jsonl")
    by_query: Dict[str, List[dict]] = {}
    if os.path.exists(combined):
        for inf in _iter_jsonl(combined):
            qid = inf.get("query_id", "")
            by_query.setdefault(qid, []).append(inf)
    return by_query


def _load_packets(packets_dir: str) -> List[dict]:
    """Load all query packets from a directory."""
    combined = os.path.join(packets_dir, "all_packets.json")
    if os.path.exists(combined):
        with open(combined, "r") as f:
            return json.load(f)
    # Fallback: load individual files
    packets = []
    for fname in sorted(os.listdir(packets_dir)):
        if fname.startswith("query_") and fname.endswith(".json"):
            with open(os.path.join(packets_dir, fname), "r") as f:
                packets.append(json.load(f))
    return packets


def _resolve_citation(
    note_id: str, grounded_idx: Dict[str, dict]
) -> dict:
    """Resolve a note_id to a citation with video_id and timestamp."""
    note = grounded_idx.get(note_id, {})
    return {
        "note_id": note_id,
        "video_id": note.get("video_id", "unknown"),
        "timestamp": note.get("timestamp_union"),
        "claim": note.get("claim", ""),
    }


# ---------------------------------------------------------------------------
# Legacy report generation (backward compat)
# ---------------------------------------------------------------------------


def _generate_note_taking_report(
    packet: dict,
    query: dict,
    grounded_idx: Dict[str, dict],
    run_id: str,
) -> dict:
    """Generate report from note-taking pipeline packet."""
    sections = []
    citations = []

    note_ids = packet.get("retrieved_note_ids", [])
    scores = packet.get("scores", [])

    for i, nid in enumerate(note_ids):
        note = grounded_idx.get(nid, {})
        claim = note.get("claim", "")
        if not claim:
            continue

        citation = _resolve_citation(nid, grounded_idx)
        citations.append(citation)

        score = scores[i] if i < len(scores) else 0.0
        sections.append({
            "text": claim,
            "citation": citation,
            "score": round(score, 4),
        })

    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": "note_taking",
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": "note_taking",
            "created_at": _utc_now_iso(),
        },
    }


def _generate_query_based_report_legacy(
    packet: dict,
    query: dict,
    run_id: str,
) -> dict:
    """Generate report from query-based pipeline packet (legacy)."""
    sections = []
    citations = []

    facts = packet.get("facts", [])

    for i, fact in enumerate(facts):
        vid = fact.get("video_id", "unknown")
        citation = {
            "note_id": f"fact-{query['query_id']}-{i:03d}",
            "video_id": vid,
            "timestamp": fact.get("timestamp"),
            "claim": fact.get("fact", ""),
        }
        citations.append(citation)

        sections.append({
            "text": fact.get("fact", ""),
            "citation": citation,
            "score": 1.0,
        })

    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": packet.get("pipeline", "query_based"),
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": packet.get("pipeline", "query_based"),
            "created_at": _utc_now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# New stream-based report generation
# ---------------------------------------------------------------------------


def _generate_general_note_report(
    packet: dict,
    query: dict,
    note_idx: Dict[str, dict],
    inferences: List[dict],
    run_id: str,
) -> dict:
    """Generate report from general-note stream."""
    sections = []
    citations = []

    if inferences:
        # Build report from inferences
        for inf in inferences:
            # Resolve inference -> source notes -> video_id + timestamp
            source_citations = []
            for sid in inf.get("source_ids", []):
                note = note_idx.get(sid, {})
                source_citations.append({
                    "note_id": sid,
                    "video_id": note.get("video_id", "unknown"),
                    "timestamp": note.get("timestamp"),
                    "claim": note.get("text", ""),
                })

            # Use first source citation as primary
            primary = source_citations[0] if source_citations else {
                "note_id": inf.get("inference_id", ""),
                "video_id": "unknown",
                "timestamp": None,
                "claim": inf.get("claim", ""),
            }
            citations.append(primary)
            citations.extend(source_citations[1:])

            sections.append({
                "text": inf.get("claim", ""),
                "citation": primary,
                "score": 1.0,
                "inference_id": inf.get("inference_id"),
                "source_citations": source_citations,
            })
    else:
        # Build directly from notes
        note_ids = packet.get("note_ids", [])
        scores = packet.get("scores", [])

        for i, nid in enumerate(note_ids):
            note = note_idx.get(nid, {})
            text = note.get("text", "")
            if not text:
                continue

            citation = {
                "note_id": nid,
                "video_id": note.get("video_id", "unknown"),
                "timestamp": note.get("timestamp"),
                "claim": text,
            }
            citations.append(citation)

            score = scores[i] if i < len(scores) else 0.0
            sections.append({
                "text": text,
                "citation": citation,
                "score": round(score, 4),
            })

    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": "general_note",
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": "general_note",
            "has_inferences": bool(inferences),
            "created_at": _utc_now_iso(),
        },
    }


def _generate_query_based_report(
    packet: dict,
    query: dict,
    claim_idx: Dict[str, dict],
    inferences: List[dict],
    run_id: str,
) -> dict:
    """Generate report from query-based stream."""
    sections = []
    citations = []

    if inferences:
        for inf in inferences:
            source_citations = []
            for sid in inf.get("source_ids", []):
                claim = claim_idx.get(sid, {})
                source_citations.append({
                    "note_id": sid,
                    "video_id": claim.get("video_id", "unknown"),
                    "timestamp": claim.get("timestamp"),
                    "claim": claim.get("claim", ""),
                })

            primary = source_citations[0] if source_citations else {
                "note_id": inf.get("inference_id", ""),
                "video_id": "unknown",
                "timestamp": None,
                "claim": inf.get("claim", ""),
            }
            citations.append(primary)
            citations.extend(source_citations[1:])

            sections.append({
                "text": inf.get("claim", ""),
                "citation": primary,
                "score": 1.0,
                "inference_id": inf.get("inference_id"),
                "source_citations": source_citations,
            })
    else:
        claim_ids = packet.get("claim_ids", [])
        scores = packet.get("scores", [])

        for i, cid in enumerate(claim_ids):
            claim = claim_idx.get(cid, {})
            text = claim.get("claim", "")
            if not text:
                continue

            citation = {
                "note_id": cid,
                "video_id": claim.get("video_id", "unknown"),
                "timestamp": claim.get("timestamp"),
                "claim": text,
            }
            citations.append(citation)

            score = scores[i] if i < len(scores) else 0.0
            sections.append({
                "text": text,
                "citation": citation,
                "score": round(score, 4),
            })

    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": "query_based",
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": "query_based",
            "has_inferences": bool(inferences),
            "created_at": _utc_now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# Shared rendering and validation
# ---------------------------------------------------------------------------


def _render_markdown(report: dict, query: dict) -> str:
    """Render a report as markdown."""
    lines = []
    lines.append(f"# Report: Query {report['query_id']}")
    lines.append(f"")
    lines.append(f"**Topic:** {report['topic']}")
    lines.append(f"**Pipeline:** {report['pipeline']}")
    lines.append(f"**Query:** {query.get('query', '')[:200]}...")
    lines.append(f"")
    lines.append(f"## Findings")
    lines.append(f"")

    for i, section in enumerate(report.get("sections", []), 1):
        cit = section.get("citation", {})
        vid = cit.get("video_id", "?")
        ts = cit.get("timestamp")
        ts_str = f" [{ts[0]:.0f}s-{ts[1]:.0f}s]" if ts else ""
        lines.append(f"{i}. {section['text']} [video:{vid}{ts_str}]")
        lines.append(f"")

    lines.append(f"## Citations")
    lines.append(f"")
    lines.append(f"| # | Video ID | Timestamp | Claim |")
    lines.append(f"|---|----------|-----------|-------|")
    for i, cit in enumerate(report.get("citations", []), 1):
        ts = cit.get("timestamp")
        ts_str = f"{ts[0]:.0f}-{ts[1]:.0f}s" if ts else "N/A"
        claim_short = (cit.get("claim", ""))[:80]
        lines.append(f"| {i} | {cit.get('video_id', '?')} | {ts_str} | {claim_short} |")

    return "\n".join(lines)


def _validate_citations(citations: list) -> list:
    """Validate all citations resolve properly."""
    errors = []
    for i, cit in enumerate(citations):
        errs = validate_report_citation(cit)
        if errs:
            errors.extend([f"citation[{i}]: {e}" for e in errs])
        if cit.get("video_id") == "unknown":
            errors.append(f"citation[{i}]: unresolved video_id")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reports from query packets")
    ap.add_argument("--packets-dir", required=True)
    ap.add_argument("--stream", choices=["general-note", "query-based"], default=None,
                     help="Stream mode (omit for legacy)")
    ap.add_argument("--grounded", default=None, help="Grounded notes JSONL (legacy note-taking pipeline)")
    ap.add_argument("--notes", default=None, help="General notes JSONL (general-note stream)")
    ap.add_argument("--claims", default=None, help="Query-conditioned claims JSONL (query-based stream)")
    ap.add_argument("--inferences", default=None, help="Inferences directory (optional)")
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on any citation error")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    manifest = build_run_manifest(
        script_name="generate_report.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "stream": args.stream,
            "strict": args.strict,
        },
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)

    queries = load_queries(args.queries_jsonl)
    query_by_id = {q["query_id"]: q for q in queries}

    packets = _load_packets(args.packets_dir)
    print(f"Loaded {len(packets)} query packets")

    # Load evidence indices depending on mode
    grounded_idx = {}
    note_idx = {}
    claim_idx = {}
    inference_idx: Dict[str, List[dict]] = {}

    if args.stream is None:
        # Legacy mode
        grounded_idx = _load_grounded_index(args.grounded)
        if grounded_idx:
            print(f"Loaded {len(grounded_idx)} grounded notes for citation resolution")
    elif args.stream == "general-note":
        note_idx = _load_note_index(args.notes)
        print(f"Loaded {len(note_idx)} general notes")
        inference_idx = _load_inference_index(args.inferences)
        if inference_idx:
            total_infs = sum(len(v) for v in inference_idx.values())
            print(f"Loaded {total_infs} inferences across {len(inference_idx)} queries")
    elif args.stream == "query-based":
        claim_idx = _load_claim_index(args.claims)
        print(f"Loaded {len(claim_idx)} query-conditioned claims")
        inference_idx = _load_inference_index(args.inferences)
        if inference_idx:
            total_infs = sum(len(v) for v in inference_idx.values())
            print(f"Loaded {total_infs} inferences across {len(inference_idx)} queries")

    os.makedirs(args.out_dir, exist_ok=True)
    all_reports = []
    total_citation_errors = 0

    pbar = tqdm(total=len(packets), desc="Generate reports", unit="query")

    for packet in packets:
        qid = packet.get("query_id", "")
        query = query_by_id.get(qid, {"query_id": qid, "query": ""})
        pbar.set_postfix(query_id=qid, reports=len(all_reports), citation_errors=total_citation_errors)

        if args.stream == "general-note":
            inferences = inference_idx.get(qid, [])
            report = _generate_general_note_report(packet, query, note_idx, inferences, manifest["run_id"])
        elif args.stream == "query-based":
            inferences = inference_idx.get(qid, [])
            report = _generate_query_based_report(packet, query, claim_idx, inferences, manifest["run_id"])
        else:
            # Legacy mode
            pipeline = packet.get("pipeline", "")
            if pipeline == "note_taking" and grounded_idx:
                report = _generate_note_taking_report(packet, query, grounded_idx, manifest["run_id"])
            else:
                report = _generate_query_based_report_legacy(packet, query, manifest["run_id"])

        # Validate citations
        cit_errors = _validate_citations(report.get("citations", []))
        if cit_errors and args.verbose:
            for e in cit_errors:
                print(f"  WARN query {qid}: {e}")
        total_citation_errors += len(cit_errors)

        all_reports.append(report)

        # Write report.json
        json_path = os.path.join(args.out_dir, f"report_{qid}.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Write report.md
        md_path = os.path.join(args.out_dir, f"report_{qid}.md")
        with open(md_path, "w") as f:
            f.write(_render_markdown(report, query))

        if args.verbose:
            n_sections = len(report.get("sections", []))
            n_cits = len(report.get("citations", []))
            print(f"  query {qid}: {n_sections} sections, {n_cits} citations")

        pbar.update(1)

    pbar.close()

    # Write combined
    combined_path = os.path.join(args.out_dir, "all_reports.json")
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] {len(all_reports)} reports generated")
    if total_citation_errors:
        print(f"[warn] {total_citation_errors} citation resolution issues")
    print(f"[ok] -> {args.out_dir}/report_{{N}}.json")
    print(f"[ok] -> {args.out_dir}/report_{{N}}.md")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")

    if args.strict and total_citation_errors > 0:
        print(f"\n[FAIL] --strict: {total_citation_errors} citation error(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
