#!/usr/bin/env python3
"""
Step 1.5+: Packet assembly for both streams.

Assembles per-query packets from general notes or query-conditioned claims.

Two modes via --stream flag:
  - general-note: reads general notes, assembles one note_packet per query
  - query-based: reads query-conditioned claims, assembles one claim_packet per query

Usage:
    python note_taking/assemble_packets.py \
        --stream general-note \
        --notes note_taking/outputs/general_notes/general_notes.jsonl \
        --out-dir note_taking/outputs/note_packets

    python note_taking/assemble_packets.py \
        --stream query-based \
        --claims note_taking/outputs/query_claims_single/query_conditioned_claims.jsonl \
        --out-dir note_taking/outputs/claim_packets
"""

import argparse
import datetime as _dt
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

from contracts import (
    DEFAULT_QUERIES_JSONL,
    DEFAULT_TOPIC_MAPPING,
    build_query_topic_map,
    load_queries,
    load_topic_mapping,
    validate_claim_packet,
    validate_note_packet,
)
from run_metadata import build_run_manifest, write_run_manifest

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _get_unli_prob(rec: dict) -> float | None:
    """Extract UNLI probability from a calibrated record, or None if absent."""
    cal = rec.get("calibration")
    if isinstance(cal, dict):
        unli = cal.get("unli")
        if isinstance(unli, dict):
            p = unli.get("prob")
            if isinstance(p, (int, float)):
                return float(p)
    return None


def _score_note_against_query(note: dict, query: dict) -> float:
    """Token-overlap score between a note and a query."""
    note_tokens = _tokenize(note.get("text", ""))
    if not note_tokens:
        return 0.0

    query_text = " ".join([
        query.get("query", ""),
        query.get("title", ""),
        query.get("persona_title", ""),
    ])
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0

    overlap = len(note_tokens & query_tokens)
    return overlap / len(note_tokens)


def assemble_note_packets(
    notes_jsonl: str,
    queries: List[dict],
    topic_map: Dict[str, List[str]],
    qtm: Dict[str, List[dict]],
    run_id: str,
    top_k: int = 50,
    unli_threshold: float | None = None,
    verbose: bool = False,
) -> List[dict]:
    """Assemble note packets from general notes."""
    # Load all notes, applying optional UNLI threshold filter
    raw_notes: List[dict] = []
    total_loaded = 0
    filtered_out = 0
    for note in _iter_jsonl(notes_jsonl):
        total_loaded += 1
        if unli_threshold is not None:
            prob = _get_unli_prob(note)
            if prob is not None and prob < unli_threshold:
                filtered_out += 1
                continue
        raw_notes.append(note)

    if unli_threshold is not None:
        print(f"  UNLI threshold={unli_threshold}: kept {total_loaded - filtered_out}/{total_loaded} notes "
              f"(filtered {filtered_out})")

    # Deduplicate notes with identical (video_id, text), keeping the first
    # occurrence and merging timestamp ranges.
    seen: Dict[tuple, dict] = {}
    dedup_count = 0
    for note in raw_notes:
        dedup_key = (note.get("video_id", ""), note.get("text", ""))
        if dedup_key in seen:
            dedup_count += 1
            existing = seen[dedup_key]
            et = existing.get("timestamp")
            nt = note.get("timestamp")
            if isinstance(et, list) and isinstance(nt, list) and len(et) == 2 and len(nt) == 2:
                existing["timestamp"] = [min(et[0], nt[0]), max(et[1], nt[1])]
        else:
            seen[dedup_key] = note

    deduped_notes = list(seen.values())
    if dedup_count > 0:
        print(f"  Dedup: {len(raw_notes)} -> {len(deduped_notes)} notes ({dedup_count} duplicates merged)")

    notes_by_topic: Dict[str, List[dict]] = defaultdict(list)
    for note in deduped_notes:
        notes_by_topic[note.get("topic", "")].append(note)

    packets = []
    total_queries = sum(len(topic_queries) for topic_queries in qtm.values())
    pbar = tqdm(total=total_queries, desc="Assemble note packets", unit="query")

    for topic, topic_queries in sorted(qtm.items()):
        topic_notes = notes_by_topic.get(topic, [])
        for q in topic_queries:
            pbar.set_postfix(topic=topic, query_id=q["query_id"])
            scored = []
            for note in topic_notes:
                overlap = _score_note_against_query(note, q)
                if overlap > 0:
                    prob = _get_unli_prob(note)
                    unli_weight = prob if prob is not None else 1.0
                    score = overlap * unli_weight
                    scored.append((score, note))
            scored.sort(key=lambda x: (-x[0], x[1].get("note_id", "")))
            top = scored[:top_k]

            packet = {
                "query_id": q["query_id"],
                "topic": topic,
                "stream": "general_note",
                "note_ids": [n.get("note_id", "") for _, n in top],
                "scores": [round(s, 4) for s, _ in top],
                "provenance": {
                    "run_id": run_id,
                    "retrieval_method": "token_overlap",
                    "top_k": top_k,
                    "unli_threshold": unli_threshold,
                    "candidate_count": len(topic_notes),
                    "created_at": _utc_now_iso(),
                },
            }
            packets.append(packet)

            if verbose:
                print(f"  query {q['query_id']} ({topic}): {len(top)} notes in packet")

                pbar.update(1)

            pbar.close()

    return packets


def assemble_claim_packets(
    claims_jsonl: str,
    queries: List[dict],
    topic_map: Dict[str, List[str]],
    qtm: Dict[str, List[dict]],
    run_id: str,
    unli_threshold: float | None = None,
    verbose: bool = False,
) -> List[dict]:
    """Assemble claim packets from query-conditioned claims."""
    # Group claims by query_id
    claims_by_query: Dict[str, List[dict]] = defaultdict(list)
    total_loaded = 0
    filtered_out = 0
    for claim in _iter_jsonl(claims_jsonl):
        total_loaded += 1
        if unli_threshold is not None:
            prob = _get_unli_prob(claim)
            if prob is not None and prob < unli_threshold:
                filtered_out += 1
                continue
        claims_by_query[claim.get("query_id", "")].append(claim)

    if unli_threshold is not None:
        print(f"  UNLI threshold={unli_threshold}: kept {total_loaded - filtered_out}/{total_loaded} claims (filtered {filtered_out})")

    packets = []
    total_queries = sum(len(topic_queries) for topic_queries in qtm.values())
    pbar = tqdm(total=total_queries, desc="Assemble claim packets", unit="query")

    for topic, topic_queries in sorted(qtm.items()):
        for q in topic_queries:
            qid = q["query_id"]
            pbar.set_postfix(topic=topic, query_id=qid)
            query_claims = claims_by_query.get(qid, [])

            # Rank by confidence (descending), then claim_id for stability
            ranked = sorted(
                query_claims,
                key=lambda c: (-(c.get("confidence") or 0.0), c.get("claim_id", "")),
            )

            packet = {
                "query_id": qid,
                "topic": topic,
                "stream": "query_based",
                "claim_ids": [c.get("claim_id", "") for c in ranked],
                "scores": [round(c.get("confidence") or 0.0, 4) for c in ranked],
                "provenance": {
                    "run_id": run_id,
                    "retrieval_method": "confidence_ranked",
                    "unli_threshold": unli_threshold,
                    "candidate_count": len(query_claims),
                    "created_at": _utc_now_iso(),
                },
            }
            packets.append(packet)

            if verbose:
                print(f"  query {qid} ({topic}): {len(ranked)} claims in packet")

                pbar.update(1)

            pbar.close()

    return packets


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble per-query packets for both streams")
    ap.add_argument("--stream", choices=["general-note", "query-based"], required=True)
    ap.add_argument("--notes", default=None, help="General notes JSONL (for general-note stream)")
    ap.add_argument("--claims", default=None, help="Query-conditioned claims JSONL (for query-based stream)")
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--mapping", default=DEFAULT_TOPIC_MAPPING)
    ap.add_argument("--top-k", type=int, default=50, help="Max notes per packet (general-note stream)")
    ap.add_argument("--unli-threshold", type=float, default=None,
                     help="Drop notes/claims with calibration.unli.prob below this value (requires calibrated JSONL)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    manifest = build_run_manifest(
        script_name="assemble_packets.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "stream": args.stream,
            "top_k": args.top_k,
            "unli_threshold": args.unli_threshold,
        },
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)

    queries = load_queries(args.queries_jsonl)
    topic_map = load_topic_mapping(args.mapping)
    qtm = build_query_topic_map(queries, list(topic_map.keys()))

    if args.stream == "general-note":
        if not args.notes:
            ap.error("--notes is required for general-note stream")
        print(f"Assembling note packets from: {args.notes}")
        packets = assemble_note_packets(
            args.notes, queries, topic_map, qtm,
            manifest["run_id"], top_k=args.top_k,
            unli_threshold=args.unli_threshold, verbose=args.verbose,
        )
        validator = validate_note_packet
    else:
        if not args.claims:
            ap.error("--claims is required for query-based stream")
        print(f"Assembling claim packets from: {args.claims}")
        packets = assemble_claim_packets(
            args.claims, queries, topic_map, qtm, manifest["run_id"],
            unli_threshold=args.unli_threshold, verbose=args.verbose,
        )
        validator = validate_claim_packet

    # Validate
    errors = []
    for packet in packets:
        errs = validator(packet)
        if errs:
            errors.extend(errs)

    if errors:
        print(f"\nValidation issues ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    # Write per-query files
    os.makedirs(args.out_dir, exist_ok=True)
    for packet in packets:
        out_path = os.path.join(args.out_dir, f"query_{packet['query_id']}.json")
        with open(out_path, "w") as f:
            json.dump(packet, f, indent=2, ensure_ascii=False)

    # Write combined
    combined_path = os.path.join(args.out_dir, "all_packets.json")
    with open(combined_path, "w") as f:
        json.dump(packets, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] {len(packets)} {args.stream} packets assembled")
    print(f"[ok] -> {args.out_dir}/query_{{N}}.json")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")


if __name__ == "__main__":
    main()
