#!/usr/bin/env python3
"""
Step 2: Higher-level inference over stream packets.

Runs LLM inference over each stream's packet to synthesize higher-level
inferences with source_ids pointing back to packet contents.

Two modes via --stream flag:
  - general-note: reads note packets, runs inference over notes
  - query-based: reads claim packets, runs inference over claims

Usage:
    python note_taking/infer_higher_level.py \
        --stream general-note \
        --packets-dir note_taking/outputs/note_packets \
        --notes note_taking/outputs/general_notes/general_notes.jsonl \
        --model Qwen/Qwen3.5-9B \
        --out-dir note_taking/outputs/inferences_note

    python note_taking/infer_higher_level.py \
        --stream query-based \
        --packets-dir note_taking/outputs/claim_packets \
        --claims note_taking/outputs/query_claims_single/query_conditioned_claims.jsonl \
        --model Qwen/Qwen3.5-9B \
        --out-dir note_taking/outputs/inferences_query
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm

from contracts import (
    DEFAULT_QUERIES_JSONL,
    DEFAULT_VLM_MODEL,
    load_queries,
    validate_higher_level_inference,
)
from prompts import (
    call_llm_with_retry,
    prompt_higher_level_inference,
    prompt_higher_level_inference_retry,
)
from run_metadata import build_run_manifest, write_run_manifest

from models.vlm import Qwen3_5_VL


def _iter_jsonl(path: str):
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


def _load_packets(packets_dir: str) -> List[dict]:
    """Load all packets from a directory."""
    combined = os.path.join(packets_dir, "all_packets.json")
    if os.path.exists(combined):
        with open(combined, "r") as f:
            return json.load(f)
    packets = []
    for fname in sorted(os.listdir(packets_dir)):
        if fname.startswith("query_") and fname.endswith(".json"):
            with open(os.path.join(packets_dir, fname), "r") as f:
                packets.append(json.load(f))
    return packets


def _build_note_index(notes_jsonl: str) -> Dict[str, dict]:
    """Index general notes by note_id."""
    idx = {}
    for note in _iter_jsonl(notes_jsonl):
        nid = note.get("note_id", "")
        if nid:
            idx[nid] = note
    return idx


def _build_claim_index(claims_jsonl: str) -> Dict[str, dict]:
    """Index query-conditioned claims by claim_id."""
    idx = {}
    for claim in _iter_jsonl(claims_jsonl):
        cid = claim.get("claim_id", "")
        if cid:
            idx[cid] = claim
    return idx


def _sanitize_source_ids(source_ids: List[str], allowed_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Keep only packet-local source ids, preserving order and uniqueness."""
    allowed = set(allowed_ids)
    valid = []
    invalid = []
    seen = set()
    for sid in source_ids or []:
        if sid in allowed:
            if sid not in seen:
                valid.append(sid)
                seen.add(sid)
        else:
            invalid.append(sid)
    return valid, invalid


def infer_over_packet(
    model,
    packet: dict,
    evidence_items: List[dict],
    query: dict,
    stream: str,
    run_id: str,
) -> Tuple[List[dict], List[str]]:
    """Run LLM inference over a single packet's evidence."""
    qid = query["query_id"]
    topic = packet.get("topic", "")
    allowed_ids = [item.get("id", "") for item in evidence_items if item.get("id")]

    prompt = prompt_higher_level_inference(
        query_id=qid,
        query=query.get("query", ""),
        topic=topic,
        evidence_items=evidence_items,
    )
    retry = prompt_higher_level_inference_retry(
        query_id=qid,
        query=query.get("query", ""),
        topic=topic,
        evidence_items=evidence_items,
    )
    result = call_llm_with_retry(
        model, prompt, retry, "inferences", video_path=None,
    )

    inferences = []
    warnings = []
    for idx, inf in enumerate(result.get("inferences", [])):
        source_ids, invalid_ids = _sanitize_source_ids(inf.get("source_ids", []), allowed_ids)
        if invalid_ids:
            warnings.append(
                f"query {qid} inference[{idx}]: dropped invalid source_ids {invalid_ids}"
            )
        if not source_ids:
            warnings.append(
                f"query {qid} inference[{idx}]: skipped because no source_ids matched packet contents"
            )
            continue
        record = {
            "inference_id": f"hli-{stream}-{qid}-{idx:03d}",
            "query_id": qid,
            "topic": topic,
            "claim": inf.get("claim", ""),
            "source_ids": source_ids,
            "confidence": None,
            "is_post_grounded": True,
            "calibration": None,
            "stream": stream,
            "run_id": run_id,
        }
        inferences.append(record)

    return inferences, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: Higher-level inference over packets")
    ap.add_argument("--stream", choices=["general-note", "query-based"], required=True)
    ap.add_argument("--packets-dir", required=True)
    ap.add_argument("--notes", default=None, help="General notes JSONL (for general-note stream)")
    ap.add_argument("--claims", default=None, help="Query-conditioned claims JSONL (for query-based stream)")
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--model", default=DEFAULT_VLM_MODEL)
    ap.add_argument("--download-dir", default="")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help=("Cap vLLM context length. Qwen3-VL-30B advertises 262K "
                          "which needs 24 GB KV cache and OOMs after FP8 weights "
                          "load. Step 5 prompts are small (packets + persona, no "
                          "video frames) so 32K-64K is plenty."))
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM GPU memory fraction. Default None lets vLLM choose.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--only-query-ids",
        default="",
        help=(
            "Optional comma-separated list of query_ids to process (e.g. '3' or "
            "'1,2,5'). Used by run_query.sh's PARALLEL_STEP5 path so each parallel "
            "worker only handles its assigned queries. Empty = process all packets."
        ),
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    # Parse --only-query-ids into a set (strings, since query_ids may be non-numeric).
    only_qids: set = set()
    if args.only_query_ids:
        only_qids = {q.strip() for q in args.only_query_ids.split(",") if q.strip()}

    manifest = build_run_manifest(
        script_name="infer_higher_level.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "stream": args.stream,
            "model": args.model,
            "temperature": args.temperature,
            "presence_penalty": args.presence_penalty,
            "max_tokens": args.max_tokens,
        },
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)

    queries = load_queries(args.queries_jsonl)
    query_by_id = {q["query_id"]: q for q in queries}

    packets = _load_packets(args.packets_dir)
    print(f"Loaded {len(packets)} packets from {args.packets_dir}")

    if only_qids:
        before = len(packets)
        packets = [p for p in packets if str(p.get("query_id", "")) in only_qids]
        print(f"Filtering to {len(packets)} of {before} packets (--only-query-ids={sorted(only_qids)})")

    # Build evidence index
    if args.stream == "general-note":
        stream_key = "general_note"
        if not args.notes:
            ap.error("--notes is required for general-note stream")
        evidence_idx = _build_note_index(args.notes)
        id_field = "note_ids"
        text_field = "text"
        id_key = "note_id"
        print(f"Indexed {len(evidence_idx)} general notes")
    else:
        stream_key = "query_based"
        if not args.claims:
            ap.error("--claims is required for query-based stream")
        evidence_idx = _build_claim_index(args.claims)
        id_field = "claim_ids"
        text_field = "claim"
        id_key = "claim_id"
        print(f"Indexed {len(evidence_idx)} query-conditioned claims")

    print(f"\nInitializing model: {args.model}")
    model = Qwen3_5_VL(
        model=args.model,
        download_dir=args.download_dir,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    all_inferences = []
    skipped_packets: List[dict] = []

    pbar = tqdm(total=len(packets), desc=f"Step 2 {args.stream} packets", unit="packet")

    for packet in packets:
        qid = packet.get("query_id", "")
        query = query_by_id.get(qid, {"query_id": qid, "query": ""})
        pbar.set_postfix(query_id=qid, inferences=len(all_inferences), skipped=len(skipped_packets))

        # Gather evidence items from packet
        item_ids = packet.get(id_field, [])
        evidence_items = []
        for eid in item_ids:
            rec = evidence_idx.get(eid, {})
            if rec:
                evidence_items.append({
                    "id": eid,
                    "text": rec.get(text_field, ""),
                    "video_id": rec.get("video_id", ""),
                })

        if not evidence_items:
            skipped_packets.append({
                "query_id": qid,
                "topic": packet.get("topic", ""),
                "reason": "no_evidence_items",
                "stream": stream_key,
                "packet_size": len(item_ids),
            })
            if args.verbose:
                print(f"  query {qid}: no evidence items, skipping")
            pbar.update(1)
            continue

        inferences, inference_warnings = infer_over_packet(
            model, packet, evidence_items, query, stream_key, manifest["run_id"],
        )
        if args.verbose:
            for warning in inference_warnings:
                print(f"  WARN {warning}")

        # Validate
        for inf in inferences:
            errs = validate_higher_level_inference(inf)
            if errs and args.verbose:
                print(f"  WARN inference {inf.get('inference_id')}: {errs}")

        all_inferences.extend(inferences)

        if args.verbose:
            print(f"  query {qid}: {len(inferences)} inferences from {len(evidence_items)} evidence items")

        pbar.update(1)

    pbar.close()

    # Write per-query files
    inferences_by_query: Dict[str, List[dict]] = {}
    for inf in all_inferences:
        qid = inf.get("query_id", "")
        inferences_by_query.setdefault(qid, []).append(inf)

    for qid, infs in inferences_by_query.items():
        out_path = os.path.join(args.out_dir, f"query_{qid}.jsonl")
        _write_jsonl(out_path, infs)

    # Write combined
    combined_path = os.path.join(args.out_dir, "inferences.jsonl")
    _write_jsonl(combined_path, all_inferences)

    skipped_path = os.path.join(args.out_dir, "skipped_packets.jsonl")
    _write_jsonl(skipped_path, skipped_packets)

    print(f"\n[ok] {len(all_inferences)} higher-level inferences from {len(packets)} packets")
    print(f"[ok] Skipped packets: {len(skipped_packets)}")
    print(f"[ok] -> {args.out_dir}/query_{{N}}.jsonl")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {skipped_path}")
    print(f"[ok] -> {manifest_path}")


if __name__ == "__main__":
    main()
