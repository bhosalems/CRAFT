#!/usr/bin/env python3
"""
Convert pipeline reports into MAGMaR submission JSONL format.

Reads all_reports.json and produces a JSONL file where each line is one
query, containing the top-level inference texts as responses with their
source video citations.

Usage:
    python format_submission.py \
        --reports outputs_query_branchv3/reports_query_based/all_reports.json \
        --team-id my_team \
        --run-id my_system-v3 \
        --task oracle \
        --out submission.jsonl
"""

import argparse
import json
import re


# Sub-clause connectors (preceded by a comma) that introduce a new
# fact-bearing clause. Splitting here turns compound sentences into
# atomic claims so the MIRAGE judge can score each piece independently
# instead of failing the whole sentence on one unsupported neighbor.
_CLAUSE_SPLIT_RE = re.compile(
    r",\s+(?="
    r"with\s+|"
    r"while(?:\s+also)?\s+|"
    r"including\s+|"
    r"resulting\s+in\s+|"
    r"causing\s+|"
    r"prompting\s+|"
    r"followed\s+by\s+|"
    r"leading\s+to\s+|"
    r"as\s+well\s+as\s+|"
    r"along\s+with\s+"
    r")",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_MIN_ATOMIC_WORDS = 4


def _split_into_atomic_claims(text: str) -> list:
    """Split a section's compound sentence into atomic claim strings.

    Heuristic: split on `.` `?` `!` first, then within each sentence on
    `,` followed by a sub-clause connector (with/while/including/...).
    Pieces shorter than _MIN_ATOMIC_WORDS are folded back into their
    parent so trivial fragments don't pollute the submission.
    Returns the original sentence as a single-item list when no split
    point is found.
    """
    text = (text or "").strip()
    if not text:
        return []

    atoms: list = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip().rstrip(",;")
        if not sentence:
            continue

        pieces = _CLAUSE_SPLIT_RE.split(sentence)
        # Re-join short fragments back into their preceding piece so a
        # stray ", with X" that happens to be 2 words doesn't become its
        # own claim.
        cleaned: list = []
        for p in pieces:
            p = p.strip().rstrip(",;")
            if not p:
                continue
            if len(p.split()) < _MIN_ATOMIC_WORDS and cleaned:
                cleaned[-1] = cleaned[-1].rstrip(",.; ") + ", " + p
            else:
                cleaned.append(p)

        for p in cleaned:
            # Make sure each atom ends with a period for clean reading.
            if not re.search(r"[.!?]$", p):
                p = p + "."
            atoms.append(p)

    return atoms or [text]


def _load_chunk_map(path: str) -> dict:
    """Load chunk_id -> {original_id,...} mapping.

    Accepts either:
    - {"created_at": ..., "chunk_map": {...}} (preferred)
    - {...} (raw mapping) for backwards compatibility.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "chunk_map" in data and isinstance(data["chunk_map"], dict):
        return data["chunk_map"]
    if isinstance(data, dict):
        return data
    return {}


def _map_video_id(video_id: str, chunk_map: dict) -> str:
    if not video_id or video_id == "unknown":
        return video_id
    rec = chunk_map.get(video_id)
    if isinstance(rec, dict):
        orig = rec.get("original_id")
        if isinstance(orig, str) and orig:
            return orig
    return video_id


def convert_report(
    report: dict,
    team_id: str,
    run_id: str,
    task: str,
    *,
    chunk_map: dict,
    atomize: bool = True,
) -> dict:
    """Convert a single report to submission format.

    When ``atomize`` is True, each section's compound sentence is split
    into atomic claims and emitted as separate ``responses`` entries
    sharing the same citations. The MIRAGE judge scores each predicted
    ``text`` as one yes/no entailment, so atomic pieces let supported
    facts count toward precision instead of being dragged down by an
    unsupported clause in the same sentence.
    """
    responses = []
    all_references = set()

    for section in report.get("sections", []):
        text = section.get("text", "").strip()
        if not text:
            continue

        # Collect video IDs from source_citations (inference sources)
        citations = set()
        for sc in section.get("source_citations", []):
            vid = sc.get("video_id", "")
            if vid and vid != "unknown":
                citations.add(_map_video_id(vid, chunk_map))

        # Fallback: primary citation if no source_citations
        if not citations:
            vid = section.get("citation", {}).get("video_id", "")
            if vid and vid != "unknown":
                citations.add(_map_video_id(vid, chunk_map))

        sorted_citations = sorted(citations)
        all_references.update(citations)

        atoms = _split_into_atomic_claims(text) if atomize else [text]
        for atom in atoms:
            responses.append({
                "text": atom,
                "citations": sorted_citations,
            })

    return {
        "metadata": {
            "run_id": run_id,
            "query_id": str(report.get("query_id", "")),
            "team_id": team_id,
            "task": task,
        },
        "responses": responses,
        "references": sorted(all_references),
    }


def main():
    ap = argparse.ArgumentParser(description="Format reports into MAGMaR submission JSONL")
    ap.add_argument("--reports", required=True, help="Path to all_reports.json")
    ap.add_argument("--team-id", required=True, help="Your team identifier")
    ap.add_argument("--run-id", required=True, help="System run identifier (e.g. system_name-config)")
    ap.add_argument("--task", choices=["oracle", "rag"], required=True, help="Task type")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--chunk-map",
        default="",
        help=(
            "Optional JSON mapping produced by chunk_videos.py. When provided, any chunked "
            "video IDs (e.g. <orig>__chunk000) are mapped back to the original video ID in the submission."
        ),
    )
    ap.add_argument(
        "--atomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Post-hoc heuristic split of compound section sentences into atomic claims. "
            "OFF by default because step 1b + step 2 prompts now produce atomic claims "
            "directly. Pass --atomize as a fallback for legacy reports whose sections "
            "still contain compound prose."
        ),
    )
    args = ap.parse_args()

    chunk_map = {}
    if args.chunk_map:
        chunk_map = _load_chunk_map(args.chunk_map)

    with open(args.reports) as f:
        reports = json.load(f)

    print(f"Loaded {len(reports)} reports from {args.reports}")

    with open(args.out, "w") as outf:
        for report in reports:
            entry = convert_report(
                report,
                args.team_id,
                args.run_id,
                args.task,
                chunk_map=chunk_map,
                atomize=args.atomize,
            )
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    print(f"Written {len(reports)} entries to {args.out}")
    print(f"  team_id: {args.team_id}")
    print(f"  run_id:  {args.run_id}")
    print(f"  task:    {args.task}")

    # Quick stats
    total_responses = 0
    total_refs = 0
    with open(args.out) as f:
        for line in f:
            entry = json.loads(line)
            total_responses += len(entry["responses"])
            total_refs += len(entry["references"])
    print(f"  total responses: {total_responses}")
    print(f"  total references: {total_refs}")


if __name__ == "__main__":
    main()
