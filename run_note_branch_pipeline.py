#!/usr/bin/env python3
"""Run the general-note downstream branch end-to-end.

Assembles note packets, runs higher-level inference, generates reports, and
evaluates the note-based branch for a provided general-notes JSONL.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from contracts import DEFAULT_QUERIES_JSONL, DEFAULT_TOPIC_MAPPING, DEFAULT_VLM_MODEL
from run_metadata import build_run_manifest, write_run_manifest


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def _run(script_rel_path: str, args: list[str]) -> None:
    cmd = [sys.executable, os.path.join(REPO_ROOT, script_rel_path), *args]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the general-note downstream branch end-to-end")
    ap.add_argument("--notes", required=True, help="General notes JSONL to use for the branch")
    ap.add_argument(
        "--query-claims",
        default=None,
        help="Query-conditioned claims JSONL for evaluation context",
    )
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--mapping", default=DEFAULT_TOPIC_MAPPING)
    ap.add_argument("--model", default=DEFAULT_VLM_MODEL)
    ap.add_argument("--download-dir", default="")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--unli-threshold", type=float, default=None,
                     help="Drop notes with calibration.unli.prob below this value")
    ap.add_argument("--packets-dir", required=True)
    ap.add_argument("--inferences-dir", required=True)
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    manifest = build_run_manifest(
        script_name="run_note_branch_pipeline.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "model": args.model,
            "top_k": args.top_k,
            "unli_threshold": args.unli_threshold,
            "notes": args.notes,
        },
    )
    manifest_path = write_run_manifest(
        args.evaluation_dir,
        manifest,
        filename="note_branch_pipeline_run_manifest.json",
    )

    assemble_args = [
        "--stream", "general-note",
        "--notes", args.notes,
        "--queries-jsonl", args.queries_jsonl,
        "--mapping", args.mapping,
        "--top-k", str(args.top_k),
        "--out-dir", args.packets_dir,
    ]
    if args.unli_threshold is not None:
        assemble_args.extend(["--unli-threshold", str(args.unli_threshold)])
    if args.verbose:
        assemble_args.append("--verbose")
    _run("assemble_packets.py", assemble_args)

    infer_args = [
        "--stream", "general-note",
        "--packets-dir", args.packets_dir,
        "--notes", args.notes,
        "--queries-jsonl", args.queries_jsonl,
        "--model", args.model,
        "--download-dir", args.download_dir,
        "--out-dir", args.inferences_dir,
    ]
    if args.verbose:
        infer_args.append("--verbose")
    _run("infer_higher_level.py", infer_args)

    report_args = [
        "--stream", "general-note",
        "--packets-dir", args.packets_dir,
        "--notes", args.notes,
        "--inferences", args.inferences_dir,
        "--queries-jsonl", args.queries_jsonl,
        "--out-dir", args.reports_dir,
    ]
    if args.verbose:
        report_args.append("--verbose")
    _run("generate_report.py", report_args)

    eval_args = [
        "--general-notes", args.notes,
        "--query-claims", args.query_claims,
        "--note-packets", args.packets_dir,
        "--inferences-note", args.inferences_dir,
        "--reports-note-based", args.reports_dir,
        "--queries-jsonl", args.queries_jsonl,
        "--mapping", args.mapping,
        "--out-dir", args.evaluation_dir,
    ]
    if args.verbose:
        eval_args.append("--verbose")
    _run("evaluate.py", eval_args)

    print(f"[ok] note-branch pipeline complete")
    print(f"[ok] -> {manifest_path}")


if __name__ == "__main__":
    main()
