#!/usr/bin/env python3
"""
Step 1.5 chunked runner: splits UNLI prediction across fresh subprocesses.

Why this exists:
    pyav / torchvision video decoding leaks mmap regions and file descriptors
    over time. After processing ~50-100 videos in a single process you can
    exhaust ``vm.max_map_count`` (default 65530 on Linux) which surfaces as:
        ERROR libav.swscaler: Failed initializing scaling graph (Resource temporarily unavailable)
        [Errno 11] Resource temporarily unavailable
    and a throughput collapse from ~1 item/s to 80+ s/item.

    We cannot reliably clear these leaks from Python (they live in libav's
    C heap and in kernel mm tracking). The simplest robust fix is to run
    ``predict_unli.py`` on small chunks of videos, each in its own subprocess,
    so the OS reclaims every mapping and descriptor when the subprocess exits.

What it does:
    1. Groups the input artifacts JSONL by video_id.
    2. Splits video_ids into chunks of --chunk-size videos.
    3. For each chunk: writes a tmp input JSONL, runs ``predict_unli.py`` as a
       subprocess on just that chunk, collects the chunk's predictions JSONL.
    4. Concatenates all chunk outputs into the final predictions JSONL.

All other arguments are forwarded verbatim to ``predict_unli.py``, so this
script is a drop-in replacement at the Hydra wrapper layer.

Usage (same args as predict_unli.py plus --chunk-size):
    python predict_unli_chunked.py \
        --artifact-type query-claims \
        --artifacts-jsonl outputs/query_claims_single/query_conditioned_claims.jsonl \
        --base-model AdoptedIrelia/UNLI --lora-path AdoptedIrelia/UNLI/lora \
        --out outputs/unli_query_claims/unli_predictions.jsonl \
        --chunk-size 40
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import List

# Path to the underlying per-chunk worker script.
PREDICT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict_unli.py")


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _append_file(src: str, dst_fh) -> int:
    """Append src file contents to dst_fh. Returns line count copied."""
    n = 0
    if not os.path.exists(src):
        return 0
    with open(src, "r") as f:
        for line in f:
            if line.strip():
                dst_fh.write(line if line.endswith("\n") else line + "\n")
                n += 1
    return n


def _build_passthrough_args(args: argparse.Namespace, chunk_in: str, chunk_out: str) -> List[str]:
    """Build the CLI arg list for predict_unli.py, substituting chunk I/O paths."""
    argv = [sys.executable, PREDICT_SCRIPT]
    argv += ["--artifact-type", args.artifact_type]
    argv += ["--scorer-backend", args.scorer_backend]
    argv += ["--artifacts-jsonl", chunk_in]
    argv += ["--video-root", args.video_root]
    if args.model:
        argv += ["--model", args.model]
    if args.base_model:
        argv += ["--base-model", args.base_model]
    if args.lora_path:
        argv += ["--lora-path", args.lora_path]
    if args.download_dir:
        argv += ["--download-dir", args.download_dir]
    argv += ["--fps", str(args.fps)]
    argv += ["--resized-height", str(args.resized_height)]
    argv += ["--resized-width", str(args.resized_width)]
    argv += ["--new-token-num", str(args.new_token_num)]
    argv += ["--new-token-prefix", args.new_token_prefix]
    argv += ["--device-map", args.device_map]
    argv += ["--torch-dtype", args.torch_dtype]
    argv += ["--attn-implementation", args.attn_implementation]
    argv += ["--max-frames", str(args.max_frames)]
    if args.enable_thinking:
        argv += ["--enable-thinking"]
    argv += ["--temperature", str(args.temperature)]
    argv += ["--top-p", str(args.top_p)]
    argv += ["--top-k", str(args.top_k)]
    argv += ["--max-tokens", str(args.max_tokens)]
    argv += ["--seed", str(args.seed)]
    argv += ["--repetition-penalty", str(args.repetition_penalty)]
    argv += ["--presence-penalty", str(args.presence_penalty)]
    argv += ["--qwen-fps", str(args.qwen_fps)]
    argv += ["--gc-every", str(args.gc_every)]
    if args.no_audio:
        argv += ["--no-audio"]
    argv += ["--out", chunk_out]
    if args.verbose:
        argv += ["--verbose"]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1.5 chunked UNLI prediction runner")
    # Chunking controls.
    ap.add_argument("--chunk-size", type=int, default=40,
                    help="Number of videos processed per subprocess (default 40). "
                         "Smaller = more subprocess overhead but fewer resource leaks "
                         "accumulated per process.")
    ap.add_argument("--chunk-tmp-dir", default=None,
                    help="Directory for per-chunk tmp inputs and outputs. "
                         "Defaults to a subdir of --out's directory.")
    ap.add_argument("--keep-chunk-outputs", action="store_true",
                    help="Do not delete per-chunk tmp files after merging (for debugging).")
    ap.add_argument("--stop-on-chunk-failure", action="store_true",
                    help="Abort the run if any chunk subprocess returns non-zero. "
                         "Default: log and continue to the next chunk.")

    # Pass-through args (must mirror predict_unli.py).
    ap.add_argument("--artifact-type", choices=["general-notes", "query-claims"], required=True)
    ap.add_argument("--scorer-backend", choices=["unli", "qwen_score"], default="unli")
    ap.add_argument("--artifacts-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--download-dir", default="")
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--resized-height", type=int, default=256)
    ap.add_argument("--resized-width", type=int, default=256)
    ap.add_argument("--new-token-num", type=int, default=100)
    ap.add_argument("--new-token-prefix", default="<CON_{idx}>")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--qwen-fps", type=float, default=1.0)
    ap.add_argument("--gc-every", type=int, default=25)
    ap.add_argument("--no-audio", action="store_true",
                    help="Disable audio decoding in the UNLI scorer (forwarded to predict_unli.py).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.chunk_size <= 0:
        print("[chunked] --chunk-size must be > 0; delegating to non-chunked predict_unli.py")
        # Fall back: single subprocess over the entire input.
        argv = _build_passthrough_args(args, args.artifacts_jsonl, args.out)
        ret = subprocess.run(argv).returncode
        sys.exit(ret)

    # --- Group the input by video_id ---
    print(f"[chunked] Loading artifacts: {args.artifacts_jsonl}")
    groups = defaultdict(list)
    for rec in _iter_jsonl(args.artifacts_jsonl):
        vid = str(rec.get("video_id") or "").strip()
        if not vid:
            continue
        groups[vid].append(rec)

    video_ids = sorted(groups.keys())
    total_items = sum(len(groups[v]) for v in video_ids)
    if not video_ids:
        print("[chunked] No artifacts with video_id found; writing empty output.")
        with open(args.out, "w") as _:
            pass
        return

    print(f"[chunked] {total_items} items across {len(video_ids)} videos; "
          f"chunk_size={args.chunk_size} → {(len(video_ids) + args.chunk_size - 1) // args.chunk_size} chunks")

    # --- Prepare tmp dir ---
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    chunk_dir = args.chunk_tmp_dir or os.path.join(out_dir, "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # --- Run chunks ---
    total_scored = 0
    failed_chunks: List[int] = []
    chunk_outputs: List[str] = []

    # Open final output fresh so we don't append to stale predictions.
    final_out_fh = open(args.out, "w")
    try:
        for chunk_idx, start in enumerate(range(0, len(video_ids), args.chunk_size)):
            chunk_video_ids = video_ids[start:start + args.chunk_size]
            chunk_records = []
            for vid in chunk_video_ids:
                chunk_records.extend(groups[vid])

            chunk_in = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}_in.jsonl")
            chunk_out = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}_out.jsonl")
            _write_jsonl(chunk_in, chunk_records)
            chunk_outputs.append(chunk_out)

            print(
                f"\n[chunked] === Chunk {chunk_idx + 1}/"
                f"{(len(video_ids) + args.chunk_size - 1) // args.chunk_size}: "
                f"{len(chunk_video_ids)} videos, {len(chunk_records)} items ==="
            )

            argv = _build_passthrough_args(args, chunk_in, chunk_out)
            try:
                # Each subprocess inherits the parent env but gets a fresh
                # libav/torchvision state, so any leaked mmaps/fds are released
                # when this subprocess exits.
                ret = subprocess.run(argv, check=False).returncode
            except KeyboardInterrupt:
                print("\n[chunked] Interrupted by user. Partial output written so far.")
                raise
            except Exception as exc:
                print(f"[chunked] Chunk {chunk_idx} launch error: {exc}")
                failed_chunks.append(chunk_idx)
                if args.stop_on_chunk_failure:
                    raise
                continue

            if ret != 0:
                print(f"[chunked] Chunk {chunk_idx} FAILED with exit code {ret}")
                failed_chunks.append(chunk_idx)
                if args.stop_on_chunk_failure:
                    sys.exit(ret)
                continue

            # Merge chunk output into final output.
            n_scored = _append_file(chunk_out, final_out_fh)
            final_out_fh.flush()
            total_scored += n_scored
            print(f"[chunked] Chunk {chunk_idx}: scored {n_scored} items "
                  f"(cumulative {total_scored}/{total_items})")

            if not args.keep_chunk_outputs:
                for p in (chunk_in, chunk_out):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    finally:
        final_out_fh.close()

    # --- Cleanup ---
    if not args.keep_chunk_outputs:
        # Only remove the chunk dir if we created it and it's now empty.
        try:
            if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
                shutil.rmtree(chunk_dir, ignore_errors=True)
        except OSError:
            pass

    # --- Summary ---
    print("\n[chunked] Done.")
    print(f"[chunked] Scored: {total_scored} / {total_items} items")
    print(f"[chunked] Chunks run: {(len(video_ids) + args.chunk_size - 1) // args.chunk_size}")
    if failed_chunks:
        print(f"[chunked] Failed chunks: {failed_chunks}")
    print(f"[chunked] Output -> {args.out}")

    if failed_chunks and args.stop_on_chunk_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
