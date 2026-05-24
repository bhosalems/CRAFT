#!/usr/bin/env python3
"""
Step 1.5: UNLI prediction runner for Step 1 artifacts.

Scores each general note or query-conditioned claim against its source video
using the UNLI model, producing a prediction JSONL consumed by calibrate_unli.py.

Usage:
    # General notes
    python note_taking/predict_unli.py \
        --artifact-type general-notes \
        --artifacts-jsonl note_taking/outputs/general_notes/general_notes.jsonl \
        --out /tmp/unli_preds_notes.jsonl

    # Query-conditioned claims
    python note_taking/predict_unli.py \
        --artifact-type query-claims \
        --artifacts-jsonl note_taking/outputs/query_claims_single/query_conditioned_claims.jsonl \
        --out /tmp/unli_preds_claims.jsonl
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from typing import Optional, Tuple
import torch
from tqdm import tqdm

from contracts import DEFAULT_UNLI_MODEL, DEFAULT_VIDEO_ROOT, DEFAULT_VLM_MODEL, resolve_video_path
from prompts import parse_qwen_score_answer, prompt_qwen_score, prompt_qwen_score_retry
from run_metadata import build_run_manifest, write_resolved_config, write_run_manifest

from models.vlm import Qwen3_5_VL, UNLI


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_unli(args):
    unli_kwargs = dict(
        model=args.model,
        download_dir=args.download_dir,
        fps=args.fps,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
        new_token_num=args.new_token_num,
        new_token_prefix=args.new_token_prefix,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        # --no-audio flips this off; long videos otherwise allocate a ~200 MB
        # float32 audio buffer that triggers libav/torchvision EAGAIN cascades.
        use_audio_in_video=not args.no_audio,
    )
    if args.base_model is not None:
        unli_kwargs["base_model"] = args.base_model
    if args.lora_path is not None:
        unli_kwargs["lora_path"] = args.lora_path
    return UNLI(**unli_kwargs)


def _build_qwen_scorer(args):
    return Qwen3_5_VL(
        model=args.model,
        download_dir=args.download_dir,
        fps=args.qwen_fps,
        max_frames=args.max_frames,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        allowed_local_media_path=args.video_root,
    )


def _score_with_qwen(model: Qwen3_5_VL, video_path: str, text: str) -> Tuple[Optional[float], Optional[str]]:
    raw_output = model.infer(video_path=video_path, query=prompt_qwen_score(text))
    prob = parse_qwen_score_answer(raw_output)
    if prob is not None:
        return prob, raw_output

    retry_output = model.infer(video_path=video_path, query=prompt_qwen_score_retry(text))
    prob = parse_qwen_score_answer(retry_output)
    return prob, retry_output


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1.5: UNLI prediction runner")
    ap.add_argument(
        "--artifact-type",
        choices=["general-notes", "query-claims"],
        required=True,
    )
    ap.add_argument("--scorer-backend", choices=["unli", "qwen_score"], default="unli")
    ap.add_argument("--artifacts-jsonl", required=True, help="Input artifacts JSONL path")
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--model", default=None, help="Merged checkpoint (default: auto-select by scorer-backend)")
    ap.add_argument("--base-model", default=None, help="Base model for LoRA mode")
    ap.add_argument("--lora-path", default=None, help="LoRA adapter path")
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
    ap.add_argument(
        "--gc-every",
        type=int,
        default=25,
        help=(
            "Run gc.collect() and (if CUDA is available) torch.cuda.empty_cache() every N items. "
            "Set to 0 to disable. This can help with video decode/scaler resource errors on busy nodes."
        ),
    )
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help=(
            "Disable audio decoding in the UNLI scorer. Recommended when any video "
            "exceeds a few minutes: the audio buffer is float32 stereo at the "
            "source sample rate (e.g. ~220 MB for a 9.5 min 48 kHz stereo clip), "
            "which can trigger libav/torchvision EAGAIN errors. Audio adds little "
            "signal for video-text NLI scoring."
        ),
    )
    ap.add_argument("--out", required=True, help="Output prediction JSONL path")
    ap.add_argument("--resolved-config-out", default="")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        args.model = DEFAULT_VLM_MODEL if args.scorer_backend == "qwen_score" else DEFAULT_UNLI_MODEL

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    manifest = build_run_manifest(
        script_name="predict_unli.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "artifact_type": args.artifact_type,
            "scorer_backend": args.scorer_backend,
            "model": args.model,
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "download_dir": args.download_dir,
            "fps": args.fps,
            "resized_height": args.resized_height,
            "resized_width": args.resized_width,
            "new_token_num": args.new_token_num,
            "new_token_prefix": args.new_token_prefix,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "use_audio_in_video": not args.no_audio,
            "qwen_fps": args.qwen_fps,
            "max_frames": args.max_frames,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "prompt_family": "qwen_step1_5_score_v1" if args.scorer_backend == "qwen_score" else "unli_conf_tokens_v1",
        },
    )
    manifest_path = write_run_manifest(out_dir, manifest, filename="predict_run_manifest.json")
    resolved_config_path = None
    if args.resolved_config_out:
        resolved_config_path = write_resolved_config(
            out_dir,
            vars(args),
            filename=args.resolved_config_out,
        )

    # Load and group artifacts by video_id
    print(f"Loading artifacts: {args.artifacts_jsonl}")
    groups = defaultdict(list)
    for rec in _iter_jsonl(args.artifacts_jsonl):
        vid = str(rec.get("video_id") or "").strip()
        if vid:
            groups[vid].append(rec)
    print(f"  {sum(len(v) for v in groups.values())} items across {len(groups)} videos")

    # Initialize scorer backend
    if args.scorer_backend == "unli" and args.base_model:
        print(f"\nInitializing UNLI model: {args.base_model} + LoRA {args.lora_path}")
    elif args.scorer_backend == "unli":
        print(f"\nInitializing UNLI model: {args.model}")
    else:
        print(f"\nInitializing Qwen scorer model: {args.model}")
    scorer = _build_unli(args) if args.scorer_backend == "unli" else _build_qwen_scorer(args)

    # Score each item
    print("\nScoring items...")
    scored = 0
    skipped_videos = 0
    skipped_items = 0
    processed_items = 0
    total_items = sum(len(v) for v in groups.values())
    pbar = tqdm(total=total_items, desc="Step 1.5 scoring", unit="item")

    with open(args.out, "w") as outf:
        for video_id in sorted(groups.keys()):
            video_path = resolve_video_path(args.video_root, video_id)
            if video_path is None:
                skipped_videos += 1
                skipped_items += len(groups[video_id])
                print(f"  WARN: video not found for {video_id}, skipping {len(groups[video_id])} items")
                pbar.update(len(groups[video_id]))
                continue

            for rec in groups[video_id]:
                pbar.set_postfix(video_id=video_id, scored=scored, skipped=skipped_items)
                # Extract text based on artifact type
                if args.artifact_type == "general-notes":
                    text = str(rec.get("text") or "").strip()
                    stable_id = rec.get("note_id")
                else:
                    text = str(rec.get("claim") or "").strip()
                    stable_id = rec.get("claim_id")

                if not text:
                    skipped_items += 1
                    processed_items += 1
                    pbar.update(1)
                    continue

                raw_output = None
                try:
                    if args.scorer_backend == "unli":
                        prob = scorer.score(video_path, text)
                    else:
                        prob, raw_output = _score_with_qwen(scorer, video_path, text)
                        if prob is None:
                            raise ValueError("Qwen scorer did not return a parseable <answer> score")
                except Exception as e:
                    skipped_items += 1
                    processed_items += 1
                    if args.verbose:
                        print(f"  WARN: scoring failed for {stable_id or video_id}: {e}")
                    pbar.update(1)
                    continue

                # Build prediction record
                pred = {"video_id": video_id, "prob": float(prob)}
                if args.artifact_type == "general-notes":
                    pred["note_id"] = stable_id
                    pred["text"] = text
                else:
                    pred["claim_id"] = stable_id
                    pred["claim"] = text
                if raw_output is not None:
                    pred["raw_output"] = raw_output

                outf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                outf.flush()
                scored += 1
                processed_items += 1
                pbar.update(1)

                if args.gc_every and processed_items % args.gc_every == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if args.verbose:
                print(f"  {video_id}: scored {len(groups[video_id])} items")

            # Free resources between videos to avoid swscaler
            # "Resource temporarily unavailable" errors from pyav.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    print(f"\n[ok] Scored: {scored}, skipped videos: {skipped_videos}, skipped items: {skipped_items}")
    print(f"[ok] -> {args.out}")
    print(f"[ok] -> {manifest_path}")
    if resolved_config_path:
        print(f"[ok] -> {resolved_config_path}")


if __name__ == "__main__":
    main()
