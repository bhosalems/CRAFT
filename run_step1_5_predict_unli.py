#!/usr/bin/env python3
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hydra_launch import dump_config, maybe_flag, maybe_kv, run_python_script


@hydra.main(version_base=None, config_path="conf", config_name="step1_5_unli_lora")
def main(cfg: DictConfig) -> None:
    if cfg.stage != "step1_5":
        raise ValueError("run_step1_5_predict_unli.py requires a Step 1.5 config preset")
    if not cfg.artifact_type:
        raise ValueError("artifact_type must be set")
    if not cfg.data.artifacts_jsonl:
        raise ValueError("data.artifacts_jsonl must be set")
    if not cfg.output.out_dir:
        raise ValueError("output.out_dir must be set")

    dump_config(cfg.output.out_dir, cfg, filename="predict_resolved_config.yaml")

    args: list[str] = []
    maybe_kv(args, "--artifact-type", cfg.artifact_type)
    maybe_kv(args, "--scorer-backend", cfg.get("scorer_backend", "unli"))
    maybe_kv(args, "--artifacts-jsonl", cfg.data.artifacts_jsonl)
    maybe_kv(args, "--video-root", cfg.data.video_root)
    maybe_kv(args, "--model", cfg.model.model)
    maybe_kv(args, "--base-model", cfg.model.base_model)
    maybe_kv(args, "--lora-path", cfg.model.lora_path)
    maybe_kv(args, "--download-dir", cfg.model.download_dir)
    maybe_kv(args, "--fps", cfg.runtime.get("fps"))
    maybe_kv(args, "--resized-height", cfg.runtime.get("resized_height"))
    maybe_kv(args, "--resized-width", cfg.runtime.get("resized_width"))
    maybe_kv(args, "--new-token-num", cfg.runtime.get("new_token_num"))
    maybe_kv(args, "--new-token-prefix", cfg.runtime.get("new_token_prefix"))
    maybe_kv(args, "--device-map", cfg.runtime.get("device_map"))
    maybe_kv(args, "--torch-dtype", cfg.runtime.get("torch_dtype"))
    maybe_kv(args, "--attn-implementation", cfg.runtime.get("attn_implementation"))
    maybe_kv(args, "--qwen-fps", cfg.runtime.get("fps"))
    maybe_kv(args, "--max-frames", cfg.runtime.get("max_frames"))
    maybe_flag(args, bool(cfg.runtime.get("enable_thinking", False)), "--enable-thinking")
    maybe_kv(args, "--temperature", cfg.runtime.get("temperature"))
    maybe_kv(args, "--top-p", cfg.runtime.get("top_p"))
    maybe_kv(args, "--top-k", cfg.runtime.get("top_k"))
    maybe_kv(args, "--max-tokens", cfg.runtime.get("max_tokens"))
    maybe_kv(args, "--seed", cfg.runtime.get("seed"))
    maybe_kv(args, "--repetition-penalty", cfg.runtime.get("repetition_penalty"))
    maybe_kv(args, "--presence-penalty", cfg.runtime.get("presence_penalty"))
    # When True, skip audio decoding (avoids ~200 MB float32 audio buffer for
    # long videos that triggers libav/torchvision EAGAIN cascades).
    maybe_flag(args, bool(cfg.get("no_audio", False)), "--no-audio")
    maybe_kv(args, "--out", f"{cfg.output.out_dir}/unli_predictions.jsonl")
    maybe_flag(args, bool(cfg.verbose), "--verbose")

    # Optional chunked mode: route to predict_unli_chunked.py when chunk_size > 0.
    # This runs predict_unli.py on chunks of videos in fresh subprocesses to
    # release pyav/libav mmap leaks that otherwise exhaust vm.max_map_count.
    chunk_size = int(cfg.get("chunk_size", 0) or 0)
    if chunk_size > 0:
        args.append("--chunk-size")
        args.append(str(chunk_size))
        run_python_script("predict_unli_chunked.py", args)
    else:
        run_python_script("predict_unli.py", args)


if __name__ == "__main__":
    main()
