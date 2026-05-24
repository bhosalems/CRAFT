#!/usr/bin/env python3
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hydra_launch import dump_config, maybe_flag, maybe_kv, run_python_script


@hydra.main(version_base=None, config_path="conf", config_name="step1_general_notes")
def main(cfg: DictConfig) -> None:
    if cfg.stage != "step1" or cfg.task != "general-notes":
        raise ValueError("run_step1_general_notes.py requires the step1_general_notes config")
    if not cfg.output.out_dir:
        raise ValueError("output.out_dir must be set")

    dump_config(cfg.output.out_dir, cfg, filename="resolved_config.yaml")

    args: list[str] = []
    maybe_kv(args, "--mapping", cfg.data.mapping)
    maybe_kv(args, "--video-root", cfg.data.video_root)
    maybe_kv(args, "--model", cfg.model.model)
    maybe_kv(args, "--download-dir", cfg.model.download_dir)
    qtz = cfg.runtime.get("quantization") if hasattr(cfg, "runtime") else None
    if qtz:
        maybe_kv(args, "--quantization", qtz)
    maybe_kv(args, "--fps", cfg.runtime.fps)
    maybe_kv(args, "--max-frames", cfg.runtime.max_frames)
    maybe_flag(args, bool(cfg.runtime.enable_thinking), "--enable-thinking")
    maybe_kv(args, "--temperature", cfg.runtime.temperature)
    maybe_kv(args, "--top-p", cfg.runtime.top_p)
    maybe_kv(args, "--top-k", cfg.runtime.top_k)
    maybe_kv(args, "--max-tokens", cfg.runtime.max_tokens)
    maybe_kv(args, "--seed", cfg.runtime.seed)
    maybe_kv(args, "--repetition-penalty", cfg.runtime.repetition_penalty)
    maybe_kv(args, "--presence-penalty", cfg.runtime.presence_penalty)
    maybe_flag(args, not bool(cfg.get("include_topic_in_prompt", True)), "--exclude-topic-in-prompt")
    maybe_kv(args, "--out-dir", cfg.output.out_dir)
    maybe_flag(args, bool(cfg.verbose), "--verbose")
    run_python_script("extract_general_notes.py", args)


if __name__ == "__main__":
    main()
