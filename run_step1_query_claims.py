#!/usr/bin/env python3
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hydra_launch import dump_config, maybe_flag, maybe_kv, run_python_script


@hydra.main(version_base=None, config_path="conf", config_name="step1_query_claims")
def main(cfg: DictConfig) -> None:
    if cfg.stage != "step1" or cfg.task != "query-claims":
        raise ValueError("run_step1_query_claims.py requires the step1_query_claims config")
    if not cfg.query_mode:
        raise ValueError("query_mode must be set")
    if not cfg.output.out_dir:
        raise ValueError("output.out_dir must be set")

    dump_config(cfg.output.out_dir, cfg, filename="resolved_config.yaml")

    args: list[str] = []
    maybe_kv(args, "--query-mode", cfg.query_mode)
    maybe_kv(args, "--queries-jsonl", cfg.data.queries_jsonl)
    maybe_kv(args, "--mapping", cfg.data.mapping)
    maybe_kv(args, "--expanded-queries", cfg.data.expanded_queries)
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
    maybe_kv(args, "--max-critic-rounds", cfg.get("max_critic_rounds", 0))
    critic_cfg = cfg.get("critic", {})
    if critic_cfg:
        maybe_kv(args, "--critic-unli-model", critic_cfg.get("unli_model"))
        maybe_kv(args, "--critic-unli-base-model", critic_cfg.get("unli_base_model"))
        maybe_kv(args, "--critic-unli-lora-path", critic_cfg.get("unli_lora_path"))
        critic_gpu = critic_cfg.get("gpu")
        if critic_gpu is not None:
            maybe_kv(args, "--critic-gpu", critic_gpu)
        # Text critic sub-checks: MNLI-screened + LLM-adjudicated contradictions,
        # and per-query LLM coverage audit with bounded targeted follow-up.
        maybe_kv(args, "--critic-nli-model", critic_cfg.get("nli_model"))
        maybe_kv(args, "--critic-nli-device", critic_cfg.get("nli_device"))
        nli_screen = critic_cfg.get("nli_screen_threshold")
        if nli_screen is not None:
            maybe_kv(args, "--critic-nli-screen-threshold", nli_screen)
        nli_max_cand = critic_cfg.get("nli_max_candidates")
        if nli_max_cand is not None:
            maybe_kv(args, "--critic-nli-max-candidates", nli_max_cand)
        nli_enabled = critic_cfg.get("nli_enabled")
        if nli_enabled is not None:
            args.append("--critic-nli-enabled" if bool(nli_enabled) else "--no-critic-nli-enabled")
        maybe_kv(args, "--critic-coverage-model", critic_cfg.get("coverage_model"))
        coverage_gpu = critic_cfg.get("coverage_gpu")
        if coverage_gpu is not None:
            maybe_kv(args, "--critic-coverage-gpu", coverage_gpu)
        coverage_enabled = critic_cfg.get("coverage_enabled")
        if coverage_enabled is not None:
            args.append("--critic-coverage-enabled" if bool(coverage_enabled) else "--no-critic-coverage-enabled")
        followup = critic_cfg.get("coverage_followup_rounds")
        if followup is not None:
            maybe_kv(args, "--coverage-followup-rounds", followup)
    # Optional vLLM GPU memory utilization override (read from runtime if present).
    gmu = cfg.runtime.get("gpu_memory_utilization") if hasattr(cfg, "runtime") else None
    if gmu is not None:
        maybe_kv(args, "--gpu-memory-utilization", gmu)
    # Optional vLLM tensor_parallel_size for sharding large models across GPUs.
    tps = cfg.runtime.get("tensor_parallel_size") if hasattr(cfg, "runtime") else None
    if tps is not None:
        maybe_kv(args, "--tensor-parallel-size", tps)
    # Optional vLLM max_model_len cap. Avoids vLLM refusing to start when the
    # model's advertised context (e.g. 262K for Qwen3-VL-30B) needs more KV
    # cache than fits after weights load.
    mml = cfg.runtime.get("max_model_len") if hasattr(cfg, "runtime") else None
    if mml is not None:
        maybe_kv(args, "--max-model-len", mml)
    maybe_kv(args, "--out-dir", cfg.output.out_dir)
    asr_dir = cfg.get("asr_dir", "") if hasattr(cfg, "get") else ""
    if asr_dir:
        maybe_kv(args, "--asr-dir", asr_dir)
    only_qids = cfg.get("only_query_ids", "") if hasattr(cfg, "get") else ""
    if only_qids:
        maybe_kv(args, "--only-query-ids", only_qids)
    maybe_flag(args, bool(cfg.verbose), "--verbose")
    run_python_script("extract_query_claims.py", args)


if __name__ == "__main__":
    main()
