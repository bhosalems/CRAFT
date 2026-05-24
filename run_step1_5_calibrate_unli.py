#!/usr/bin/env python3
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hydra_launch import dump_config, maybe_flag, maybe_kv, run_python_script


@hydra.main(version_base=None, config_path="conf", config_name="step1_5_unli_lora")
def main(cfg: DictConfig) -> None:
    if cfg.stage != "step1_5":
        raise ValueError("run_step1_5_calibrate_unli.py requires a Step 1.5 config preset")
    if not cfg.artifact_type:
        raise ValueError("artifact_type must be set")
    if not cfg.data.claims_jsonl:
        raise ValueError("data.claims_jsonl must be set")
    if not cfg.data.unli_jsonl:
        raise ValueError("data.unli_jsonl must be set")
    if not cfg.output.out_dir:
        raise ValueError("output.out_dir must be set")

    dump_config(cfg.output.out_dir, cfg, filename="calibrate_resolved_config.yaml")

    output_name = "general_notes_calibrated.jsonl" if cfg.artifact_type == "general-notes" else "query_conditioned_claims_calibrated.jsonl" if cfg.artifact_type == "query-claims" else "calibrated.jsonl"

    args: list[str] = []
    maybe_kv(args, "--artifact-type", cfg.artifact_type)
    maybe_kv(args, "--claims-jsonl", cfg.data.claims_jsonl)
    maybe_kv(args, "--unli-jsonl", cfg.data.unli_jsonl)
    maybe_kv(args, "--out", f"{cfg.output.out_dir}/{output_name}")
    run_python_script("calibrate_unli.py", args)


if __name__ == "__main__":
    main()
