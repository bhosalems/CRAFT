#!/usr/bin/env python3
"""Helpers for recording reproducible run metadata alongside generated outputs."""

from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import socket
import subprocess
import sys
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(script_name: str, created_at: Optional[str] = None) -> str:
    created_at = created_at or utc_now_iso()
    stamp = created_at.replace("+00:00", "Z").replace("-", "").replace(":", "")
    script_tag = os.path.splitext(os.path.basename(script_name))[0]
    return f"{stamp}-{script_tag}"


def _git_value(args: list[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value or None


def build_run_manifest(
    *,
    script_name: str,
    argv: list[str],
    args_dict: Dict[str, Any],
    run_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    created_at = utc_now_iso()
    return {
        "run_id": make_run_id(script_name, created_at),
        "script": script_name,
        "created_at": created_at,
        "cwd": os.getcwd(),
        "command": " ".join(shlex.quote(part) for part in argv),
        "args": args_dict,
        "run_config": run_config or {},
        "git": {
            "commit": _git_value(["git", "rev-parse", "HEAD"]),
            "branch": _git_value(["git", "branch", "--show-current"]),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        },
        "extra": extra or {},
    }


def write_run_manifest(
    out_dir: str,
    manifest: Dict[str, Any],
    filename: str = "run_manifest.json",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def write_resolved_config(
    out_dir: str,
    config_obj: Any,
    filename: str = "resolved_config.yaml",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(config_obj, DictConfig):
            rendered = OmegaConf.to_yaml(config_obj, resolve=True)
        else:
            rendered = OmegaConf.to_yaml(OmegaConf.create(config_obj), resolve=True)
        with open(path, "w") as f:
            f.write(rendered)
        return path
    except Exception:
        pass

    try:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(config_obj, f, sort_keys=False, allow_unicode=True)
        return path
    except Exception:
        with open(path, "w") as f:
            json.dump(config_obj, f, indent=2, ensure_ascii=False)
        return path
