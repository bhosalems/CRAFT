#!/usr/bin/env python3
"""Helpers for config-driven note_taking entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable

from run_metadata import write_resolved_config


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def _stringify(value) -> str:
    if value is None:
        return ""
    return str(value)


def maybe_flag(args: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        args.append(flag)


def maybe_kv(args: list[str], flag: str, value) -> None:
    if value is None:
        return
    args.extend([flag, _stringify(value)])


def run_python_script(script_rel_path: str, args: Iterable[str]) -> None:
    cmd = [sys.executable, os.path.join(REPO_ROOT, script_rel_path), *list(args)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def dump_config(out_dir: str, cfg, filename: str) -> str:
    return write_resolved_config(out_dir, cfg, filename=filename)
