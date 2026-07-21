"""Portable repo paths and default model location."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def results_dir() -> Path:
    env = os.environ.get("BITX_RESULTS", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "kef_results").resolve()


def default_model() -> str:
    env = os.environ.get("BITX_MODEL", "").strip()
    if env:
        return str(Path(env).expanduser())
    return ""


def result_path(*parts: str) -> str:
    return str(results_dir().joinpath(*parts))


def ensure_results_dir() -> Path:
    d = results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
