"""Small IO helpers for staged, memory-minimal experiments.

The hard rule (learned from an OOM): only ONE big model alive at a time. Stages
run as separate processes and pass results through small JSON files in a results
directory.
"""
import json
import os
from typing import Any

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kef_results")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(name: str, obj: Any) -> str:
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name: str) -> Any:
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def exists(name: str) -> bool:
    return os.path.exists(os.path.join(RESULTS_DIR, name))
