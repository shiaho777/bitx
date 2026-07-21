import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import List, Optional


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return r / (1024 * 1024)
    return r / 1024


def git_commit(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


@dataclass
class BenchRecord:
    run_id: str
    created_at: str
    git_commit: str
    model_id: str
    backend: str
    quantization_recipe: str
    adapter_id: str
    task_name: str
    prompt_count: int
    raw_predictions_path: str
    score: float
    tokens_per_second: float
    first_token_latency_s: float
    peak_rss_mb: float
    wall_time_s: float
    notes: str
    metrics: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_index_probe(store_size: int) -> int:
    if store_size <= 0:
        return 1
    return max(4, min(32, round(store_size / 512)))


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_row(f, row: dict) -> int:
    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 1


def append_record(path: str, record: BenchRecord) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def append_records(path: str, records: list) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")


def make_record(
    run_id: str,
    model_id: str,
    backend: str,
    task_name: str,
    raw_path: str,
    score: float,
    prompt_count: int,
    tokens_per_second: float,
    first_token_latency_s: float,
    wall_time_s: float,
    notes: str,
    metrics: dict,
    cwd: Optional[str] = None,
    quantization_recipe: str = "none",
    adapter_id: str = "none",
) -> BenchRecord:
    return BenchRecord(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=git_commit(cwd),
        model_id=model_id,
        backend=backend,
        quantization_recipe=quantization_recipe,
        adapter_id=adapter_id,
        task_name=task_name,
        prompt_count=prompt_count,
        raw_predictions_path=raw_path,
        score=score,
        tokens_per_second=tokens_per_second,
        first_token_latency_s=first_token_latency_s,
        peak_rss_mb=peak_rss_mb(),
        wall_time_s=wall_time_s,
        notes=notes,
        metrics=metrics,
    )
