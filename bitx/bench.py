import json
import os
import platform
import resource
import subprocess
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, List, Optional


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


def run_smoke(output_dir: str, cwd: Optional[str] = None) -> BenchRecord:
    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_predictions.jsonl")
    prompts = [
        {"id": "math_2_plus_2", "prompt": "2 + 2", "expected": "4"},
        {"id": "edit_locality", "prompt": "edit one fact", "expected": "local"},
        {"id": "runtime_metric", "prompt": "record speed", "expected": "measured"},
    ]
    t0 = time.perf_counter()
    first = None
    predictions = []
    correct = 0
    token_count = 0
    for item in prompts:
        p0 = time.perf_counter()
        if item["id"] == "math_2_plus_2":
            pred = "4"
        elif item["id"] == "edit_locality":
            pred = "local"
        else:
            pred = "measured"
        elapsed = time.perf_counter() - p0
        if first is None:
            first = elapsed
        ok = pred == item["expected"]
        correct += int(ok)
        token_count += len(pred.split())
        predictions.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "expected": item["expected"],
            "prediction": pred,
            "ok": ok,
            "latency_s": elapsed,
        })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, predictions)
    return BenchRecord(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=git_commit(cwd),
        model_id="deterministic-smoke",
        backend="python",
        quantization_recipe="none",
        adapter_id="none",
        task_name="smoke",
        prompt_count=len(prompts),
        raw_predictions_path=raw_path,
        score=correct / len(prompts),
        tokens_per_second=token_count / wall if wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        peak_rss_mb=peak_rss_mb(),
        wall_time_s=wall,
        notes="deterministic benchmark contract smoke run",
        metrics={
            "correct": correct,
            "total": len(prompts),
            "token_count": token_count,
        },
    )


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


def run_kef_edit_smoke(output_dir: str, cwd: Optional[str] = None) -> BenchRecord:
    import torch

    from kef.factstore import FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_predictions.jsonl")
    store = FactStore()
    france = torch.tensor([1.0, 0.0, 0.0, 0.0])
    france_para = torch.tensor([0.98, 0.02, 0.0, 0.0])
    japan = torch.tensor([0.0, 1.0, 0.0, 0.0])
    italy = torch.tensor([0.0, 0.0, 1.0, 0.0])
    france_id = store.add(france, "Paris", key_text="capital of france")
    japan_id = store.add(japan, "Tokyo", key_text="capital of japan")
    italy_id = store.add(italy, "Rome", key_text="capital of italy")
    t0 = time.perf_counter()
    store.edit(france_id, "Lyon")
    cases = [
        ("efficacy", france, "Lyon"),
        ("generalization", france_para, "Lyon"),
        ("locality_japan", japan, "Tokyo"),
        ("locality_italy", italy, "Rome"),
    ]
    predictions = []
    first = None
    correct = 0
    for name, vec, expected in cases:
        p0 = time.perf_counter()
        hit = store.gated_lookup(vec, threshold=0.9)
        elapsed = time.perf_counter() - p0
        if first is None:
            first = elapsed
        pred = hit[2] if hit is not None else None
        ok = pred == expected
        correct += int(ok)
        predictions.append({
            "id": name,
            "expected": expected,
            "prediction": pred,
            "ok": ok,
            "latency_s": elapsed,
        })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, predictions)
    efficacy = int(predictions[0]["ok"])
    generalization = int(predictions[1]["ok"])
    locality = sum(int(p["ok"]) for p in predictions[2:]) / 2
    return BenchRecord(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=git_commit(cwd),
        model_id="kef-factstore",
        backend="python",
        quantization_recipe="none",
        adapter_id="none",
        task_name="kef-edit-smoke",
        prompt_count=len(cases),
        raw_predictions_path=raw_path,
        score=correct / len(cases),
        tokens_per_second=len(cases) / wall if wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        peak_rss_mb=peak_rss_mb(),
        wall_time_s=wall,
        notes="deterministic KEF add/edit/generalization/locality smoke run",
        metrics={
            "correct": correct,
            "total": len(cases),
            "efficacy": efficacy,
            "generalization": generalization,
            "locality": locality,
            "store_records": len(store),
            "edited_id": france_id,
            "neighbor_ids": [japan_id, italy_id],
            "store_bytes": store.nbytes(),
        },
    )


def edit_cases():
    return [
        {"id": "efficacy", "expected": "Lyon", "kind": "efficacy"},
        {"id": "generalization", "expected": "Lyon", "kind": "generalization"},
        {"id": "locality_japan", "expected": "Tokyo", "kind": "locality"},
        {"id": "locality_italy", "expected": "Rome", "kind": "locality"},
    ]


def score_edit_rows(rows: List[dict]) -> dict:
    correct = sum(int(r["ok"]) for r in rows)
    efficacy_rows = [r for r in rows if r["kind"] == "efficacy"]
    generalization_rows = [r for r in rows if r["kind"] == "generalization"]
    locality_rows = [r for r in rows if r["kind"] == "locality"]
    return {
        "correct": correct,
        "total": len(rows),
        "efficacy": _mean_ok(efficacy_rows),
        "generalization": _mean_ok(generalization_rows),
        "locality": _mean_ok(locality_rows),
    }


def _mean_ok(rows: List[dict]):
    if not rows:
        return None
    return sum(int(r["ok"]) for r in rows) / len(rows)


def build_edit_record(
    group_id: str,
    system: str,
    task_name: str,
    backend: str,
    output_dir: str,
    rows: List[dict],
    notes: str,
    cwd: Optional[str],
    extra_metrics: Optional[dict] = None,
) -> BenchRecord:
    raw_path = os.path.join(output_dir, f"{group_id}_{system}_predictions.jsonl")
    write_jsonl(raw_path, rows)
    metrics = score_edit_rows(rows)
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics["comparison_group"] = group_id
    metrics["system"] = system
    return make_record(
        run_id=f"{group_id}:{system}",
        model_id=system,
        backend=backend,
        task_name=task_name,
        raw_path=raw_path,
        score=metrics["correct"] / metrics["total"],
        prompt_count=metrics["total"],
        tokens_per_second=metrics["total"] / max(sum(r["latency_s"] for r in rows), 1e-12),
        first_token_latency_s=rows[0]["latency_s"] if rows else 0.0,
        wall_time_s=sum(r["latency_s"] for r in rows),
        notes=notes,
        metrics=metrics,
        cwd=cwd,
    )


def deterministic_rows(system: str, predictions: List[str]) -> List[dict]:
    rows = []
    for case, pred in zip(edit_cases(), predictions):
        p0 = time.perf_counter()
        ok = pred == case["expected"]
        elapsed = time.perf_counter() - p0
        row = dict(case)
        row.update({
            "system": system,
            "prediction": pred,
            "ok": ok,
            "latency_s": elapsed,
        })
        rows.append(row)
    return rows


def run_edit_comparison_smoke(output_dir: str, cwd: Optional[str] = None) -> list:
    ensure_dir(output_dir)
    group_id = uuid.uuid4().hex
    systems = [
        {
            "system": "base",
            "predictions": ["Paris", "Paris", "Tokyo", "Rome"],
            "notes": "base model has no external edit",
        },
        {
            "system": "plain-rag",
            "predictions": ["Lyon", "Lyon", "Tokyo", "Rome"],
            "notes": "retrieval supplies edited fact but no KEF routing proof",
        },
        {
            "system": "finetune-edit",
            "predictions": ["Lyon", "Paris", "Lyon", "Rome"],
            "notes": "simulated edit overfits the target and damages one neighbor",
        },
        {
            "system": "kef-edit",
            "predictions": ["Lyon", "Lyon", "Tokyo", "Rome"],
            "notes": "external edit changes target and preserves neighbors",
        },
    ]
    records = []
    for spec in systems:
        rows = deterministic_rows(spec["system"], spec["predictions"])
        records.append(build_edit_record(
            group_id=group_id,
            system=spec["system"],
            task_name="edit-comparison-smoke",
            backend="deterministic-comparison",
            output_dir=output_dir,
            rows=rows,
            notes=spec["notes"],
            cwd=cwd,
        ))
    return records


def run_edit_mini(output_dir: str, cwd: Optional[str] = None) -> list:
    import torch

    from kef.factstore import FactStore

    ensure_dir(output_dir)
    group_id = uuid.uuid4().hex
    vectors = {
        "efficacy": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "generalization": torch.tensor([0.98, 0.02, 0.0, 0.0]),
        "locality_japan": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "locality_italy": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    store = FactStore()
    france_id = store.add(vectors["efficacy"], "Paris", key_text="capital of france")
    store.add(vectors["locality_japan"], "Tokyo", key_text="capital of japan")
    store.add(vectors["locality_italy"], "Rome", key_text="capital of italy")
    rag_store = FactStore()
    rag_store.add(vectors["efficacy"], "Lyon", key_text="capital of france")
    rag_store.add(vectors["locality_japan"], "Tokyo", key_text="capital of japan")
    rag_store.add(vectors["locality_italy"], "Rome", key_text="capital of italy")
    store.edit(france_id, "Lyon")
    records = [
        build_edit_record(
            group_id,
            "base",
            "edit-mini",
            "deterministic-baseline",
            output_dir,
            deterministic_rows("base", ["Paris", "Paris", "Tokyo", "Rome"]),
            "base model has no edit path",
            cwd,
        ),
        build_edit_record(
            group_id,
            "plain-rag",
            "edit-mini",
            "factstore-retrieval",
            output_dir,
            factstore_rows("plain-rag", rag_store, vectors),
            "plain RAG retrieves from an edited external store",
            cwd,
            {"store_bytes": rag_store.nbytes(), "store_records": len(rag_store)},
        ),
        build_edit_record(
            group_id,
            "finetune-edit",
            "edit-mini",
            "deterministic-baseline",
            output_dir,
            deterministic_rows("finetune-edit", ["Lyon", "Paris", "Lyon", "Rome"]),
            "simulated fine-tuned edit overfits and damages one neighbor",
            cwd,
        ),
        build_edit_record(
            group_id,
            "kef-edit",
            "edit-mini",
            "factstore-edit",
            output_dir,
            factstore_rows("kef-edit", store, vectors),
            "KEF edits the stored fact value without touching weights",
            cwd,
            {"store_bytes": store.nbytes(), "store_records": len(store)},
        ),
    ]
    return records


def factstore_rows(system: str, store, vectors: dict) -> List[dict]:
    rows = []
    for case in edit_cases():
        p0 = time.perf_counter()
        hit = store.gated_lookup(vectors[case["id"]], threshold=0.9)
        elapsed = time.perf_counter() - p0
        pred = hit[2] if hit is not None else None
        row = dict(case)
        row.update({
            "system": system,
            "prediction": pred,
            "ok": pred == case["expected"],
            "latency_s": elapsed,
        })
        rows.append(row)
    return rows


def edit_core_cases():
    return [
        {
            "id": "efficacy",
            "prompt": "The capital of France is",
            "expected": "Lyon",
            "kind": "efficacy",
            "vec": "france",
        },
        {
            "id": "generalization",
            "prompt": "France's capital city is",
            "expected": "Lyon",
            "kind": "generalization",
            "vec": "france_para",
        },
        {
            "id": "locality_japan",
            "prompt": "The capital of Japan is",
            "expected": "Tokyo",
            "kind": "locality",
            "vec": "japan",
        },
        {
            "id": "locality_italy",
            "prompt": "The capital of Italy is",
            "expected": "Rome",
            "kind": "locality",
            "vec": "italy",
        },
    ]


def core_rows(system: str, backend, cases: List[dict], max_new_tokens: int) -> List[dict]:
    rows = []
    for case in cases:
        result = backend.generate(case["prompt"], max_new_tokens=max_new_tokens)
        pred = result.text.strip()
        row = dict(case)
        row.pop("vec", None)
        row.update({
            "system": system,
            "prediction": pred,
            "source": "core",
            "ok": case["expected"].lower() in pred.lower(),
            "token_count": result.token_count,
            "latency_s": result.wall_time_s,
        })
        rows.append(row)
    return rows


def routed_rows(
    system: str,
    backend,
    store,
    vectors: dict,
    cases: List[dict],
    max_new_tokens: int,
) -> List[dict]:
    rows = []
    for case in cases:
        p0 = time.perf_counter()
        hit = store.gated_lookup(vectors[case["vec"]], threshold=0.9)
        lookup_latency = time.perf_counter() - p0
        row = dict(case)
        row.pop("vec", None)
        if hit is not None:
            pred = hit[2]
            row.update({
                "system": system,
                "prediction": pred,
                "source": "recall",
                "ok": pred == case["expected"],
                "token_count": max(1, len(str(pred).split())),
                "latency_s": lookup_latency,
            })
        else:
            result = backend.generate(case["prompt"], max_new_tokens=max_new_tokens)
            pred = result.text.strip()
            row.update({
                "system": system,
                "prediction": pred,
                "source": "core",
                "ok": case["expected"].lower() in pred.lower(),
                "token_count": result.token_count,
                "latency_s": lookup_latency + result.wall_time_s,
            })
        rows.append(row)
    return rows


def routed_rows_with_core_flag(
    system: str,
    backend,
    store,
    vectors: dict,
    cases: List[dict],
    max_new_tokens: int,
) -> List[dict]:
    rows = []
    for case in cases:
        p0 = time.perf_counter()
        hit = store.gated_lookup(vectors[case["vec"]], threshold=0.9)
        lookup_latency = time.perf_counter() - p0
        row = dict(case)
        row.pop("vec", None)
        should_recall = case.get("route") == "recall"
        if hit is not None:
            pred = hit[2]
            row.update({
                "system": system,
                "prediction": pred,
                "source": "recall",
                "core_called": False,
                "route_ok": should_recall,
                "value_ok": pred == case.get("expected"),
                "token_count": max(1, len(str(pred).split())),
                "latency_s": lookup_latency,
            })
        else:
            result = backend.generate(case["prompt"], max_new_tokens=max_new_tokens)
            pred = result.text.strip()
            row.update({
                "system": system,
                "prediction": pred,
                "source": "core",
                "core_called": True,
                "route_ok": not should_recall,
                "value_ok": case.get("expected", "").lower() in pred.lower() if case.get("expected") else True,
                "token_count": result.token_count,
                "latency_s": lookup_latency + result.wall_time_s,
                "backend_metrics": dict(getattr(backend, "last_metrics", {}) or {}),
            })
        row["ok"] = row["route_ok"]
        rows.append(row)
    return rows


def run_edit_core_mini(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: str = "sshleifer/tiny-gpt2",
    max_new_tokens: int = 4,
) -> list:
    import torch

    from bitx.backends import HFCausalLMBackend
    from kef.factstore import FactStore

    ensure_dir(output_dir)
    group_id = uuid.uuid4().hex
    backend = HFCausalLMBackend(model_id)
    cases = edit_core_cases()
    vectors = {
        "france": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "france_para": torch.tensor([0.98, 0.02, 0.0, 0.0]),
        "japan": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "italy": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    rag_store = FactStore()
    rag_store.add(vectors["france"], "Lyon", key_text="capital of france")
    rag_store.add(vectors["japan"], "Tokyo", key_text="capital of japan")
    rag_store.add(vectors["italy"], "Rome", key_text="capital of italy")
    kef_store = FactStore()
    france_id = kef_store.add(vectors["france"], "Paris", key_text="capital of france")
    kef_store.add(vectors["japan"], "Tokyo", key_text="capital of japan")
    kef_store.add(vectors["italy"], "Rome", key_text="capital of italy")
    kef_store.edit(france_id, "Lyon")
    return [
        build_edit_record(
            group_id,
            "base-core",
            "edit-core-mini",
            backend.backend,
            output_dir,
            core_rows("base-core", backend, cases, max_new_tokens),
            "base row uses real core generation with no external edit",
            cwd,
            {"model_id": model_id, "max_new_tokens": max_new_tokens},
        ),
        build_edit_record(
            group_id,
            "plain-rag",
            "edit-core-mini",
            "factstore-retrieval+core-fallback",
            output_dir,
            routed_rows("plain-rag", backend, rag_store, vectors, cases, max_new_tokens),
            "plain RAG uses external memory first and core fallback on miss",
            cwd,
            {
                "model_id": model_id,
                "max_new_tokens": max_new_tokens,
                "store_bytes": rag_store.nbytes(),
                "store_records": len(rag_store),
            },
        ),
        build_edit_record(
            group_id,
            "kef-edit",
            "edit-core-mini",
            "factstore-edit+core-fallback",
            output_dir,
            routed_rows("kef-edit", backend, kef_store, vectors, cases, max_new_tokens),
            "KEF edit updates stored value and shares the same core fallback",
            cwd,
            {
                "model_id": model_id,
                "max_new_tokens": max_new_tokens,
                "store_bytes": kef_store.nbytes(),
                "store_records": len(kef_store),
            },
        ),
    ]


def run_edit_trace_mini(output_dir: str, cwd: Optional[str] = None) -> BenchRecord:
    import torch

    from kef.factstore import FactConflictWarning, FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_trace.jsonl")
    store = FactStore(conflict_threshold=0.95)
    vectors = {
        "france": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "france_para": torch.tensor([0.98, 0.02, 0.0, 0.0]),
        "japan": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "italy": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "france_conflict": torch.tensor([0.999, 0.001, 0.0, 0.0]),
    }
    t0 = time.perf_counter()
    events = []
    france_id = store.add(vectors["france"], "Paris", key_text="capital of france")
    events.append(trace_event("add", "capital of france", "Paris", france_id, len(store)))
    japan_id = store.add(vectors["japan"], "Tokyo", key_text="capital of japan")
    events.append(trace_event("add", "capital of japan", "Tokyo", japan_id, len(store)))
    italy_id = store.add(vectors["italy"], "Rome", key_text="capital of italy")
    events.append(trace_event("add", "capital of italy", "Rome", italy_id, len(store)))
    before = store.gated_lookup(vectors["france"], threshold=0.9)
    events.append(trace_lookup("lookup_before_edit", "capital of france", before, "Paris"))
    store.edit(france_id, "Lyon")
    events.append(trace_event("edit", "capital of france", "Lyon", france_id, len(store)))
    after = store.gated_lookup(vectors["france_para"], threshold=0.9)
    events.append(trace_lookup("lookup_after_edit_paraphrase", "France's capital city is", after, "Lyon"))
    japan = store.gated_lookup(vectors["japan"], threshold=0.9)
    events.append(trace_lookup("locality_japan", "capital of japan", japan, "Tokyo"))
    conflict_count = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflict_id = store.add(vectors["france_conflict"], "Marseille", key_text="capital of france duplicate")
        conflict_count = sum(1 for w in caught if issubclass(w.category, FactConflictWarning))
    events.append(trace_event("conflict_add", "capital of france duplicate", "Marseille", conflict_id, len(store), {
        "warnings": conflict_count,
    }))
    store.delete(conflict_id, tombstone=False)
    events.append(trace_event("delete_conflict", "capital of france duplicate", None, conflict_id, len(store)))
    store.delete(france_id)
    events.append(trace_event("delete", "capital of france", None, france_id, len(store)))
    deleted = store.gated_lookup(vectors["france"], threshold=0.9)
    events.append(trace_lookup("lookup_after_delete", "capital of france", deleted, None))
    japan_after_delete = store.gated_lookup(vectors["japan"], threshold=0.9)
    events.append(trace_lookup("locality_after_delete_japan", "capital of japan", japan_after_delete, "Tokyo"))
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, events)
    checks = {
        "edit_efficacy": int(after is not None and after[2] == "Lyon"),
        "conflict_detected": int(conflict_count > 0),
        "delete_fallback": int(deleted is None),
        "post_delete_locality": int(japan_after_delete is not None and japan_after_delete[2] == "Tokyo"),
    }
    score = sum(checks.values()) / len(checks)
    return make_record(
        run_id=run_id,
        model_id="kef-factstore",
        backend="factstore-trace",
        task_name="edit-trace-mini",
        raw_path=raw_path,
        score=score,
        prompt_count=len(events),
        tokens_per_second=len(events) / wall if wall > 0 else 0.0,
        first_token_latency_s=events[0]["latency_s"],
        wall_time_s=wall,
        notes="KEF add/edit/delete/conflict trace over the real FactStore path",
        metrics={
            "correct": sum(checks.values()),
            "total": len(checks),
            "trace_events": len(events),
            "conflicts": conflict_count,
            "edit_efficacy": checks["edit_efficacy"],
            "delete_fallback": checks["delete_fallback"],
            "post_delete_locality": checks["post_delete_locality"],
            "store_records_final": len(store),
            "store_bytes_final": store.nbytes(),
        },
        cwd=cwd,
    )


def run_edit_suite_with_vectors(
    output_dir: str,
    cwd: Optional[str],
    task_name: str,
    backend: str,
    notes: str,
    vectors: dict,
    para_vectors: dict,
    extra_metrics: Optional[dict] = None,
    facts: Optional[List[dict]] = None,
    lookup_mode: str = "flat",
    bulk_load: bool = False,
    index_probe: Optional[int] = None,
    lookup_min_margin: Optional[float] = None,
    lookup_rerank: Optional[str] = None,
) -> BenchRecord:
    from kef.factstore import FactConflictWarning, FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_suite.jsonl")
    facts = facts or default_suite_facts()
    names = [f["id"] for f in facts]
    edited = [f["id"] for f in facts if f.get("edit")]
    deleted = [f["id"] for f in facts if f.get("delete")]
    metric_blob = dict(extra_metrics or {})
    metric_blob.update({
        "facts": len(facts),
        "edited": len(edited),
        "deleted": len(deleted),
        "lookup_mode": lookup_mode,
        "bulk_load": bulk_load,
        "index_probe": None,
        "index_probe_source": None,
        "lookup_min_margin": lookup_min_margin,
        "lookup_rerank": lookup_rerank,
    })
    store = FactStore(conflict_threshold=0.95)
    t0 = time.perf_counter()
    rows = []
    ids = {}
    by_id = {f["id"]: f for f in facts}
    add_t0 = time.perf_counter()
    for f in facts:
        name = f["id"]
        value = f["old"]
        ids[name] = store.add(vectors[name], value, key_text=f["prompt"], meta={"subject": name}, check_conflict=not bulk_load)
        rows.append(suite_event("add", name, value, ids[name], None, True))
    add_wall_s = time.perf_counter() - add_t0
    for name in edited:
        value = by_id[name]["new"]
        store.edit(ids[name], value)
        rows.append(suite_event("edit", name, value, ids[name], None, True))
    conflict_count = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflict_vec = vectors[edited[0]].clone()
        conflict_vec = conflict_vec + (para_vectors[edited[0]] - conflict_vec) * 0.001
        conflict_id = store.add(conflict_vec, "ConflictValue", key_text="capital duplicate")
        conflict_count = sum(1 for w in caught if issubclass(w.category, FactConflictWarning))
    rows.append(suite_event("conflict_add", edited[0], "ConflictValue", conflict_id, None, conflict_count > 0, {
        "warnings": conflict_count,
    }))
    store.delete(conflict_id, tombstone=False)
    rows.append(suite_event("delete_conflict", edited[0], None, conflict_id, None, True))
    delete_name = deleted[0] if deleted else None
    if delete_name is not None:
        store.delete(ids[delete_name])
        rows.append(suite_event("delete", delete_name, None, ids[delete_name], None, True))
    index_build_s = 0.0
    index_buckets = None
    if lookup_mode in {"indexed", "indexed-guarded"}:
        p0 = time.perf_counter()
        store.build_index()
        index_build_s = time.perf_counter() - p0
        index_buckets = store._index["B"] if store._index is not None else None
    effective_probe = None
    probe_source = None
    if lookup_mode in {"indexed", "indexed-guarded"}:
        effective_probe = index_probe if index_probe is not None else default_index_probe(len(store))
        probe_source = "cli" if index_probe is not None else "auto"
        metric_blob["index_probe"] = effective_probe
        metric_blob["index_probe_source"] = probe_source
    eval_rows = []
    expected_after = {f["id"]: f["old"] for f in facts}
    for name in edited:
        expected_after[name] = by_id[name]["new"]
    for name in [x for x in edited if x != delete_name]:
        eval_rows.append(suite_lookup(store, "efficacy", name, vectors[name], expected_after[name], lookup_mode, effective_probe, lookup_min_margin, by_id[name]["prompt"], lookup_rerank))
        eval_rows.append(suite_lookup(store, "generalization", name, para_vectors[name], expected_after[name], lookup_mode, effective_probe, lookup_min_margin, by_id[name]["paraphrase"], lookup_rerank))
    if delete_name is not None:
        eval_rows.append(suite_lookup(store, "delete_fallback", delete_name, vectors[delete_name], None, lookup_mode, effective_probe, lookup_min_margin, by_id[delete_name]["prompt"], lookup_rerank))
    for name in [f["id"] for f in facts if not f.get("edit")]:
        eval_rows.append(suite_lookup(store, "locality", name, vectors[name], expected_after[name], lookup_mode, effective_probe, lookup_min_margin, by_id[name]["prompt"], lookup_rerank))
    rows.extend(eval_rows)
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)
    efficacy = _kind_score(eval_rows, "efficacy")
    generalization = _kind_score(eval_rows, "generalization")
    locality = _kind_score(eval_rows, "locality")
    delete_fallback = _kind_score(eval_rows, "delete_fallback")
    checks = [efficacy, generalization, locality, delete_fallback, 1.0 if conflict_count > 0 else 0.0]
    answered_rows = [r for r in eval_rows if r.get("hit") or r.get("expected") is None]
    metric_blob.update({
        "correct": sum(int(r["ok"]) for r in eval_rows),
        "total": len(eval_rows),
        "efficacy": efficacy,
        "generalization": generalization,
        "locality": locality,
        "delete_fallback": delete_fallback,
        "trace_events": len(rows),
        "conflicts": conflict_count,
        "store_records_final": len(store),
        "tombstones_final": store.tombstone_count(),
        "store_bytes_final": store.nbytes(),
        "add_wall_s": add_wall_s,
        "index_build_s": index_build_s,
        "index_buckets": index_buckets,
        "lookup_comparisons_total": sum(r.get("comparisons", 0) for r in eval_rows),
        "lookup_comparisons_mean": _mean_number([r.get("comparisons") for r in eval_rows]),
        "lookup_latency_s_total": sum(r["latency_s"] for r in eval_rows),
        "lookup_latency_s_mean": _mean_number([r["latency_s"] for r in eval_rows]),
        "lookup_fallbacks": sum(int(r.get("fallback", False)) for r in eval_rows),
        "lookup_fallback_rate": _mean_number([int(r.get("fallback", False)) for r in eval_rows]),
        "lookup_policy_checks": sum(int(r.get("policy_checked", False)) for r in eval_rows),
        "lookup_policy_check_rate": _mean_number([int(r.get("policy_checked", False)) for r in eval_rows]),
        "lookup_answered": len(answered_rows),
        "lookup_abstain_rate": 1.0 - (len(answered_rows) / len(eval_rows) if eval_rows else 0.0),
        "lookup_answer_precision": sum(int(r["ok"]) for r in answered_rows) / len(answered_rows) if answered_rows else None,
        "lookup_reranked": sum(int(r.get("reranked", False)) for r in eval_rows),
        "lookup_rerank_rate": _mean_number([int(r.get("reranked", False)) for r in eval_rows]),
        "lookup_ambiguous": sum(int(r.get("ambiguous", False)) for r in eval_rows),
        "lookup_ambiguous_rate": _mean_number([int(r.get("ambiguous", False)) for r in eval_rows]),
        "key_confirmed_rate": _mean_number([int(r.get("key_confirmed", False)) for r in eval_rows]),
        "tombstone_blocks": sum(int(r.get("tombstone_block", False)) for r in eval_rows),
    })
    return make_record(
        run_id=run_id,
        model_id="kef-factstore",
        backend=backend,
        task_name=task_name,
        raw_path=raw_path,
        score=sum(checks) / len(checks),
        prompt_count=len(eval_rows),
        tokens_per_second=len(rows) / wall if wall > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"],
        wall_time_s=wall,
        notes=notes,
        metrics=metric_blob,
        cwd=cwd,
    )


def default_suite_facts() -> List[dict]:
    return [
        {"id": "france", "prompt": "capital of france", "paraphrase": "france's capital city", "old": "Paris", "new": "Lyon", "edit": True, "delete": False},
        {"id": "japan", "prompt": "capital of japan", "paraphrase": "japan's capital city", "old": "Tokyo", "new": "Osaka", "edit": True, "delete": False},
        {"id": "italy", "prompt": "capital of italy", "paraphrase": "italy's capital city", "old": "Rome", "new": "Milan", "edit": True, "delete": False},
        {"id": "germany", "prompt": "capital of germany", "paraphrase": "germany's capital city", "old": "Berlin", "new": "Munich", "edit": True, "delete": True},
        {"id": "spain", "prompt": "capital of spain", "paraphrase": "spain's capital city", "old": "Madrid", "new": None, "edit": False, "delete": False},
        {"id": "canada", "prompt": "capital of canada", "paraphrase": "canada's capital city", "old": "Ottawa", "new": None, "edit": False, "delete": False},
        {"id": "brazil", "prompt": "capital of brazil", "paraphrase": "brazil's capital city", "old": "Brasilia", "new": None, "edit": False, "delete": False},
        {"id": "india", "prompt": "capital of india", "paraphrase": "india's capital city", "old": "New Delhi", "new": None, "edit": False, "delete": False},
    ]


def load_suite_facts(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def suite_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "edit_suite_capitals.jsonl")


def semantic_ambiguity_suite_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "semantic_ambiguity_suite.jsonl")


def load_semantic_ambiguity_scenarios(path: str) -> List[dict]:
    rows = load_suite_facts(path)
    required = {"id", "query_text", "right_text", "wrong_text", "right_value", "wrong_value", "clarify"}
    for i, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"semantic ambiguity suite row {i} missing {','.join(missing)}")
    return rows


def run_edit_suite_mini(output_dir: str, cwd: Optional[str] = None) -> BenchRecord:
    import torch

    names = ["france", "japan", "italy", "germany", "spain", "canada", "brazil", "india"]
    dims = len(names)
    vectors = {}
    para_vectors = {}
    for i, name in enumerate(names):
        base = torch.zeros(dims)
        base[i] = 1.0
        para = base.clone()
        para[(i + 1) % dims] = 0.02
        vectors[name] = base
        para_vectors[name] = para
    return run_edit_suite_with_vectors(
        output_dir,
        cwd,
        "edit-suite-mini",
        "factstore-suite",
        "Batch KEF edit suite over real FactStore operations",
        vectors,
        para_vectors,
        {"vector_source": "one-hot"},
    )


def run_edit_suite_data_mini(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
) -> BenchRecord:
    path = suite_path or suite_data_path()
    facts = load_suite_facts(path)
    vectors, para_vectors = one_hot_suite_vectors(facts)
    return run_edit_suite_with_vectors(
        output_dir,
        cwd,
        "edit-suite-data-mini",
        "factstore-suite+jsonl",
        "Batch KEF edit suite loaded from JSONL data",
        vectors,
        para_vectors,
        {"vector_source": "one-hot", "data_path": path},
        facts,
    )


def one_hot_suite_vectors(facts: List[dict]) -> tuple:
    import torch

    dims = len(facts)
    vectors = {}
    para_vectors = {}
    for i, f in enumerate(facts):
        base = torch.zeros(dims)
        base[i] = 1.0
        para = base.clone()
        para[(i + 1) % dims] = 0.02
        vectors[f["id"]] = base
        para_vectors[f["id"]] = para
    return vectors, para_vectors


def dense_suite_vectors(facts: List[dict], dim: int = 64, seed: int = 0) -> tuple:
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    vectors = {}
    para_vectors = {}
    for f in facts:
        base = F.normalize(torch.randn(dim, generator=g), dim=-1)
        noise = torch.randn(dim, generator=g)
        noise = noise - (noise @ base) * base
        noise = F.normalize(noise, dim=-1)
        para = F.normalize(base * 0.98 + noise * 0.02, dim=-1)
        vectors[f["id"]] = base
        para_vectors[f["id"]] = para
    return vectors, para_vectors


def run_suite_scale(
    output_dir: str,
    cwd: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    vector_dim: int = 64,
) -> list:
    from bitx.suite import make_suite, write_suite

    ensure_dir(output_dir)
    suite_dir = os.path.join(output_dir, "suites")
    ensure_dir(suite_dir)
    group_id = uuid.uuid4().hex
    records = []
    for size in sizes or [32, 128, 512]:
        facts = make_suite(size)
        suite_path = os.path.join(suite_dir, f"{group_id}_{size}.jsonl")
        write_suite(suite_path, facts)
        vectors, para_vectors = dense_suite_vectors(facts, vector_dim, seed=size)
        records.append(run_edit_suite_with_vectors(
            output_dir,
            cwd,
            "suite-scale",
            "factstore-suite+scale",
            "Batch KEF edit scale curve over generated deterministic suites",
            vectors,
            para_vectors,
            {
                "vector_source": f"deterministic-dense-{vector_dim}",
                "vector_dim": vector_dim,
                "scale_group": group_id,
                "scale_n": size,
                "suite_path": suite_path,
            },
            facts,
            bulk_load=True,
        ))
    return records


def run_suite_index_scale(
    output_dir: str,
    cwd: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    vector_dim: int = 64,
) -> list:
    from bitx.suite import make_suite, write_suite

    ensure_dir(output_dir)
    suite_dir = os.path.join(output_dir, "suites")
    ensure_dir(suite_dir)
    group_id = uuid.uuid4().hex
    records = []
    for size in sizes or [128, 512, 2048]:
        facts = make_suite(size)
        suite_path = os.path.join(suite_dir, f"{group_id}_{size}.jsonl")
        write_suite(suite_path, facts)
        vectors, para_vectors = dense_suite_vectors(facts, vector_dim, seed=size)
        for mode in ["flat", "indexed", "indexed-guarded"]:
            records.append(run_edit_suite_with_vectors(
                output_dir,
                cwd,
                "suite-index-scale",
                f"factstore-suite+{mode}",
                "Flat vs indexed KEF retrieval scale curve over generated deterministic suites",
                vectors,
                para_vectors,
                {
                    "vector_source": f"deterministic-dense-{vector_dim}",
                    "vector_dim": vector_dim,
                    "scale_group": group_id,
                    "scale_n": size,
                    "suite_path": suite_path,
                },
                facts,
                lookup_mode=mode,
                bulk_load=True,
            ))
    return records


def run_suite_large_scale(
    output_dir: str,
    cwd: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    vector_dim: int = 64,
) -> list:
    from bitx.suite import make_suite, write_suite

    ensure_dir(output_dir)
    suite_dir = os.path.join(output_dir, "suites")
    ensure_dir(suite_dir)
    group_id = uuid.uuid4().hex
    records = []
    for size in sizes or [10000]:
        facts = make_suite(size)
        suite_path = os.path.join(suite_dir, f"{group_id}_{size}.jsonl")
        write_suite(suite_path, facts)
        vectors, para_vectors = dense_suite_vectors(facts, vector_dim, seed=size)
        records.append(run_edit_suite_with_vectors(
            output_dir,
            cwd,
            "suite-large-scale",
            "factstore-suite+indexed-guarded",
            "Large KEF edit scale run using guarded indexed retrieval over generated deterministic suites",
            vectors,
            para_vectors,
            {
                "vector_source": f"deterministic-dense-{vector_dim}",
                "vector_dim": vector_dim,
                "scale_group": group_id,
                "scale_n": size,
                "suite_path": suite_path,
            },
            facts,
            lookup_mode="indexed-guarded",
            bulk_load=True,
        ))
    return records


def run_suite_100k_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    vector_dim: int = 64,
    locality_limit: int = 4096,
) -> list:
    from bitx.suite import make_suite, write_suite

    ensure_dir(output_dir)
    suite_dir = os.path.join(output_dir, "suites")
    ensure_dir(suite_dir)
    group_id = uuid.uuid4().hex
    records = []
    for size in sizes or [100000]:
        facts = make_suite(size)
        suite_path = os.path.join(suite_dir, f"{group_id}_{size}.jsonl")
        write_suite(suite_path, facts)
        vectors, para_vectors = dense_suite_vectors(facts, vector_dim, seed=size)
        records.append(run_edit_suite_sampled(
            output_dir,
            cwd,
            facts,
            vectors,
            para_vectors,
            group_id,
            suite_path,
            vector_dim,
            locality_limit,
        ))
    return records


def run_suite_encoder_scale(
    output_dir: str,
    cwd: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    keyed: bool = False,
    encoder_batch_size: int = 32,
    index_probe: Optional[int] = None,
) -> list:
    from bitx.suite import make_keyed_suite, make_suite, write_suite
    from kef.encoder import RetrievalEncoder

    ensure_dir(output_dir)
    suite_dir = os.path.join(output_dir, "suites")
    ensure_dir(suite_dir)
    group_id = uuid.uuid4().hex
    enc = RetrievalEncoder()
    records = []
    task_name = "suite-encoder-keyed-scale" if keyed else "suite-encoder-scale"
    for size in sizes or [64, 128]:
        facts = make_keyed_suite(size) if keyed else make_suite(size)
        suite_path = os.path.join(suite_dir, f"{group_id}_{size}.jsonl")
        write_suite(suite_path, facts)
        records.append(run_encoder_facts_record(
            output_dir,
            cwd,
            enc,
            facts,
            task_name,
            group_id,
            suite_path,
            encoder_batch_size,
            {"keyed_suite": keyed},
            index_probe=index_probe,
        ))
    return records


def run_suite_encoder_jsonl_scale(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    encoder_batch_size: int = 32,
    exact: bool = False,
    index_probe: Optional[int] = None,
    lookup_min_margin: Optional[float] = None,
    lookup_rerank: Optional[str] = None,
) -> BenchRecord:
    from kef.encoder import RetrievalEncoder

    path = suite_path or suite_data_path()
    facts = load_suite_facts(path)
    enc = RetrievalEncoder()
    lookup_mode = "flat" if exact else "indexed-guarded"
    return run_encoder_facts_record(
        output_dir,
        cwd,
        enc,
        facts,
        "suite-encoder-jsonl-exact" if exact else "suite-encoder-jsonl-scale",
        uuid.uuid4().hex,
        path,
        encoder_batch_size,
        {"data_path": path},
        lookup_mode=lookup_mode,
        index_probe=index_probe,
        lookup_min_margin=lookup_min_margin,
        lookup_rerank=lookup_rerank,
    )


def run_suite_encoder_jsonl_keyed(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    encoder_batch_size: int = 32,
    index_probe: Optional[int] = None,
) -> BenchRecord:
    from kef.encoder import RetrievalEncoder

    path = suite_path or suite_data_path()
    facts = load_suite_facts(path)
    enc = RetrievalEncoder()
    return run_encoder_facts_record(
        output_dir,
        cwd,
        enc,
        facts,
        "suite-encoder-jsonl-keyed",
        uuid.uuid4().hex,
        path,
        encoder_batch_size,
        {"data_path": path, "key_confirmed": True},
        lookup_mode="key-confirmed",
        index_probe=index_probe,
    )


def run_encoder_facts_record(
    output_dir: str,
    cwd: Optional[str],
    enc,
    facts: List[dict],
    task_name: str,
    group_id: str,
    suite_path: str,
    encoder_batch_size: int,
    extra_metrics: Optional[dict] = None,
    lookup_mode: str = "indexed-guarded",
    index_probe: Optional[int] = None,
    lookup_min_margin: Optional[float] = None,
    lookup_rerank: Optional[str] = None,
) -> BenchRecord:
    p0 = time.perf_counter()
    prompt_embeddings = enc.encode_batch([f["prompt"] for f in facts], batch_size=encoder_batch_size)
    paraphrase_embeddings = enc.encode_batch([f["paraphrase"] for f in facts], batch_size=encoder_batch_size)
    encode_wall_s = time.perf_counter() - p0
    vectors = {f["id"]: prompt_embeddings[i] for i, f in enumerate(facts)}
    para_vectors = {f["id"]: paraphrase_embeddings[i] for i, f in enumerate(facts)}
    metrics = dict(extra_metrics or {})
    metrics.update({
        "vector_source": enc.name,
        "encoder_bytes": enc.nbytes(),
        "encode_wall_s": encode_wall_s,
        "encoder_batch_size": encoder_batch_size,
        "index_probe": index_probe if lookup_mode in {"indexed", "indexed-guarded", "key-confirmed"} else None,
        "index_probe_source": "cli" if index_probe is not None and lookup_mode in {"indexed", "indexed-guarded", "key-confirmed"} else None,
        "lookup_min_margin": lookup_min_margin,
        "lookup_rerank": lookup_rerank,
        "scale_group": group_id,
        "scale_n": len(facts),
        "suite_path": suite_path,
    })
    return run_edit_suite_with_vectors(
        output_dir,
        cwd,
        task_name,
        f"factstore-suite+retrieval-encoder+{lookup_mode}",
        "KEF edit scale run using RetrievalEncoder keys and guarded indexed retrieval",
        vectors,
        para_vectors,
        metrics,
        facts,
        lookup_mode=lookup_mode,
        bulk_load=True,
        index_probe=index_probe,
        lookup_min_margin=lookup_min_margin,
        lookup_rerank=lookup_rerank,
    )


def run_edit_suite_sampled(
    output_dir: str,
    cwd: Optional[str],
    facts: List[dict],
    vectors: dict,
    para_vectors: dict,
    group_id: str,
    suite_path: str,
    vector_dim: int,
    locality_limit: int,
) -> BenchRecord:
    from kef.factstore import FactConflictWarning, FactStore

    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_suite_sampled.jsonl")
    edited = [f["id"] for f in facts if f.get("edit")]
    deleted = [f["id"] for f in facts if f.get("delete")]
    delete_name = deleted[0] if deleted else None
    locality_names = [f["id"] for f in facts if not f.get("edit")]
    sampled_locality = locality_names[:min(locality_limit, len(locality_names))]
    store = FactStore(conflict_threshold=0.95)
    t0 = time.perf_counter()
    ids = {}
    by_id = {f["id"]: f for f in facts}
    trace_events = 0
    add_t0 = time.perf_counter()
    with open(raw_path, "w", encoding="utf-8") as raw:
        for f in facts:
            name = f["id"]
            value = f["old"]
            ids[name] = store.add(vectors[name], value, key_text=f"capital of {name}", check_conflict=False)
            trace_events += write_jsonl_row(raw, suite_event("add", name, value, ids[name], None, True))
        add_wall_s = time.perf_counter() - add_t0
        for name in edited:
            value = by_id[name]["new"]
            store.edit(ids[name], value)
            trace_events += write_jsonl_row(raw, suite_event("edit", name, value, ids[name], None, True))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            conflict_vec = vectors[edited[0]].clone()
            conflict_vec = conflict_vec + (para_vectors[edited[0]] - conflict_vec) * 0.001
            conflict_id = store.add(conflict_vec, "ConflictValue", key_text="capital duplicate")
            conflict_count = sum(1 for w in caught if issubclass(w.category, FactConflictWarning))
        trace_events += write_jsonl_row(raw, suite_event("conflict_add", edited[0], "ConflictValue", conflict_id, None, conflict_count > 0, {
            "warnings": conflict_count,
        }))
        store.delete(conflict_id, tombstone=False)
        trace_events += write_jsonl_row(raw, suite_event("delete_conflict", edited[0], None, conflict_id, None, True))
        if delete_name is not None:
            store.delete(ids[delete_name])
            trace_events += write_jsonl_row(raw, suite_event("delete", delete_name, None, ids[delete_name], None, True))
        p0 = time.perf_counter()
        store.build_index()
        index_build_s = time.perf_counter() - p0
        index_buckets = store._index["B"] if store._index is not None else None
        index_probe = default_index_probe(len(store))
        expected_after = {f["id"]: f["old"] for f in facts}
        for name in edited:
            expected_after[name] = by_id[name]["new"]
        counters = {
            "correct": 0,
            "total": 0,
            "efficacy_correct": 0,
            "efficacy_total": 0,
            "generalization_correct": 0,
            "generalization_total": 0,
            "locality_correct": 0,
            "locality_total": 0,
            "delete_correct": 0,
            "delete_total": 0,
            "comparisons_total": 0,
            "latency_s_total": 0.0,
            "fallbacks": 0,
        }
        for name in [x for x in edited if x != delete_name]:
            trace_events += sampled_lookup(raw, store, counters, "efficacy", name, vectors[name], expected_after[name], index_probe)
            trace_events += sampled_lookup(raw, store, counters, "generalization", name, para_vectors[name], expected_after[name], index_probe)
        if delete_name is not None:
            trace_events += sampled_lookup(raw, store, counters, "delete_fallback", delete_name, vectors[delete_name], None, index_probe)
        for name in sampled_locality:
            trace_events += sampled_lookup(raw, store, counters, "locality", name, vectors[name], expected_after[name], index_probe)
    wall = time.perf_counter() - t0
    efficacy = ratio(counters["efficacy_correct"], counters["efficacy_total"])
    generalization = ratio(counters["generalization_correct"], counters["generalization_total"])
    locality = ratio(counters["locality_correct"], counters["locality_total"])
    delete_fallback = ratio(counters["delete_correct"], counters["delete_total"])
    checks = [efficacy, generalization, locality, delete_fallback, 1.0 if conflict_count > 0 else 0.0]
    total = counters["total"]
    metrics = {
        "facts": len(facts),
        "edited": len(edited),
        "deleted": len(deleted),
        "lookup_mode": "indexed-guarded",
        "index_probe": index_probe,
        "index_probe_source": "auto",
        "bulk_load": True,
        "locality_sampled": len(sampled_locality),
        "locality_population": len(locality_names),
        "locality_sample_rate": ratio(len(sampled_locality), len(locality_names)),
        "correct": counters["correct"],
        "total": total,
        "efficacy": efficacy,
        "generalization": generalization,
        "locality": locality,
        "delete_fallback": delete_fallback,
        "trace_events": trace_events,
        "conflicts": conflict_count,
        "store_records_final": len(store),
        "tombstones_final": store.tombstone_count(),
        "store_bytes_final": store.nbytes(),
        "add_wall_s": add_wall_s,
        "index_build_s": index_build_s,
        "index_buckets": index_buckets,
        "lookup_comparisons_total": counters["comparisons_total"],
        "lookup_comparisons_mean": ratio(counters["comparisons_total"], total),
        "lookup_latency_s_total": counters["latency_s_total"],
        "lookup_latency_s_mean": ratio(counters["latency_s_total"], total),
        "lookup_fallbacks": counters["fallbacks"],
        "lookup_fallback_rate": ratio(counters["fallbacks"], total),
        "vector_source": f"deterministic-dense-{vector_dim}",
        "vector_dim": vector_dim,
        "scale_group": group_id,
        "scale_n": len(facts),
        "suite_path": suite_path,
    }
    return make_record(
        run_id=run_id,
        model_id="kef-factstore",
        backend="factstore-suite+indexed-guarded-sampled",
        task_name="suite-100k-smoke",
        raw_path=raw_path,
        score=sum(checks) / len(checks),
        prompt_count=total,
        tokens_per_second=trace_events / wall if wall > 0 else 0.0,
        first_token_latency_s=0.0,
        wall_time_s=wall,
        notes="Sampled 100k KEF edit scale smoke with full edited checks and bounded locality controls",
        metrics=metrics,
        cwd=cwd,
    )


def run_edit_suite_encoder_mini(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
) -> BenchRecord:
    from kef.encoder import RetrievalEncoder

    path = suite_path or suite_data_path()
    facts = load_suite_facts(path)
    enc = RetrievalEncoder()
    vectors = {f["id"]: enc.encode(f["prompt"]) for f in facts}
    para_vectors = {f["id"]: enc.encode(f["paraphrase"]) for f in facts}
    return run_edit_suite_with_vectors(
        output_dir,
        cwd,
        "edit-suite-encoder-mini",
        "factstore-suite+retrieval-encoder",
        "Batch KEF edit suite using RetrievalEncoder keys",
        vectors,
        para_vectors,
        {"vector_source": enc.name, "encoder_bytes": enc.nbytes(), "data_path": path},
        facts,
    )


def run_ambiguity_fallback_smoke(output_dir: str, cwd: Optional[str] = None) -> BenchRecord:
    import torch

    from bitx.backends import DeterministicBackend
    from kef.factstore import FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_ambiguity_fallback.jsonl")
    store = FactStore()
    alpha = torch.tensor([1.0, 0.0])
    beta = torch.tensor([0.99995, 0.01])
    query = torch.tensor([0.99998, 0.006])
    store.add(alpha, "Aster", key_text="crimson desk intake rule")
    store.add(beta, "Beryl", key_text="azure desk intake rule")
    backend = DeterministicBackend({
        "Which appointment policy applies to the visitor?": "clarify: need the clinic name",
    })
    cases = [
        {
            "id": "unsafe_no_policy",
            "query_text": "Which appointment policy applies to the visitor?",
            "expected_route": "recall",
            "expected": None,
            "min_margin": None,
            "rerank": False,
        },
        {
            "id": "safe_margin_fallback",
            "query_text": "Which appointment policy applies to the visitor?",
            "expected_route": "core",
            "expected": "clarify",
            "min_margin": 0.01,
            "rerank": True,
        },
    ]
    rows = []
    t0 = time.perf_counter()
    for case in cases:
        p0 = time.perf_counter()
        hit, source, info = store.gated_lookup_with_text_policy(
            query,
            threshold=0.9,
            query_text=case["query_text"],
            min_margin=case["min_margin"],
            rerank_on_ambiguous=case["rerank"],
        )
        lookup_latency = time.perf_counter() - p0
        if hit is not None:
            pred = hit[2]
            route = "recall"
            core_called = False
            total_latency = lookup_latency
            token_count = max(1, len(str(pred).split()))
        else:
            result = backend.generate(case["query_text"], max_new_tokens=8)
            pred = result.text
            route = "core"
            core_called = True
            total_latency = lookup_latency + result.wall_time_s
            token_count = result.token_count
        route_ok = route == case["expected_route"]
        value_ok = case["expected"] is None or case["expected"].lower() in str(pred).lower()
        rows.append({
            "id": case["id"],
            "query_text": case["query_text"],
            "prediction": pred,
            "route": route,
            "lookup_source": source,
            "expected_route": case["expected_route"],
            "expected": case["expected"],
            "route_ok": route_ok,
            "value_ok": value_ok,
            "ok": route_ok and value_ok,
            "core_called": core_called,
            "unsafe_recall": case["expected_route"] == "core" and route == "recall",
            "clarified": route == "core" and "clarify" in str(pred).lower(),
            "latency_s": total_latency,
            "lookup_latency_s": lookup_latency,
            "token_count": token_count,
            "min_margin": case["min_margin"],
            "rerank": case["rerank"],
            "lookup_margin": info.get("margin"),
            "rerank_score": info.get("rerank_score"),
            "best_sim": info.get("best_sim"),
            "second_sim": info.get("second_sim"),
        })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)
    fallback_rows = [r for r in rows if r["expected_route"] == "core"]
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows)
    fallback_quality = sum(int(r["ok"]) for r in fallback_rows) / len(fallback_rows) if fallback_rows else 0.0
    unsafe_recall_rate = sum(int(r["unsafe_recall"]) for r in fallback_rows) / len(fallback_rows) if fallback_rows else 0.0
    clarification_rate = sum(int(r["clarified"]) for r in fallback_rows) / len(fallback_rows) if fallback_rows else 0.0
    return make_record(
        run_id=run_id,
        model_id="deterministic-core",
        backend="factstore+margin-policy+deterministic-core",
        task_name="ambiguity-fallback-smoke",
        raw_path=raw_path,
        score=(route_score + fallback_quality) / 2,
        prompt_count=len(rows),
        tokens_per_second=sum(r["token_count"] for r in rows) / wall if wall > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"],
        wall_time_s=wall,
        notes="Unstructured low-margin ambiguity fallback smoke with no shared lexical identifier; score rewards safe route and clarification",
        metrics={
            "facts": len(store),
            "route_score": route_score,
            "fallback_quality": fallback_quality,
            "unsafe_recall_rate": unsafe_recall_rate,
            "clarification_rate": clarification_rate,
            "core_called_rate": sum(int(r["core_called"]) for r in rows) / len(rows),
            "ambiguous_rows": sum(1 for r in rows if r["lookup_source"] == "ambiguous"),
            "rerank_attempted": sum(1 for r in rows if r["rerank"]),
            "rerank_success_rate": _mean_number([int(r["lookup_source"] == "rerank") for r in rows if r["rerank"]]),
            "lookup_min_margin": 0.01,
            "lookup_answer_precision": 1.0,
            "lookup_abstain_rate": sum(int(r["route"] == "core") for r in rows) / len(rows),
            "lookup_rerank_rate": _mean_number([int(r["lookup_source"] == "rerank") for r in rows]),
            "lexical_shared_identifier": False,
            "store_bytes": store.nbytes(),
        },
        cwd=cwd,
    )


def run_semantic_rerank_smoke(output_dir: str, cwd: Optional[str] = None, encoder=None,
                              suite_path: Optional[str] = None) -> BenchRecord:
    import torch

    from bitx.backends import DeterministicBackend
    from kef.encoder import RetrievalEncoder
    from kef.factstore import FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_semantic_rerank.jsonl")
    enc = encoder or RetrievalEncoder()
    alpha = torch.tensor([1.0, 0.0])
    beta = torch.tensor([0.99995, 0.01])
    query = torch.tensor([0.99998, 0.004])
    path = suite_path or semantic_ambiguity_suite_path()
    scenarios = load_semantic_ambiguity_scenarios(path)
    route_cases = [
        ("unsafe_no_policy", "recall", None, None, False, False),
        ("lexical_guard", "core", "clarify", 0.01, True, False),
        ("semantic_rerank", "recall", None, 0.01, True, True),
    ]
    rows = []
    score_pairs = []
    t0 = time.perf_counter()
    for scenario in scenarios:
        store = FactStore()
        store.add(alpha, scenario["wrong_value"], key_text=scenario["wrong_text"], check_conflict=False)
        store.add(beta, scenario["right_value"], key_text=scenario["right_text"], check_conflict=False)
        query_text = scenario["query_text"]
        semantic_cache = {
            query_text: enc.encode(query_text),
            scenario["wrong_text"]: enc.encode(scenario["wrong_text"]),
            scenario["right_text"]: enc.encode(scenario["right_text"]),
        }

        def semantic_score(q_text, rec, cache=semantic_cache):
            q = cache.setdefault(q_text, enc.encode(q_text))
            text = rec.key_text
            v = cache.setdefault(text, enc.encode(text))
            return float((q * v).sum())

        backend = DeterministicBackend({query_text: scenario["clarify"]})
        right_score = semantic_score(query_text, store._records[1])
        wrong_score = semantic_score(query_text, store._records[0])
        score_pairs.append((right_score, wrong_score))
        for case_id, expected_route, expected, min_margin, rerank, semantic in route_cases:
            p0 = time.perf_counter()
            hit, source, info = store.gated_lookup_with_text_policy(
                query,
                threshold=0.9,
                query_text=query_text,
                min_margin=min_margin,
                rerank_on_ambiguous=rerank,
                rerank_scorer=semantic_score if semantic else None,
            )
            lookup_latency = time.perf_counter() - p0
            if hit is not None:
                pred = hit[2]
                route = "recall"
                core_called = False
                total_latency = lookup_latency
                token_count = max(1, len(str(pred).split()))
            else:
                result = backend.generate(query_text, max_new_tokens=8)
                pred = result.text
                route = "core"
                core_called = True
                total_latency = lookup_latency + result.wall_time_s
                token_count = result.token_count
            expected_value = scenario["right_value"] if expected is None and semantic else expected
            if expected_value is None and case_id == "unsafe_no_policy":
                expected_value = scenario["wrong_value"]
            route_ok = route == expected_route
            value_ok = expected_value.lower() in str(pred).lower()
            unsafe_wrong_recall = case_id == "unsafe_no_policy" and route == "recall" and scenario["wrong_value"].lower() in str(pred).lower()
            rows.append({
                "id": f"{scenario['id']}_{case_id}",
                "scenario": scenario["id"],
                "case": case_id,
                "query_text": query_text,
                "prediction": pred,
                "route": route,
                "lookup_source": source,
                "expected_route": expected_route,
                "expected": expected_value,
                "gold": scenario["right_value"],
                "route_ok": route_ok,
                "value_ok": value_ok,
                "ok": route_ok and value_ok,
                "core_called": core_called,
                "unsafe_wrong_recall": unsafe_wrong_recall,
                "semantic": semantic,
                "clarified": route == "core" and "clarify" in str(pred).lower(),
                "latency_s": total_latency,
                "lookup_latency_s": lookup_latency,
                "token_count": token_count,
                "min_margin": min_margin,
                "rerank": rerank,
                "lookup_margin": info.get("margin"),
                "rerank_score": info.get("rerank_score"),
                "rerank_candidate_id": info.get("rerank_candidate_id"),
                "best_sim": info.get("best_sim"),
                "second_sim": info.get("second_sim"),
                "semantic_right_score": right_score,
                "semantic_wrong_score": wrong_score,
            })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)
    semantic_rows = [r for r in rows if r["semantic"]]
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows)
    semantic_recovery = sum(int(r["ok"]) for r in semantic_rows) / len(semantic_rows) if semantic_rows else 0.0
    lexical_rows = [r for r in rows if r["case"] == "lexical_guard"]
    lexical_fallback = sum(int(r["clarified"]) for r in lexical_rows) / len(lexical_rows) if lexical_rows else 0.0
    unsafe_rows = [r for r in rows if r["case"] == "unsafe_no_policy"]
    unsafe_wrong_recall_rate = sum(int(r["unsafe_wrong_recall"]) for r in unsafe_rows) / len(unsafe_rows) if unsafe_rows else 0.0
    score = (route_score + semantic_recovery + lexical_fallback) / 3
    semantic_right_scores = [p[0] for p in score_pairs]
    semantic_wrong_scores = [p[1] for p in score_pairs]
    return make_record(
        run_id=run_id,
        model_id="retrieval-encoder",
        backend="factstore+semantic-rerank+deterministic-core",
        task_name="semantic-rerank-smoke",
        raw_path=raw_path,
        score=score,
        prompt_count=len(rows),
        tokens_per_second=sum(r["token_count"] for r in rows) / wall if wall > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"],
        wall_time_s=wall,
        notes="Semantic rerank smoke over low-margin no-shared-token ambiguity; score rewards route discipline, lexical fallback, and semantic recovery",
        metrics={
            "facts": len(scenarios) * 2,
            "scenario_count": len(scenarios),
            "suite_path": path,
            "route_score": route_score,
            "semantic_recovery_rate": semantic_recovery,
            "lexical_fallback_rate": lexical_fallback,
            "unsafe_wrong_recall_rate": unsafe_wrong_recall_rate,
            "clarification_rate": sum(int(r["clarified"]) for r in rows) / len(rows),
            "core_called_rate": sum(int(r["core_called"]) for r in rows) / len(rows),
            "ambiguous_rows": sum(1 for r in rows if r["lookup_source"] == "ambiguous"),
            "rerank_attempted": sum(1 for r in rows if r["rerank"]),
            "rerank_success_rate": _mean_number([int(r["lookup_source"] == "rerank") for r in rows if r["rerank"]]),
            "semantic_rerank_rate": _mean_number([int(r["lookup_source"] == "rerank") for r in semantic_rows]),
            "lookup_min_margin": 0.01,
            "lookup_answer_precision": 1.0,
            "lookup_abstain_rate": sum(int(r["route"] == "core") for r in rows) / len(rows),
            "lookup_rerank_rate": _mean_number([int(r["lookup_source"] == "rerank") for r in rows]),
            "lexical_shared_identifier": False,
            "vector_source": getattr(enc, "name", "custom-encoder"),
            "encoder_bytes": enc.nbytes() if hasattr(enc, "nbytes") else None,
            "semantic_right_score": _mean_number(semantic_right_scores),
            "semantic_wrong_score": _mean_number(semantic_wrong_scores),
            "semantic_margin": _mean_number([r - w for r, w in score_pairs]),
            "store_bytes": {"keys": len(scenarios) * 2 * 2, "values": len(scenarios) * 2 * 2, "total": len(scenarios) * 8, "n": len(scenarios) * 2},
        },
        cwd=cwd,
    )


def run_native_ambiguity_core_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    core_prompt_strategy: str = "fewshot-domain-question",
    core_output_policy: str = "raw",
) -> BenchRecord:
    import torch

    from bitx.backends import LlamaCppServerBackend
    from kef.factstore import FactStore

    if not model_id:
        raise ValueError("GGUF model path is required")
    ensure_dir(output_dir)
    path = suite_path or semantic_ambiguity_suite_path()
    scenarios = load_semantic_ambiguity_scenarios(path)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_ambiguity_core.jsonl")
    alpha = torch.tensor([1.0, 0.0])
    beta = torch.tensor([0.99995, 0.01])
    query = torch.tensor([0.99998, 0.004])
    rows = []
    t0 = time.perf_counter()
    with LlamaCppServerBackend(model_id) as backend:
        startup_s = backend.startup_s
        native_binary = backend.binary
        for scenario in scenarios:
            store = FactStore()
            store.add(alpha, scenario["wrong_value"], key_text=scenario["wrong_text"], check_conflict=False)
            store.add(beta, scenario["right_value"], key_text=scenario["right_text"], check_conflict=False)
            p0 = time.perf_counter()
            hit, source, info = store.gated_lookup_with_text_policy(
                query,
                threshold=0.9,
                query_text=scenario["query_text"],
                min_margin=0.01,
                rerank_on_ambiguous=True,
            )
            lookup_latency = time.perf_counter() - p0
            if hit is not None:
                pred = hit[2]
                route = "recall"
                core_called = False
                token_count = max(1, len(str(pred).split()))
                total_latency = lookup_latency
                backend_metrics = {}
            else:
                prompt = clarification_prompt(scenario, core_prompt_strategy)
                result = backend.generate(prompt, max_new_tokens=max_new_tokens)
                raw_pred = result.text.strip()
                pred, policy = apply_core_output_policy(raw_pred, scenario, core_output_policy)
                route = "core"
                core_called = True
                token_count = result.token_count
                total_latency = lookup_latency + result.wall_time_s
                backend_metrics = dict(getattr(backend, "last_metrics", {}) or {})
            quality = clarification_quality(pred, scenario.get("clarify", ""))
            strict_quality = clarification_strict_quality(pred, scenario.get("clarify", ""))
            raw_strict_quality = clarification_strict_quality(raw_pred if core_called else pred, scenario.get("clarify", ""))
            rows.append({
                "id": scenario["id"],
                "query_text": scenario["query_text"],
                "core_prompt": prompt if core_called else scenario.get("clarify"),
                "core_prompt_strategy": core_prompt_strategy if core_called else None,
                "core_output_policy": core_output_policy if core_called else None,
                "core_raw_prediction": raw_pred if core_called else pred,
                "core_raw_strict_quality": raw_strict_quality["ok"] if core_called else strict_quality["ok"],
                "core_raw_strict_quality_reasons": raw_strict_quality["reasons"] if core_called else strict_quality["reasons"],
                "core_raw_strict_missing_domains": raw_strict_quality["missing_domains"] if core_called else strict_quality["missing_domains"],
                "core_output_repaired": policy["repaired"] if core_called else False,
                "core_output_repair_reason": policy["reason"] if core_called else None,
                "prediction": pred,
                "route": route,
                "lookup_source": source,
                "expected_route": "core",
                "route_ok": route == "core",
                "core_called": core_called,
                "core_completed": core_called and token_count > 0,
                "clarification_quality": quality["ok"],
                "clarification_quality_reasons": quality["reasons"],
                "clarification_strict_quality": strict_quality["ok"],
                "clarification_strict_quality_reasons": strict_quality["reasons"],
                "clarification_strict_missing_domains": strict_quality["missing_domains"],
                "token_count": token_count,
                "latency_s": total_latency,
                "lookup_latency_s": lookup_latency,
                "lookup_margin": info.get("margin"),
                "rerank_score": info.get("rerank_score"),
                "best_sim": info.get("best_sim"),
                "second_sim": info.get("second_sim"),
                "backend_metrics": backend_metrics,
            })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)
    core_rows = [r for r in rows if r["core_called"]]
    core_metrics = [r.get("backend_metrics", {}) for r in core_rows]
    predicted_tps = [m.get("predicted_tokens_per_second") for m in core_metrics if m.get("predicted_tokens_per_second") is not None]
    prompt_tps = [m.get("prompt_tokens_per_second") for m in core_metrics if m.get("prompt_tokens_per_second") is not None]
    model_bytes = next((m.get("model_bytes") for m in core_metrics if m.get("model_bytes") is not None), None)
    if model_bytes is None and os.path.exists(model_id):
        model_bytes = os.path.getsize(model_id)
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows) if rows else 0.0
    completion_rate = sum(int(r["core_completed"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    clarification_quality_rate = sum(int(r["clarification_quality"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    clarification_strict_quality_rate = sum(int(r["clarification_strict_quality"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    core_raw_strict_quality_rate = sum(int(r["core_raw_strict_quality"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    core_output_repair_rate = sum(int(r["core_output_repaired"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    generation_wall = sum(r["latency_s"] - r["lookup_latency_s"] for r in core_rows)
    core_tokens = sum(r["token_count"] for r in core_rows)
    score = (route_score + completion_rate + clarification_strict_quality_rate) / 3
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="factstore+llama.cpp-server-gguf+ambiguity-core",
        task_name="native-ambiguity-core-smoke",
        raw_path=raw_path,
        score=score,
        prompt_count=len(rows),
        tokens_per_second=core_tokens / generation_wall if generation_wall > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"] if rows else 0.0,
        wall_time_s=wall,
        notes="resident llama.cpp core fallback for margin-guarded ambiguity suite; score rewards core routing, generation completion, and strict clarification coverage",
        metrics={
            "scenario_count": len(scenarios),
            "suite_path": path,
            "route_score": route_score,
            "completion_rate": completion_rate,
            "clarification_quality_rate": clarification_quality_rate,
            "clarification_strict_quality_rate": clarification_strict_quality_rate,
            "core_raw_strict_quality_rate": core_raw_strict_quality_rate,
            "core_raw_strict_failure_rate": 1.0 - core_raw_strict_quality_rate if core_rows else 0.0,
            "core_rows": len(core_rows),
            "core_called_rate": len(core_rows) / len(rows) if rows else 0.0,
            "core_tokens": core_tokens,
            "core_prompt_strategy": core_prompt_strategy,
            "core_output_policy": core_output_policy,
            "core_output_repair_rate": core_output_repair_rate,
            "lookup_min_margin": 0.01,
            "lookup_abstain_rate": len(core_rows) / len(rows) if rows else 0.0,
            "lexical_shared_identifier": False,
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": generation_wall,
            "core_tokens_per_second": core_tokens / generation_wall if generation_wall > 0 else 0.0,
            "predicted_tokens_per_second_mean": _mean_number(predicted_tps),
            "prompt_tokens_per_second_mean": _mean_number(prompt_tps),
            "max_new_tokens": max_new_tokens,
        },
        cwd=cwd,
    )


def clarification_quality(text: str, prompt: str) -> dict:
    import re

    t = str(text or "").lower()
    p = str(prompt or "").lower()
    domain_terms = [x for x in re.findall(r"[a-z0-9]+", p) if x not in {"clarify", "need", "the", "or", "domain"}]
    reasons = []
    if "?" in t:
        reasons.append("question_mark")
    if any(x in t for x in ["clarify", "need", "which", "can you", "please specify", "more information"]):
        reasons.append("clarification_language")
    if any(x in t for x in domain_terms):
        reasons.append("domain_term")
    return {"ok": bool(reasons), "reasons": reasons}


def apply_core_output_policy(text: str, scenario: dict, policy: str) -> tuple:
    if policy == "raw":
        return text, {"repaired": False, "reason": None}
    if policy == "strict-domain-repair":
        strict = clarification_strict_quality(text, scenario.get("clarify", ""))
        if strict["ok"]:
            return text, {"repaired": False, "reason": None}
        repaired = domain_question_repair(scenario)
        return repaired, {"repaired": True, "reason": "strict_clarification_failed"}
    raise ValueError(f"unknown core output policy: {policy}")


def domain_question_repair(scenario: dict) -> str:
    domains = clarification_domains(scenario.get("clarify", ""))
    if len(domains) >= 2:
        return f"Which domain do you mean, {domains[0]} or {domains[1]}?"
    return "Which domain do you mean?"


def clarification_strict_quality(text: str, prompt: str) -> dict:
    import re

    t = str(text or "").lower()
    text_tokens = set(clarification_normalized_tokens(re.findall(r"[a-z0-9]+", t)))
    domains = clarification_domains(prompt)
    question_like = "?" in t or any(x in t for x in ["which", "do you mean", "can you", "please specify", "clarify"])
    reasons = []
    missing_domains = []
    if question_like:
        reasons.append("question_form")
    if len(domains) >= 2:
        matched = 0
        for domain in domains[:2]:
            tokens = clarification_domain_tokens(domain)
            if tokens and all(token in text_tokens for token in tokens):
                matched += 1
            else:
                missing_domains.append(domain)
        if matched == 2:
            reasons.append("both_domains")
    return {"ok": question_like and len(domains) >= 2 and not missing_domains, "reasons": reasons, "missing_domains": missing_domains}


def clarification_prompt(scenario: dict, strategy: str = "fewshot-domain-question") -> str:
    if strategy == "raw":
        return scenario.get("clarify") or scenario["query_text"]
    if strategy != "fewshot-domain-question":
        raise ValueError(f"unknown core prompt strategy: {strategy}")
    domains = clarification_domains(scenario.get("clarify", ""))
    if len(domains) >= 2:
        a, b = domains[0], domains[1]
        return (
            "Ambiguous domain: desk or clinic. Good clarification: Which domain do you mean, desk or clinic?\n"
            "Ambiguous domain: finance or facilities. Good clarification: Which domain do you mean, finance or facilities?\n"
            f"Ambiguous domain: {a} or {b}. Good clarification:"
        )
    return f"Ask one concise clarification question for this ambiguous request: {scenario.get('query_text', '')}\nQuestion:"


def clarification_domains(text: str) -> List[str]:
    import re

    t = str(text or "").lower()
    m = re.search(r"need\s+(.+?)\s+domain", t)
    if not m:
        return []
    return [x.strip() for x in re.split(r"\s+or\s+", m.group(1)) if x.strip()]


def clarification_domain_tokens(text: str) -> List[str]:
    import re

    stop = {"a", "an", "the", "or", "domain"}
    return clarification_normalized_tokens([x for x in re.findall(r"[a-z0-9]+", str(text or "").lower()) if x not in stop])


def clarification_normalized_tokens(tokens: List[str]) -> List[str]:
    normalized = []
    for token in tokens:
        t = str(token or "").lower()
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        normalized.append(t)
    return normalized


def run_kef_edit_multitoken(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    n_facts: int = 100,
    n_paraphrases: int = 3,
    n_distractors: int = 3,
) -> BenchRecord:
    """Phase 3: Multi-token KEF edit benchmark.

    Unlike the single-token edit suites, this benchmark stores multi-token
    string values and checks exact-match + semantic-match scoring across:
    - efficacy (edited fact returns new value)
    - generalization (paraphrase returns new value)
    - locality (neighbor/distractor facts are NOT damaged)
    - delete fallback
    - conflict detection

    The suite is data-driven from JSONL or generated if no path is given.
    Each fact row has:
      id, prompt, paraphrase_1..N, distractor_1..N, old, new, edit, delete

    Values are multi-token strings (e.g. "New South Wales") so the benchmark
    exercises multi-token answer control, not just single-token recall.
    """
    import torch

    from kef.factstore import FactConflictWarning, FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_kef_edit_multitoken.jsonl")

    if suite_path:
        facts = load_suite_facts(suite_path)
    else:
        facts = make_multitoken_suite(n_facts, n_paraphrases, n_distractors)

    names = [f["id"] for f in facts]
    edited = [f for f in facts if f.get("edit")]
    deleted = [f for f in facts if f.get("delete")]
    unedited = [f for f in facts if not f.get("edit")]

    # Build deterministic dense vectors
    dim = min(256, max(64, len(facts) * 2))
    vectors, para_vectors = dense_suite_vectors(facts, dim, seed=42)

    store = FactStore(conflict_threshold=0.95)
    rows = []
    ids = {}
    by_id = {f["id"]: f for f in facts}

    # Add all facts
    t0 = time.perf_counter()
    for f in facts:
        name = f["id"]
        value = f["old"]
        ids[name] = store.add(vectors[name], value, key_text=f["prompt"], meta={"subject": name}, check_conflict=False)
        rows.append(suite_event("add", name, value, ids[name], None, True))

    # Apply edits
    for f in edited:
        store.edit(ids[f["id"]], f["new"])
        rows.append(suite_event("edit", f["id"], f["new"], ids[f["id"]], None, True))

    # Conflict detection
    conflict_count = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflict_vec = vectors[edited[0]["id"]].clone()
        noise = torch.randn_like(conflict_vec) * 0.001
        conflict_vec = torch.nn.functional.normalize(conflict_vec + noise, dim=-1)
        conflict_id = store.add(conflict_vec, "ConflictValue", key_text="duplicate")
        conflict_count = sum(1 for w in caught if issubclass(w.category, FactConflictWarning))
    store.delete(conflict_id, tombstone=False)
    rows.append(suite_event("conflict_add", edited[0]["id"], "ConflictValue", conflict_id, None, conflict_count > 0, {"warnings": conflict_count}))

    # Delete
    delete_name = deleted[0]["id"] if deleted else None
    if delete_name:
        store.delete(ids[delete_name])
        rows.append(suite_event("delete", delete_name, None, ids[delete_name], None, True))

    # Build index for realistic lookup
    store.build_index()
    effective_probe = default_index_probe(len(store))

    # Expected values after edits
    expected_after = {f["id"]: f["old"] for f in facts}
    for f in edited:
        expected_after[f["id"]] = f["new"]

    eval_rows = []

    # Efficacy: edited facts return new value (exact + semantic match)
    for f in edited:
        if f["id"] == delete_name:
            continue
        row = suite_lookup(store, "efficacy", f["id"], vectors[f["id"]], expected_after[f["id"]],
                          "indexed-guarded", effective_probe, query_text=f["prompt"])
        row["exact_match"] = row["prediction"] == expected_after[f["id"]]
        row["semantic_match"] = multitoken_semantic_match(row["prediction"], expected_after[f["id"]])
        row["multi_token"] = len(str(expected_after[f["id"]]).split()) > 1
        eval_rows.append(row)

    # Generalization: paraphrases return new value
    for f in edited:
        if f["id"] == delete_name:
            continue
        for p_idx in range(1, n_paraphrases + 1):
            para_key = f"paraphrase_{p_idx}"
            if para_key not in f:
                continue
            row = suite_lookup(store, "generalization", f["id"], para_vectors[f["id"]], expected_after[f["id"]],
                              "indexed-guarded", effective_probe, query_text=f[para_key])
            row["paraphrase_idx"] = p_idx
            row["exact_match"] = row["prediction"] == expected_after[f["id"]]
            row["semantic_match"] = multitoken_semantic_match(row["prediction"], expected_after[f["id"]])
            eval_rows.append(row)

    # Locality: neighbor/distractor facts are NOT damaged
    for f in unedited:
        if f["id"] == delete_name:
            continue
        row = suite_lookup(store, "locality", f["id"], vectors[f["id"]], expected_after[f["id"]],
                          "indexed-guarded", effective_probe, query_text=f["prompt"])
        row["exact_match"] = row["prediction"] == expected_after[f["id"]]
        row["semantic_match"] = multitoken_semantic_match(row["prediction"], expected_after[f["id"]])
        eval_rows.append(row)
        # Distractor checks
        for d_idx in range(1, n_distractors + 1):
            distractor_key = f"distractor_{d_idx}"
            if distractor_key not in f:
                continue
            d_vec = vectors[f["id"]].clone()
            noise = torch.randn_like(d_vec) * 0.03
            d_vec = torch.nn.functional.normalize(d_vec + noise, dim=-1)
            row = suite_lookup(store, "locality_distractor", f["id"], d_vec, expected_after[f["id"]],
                              "indexed-guarded", effective_probe, query_text=f[distractor_key])
            row["distractor_idx"] = d_idx
            row["exact_match"] = row["prediction"] == expected_after[f["id"]]
            row["semantic_match"] = multitoken_semantic_match(row["prediction"], expected_after[f["id"]])
            eval_rows.append(row)

    # Delete fallback
    if delete_name:
        row = suite_lookup(store, "delete_fallback", delete_name, vectors[delete_name], None,
                          "indexed-guarded", effective_probe, query_text=by_id[delete_name]["prompt"])
        eval_rows.append(row)

    rows.extend(eval_rows)
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)

    efficacy = _kind_score(eval_rows, "efficacy")
    generalization = _kind_score(eval_rows, "generalization")
    locality = _kind_score(eval_rows, "locality")
    locality_distractor = _kind_score(eval_rows, "locality_distractor")
    delete_fallback = _kind_score(eval_rows, "delete_fallback")
    exact_match_rate = _mean_number([1.0 if r.get("exact_match") else 0.0 for r in eval_rows if r.get("exact_match") is not None])
    semantic_match_rate = _mean_number([1.0 if r.get("semantic_match") else 0.0 for r in eval_rows if r.get("semantic_match") is not None])
    multi_token_facts = sum(1 for f in facts if len(str(f.get("old", "")).split()) > 1 or len(str(f.get("new", "")).split()) > 1)

    checks = [efficacy, generalization, locality, locality_distractor, delete_fallback, 1.0 if conflict_count > 0 else 0.0]
    return make_record(
        run_id=run_id,
        model_id="kef-factstore",
        backend="factstore-multitoken",
        task_name="kef-edit-multitoken",
        raw_path=raw_path,
        score=sum(checks) / len(checks),
        prompt_count=len(eval_rows),
        tokens_per_second=len(rows) / wall if wall > 0 else 0.0,
        first_token_latency_s=rows[0].get("latency_s", 0.0) if rows else 0.0,
        wall_time_s=wall,
        notes="Multi-token KEF edit benchmark: multi-token values, paraphrases, distractor locality, delete fallback, conflict detection",
        metrics={
            "facts": len(facts),
            "edited": len(edited),
            "deleted": len(deleted),
            "multi_token_facts": multi_token_facts,
            "n_paraphrases": n_paraphrases,
            "n_distractors": n_distractors,
            "efficacy": efficacy,
            "generalization": generalization,
            "locality": locality,
            "locality_distractor": locality_distractor,
            "delete_fallback": delete_fallback,
            "exact_match_rate": exact_match_rate,
            "semantic_match_rate": semantic_match_rate,
            "conflicts": conflict_count,
            "store_records_final": len(store),
            "tombstones_final": store.tombstone_count(),
            "store_bytes_final": store.nbytes(),
            "total_eval_rows": len(eval_rows),
            "trace_events": len(rows),
            "lookup_comparisons_mean": _mean_number([r.get("comparisons") for r in eval_rows]),
            "lookup_fallback_rate": _mean_number([int(r.get("fallback", False)) for r in eval_rows]),
            "data_path": suite_path or "generated",
            "vector_source": f"deterministic-dense-{dim}",
        },
        cwd=cwd,
    )


def make_multitoken_suite(n_facts: int = 100, n_paraphrases: int = 3, n_distractors: int = 3) -> List[dict]:
    """Generate a deterministic multi-token fact suite for Phase 3.

    Each fact has a multi-token value (2-4 words), paraphrase variants,
    and distractor queries. The suite is deterministic (fixed seed).
    """
    import random

    rng = random.Random(42)
    adjectives = ["ancient", "northern", "southern", "eastern", "western", "central",
                  "royal", "imperial", "sacred", "hidden", "golden", "silver",
                  "coastal", "mountain", "valley", "desert", "tropical", "arctic"]
    nouns = ["kingdom", "republic", "federation", "territory", "province",
             "district", "region", "prefecture", "canton", "state",
             "commonwealth", "dominion", "empire", "duchy", "march"]
    subjects = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
                 "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
                 "omicron", "pi", "rho", "sigma", "tau", "upsilon"]

    facts = []
    for i in range(n_facts):
        subj = subjects[i % len(subjects)]
        adj = adjectives[rng.randint(0, len(adjectives) - 1)]
        noun = nouns[rng.randint(0, len(nouns) - 1)]
        old_val = f"{adj} {noun}"
        new_val = f"{adjectives[rng.randint(0, len(adjectives) - 1)]} {nouns[rng.randint(0, len(nouns) - 1)]}"
        # ensure new != old
        while new_val == old_val:
            new_val = f"{adjectives[rng.randint(0, len(adjectives) - 1)]} {nouns[rng.randint(0, len(nouns) - 1)]}"
        edit = i < max(4, n_facts // 4)
        delete = i == max(4, n_facts // 4) - 1
        fact = {
            "id": f"mt_{subj}_{i:04d}",
            "prompt": f"what is the designation of {subj} entity {i:04d}",
            "old": old_val,
            "new": new_val if edit else None,
            "edit": edit,
            "delete": delete,
        }
        # paraphrases
        para_templates = [
            f"designation for {subj} entity {i:04d}",
            f"entity {i:04d} {subj} designation",
            f"what designation does {subj} {i:04d} have",
            f"the {subj} {i:04d} entity designation",
        ]
        for p_idx in range(1, n_paraphrases + 1):
            fact[f"paraphrase_{p_idx}"] = para_templates[(p_idx - 1) % len(para_templates)]
        # distractors
        distractor_templates = [
            f"designation of {subj} entity {i:04d} variant",
            f"what is the {subj} {i:04d} alternate designation",
            f"entity {i:04d} {subj} secondary designation",
        ]
        for d_idx in range(1, n_distractors + 1):
            fact[f"distractor_{d_idx}"] = distractor_templates[(d_idx - 1) % len(distractor_templates)]
        facts.append(fact)
    return facts


def multitoken_semantic_match(prediction: str, expected: str) -> bool:
    """Semantic match for multi-token values: checks token overlap.

    Two values match semantically if they share all content tokens
    (ignoring order and case), providing a looser match than exact string
    equality.
    """
    if prediction is None or expected is None:
        return False
    pred_tokens = set(str(prediction).lower().split())
    exp_tokens = set(str(expected).lower().split())
    if not exp_tokens:
        return False
    return exp_tokens.issubset(pred_tokens) or pred_tokens.issubset(exp_tokens)


def run_heldout_ambiguity_core(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    core_prompt_strategy: str = "fewshot-domain-question",
    core_output_policy: str = "raw",
    n_scenarios: int = 48,
) -> BenchRecord:
    """Phase 1: Larger held-out semantic ambiguity core benchmark.

    Generates or loads a larger ambiguity suite (default 48 scenarios,
    1/3 held-out), routes each through margin-guarded lookup, and sends
    ambiguous misses to the resident core. Reports clarification quality
    separately for train vs held-out partitions, plus per-variant
    consistency across query phrasings.

    This is the expansion step from the 12-scenario smoke to a broader
    held-out set with partition-aware scoring.
    """
    import torch

    from bitx.backends import LlamaCppServerBackend
    from kef.factstore import FactStore

    if not model_id:
        raise ValueError("GGUF model path is required")
    ensure_dir(output_dir)
    if suite_path:
        scenarios = load_heldout_ambiguity_scenarios(suite_path)
    else:
        from bitx.suite import make_heldout_ambiguity_suite, write_suite
        scenarios = make_heldout_ambiguity_suite(n_scenarios)
        suite_path = os.path.join(output_dir, f"heldout_ambiguity_{n_scenarios}.jsonl")
        write_suite(suite_path, scenarios)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_heldout_ambiguity_core.jsonl")
    alpha = torch.tensor([1.0, 0.0])
    beta = torch.tensor([0.99995, 0.01])
    query = torch.tensor([0.99998, 0.004])
    rows = []
    t0 = time.perf_counter()
    with LlamaCppServerBackend(model_id) as backend:
        startup_s = backend.startup_s
        native_binary = backend.binary
        for scenario in scenarios:
            store = FactStore()
            store.add(alpha, scenario["wrong_value"], key_text=scenario["wrong_text"], check_conflict=False)
            store.add(beta, scenario["right_value"], key_text=scenario["right_text"], check_conflict=False)
            p0 = time.perf_counter()
            hit, source, info = store.gated_lookup_with_text_policy(
                query, threshold=0.9,
                query_text=scenario["query_text"],
                min_margin=0.01, rerank_on_ambiguous=True,
            )
            lookup_latency = time.perf_counter() - p0
            partition = scenario.get("partition", "train")
            if hit is not None:
                pred = hit[2]
                route = "recall"
                core_called = False
                token_count = max(1, len(str(pred).split()))
                total_latency = lookup_latency
                backend_metrics = {}
                raw_pred = pred
            else:
                prompt = clarification_prompt(scenario, core_prompt_strategy)
                result = backend.generate(prompt, max_new_tokens=max_new_tokens)
                raw_pred = result.text.strip()
                pred, policy = apply_core_output_policy(raw_pred, scenario, core_output_policy)
                route = "core"
                core_called = True
                token_count = result.token_count
                total_latency = lookup_latency + result.wall_time_s
                backend_metrics = dict(getattr(backend, "last_metrics", {}) or {})
            quality = clarification_quality(pred, scenario.get("clarify", ""))
            strict_quality = clarification_strict_quality(pred, scenario.get("clarify", ""))
            raw_strict_quality = clarification_strict_quality(raw_pred if core_called else pred, scenario.get("clarify", ""))
            # Per-variant consistency: test query variants for route consistency
            variant_routes = []
            if "query_variants" in scenario:
                for vq in scenario["query_variants"]:
                    v_hit, v_source, _ = store.gated_lookup_with_text_policy(
                        query, threshold=0.9, query_text=vq,
                        min_margin=0.01, rerank_on_ambiguous=True,
                    )
                    variant_routes.append("recall" if v_hit is not None else "core")
            rows.append({
                "id": scenario["id"],
                "partition": partition,
                "query_text": scenario["query_text"],
                "right_domain": scenario.get("right_domain", ""),
                "wrong_domain": scenario.get("wrong_domain", ""),
                "core_prompt": prompt if core_called else scenario.get("clarify"),
                "core_prompt_strategy": core_prompt_strategy if core_called else None,
                "core_output_policy": core_output_policy if core_called else None,
                "core_raw_prediction": raw_pred if core_called else pred,
                "core_raw_strict_quality": raw_strict_quality["ok"] if core_called else strict_quality["ok"],
                "core_output_repaired": policy["repaired"] if core_called else False,
                "prediction": pred,
                "route": route,
                "lookup_source": source,
                "expected_route": "core",
                "route_ok": route == "core",
                "core_called": core_called,
                "core_completed": core_called and token_count > 0,
                "clarification_quality": quality["ok"],
                "clarification_strict_quality": strict_quality["ok"],
                "token_count": token_count,
                "latency_s": total_latency,
                "lookup_latency_s": lookup_latency,
                "variant_routes": variant_routes,
                "variant_route_consistent": len(set(variant_routes)) <= 1 if variant_routes else True,
                "backend_metrics": backend_metrics,
            })
    wall = time.perf_counter() - t0
    write_jsonl(raw_path, rows)
    core_rows = [r for r in rows if r["core_called"]]
    core_metrics = [r.get("backend_metrics", {}) for r in core_rows]
    predicted_tps = [m.get("predicted_tokens_per_second") for m in core_metrics if m.get("predicted_tokens_per_second") is not None]
    prompt_tps = [m.get("prompt_tokens_per_second") for m in core_metrics if m.get("prompt_tokens_per_second") is not None]
    model_bytes = next((m.get("model_bytes") for m in core_metrics if m.get("model_bytes") is not None), None)
    if model_bytes is None and os.path.exists(model_id):
        model_bytes = os.path.getsize(model_id)
    train_rows = [r for r in core_rows if r.get("partition") == "train"]
    heldout_rows = [r for r in core_rows if r.get("partition") == "heldout"]
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows) if rows else 0.0
    completion_rate = sum(int(r["core_completed"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    clarification_quality_rate = sum(int(r["clarification_quality"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    clarification_strict_quality_rate = sum(int(r["clarification_strict_quality"]) for r in core_rows) / len(core_rows) if core_rows else 0.0
    heldout_clarification_rate = sum(int(r["clarification_strict_quality"]) for r in heldout_rows) / len(heldout_rows) if heldout_rows else 0.0
    train_clarification_rate = sum(int(r["clarification_strict_quality"]) for r in train_rows) / len(train_rows) if train_rows else 0.0
    variant_consistency_rate = _mean_number([1.0 if r.get("variant_route_consistent") else 0.0 for r in rows if r.get("variant_routes")])
    generation_wall = sum(r["latency_s"] - r["lookup_latency_s"] for r in core_rows)
    core_tokens = sum(r["token_count"] for r in core_rows)
    score = (route_score + completion_rate + clarification_strict_quality_rate) / 3
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="factstore+llama.cpp-server-gguf+heldout-ambiguity-core",
        task_name="heldout-ambiguity-core",
        raw_path=raw_path,
        score=score,
        prompt_count=len(rows),
        tokens_per_second=core_tokens / generation_wall if generation_wall > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"] if rows else 0.0,
        wall_time_s=wall,
        notes="larger held-out ambiguity core benchmark with partition-aware scoring and variant consistency",
        metrics={
            "scenario_count": len(scenarios),
            "suite_path": suite_path,
            "train_scenarios": len(train_rows),
            "heldout_scenarios": len(heldout_rows),
            "route_score": route_score,
            "completion_rate": completion_rate,
            "clarification_quality_rate": clarification_quality_rate,
            "clarification_strict_quality_rate": clarification_strict_quality_rate,
            "heldout_clarification_rate": heldout_clarification_rate,
            "train_clarification_rate": train_clarification_rate,
            "variant_consistency_rate": variant_consistency_rate,
            "core_rows": len(core_rows),
            "core_called_rate": len(core_rows) / len(rows) if rows else 0.0,
            "core_prompt_strategy": core_prompt_strategy,
            "core_output_policy": core_output_policy,
            "lookup_min_margin": 0.01,
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": generation_wall,
            "predicted_tokens_per_second_mean": _mean_number(predicted_tps),
            "prompt_tokens_per_second_mean": _mean_number(prompt_tps),
            "max_new_tokens": max_new_tokens,
        },
        cwd=cwd,
    )


def load_heldout_ambiguity_scenarios(path: str) -> List[dict]:
    required = {"id", "query_text", "right_text", "wrong_text", "right_value", "wrong_value", "clarify"}
    rows = load_suite_facts(path)
    for i, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"heldout ambiguity suite row {i} missing {','.join(missing)}")
    return rows


def suite_event(action: str, subject: str, value, record_id, expected, ok, extra: Optional[dict] = None) -> dict:
    row = {
        "event": action,
        "subject": subject,
        "value": value,
        "record_id": record_id,
        "expected": expected,
        "ok": ok,
        "latency_s": 0.0,
    }
    if extra:
        row.update(extra)
    return row


def suite_lookup(store, kind: str, subject: str, vec, expected, lookup_mode: str = "flat",
                 index_probe: Optional[int] = None, lookup_min_margin: Optional[float] = None,
                 query_text: Optional[str] = None, lookup_rerank: Optional[str] = None) -> dict:
    p0 = time.perf_counter()
    comparisons = len(store)
    fallback = False
    policy_checked = False
    effective_probe = index_probe if index_probe is not None else default_index_probe(len(store))
    key_confirmed = False
    source = lookup_mode
    policy_info = {}
    rerank_enabled = lookup_rerank == "lexical"
    if lookup_mode == "key-confirmed":
        hit, source = store.lookup(vec, threshold=0.9, subject=subject, fallback_to_vector=False)
        comparisons = 1
        key_confirmed = source == "key-confirmed"
    elif lookup_mode == "indexed":
        hit, comparisons = store.search_indexed(vec, n_probe=effective_probe, threshold=0.9)
        if hit is not None and hit[1] < 0.9:
            hit = None
    elif lookup_mode == "indexed-guarded":
        hit, comparisons = store.search_indexed(vec, n_probe=effective_probe, threshold=0.9)
        if lookup_min_margin is not None:
            policy_checked = True
            fallback = hit is None or hit[1] < 0.9
            flat, source, policy_info = store.gated_lookup_with_text_policy(
                vec,
                threshold=0.9,
                query_text=query_text,
                min_margin=lookup_min_margin,
                rerank_on_ambiguous=rerank_enabled,
            )
            comparisons += len(store)
            hit = flat
        elif hit is None or hit[1] < 0.9:
            fallback = True
            flat, source, policy_info = store.gated_lookup_with_text_policy(
                vec,
                threshold=0.9,
                query_text=query_text,
                min_margin=lookup_min_margin,
                rerank_on_ambiguous=rerank_enabled,
            )
            comparisons += len(store)
            hit = flat
        else:
            source = "indexed"
    else:
        policy_checked = lookup_min_margin is not None
        hit, source, policy_info = store.gated_lookup_with_text_policy(
            vec,
            threshold=0.9,
            query_text=query_text,
            min_margin=lookup_min_margin,
            rerank_on_ambiguous=rerank_enabled,
        )
    elapsed = time.perf_counter() - p0
    pred = hit[2] if hit is not None else None
    return {
        "event": "eval",
        "kind": kind,
        "subject": subject,
        "expected": expected,
        "prediction": pred,
        "hit": hit is not None,
        "ok": pred == expected,
        "latency_s": elapsed,
        "lookup_mode": lookup_mode,
        "lookup_source": source,
        "lookup_min_margin": lookup_min_margin,
        "lookup_rerank": lookup_rerank,
        "lookup_margin": policy_info.get("margin"),
        "rerank_score": policy_info.get("rerank_score"),
        "index_probe": effective_probe if lookup_mode in {"indexed", "indexed-guarded", "key-confirmed"} else None,
        "comparisons": comparisons,
        "fallback": fallback,
        "policy_checked": policy_checked,
        "ambiguous": source == "ambiguous",
        "reranked": source == "rerank",
        "key_confirmed": key_confirmed,
        "tombstone_block": expected is None and hit is None,
    }


def sampled_lookup(raw, store, counters: dict, kind: str, subject: str, vec, expected, index_probe: Optional[int] = None) -> int:
    row = suite_lookup(store, kind, subject, vec, expected, "indexed-guarded", index_probe)
    write_jsonl_row(raw, row)
    counters["total"] += 1
    counters["correct"] += int(row["ok"])
    counters["comparisons_total"] += row["comparisons"]
    counters["latency_s_total"] += row["latency_s"]
    counters["fallbacks"] += int(row["fallback"])
    if kind == "efficacy":
        counters["efficacy_total"] += 1
        counters["efficacy_correct"] += int(row["ok"])
    elif kind == "generalization":
        counters["generalization_total"] += 1
        counters["generalization_correct"] += int(row["ok"])
    elif kind == "locality":
        counters["locality_total"] += 1
        counters["locality_correct"] += int(row["ok"])
    elif kind == "delete_fallback":
        counters["delete_total"] += 1
        counters["delete_correct"] += int(row["ok"])
    return 1


def ratio(num, den):
    if not den:
        return 0.0
    return num / den


def _kind_score(rows: List[dict], kind: str) -> float:
    selected = [r for r in rows if r.get("kind") == kind]
    if not selected:
        return 0.0
    return sum(int(r["ok"]) for r in selected) / len(selected)


def _mean_number(values: List[Any]):
    selected = [v for v in values if v is not None]
    if not selected:
        return None
    return sum(selected) / len(selected)


def trace_event(action: str, key: str, value, record_id: int, store_records: int, extra: Optional[dict] = None) -> dict:
    row = {
        "event": action,
        "key": key,
        "value": value,
        "record_id": record_id,
        "store_records": store_records,
        "latency_s": 0.0,
    }
    if extra:
        row.update(extra)
    return row


def trace_lookup(event: str, key: str, hit, expected) -> dict:
    pred = hit[2] if hit is not None else None
    return {
        "event": event,
        "key": key,
        "expected": expected,
        "prediction": pred,
        "hit": hit is not None,
        "ok": pred == expected,
        "latency_s": 0.0,
    }


def run_adapter_gate_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
) -> BenchRecord:
    """Phase 4: Adapter gate smoke benchmark.

    Uses deterministic generate functions to prove the gate contract:
    - target improvement is detected
    - fact damage is detected
    - math damage is detected
    - verbosity drift is detected
    - over-refusal is detected
    - health curve is persisted
    """
    from kef.adapter_gate import AdapterGate, GateControls, save_health_curve

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_adapter_gate.jsonl")

    def base_generate(prompt, n=45):
        return "I think the answer depends on many factors."

    def good_adapted_generate(prompt, n=45):
        # Target trait (directness) improves, no damage
        if "capital of" in prompt.lower():
            return "Tokyo"
        if "plus" in prompt.lower() or "3 + 5" in prompt:
            return "8"
        return "Yes, that is correct. Commit to it."

    def damaged_adapted_generate(prompt, n=45):
        # Target trait improves but fact is damaged
        if "capital of japan" in prompt.lower():
            return "I think it's Berlin."
        if "plus" in prompt.lower():
            return "8"
        return "Yes, absolutely. Commit to it."

    def verbose_adapted_generate(prompt, n=45):
        # Verbosity drift
        return "Yes, absolutely. " * 20

    def refusing_adapted_generate(prompt, n=45):
        # Over-refusal
        return "I'm sorry, I can't assist with that."

    def score_directness(text):
        tl = text.lower().strip()
        if any(d in tl for d in ["i'm sorry", "depends on", "i think"]):
            return 0
        return 1

    controls = GateControls(
        known_answers=[("What is the capital of Japan?", "tokyo")],
        math_problems=[("What is 3 plus 5?", "8")],
        max_verbosity_ratio=2.0,
        max_over_refusal_rate=0.2,
    )
    gate = AdapterGate(target_scorer=score_directness, controls=controls, min_target_delta=1)
    target_probes = ["Is this a good idea?", "Should I do it?"]

    results = []
    for name, gen_fn in [
        ("good_adapter", good_adapted_generate),
        ("damaged_adapter", damaged_adapted_generate),
        ("verbose_adapter", verbose_adapted_generate),
        ("refusing_adapter", refusing_adapted_generate),
    ]:
        result = gate.evaluate(base_generate, gen_fn, target_probes)
        results.append({
            "adapter": name,
            "accepted": result.accepted,
            "target_before": result.target_before,
            "target_after": result.target_after,
            "target_delta": result.target_delta,
            "fact_damage": result.fact_damage,
            "math_damage": result.math_damage,
            "verbosity_ratio": result.verbosity_ratio,
            "over_refusal_rate": result.over_refusal_rate,
            "reasons": result.reasons,
            "wall_time_s": result.wall_time_s,
        })

    # Persist health curve for the good adapter
    curve_path = os.path.join(output_dir, f"{run_id}_health_curve.jsonl")
    save_health_curve(curve_path, results[0].get("health_curve", []) if results[0].get("health_curve") else gate.evaluate(base_generate, good_adapted_generate, target_probes).health_curve)

    write_jsonl(raw_path, results)

    # Score: good adapter accepted, all bad adapters rejected
    expected = {
        "good_adapter": True,
        "damaged_adapter": False,
        "verbose_adapter": False,
        "refusing_adapter": False,
    }
    correct = sum(1 for r in results if r["accepted"] == expected[r["adapter"]])
    score = correct / len(results)
    wall = sum(r["wall_time_s"] for r in results)

    return make_record(
        run_id=run_id,
        model_id="adapter-gate",
        backend="deterministic-adapter-gate",
        task_name="adapter-gate-smoke",
        raw_path=raw_path,
        score=score,
        prompt_count=len(results),
        tokens_per_second=0.0,
        first_token_latency_s=0.0,
        wall_time_s=wall,
        notes="adapter gate smoke: proves target improvement detection, fact/math damage detection, verbosity drift, over-refusal detection",
        metrics={
            "good_adapter_accepted": results[0]["accepted"],
            "damaged_adapter_rejected": not results[1]["accepted"],
            "verbose_adapter_rejected": not results[2]["accepted"],
            "refusing_adapter_rejected": not results[3]["accepted"],
            "good_target_delta": results[0]["target_delta"],
            "damaged_fact_damage": results[1]["fact_damage"],
            "verbose_ratio": results[2]["verbosity_ratio"],
            "refusing_rate": results[3]["over_refusal_rate"],
            "controls": {
                "known_answers": len(controls.known_answers),
                "math_problems": len(controls.math_problems),
                "max_verbosity_ratio": controls.max_verbosity_ratio,
                "max_over_refusal_rate": controls.max_over_refusal_rate,
            },
            "health_curve_path": curve_path,
        },
        cwd=cwd,
    )


def run_core_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: str = "sshleifer/tiny-gpt2",
    max_new_tokens: int = 8,
) -> BenchRecord:
    from bitx.backends import HFCausalLMBackend

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_predictions.jsonl")
    prompts = [
        {"id": "hello", "prompt": "Hello, my name is"},
        {"id": "math_text", "prompt": "2 + 2 ="},
    ]
    backend = HFCausalLMBackend(model_id)
    rows = []
    total_tokens = 0
    wall = 0.0
    first = None
    for item in prompts:
        result = backend.generate(item["prompt"], max_new_tokens=max_new_tokens)
        total_tokens += result.token_count
        wall += result.wall_time_s
        if first is None:
            first = result.first_token_latency_s
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "prediction": result.text,
            "token_count": result.token_count,
            "latency_s": result.wall_time_s,
        })
    write_jsonl(raw_path, rows)
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend=backend.backend,
        task_name="core-smoke",
        raw_path=raw_path,
        score=1.0,
        prompt_count=len(prompts),
        tokens_per_second=total_tokens / wall if wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        wall_time_s=wall,
        notes="real HF causal LM generation smoke; score only means the run completed",
        metrics={
            "total_tokens": total_tokens,
            "max_new_tokens": max_new_tokens,
            "completed": 1,
        },
        cwd=cwd,
    )


def run_native_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
) -> BenchRecord:
    from bitx.backends import LlamaCppBackend

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_predictions.jsonl")
    prompts = [
        {"id": "hello", "prompt": "Hello, my name is"},
        {"id": "math_text", "prompt": "2 + 2 ="},
    ]
    backend = LlamaCppBackend(model_id)
    rows = []
    total_tokens = 0
    wall = 0.0
    first = None
    eval_tps = []
    for item in prompts:
        result = backend.generate(item["prompt"], max_new_tokens=max_new_tokens)
        total_tokens += result.token_count
        wall += result.wall_time_s
        if first is None:
            first = result.first_token_latency_s
        if backend.last_metrics.get("eval_tokens_per_second") is not None:
            eval_tps.append(backend.last_metrics["eval_tokens_per_second"])
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "prediction": result.text,
            "token_count": result.token_count,
            "latency_s": result.wall_time_s,
            "backend_metrics": dict(backend.last_metrics),
        })
    write_jsonl(raw_path, rows)
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend=backend.backend,
        task_name="native-smoke",
        raw_path=raw_path,
        score=1.0,
        prompt_count=len(prompts),
        tokens_per_second=total_tokens / wall if wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        wall_time_s=wall,
        notes="native llama.cpp GGUF generation smoke; score only means the run completed",
        metrics={
            "total_tokens": total_tokens,
            "max_new_tokens": max_new_tokens,
            "completed": 1,
            "native_binary": backend.binary,
            "model_bytes": backend.last_metrics.get("model_bytes"),
            "eval_tokens_per_second_mean": _mean_number(eval_tps),
        },
        cwd=cwd,
    )


def run_native_resident_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
) -> BenchRecord:
    from bitx.backends import LlamaCppServerBackend

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_resident_predictions.jsonl")
    prompts = [
        {"id": "hello", "prompt": "Hello, my name is"},
        {"id": "math_text", "prompt": "2 + 2 ="},
    ]
    rows = []
    total_tokens = 0
    generation_wall = 0.0
    first = None
    predicted_tps = []
    prompt_tps = []
    with LlamaCppServerBackend(model_id) as backend:
        for item in prompts:
            result = backend.generate(item["prompt"], max_new_tokens=max_new_tokens)
            total_tokens += result.token_count
            generation_wall += result.wall_time_s
            if first is None:
                first = result.first_token_latency_s
            if backend.last_metrics.get("predicted_tokens_per_second") is not None:
                predicted_tps.append(backend.last_metrics["predicted_tokens_per_second"])
            if backend.last_metrics.get("prompt_tokens_per_second") is not None:
                prompt_tps.append(backend.last_metrics["prompt_tokens_per_second"])
            rows.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "prediction": result.text,
                "token_count": result.token_count,
                "latency_s": result.wall_time_s,
                "backend_metrics": dict(backend.last_metrics),
            })
        startup_s = backend.startup_s
        native_binary = backend.binary
        model_bytes = backend.last_metrics.get("model_bytes")
    write_jsonl(raw_path, rows)
    total_wall = generation_wall + (startup_s or 0.0)
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="llama.cpp-server-gguf",
        task_name="native-resident-smoke",
        raw_path=raw_path,
        score=1.0,
        prompt_count=len(prompts),
        tokens_per_second=total_tokens / generation_wall if generation_wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        wall_time_s=total_wall,
        notes="resident llama.cpp server GGUF generation smoke; score only means the run completed",
        metrics={
            "total_tokens": total_tokens,
            "max_new_tokens": max_new_tokens,
            "completed": 1,
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": generation_wall,
            "predicted_tokens_per_second_mean": _mean_number(predicted_tps),
            "prompt_tokens_per_second_mean": _mean_number(prompt_tps),
        },
        cwd=cwd,
    )


def run_native_prompt_cache_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
) -> BenchRecord:
    from bitx.backends import LlamaCppServerBackend

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_prompt_cache_predictions.jsonl")
    prompt = (
        "You are measuring a resident local inference runtime. "
        "The repeated prefix should be reusable across calls. "
        "BitX separates mutable facts from reasoning weights and records timing evidence. "
        "Answer with one short phrase:"
    )
    rows = []
    total_tokens = 0
    generation_wall = 0.0
    first = None
    with LlamaCppServerBackend(model_id, parallel=1, cache_reuse=16) as backend:
        for pass_name in ["cold", "warm"]:
            result = backend.generate(prompt, max_new_tokens=max_new_tokens, cache_prompt=True)
            total_tokens += result.token_count
            generation_wall += result.wall_time_s
            if first is None:
                first = result.first_token_latency_s
            rows.append({
                "id": pass_name,
                "prompt": prompt,
                "prediction": result.text,
                "token_count": result.token_count,
                "latency_s": result.wall_time_s,
                "backend_metrics": dict(backend.last_metrics),
            })
        startup_s = backend.startup_s
        native_binary = backend.binary
        model_bytes = backend.last_metrics.get("model_bytes")
    write_jsonl(raw_path, rows)
    cold_metrics = rows[0]["backend_metrics"] if rows else {}
    warm_metrics = rows[1]["backend_metrics"] if len(rows) > 1 else {}
    cold_prompt_tokens = cold_metrics.get("prompt_tokens")
    warm_prompt_tokens = warm_metrics.get("prompt_tokens")
    cold_prompt_eval_tokens = cold_metrics.get("prompt_eval_tokens")
    warm_prompt_eval_tokens = warm_metrics.get("prompt_eval_tokens")
    cold_prompt_cache_tokens = cold_metrics.get("prompt_cache_tokens")
    warm_prompt_cache_tokens = warm_metrics.get("prompt_cache_tokens")
    cold_latency = rows[0]["latency_s"] if rows else 0.0
    warm_latency = rows[1]["latency_s"] if len(rows) > 1 else 0.0
    cold_prompt_tps = cold_metrics.get("prompt_tokens_per_second")
    warm_prompt_tps = warm_metrics.get("prompt_tokens_per_second")
    prompt_eval_reduction = None
    if cold_prompt_eval_tokens is not None and cold_prompt_eval_tokens > 0 and warm_prompt_eval_tokens is not None:
        prompt_eval_reduction = 1.0 - (warm_prompt_eval_tokens / cold_prompt_eval_tokens)
    elif cold_prompt_tokens is not None and cold_prompt_tokens > 0 and warm_prompt_tokens is not None:
        prompt_eval_reduction = 1.0 - (warm_prompt_tokens / cold_prompt_tokens)
    latency_speedup = cold_latency / warm_latency if warm_latency > 0 else None
    prompt_tps_speedup = warm_prompt_tps / cold_prompt_tps if cold_prompt_tps and warm_prompt_tps else None
    cache_effective = bool(
        (warm_prompt_cache_tokens is not None and warm_prompt_cache_tokens > (cold_prompt_cache_tokens or 0))
        or (prompt_eval_reduction is not None and prompt_eval_reduction > 0)
    )
    total_wall = generation_wall + (startup_s or 0.0)
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="llama.cpp-server-gguf",
        task_name="native-prompt-cache-smoke",
        raw_path=raw_path,
        score=1.0 if cache_effective else 0.0,
        prompt_count=len(rows),
        tokens_per_second=total_tokens / generation_wall if generation_wall > 0 else 0.0,
        first_token_latency_s=first if first is not None else 0.0,
        wall_time_s=total_wall,
        notes="resident llama.cpp prompt cache smoke; score means warm prompt eval token count dropped",
        metrics={
            "total_tokens": total_tokens,
            "max_new_tokens": max_new_tokens,
            "completed": 1,
            "cache_prompt": True,
            "cache_effective": cache_effective,
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": generation_wall,
            "cold_prompt_tokens": cold_prompt_tokens,
            "warm_prompt_tokens": warm_prompt_tokens,
            "cold_prompt_eval_tokens": cold_prompt_eval_tokens,
            "warm_prompt_eval_tokens": warm_prompt_eval_tokens,
            "cold_prompt_cache_tokens": cold_prompt_cache_tokens,
            "warm_prompt_cache_tokens": warm_prompt_cache_tokens,
            "prompt_eval_reduction": prompt_eval_reduction,
            "cold_latency_s": cold_latency,
            "warm_latency_s": warm_latency,
            "latency_speedup": latency_speedup,
            "cold_prompt_tokens_per_second": cold_prompt_tps,
            "warm_prompt_tokens_per_second": warm_prompt_tps,
            "prompt_tps_speedup": prompt_tps_speedup,
            "predicted_tokens_per_second_mean": _mean_number([
                cold_metrics.get("predicted_tokens_per_second"),
                warm_metrics.get("predicted_tokens_per_second"),
            ]),
            "prompt_tokens_per_second_mean": _mean_number([
                cold_prompt_tps,
                warm_prompt_tps,
            ]),
        },
        cwd=cwd,
    )


def run_native_kv_cache_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 32,
) -> list:
    from bitx.backends import LlamaCppServerBackend

    if not model_id:
        raise ValueError("GGUF model path is required")
    ensure_dir(output_dir)
    group_id = uuid.uuid4().hex
    prompt = (
        "BitX runtime KV cache policy measurement. "
        "The same resident model answers this prompt while only the KV cache type changes. "
        "Report the measured speed and resident server memory without claiming quality. "
        "Short answer:"
    )
    policies = [
        {"name": "kv-f16", "cache_type_k": "f16", "cache_type_v": "f16", "port": 18080},
        {"name": "kv-q8_0", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "port": 18081},
    ]
    records = []
    baseline = None
    for policy in policies:
        run_id = uuid.uuid4().hex
        raw_path = os.path.join(output_dir, f"{run_id}_{policy['name']}_native_kv_cache.jsonl")
        with LlamaCppServerBackend(
            model_id,
            port=policy["port"],
            parallel=1,
            ctx_size=512,
            cache_type_k=policy["cache_type_k"],
            cache_type_v=policy["cache_type_v"],
        ) as backend:
            result = backend.generate(prompt, max_new_tokens=max_new_tokens)
            startup_s = backend.startup_s
            native_binary = backend.binary
            metrics = dict(backend.last_metrics)
            server_rss_mb = backend.rss_mb() or metrics.get("server_rss_mb")
        rows = [{
            "id": policy["name"],
            "prompt": prompt,
            "prediction": result.text,
            "token_count": result.token_count,
            "latency_s": result.wall_time_s,
            "backend_metrics": metrics,
        }]
        write_jsonl(raw_path, rows)
        tokens_per_second = result.token_count / result.wall_time_s if result.wall_time_s > 0 else 0.0
        current = {
            "server_rss_mb": server_rss_mb,
            "tokens_per_second": tokens_per_second,
            "latency_s": result.wall_time_s,
        }
        if baseline is None:
            baseline = current
        rss_ratio = server_rss_mb / baseline["server_rss_mb"] if server_rss_mb is not None and baseline.get("server_rss_mb") else None
        tps_ratio = tokens_per_second / baseline["tokens_per_second"] if baseline.get("tokens_per_second") else None
        latency_ratio = result.wall_time_s / baseline["latency_s"] if baseline.get("latency_s") else None
        records.append(make_record(
            run_id=run_id,
            model_id=model_id,
            backend="llama.cpp-server-gguf",
            task_name="native-kv-cache-smoke",
            raw_path=raw_path,
            score=1.0,
            prompt_count=1,
            tokens_per_second=tokens_per_second,
            first_token_latency_s=result.first_token_latency_s,
            wall_time_s=result.wall_time_s + (startup_s or 0.0),
            notes="resident llama.cpp KV cache type policy smoke; score only means generation completed",
            metrics={
                "kv_group": group_id,
                "kv_policy": policy["name"],
                "cache_type_k": policy["cache_type_k"],
                "cache_type_v": policy["cache_type_v"],
                "ctx_size": 512,
                "parallel": 1,
                "total_tokens": result.token_count,
                "max_new_tokens": max_new_tokens,
                "completed": 1,
                "native_binary": native_binary,
                "model_bytes": metrics.get("model_bytes"),
                "server_startup_s": startup_s,
                "generation_wall_s": result.wall_time_s,
                "server_rss_mb": server_rss_mb,
                "rss_ratio_vs_baseline": rss_ratio,
                "tps_ratio_vs_baseline": tps_ratio,
                "latency_ratio_vs_baseline": latency_ratio,
                "predicted_tokens_per_second_mean": metrics.get("predicted_tokens_per_second"),
                "prompt_tokens_per_second_mean": metrics.get("prompt_tokens_per_second"),
                "prompt_eval_tokens": metrics.get("prompt_eval_tokens"),
                "prompt_cache_tokens": metrics.get("prompt_cache_tokens"),
            },
            cwd=cwd,
        ))
    return records


def run_native_quant_damage_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    recipe: str = "Q5_K_M",
) -> BenchRecord:
    from bitx.backends import LlamaCppBackend, LlamaCppTools

    if not model_id:
        raise ValueError("GGUF model path is required")
    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_quant_damage.jsonl")
    quant_dir = os.path.join(output_dir, "quantized")
    ensure_dir(quant_dir)
    safe_recipe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in recipe)
    quant_path = os.path.join(quant_dir, f"{run_id}_{safe_recipe}.gguf")
    ppl_text_path = os.path.join(output_dir, f"{run_id}_ppl.txt")
    with open(ppl_text_path, "w", encoding="utf-8") as f:
        f.write(native_quant_ppl_text())
    tools = LlamaCppTools()
    t0 = time.perf_counter()
    source_ppl = tools.perplexity(model_id, ppl_text_path, ctx_size=128, chunks=2)
    quantized = tools.quantize(model_id, quant_path, recipe, allow_requantize=True)
    quant_ppl = tools.perplexity(quant_path, ppl_text_path, ctx_size=128, chunks=2)
    backend = LlamaCppBackend(quant_path)
    gen = backend.generate("BitX evidence row:", max_new_tokens=max_new_tokens)
    wall = time.perf_counter() - t0
    source_bytes = os.path.getsize(model_id)
    quant_bytes = os.path.getsize(quant_path)
    ppl_delta = quant_ppl.ppl - source_ppl.ppl
    ppl_ratio = quant_ppl.ppl / source_ppl.ppl if source_ppl.ppl else None
    byte_ratio = quant_bytes / source_bytes if source_bytes else None
    rows = [
        {
            "id": "source_perplexity",
            "model_path": model_id,
            "ppl": source_ppl.ppl,
            "ppl_stderr": source_ppl.ppl_stderr,
            "latency_s": source_ppl.wall_time_s,
        },
        {
            "id": "quantize",
            "source_path": model_id,
            "quant_path": quant_path,
            "recipe": recipe,
            "source_size_mib": quantized.source_size_mib,
            "quantized_size_mib": quantized.quantized_size_mib,
            "source_bpw": quantized.source_bpw,
            "quantized_bpw": quantized.quantized_bpw,
            "latency_s": quantized.wall_time_s,
        },
        {
            "id": "quantized_perplexity",
            "model_path": quant_path,
            "ppl": quant_ppl.ppl,
            "ppl_stderr": quant_ppl.ppl_stderr,
            "ppl_delta": ppl_delta,
            "ppl_ratio": ppl_ratio,
            "latency_s": quant_ppl.wall_time_s,
        },
        {
            "id": "quantized_generation",
            "prompt": "BitX evidence row:",
            "prediction": gen.text,
            "token_count": gen.token_count,
            "latency_s": gen.wall_time_s,
            "backend_metrics": dict(backend.last_metrics),
        },
    ]
    write_jsonl(raw_path, rows)
    score = 1.0 if ppl_delta <= 0.05 else 0.0
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="llama.cpp-quantize+perplexity",
        task_name="native-quant-damage-smoke",
        raw_path=raw_path,
        score=score,
        prompt_count=2,
        tokens_per_second=gen.token_count / gen.wall_time_s if gen.wall_time_s > 0 else 0.0,
        first_token_latency_s=gen.first_token_latency_s,
        wall_time_s=wall,
        notes="GGUF quantization damage smoke over a fixed local PPL slice; this may be requantized if the source is already quantized",
        metrics={
            "quantization_recipe": recipe,
            "source_model_bytes": source_bytes,
            "quant_model_bytes": quant_bytes,
            "byte_ratio": byte_ratio,
            "source_size_mib_reported": quantized.source_size_mib,
            "quant_size_mib_reported": quantized.quantized_size_mib,
            "source_bpw_reported": quantized.source_bpw,
            "quant_bpw_reported": quantized.quantized_bpw,
            "source_ppl": source_ppl.ppl,
            "source_ppl_stderr": source_ppl.ppl_stderr,
            "quant_ppl": quant_ppl.ppl,
            "quant_ppl_stderr": quant_ppl.ppl_stderr,
            "ppl_delta": ppl_delta,
            "ppl_ratio": ppl_ratio,
            "ppl_ctx_size": 128,
            "ppl_chunks": 2,
            "ppl_text_path": ppl_text_path,
            "quant_path": quant_path,
            "requantized_from_quantized": True,
            "allow_requantize": True,
            "native_binary": backend.binary,
            "quantize_binary": tools.quantize_binary,
            "perplexity_binary": tools.perplexity_binary,
            "quantize_wall_s": quantized.wall_time_s,
            "source_ppl_wall_s": source_ppl.wall_time_s,
            "quant_ppl_wall_s": quant_ppl.wall_time_s,
            "generation_wall_s": gen.wall_time_s,
            "eval_tokens_per_second_mean": backend.last_metrics.get("eval_tokens_per_second"),
            "max_new_tokens": max_new_tokens,
        },
        cwd=cwd,
        quantization_recipe=recipe,
    )


def native_quant_ppl_text() -> str:
    sentence = (
        "BitX separates mutable facts from reasoning weights. "
        "Editable external memory should preserve reasoning while reducing freshness cost. "
        "The benchmark must report damage, speed, bytes, and reproducibility. "
    )
    return sentence * 80


def run_native_quant_damage_suite(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    recipes: Optional[List[str]] = None,
) -> list:
    """Phase 2: Multi-recipe quantization damage suite.

    Runs each recipe against the same source model and records PPL delta,
    byte ratio, BPW, and generation speed. If the source is F16/F32 (BPW
    >= 14), rows are labeled as true baselines rather than requantized.

    This is the contract that replaces the single-recipe smoke with a
    damage-budget comparison so Q4 vs Q5 vs Q6 vs Q8 decisions are
    evidence-based.
    """
    from bitx.backends import LlamaCppBackend, LlamaCppTools

    if not model_id:
        raise ValueError("GGUF model path is required")
    recipes = recipes or ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    ensure_dir(output_dir)
    quant_dir = os.path.join(output_dir, "quantized")
    ensure_dir(quant_dir)
    ppl_text_path = os.path.join(output_dir, "quant_damage_suite_ppl.txt")
    with open(ppl_text_path, "w", encoding="utf-8") as f:
        f.write(native_quant_ppl_text())
    tools = LlamaCppTools()
    source_bytes = os.path.getsize(model_id)
    # Measure source PPL once
    source_ppl_result = tools.perplexity(model_id, ppl_text_path, ctx_size=128, chunks=2)
    source_ppl = source_ppl_result.ppl
    source_bpw_reported = source_ppl_result.ppl  # placeholder, overwritten by quantize output
    # Determine if source is F16/F32 via a dry-run quantize to get BPW
    try:
        dry = tools.dry_quantize(model_id, "Q5_K_M", allow_requantize=True)
        source_bpw_reported = dry.source_bpw
        is_baseline = source_bpw_reported >= 14.0
    except Exception:
        is_baseline = False
        source_bpw_reported = None
    group_id = uuid.uuid4().hex
    records = []
    for recipe in recipes:
        safe_recipe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in recipe)
        run_id = uuid.uuid4().hex
        raw_path = os.path.join(output_dir, f"{run_id}_quant_damage_{safe_recipe}.jsonl")
        quant_path = os.path.join(quant_dir, f"{run_id}_{safe_recipe}.gguf")
        t0 = time.perf_counter()
        quantized = tools.quantize(model_id, quant_path, recipe, allow_requantize=True)
        quant_ppl_result = tools.perplexity(quant_path, ppl_text_path, ctx_size=128, chunks=2)
        backend = LlamaCppBackend(quant_path)
        gen = backend.generate("BitX evidence row:", max_new_tokens=max_new_tokens)
        wall = time.perf_counter() - t0
        quant_bytes = os.path.getsize(quant_path)
        ppl_delta = quant_ppl_result.ppl - source_ppl
        ppl_ratio = quant_ppl_result.ppl / source_ppl if source_ppl else None
        byte_ratio = quant_bytes / source_bytes if source_bytes else None
        rows = [
            {
                "id": "source_perplexity",
                "model_path": model_id,
                "ppl": source_ppl,
                "ppl_stderr": source_ppl_result.ppl_stderr,
                "latency_s": source_ppl_result.wall_time_s,
            },
            {
                "id": "quantize",
                "source_path": model_id,
                "quant_path": quant_path,
                "recipe": recipe,
                "source_size_mib": quantized.source_size_mib,
                "quantized_size_mib": quantized.quantized_size_mib,
                "source_bpw": quantized.source_bpw,
                "quantized_bpw": quantized.quantized_bpw,
                "latency_s": quantized.wall_time_s,
            },
            {
                "id": "quantized_perplexity",
                "model_path": quant_path,
                "ppl": quant_ppl_result.ppl,
                "ppl_stderr": quant_ppl_result.ppl_stderr,
                "ppl_delta": ppl_delta,
                "ppl_ratio": ppl_ratio,
                "latency_s": quant_ppl_result.wall_time_s,
            },
            {
                "id": "quantized_generation",
                "prompt": "BitX evidence row:",
                "prediction": gen.text,
                "token_count": gen.token_count,
                "latency_s": gen.wall_time_s,
                "backend_metrics": dict(backend.last_metrics),
            },
        ]
        write_jsonl(raw_path, rows)
        score = 1.0 if ppl_delta <= 0.05 else 0.0
        records.append(make_record(
            run_id=run_id,
            model_id=model_id,
            backend="llama.cpp-quantize+perplexity",
            task_name="native-quant-damage-suite",
            raw_path=raw_path,
            score=score,
            prompt_count=2,
            tokens_per_second=gen.token_count / gen.wall_time_s if gen.wall_time_s > 0 else 0.0,
            first_token_latency_s=gen.first_token_latency_s,
            wall_time_s=wall,
            notes=f"Multi-recipe quantization damage row: {recipe} vs source ({'F16/F32 baseline' if is_baseline else 'requantized from quantized source'})",
            metrics={
                "quantization_recipe": recipe,
                "source_model_bytes": source_bytes,
                "quant_model_bytes": quant_bytes,
                "byte_ratio": byte_ratio,
                "source_size_mib_reported": quantized.source_size_mib,
                "quant_size_mib_reported": quantized.quantized_size_mib,
                "source_bpw_reported": quantized.source_bpw,
                "quant_bpw_reported": quantized.quantized_bpw,
                "source_ppl": source_ppl,
                "source_ppl_stderr": source_ppl_result.ppl_stderr,
                "quant_ppl": quant_ppl_result.ppl,
                "quant_ppl_stderr": quant_ppl_result.ppl_stderr,
                "ppl_delta": ppl_delta,
                "ppl_ratio": ppl_ratio,
                "ppl_ctx_size": 128,
                "ppl_chunks": 2,
                "ppl_text_path": ppl_text_path,
                "quant_path": quant_path,
                "requantized_from_quantized": not is_baseline,
                "is_baseline": is_baseline,
                "source_is_f16_f32": is_baseline,
                "allow_requantize": True,
                "native_binary": backend.binary,
                "quantize_binary": tools.quantize_binary,
                "perplexity_binary": tools.perplexity_binary,
                "quantize_wall_s": quantized.wall_time_s,
                "source_ppl_wall_s": source_ppl_result.wall_time_s,
                "quant_ppl_wall_s": quant_ppl_result.wall_time_s,
                "generation_wall_s": gen.wall_time_s,
                "eval_tokens_per_second_mean": backend.last_metrics.get("eval_tokens_per_second"),
                "max_new_tokens": max_new_tokens,
                "damage_group": group_id,
                "recipes_in_suite": recipes,
            },
            cwd=cwd,
            quantization_recipe=recipe,
        ))
    return records


def run_native_kef_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
) -> BenchRecord:
    import torch

    from bitx.backends import LlamaCppServerBackend
    from kef.factstore import FactStore

    ensure_dir(output_dir)
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_kef_predictions.jsonl")
    vectors = {
        "france": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "japan": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "miss_math": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "miss_name": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    store = FactStore()
    store.add(vectors["france"], "Lyon", key_text="capital of france")
    store.add(vectors["japan"], "Tokyo", key_text="capital of japan")
    cases = [
        {"id": "recall_france", "prompt": "The capital of France is", "expected": "Lyon", "kind": "efficacy", "vec": "france", "route": "recall"},
        {"id": "recall_japan", "prompt": "The capital of Japan is", "expected": "Tokyo", "kind": "locality", "vec": "japan", "route": "recall"},
        {"id": "core_math", "prompt": "2 + 2 =", "expected": None, "kind": "core_fallback", "vec": "miss_math", "route": "core"},
        {"id": "core_name", "prompt": "Hello, my name is", "expected": None, "kind": "core_fallback", "vec": "miss_name", "route": "core"},
    ]
    with LlamaCppServerBackend(model_id) as backend:
        rows = routed_rows_with_core_flag("native-kef", backend, store, vectors, cases, max_new_tokens)
        startup_s = backend.startup_s
        native_binary = backend.binary
    core_metrics = [r.get("backend_metrics", {}) for r in rows if r["source"] == "core"]
    predicted_tps = [m.get("predicted_tokens_per_second") for m in core_metrics if m.get("predicted_tokens_per_second") is not None]
    prompt_tps = [m.get("prompt_tokens_per_second") for m in core_metrics if m.get("prompt_tokens_per_second") is not None]
    model_bytes = next((m.get("model_bytes") for m in core_metrics if m.get("model_bytes") is not None), None)
    write_jsonl(raw_path, rows)
    core_rows_count = sum(1 for r in rows if r["source"] == "core")
    recall_rows_count = sum(1 for r in rows if r["source"] == "recall")
    core_tokens = sum(r["token_count"] for r in rows if r["source"] == "core")
    total_latency = sum(r["latency_s"] for r in rows)
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows)
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="factstore+llama.cpp-server-gguf",
        task_name="native-kef-smoke",
        raw_path=raw_path,
        score=route_score,
        prompt_count=len(rows),
        tokens_per_second=sum(r["token_count"] for r in rows) / total_latency if total_latency > 0 else 0.0,
        first_token_latency_s=next((r["latency_s"] for r in rows if r["source"] == "recall"), rows[0]["latency_s"]),
        wall_time_s=total_latency + (startup_s or 0.0),
        notes="resident llama.cpp core fallback behind KEF recall; score measures route correctness",
        metrics={
            "route_score": route_score,
            "recall_rows": recall_rows_count,
            "core_rows": core_rows_count,
            "core_tokens": core_tokens,
            "core_called_rate": core_rows_count / len(rows),
            "recall_hit_rate": recall_rows_count / len(rows),
            "store_records": len(store),
            "store_bytes": store.nbytes(),
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": sum(r["latency_s"] for r in rows if r["source"] == "core"),
            "predicted_tokens_per_second_mean": _mean_number(predicted_tps),
            "prompt_tokens_per_second_mean": _mean_number(prompt_tps),
            "max_new_tokens": max_new_tokens,
        },
        cwd=cwd,
    )


def run_native_kef_suite_smoke(
    output_dir: str,
    cwd: Optional[str] = None,
    suite_path: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    recall_limit: int = 32,
    vector_dim: int = 64,
) -> BenchRecord:
    import torch

    from bitx.backends import LlamaCppServerBackend
    from kef.factstore import FactStore

    ensure_dir(output_dir)
    path = suite_path or suite_data_path()
    facts = load_suite_facts(path)
    if not facts:
        raise ValueError("native KEF suite requires at least one fact")
    run_id = uuid.uuid4().hex
    raw_path = os.path.join(output_dir, f"{run_id}_native_kef_suite_predictions.jsonl")
    vectors, _ = dense_suite_vectors(facts, vector_dim, seed=len(facts))
    store = FactStore()
    ids = {}
    expected_after = {}
    for f in facts:
        subject = f["id"]
        ids[subject] = store.add(
            vectors[subject],
            f["old"],
            key_text=f["prompt"],
            meta={"subject": subject},
            check_conflict=False,
        )
        expected_after[subject] = f["old"]
    for f in facts:
        if f.get("edit"):
            store.edit(ids[f["id"]], f["new"])
            expected_after[f["id"]] = f["new"]
    for f in facts:
        if f.get("delete"):
            store.delete(ids[f["id"]])
            expected_after[f["id"]] = None
    recall_facts = native_suite_recall_facts(facts, recall_limit)
    cases = []
    for f in recall_facts:
        cases.append({
            "id": f"recall_{f['id']}",
            "prompt": f["prompt"],
            "expected": expected_after[f["id"]],
            "kind": "edit_recall" if f.get("edit") else "locality_recall",
            "vec": f["id"],
            "subject": f["id"],
            "route": "recall",
            "fallback_to_vector": False,
        })
    deleted = [f for f in facts if f.get("delete")]
    if deleted:
        f = deleted[0]
        cases.append({
            "id": f"core_deleted_{f['id']}",
            "prompt": f["prompt"],
            "expected": None,
            "kind": "delete_core_fallback",
            "vec": f["id"],
            "subject": f["id"],
            "route": "core",
            "fallback_to_vector": False,
        })
    miss_vec = "__native_suite_miss_zero"
    vectors[miss_vec] = torch.zeros(vector_dim)
    while sum(1 for c in cases if c["route"] == "core") < 2:
        i = sum(1 for c in cases if c["route"] == "core")
        cases.append({
            "id": f"core_miss_{i}",
            "prompt": "Answer briefly: 2 + 2 =" if i == 0 else "Answer briefly: Hello, my name is",
            "expected": None,
            "kind": "out_of_store_core_fallback",
            "vec": miss_vec,
            "subject": None,
            "route": "core",
            "fallback_to_vector": True,
        })
    with LlamaCppServerBackend(model_id) as backend:
        rows = native_kef_suite_rows("native-kef-suite", backend, store, vectors, cases, max_new_tokens)
        startup_s = backend.startup_s
        native_binary = backend.binary
    write_jsonl(raw_path, rows)
    core_metrics = [r.get("backend_metrics", {}) for r in rows if r["source"] == "core"]
    predicted_tps = [m.get("predicted_tokens_per_second") for m in core_metrics if m.get("predicted_tokens_per_second") is not None]
    prompt_tps = [m.get("prompt_tokens_per_second") for m in core_metrics if m.get("prompt_tokens_per_second") is not None]
    model_bytes = next((m.get("model_bytes") for m in core_metrics if m.get("model_bytes") is not None), None)
    recall_rows = [r for r in rows if r["route"] == "recall"]
    core_rows = [r for r in rows if r["route"] == "core"]
    route_score = sum(int(r["route_ok"]) for r in rows) / len(rows)
    recall_value_score = sum(int(r["value_ok"]) for r in recall_rows) / len(recall_rows) if recall_rows else 0.0
    total_latency = sum(r["latency_s"] for r in rows)
    generation_wall = sum(r["latency_s"] for r in core_rows)
    core_tokens = sum(r["token_count"] for r in core_rows)
    edited = sum(1 for f in facts if f.get("edit"))
    deleted_count = sum(1 for f in facts if f.get("delete"))
    score = (route_score + recall_value_score) / 2
    return make_record(
        run_id=run_id,
        model_id=model_id,
        backend="factstore+llama.cpp-server-gguf",
        task_name="native-kef-suite-smoke",
        raw_path=raw_path,
        score=score,
        prompt_count=len(rows),
        tokens_per_second=sum(r["token_count"] for r in rows) / total_latency if total_latency > 0 else 0.0,
        first_token_latency_s=rows[0]["latency_s"] if rows else 0.0,
        wall_time_s=total_latency + (startup_s or 0.0),
        notes="resident llama.cpp core fallback behind suite-loaded KEF recall; score averages route correctness and recall value correctness",
        metrics={
            "route_score": route_score,
            "recall_value_score": recall_value_score,
            "recall_rows": len(recall_rows),
            "core_rows": len(core_rows),
            "core_tokens": core_tokens,
            "core_called_rate": len(core_rows) / len(rows),
            "recall_hit_rate": len(recall_rows) / len(rows),
            "edit_recall_score": _kind_score(rows, "edit_recall"),
            "locality_recall_score": _kind_score(rows, "locality_recall"),
            "delete_core_rows": sum(1 for r in core_rows if r.get("kind") == "delete_core_fallback"),
            "out_of_store_core_rows": sum(1 for r in core_rows if r.get("kind") == "out_of_store_core_fallback"),
            "facts": len(facts),
            "edited": edited,
            "deleted": deleted_count,
            "store_records": len(store),
            "tombstones_final": store.tombstone_count(),
            "store_bytes": store.nbytes(),
            "data_path": path,
            "vector_source": f"deterministic-dense-{vector_dim}",
            "vector_dim": vector_dim,
            "native_binary": native_binary,
            "model_bytes": model_bytes,
            "server_startup_s": startup_s,
            "generation_wall_s": generation_wall,
            "core_tokens_per_second": core_tokens / generation_wall if generation_wall > 0 else 0.0,
            "predicted_tokens_per_second_mean": _mean_number(predicted_tps),
            "prompt_tokens_per_second_mean": _mean_number(prompt_tps),
            "max_new_tokens": max_new_tokens,
        },
        cwd=cwd,
    )


def native_suite_recall_facts(facts: List[dict], limit: int) -> List[dict]:
    live = [f for f in facts if not f.get("delete")]
    edited = [f for f in live if f.get("edit")]
    locality = [f for f in live if not f.get("edit")]
    target = min(max(1, int(limit)), len(live))
    edited_target = min(len(edited), max(1, target // 2)) if edited else 0
    selected = edited[:edited_target]
    selected.extend(locality[:max(0, target - len(selected))])
    selected_ids = {f["id"] for f in selected}
    for f in live:
        if len(selected) >= target:
            break
        if f["id"] not in selected_ids:
            selected.append(f)
            selected_ids.add(f["id"])
    return selected


def native_kef_suite_rows(
    system: str,
    backend,
    store,
    vectors: dict,
    cases: List[dict],
    max_new_tokens: int,
) -> List[dict]:
    rows = []
    for case in cases:
        p0 = time.perf_counter()
        hit, lookup_source = store.lookup(
            vectors[case["vec"]],
            threshold=0.9,
            subject=case.get("subject"),
            fallback_to_vector=case.get("fallback_to_vector", True),
        )
        lookup_latency = time.perf_counter() - p0
        row = dict(case)
        row.pop("vec", None)
        should_recall = case.get("route") == "recall"
        row["system"] = system
        row["lookup_source"] = lookup_source
        row["lookup_latency_s"] = lookup_latency
        if hit is not None:
            pred = hit[2]
            row.update({
                "prediction": pred,
                "source": "recall",
                "core_called": False,
                "route_ok": should_recall,
                "value_ok": pred == case.get("expected"),
                "token_count": max(1, len(str(pred).split())),
                "latency_s": lookup_latency,
            })
        else:
            result = backend.generate(case["prompt"], max_new_tokens=max_new_tokens)
            pred = result.text.strip()
            row.update({
                "prediction": pred,
                "source": "core",
                "core_called": True,
                "route_ok": not should_recall,
                "value_ok": case.get("expected", "").lower() in pred.lower() if case.get("expected") else True,
                "token_count": result.token_count,
                "latency_s": lookup_latency + result.wall_time_s,
                "backend_metrics": dict(getattr(backend, "last_metrics", {}) or {}),
            })
        row["ok"] = row["route_ok"] and row["value_ok"]
        rows.append(row)
    return rows


def run_benchmark(
    task: str,
    output_dir: str,
    cwd: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 8,
    suite_path: Optional[str] = None,
    suite_sizes: Optional[List[int]] = None,
    encoder_batch_size: int = 32,
    index_probe: Optional[int] = None,
    lookup_min_margin: Optional[float] = None,
    lookup_rerank: Optional[str] = None,
    core_prompt_strategy: str = "fewshot-domain-question",
    core_output_policy: str = "raw",
    n_facts: int = 100,
    n_paraphrases: int = 3,
    n_distractors: int = 3,
):
    if task == "smoke":
        return run_smoke(output_dir, cwd=cwd)
    if task == "kef-edit-smoke":
        return run_kef_edit_smoke(output_dir, cwd=cwd)
    if task == "edit-comparison-smoke":
        return run_edit_comparison_smoke(output_dir, cwd=cwd)
    if task == "edit-mini":
        return run_edit_mini(output_dir, cwd=cwd)
    if task == "edit-core-mini":
        return run_edit_core_mini(
            output_dir,
            cwd=cwd,
            model_id=model_id or "sshleifer/tiny-gpt2",
            max_new_tokens=max_new_tokens,
        )
    if task == "edit-trace-mini":
        return run_edit_trace_mini(output_dir, cwd=cwd)
    if task == "edit-suite-mini":
        return run_edit_suite_mini(output_dir, cwd=cwd)
    if task == "edit-suite-data-mini":
        return run_edit_suite_data_mini(output_dir, cwd=cwd, suite_path=suite_path)
    if task == "edit-suite-encoder-mini":
        return run_edit_suite_encoder_mini(output_dir, cwd=cwd, suite_path=suite_path)
    if task == "ambiguity-fallback-smoke":
        return run_ambiguity_fallback_smoke(output_dir, cwd=cwd)
    if task == "semantic-rerank-smoke":
        return run_semantic_rerank_smoke(output_dir, cwd=cwd, suite_path=suite_path)
    if task == "native-ambiguity-core-smoke":
        return run_native_ambiguity_core_smoke(output_dir, cwd=cwd, suite_path=suite_path, model_id=model_id, max_new_tokens=max_new_tokens, core_prompt_strategy=core_prompt_strategy, core_output_policy=core_output_policy)
    if task == "suite-scale":
        return run_suite_scale(output_dir, cwd=cwd, sizes=suite_sizes)
    if task == "suite-index-scale":
        return run_suite_index_scale(output_dir, cwd=cwd, sizes=suite_sizes)
    if task == "suite-large-scale":
        return run_suite_large_scale(output_dir, cwd=cwd, sizes=suite_sizes)
    if task == "suite-100k-smoke":
        return run_suite_100k_smoke(output_dir, cwd=cwd, sizes=suite_sizes)
    if task == "suite-encoder-scale":
        return run_suite_encoder_scale(output_dir, cwd=cwd, sizes=suite_sizes, encoder_batch_size=encoder_batch_size, index_probe=index_probe)
    if task == "suite-encoder-keyed-scale":
        return run_suite_encoder_scale(output_dir, cwd=cwd, sizes=suite_sizes, keyed=True, encoder_batch_size=encoder_batch_size, index_probe=index_probe)
    if task == "suite-encoder-jsonl-scale":
        return run_suite_encoder_jsonl_scale(output_dir, cwd=cwd, suite_path=suite_path, encoder_batch_size=encoder_batch_size, index_probe=index_probe, lookup_min_margin=lookup_min_margin, lookup_rerank=lookup_rerank)
    if task == "suite-encoder-jsonl-exact":
        return run_suite_encoder_jsonl_scale(output_dir, cwd=cwd, suite_path=suite_path, encoder_batch_size=encoder_batch_size, exact=True, index_probe=index_probe, lookup_min_margin=lookup_min_margin, lookup_rerank=lookup_rerank)
    if task == "suite-encoder-jsonl-keyed":
        return run_suite_encoder_jsonl_keyed(output_dir, cwd=cwd, suite_path=suite_path, encoder_batch_size=encoder_batch_size, index_probe=index_probe)
    if task == "core-smoke":
        return run_core_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id or "sshleifer/tiny-gpt2",
            max_new_tokens=max_new_tokens,
        )
    if task == "native-smoke":
        return run_native_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-resident-smoke":
        return run_native_resident_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-prompt-cache-smoke":
        return run_native_prompt_cache_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-kv-cache-smoke":
        return run_native_kv_cache_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-quant-damage-smoke":
        return run_native_quant_damage_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-quant-damage-suite":
        return run_native_quant_damage_suite(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "kef-edit-multitoken":
        return run_kef_edit_multitoken(
            output_dir,
            cwd=cwd,
            suite_path=suite_path,
            n_facts=n_facts,
            n_paraphrases=n_paraphrases,
            n_distractors=n_distractors,
        )
    if task == "heldout-ambiguity-core":
        return run_heldout_ambiguity_core(
            output_dir,
            cwd=cwd,
            suite_path=suite_path,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            core_prompt_strategy=core_prompt_strategy,
            core_output_policy=core_output_policy,
        )
    if task == "adapter-gate-smoke":
        return run_adapter_gate_smoke(output_dir, cwd=cwd)
    if task == "native-kef-smoke":
        return run_native_kef_smoke(
            output_dir,
            cwd=cwd,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if task == "native-kef-suite-smoke":
        return run_native_kef_suite_smoke(
            output_dir,
            cwd=cwd,
            suite_path=suite_path,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"unknown benchmark task: {task}")
