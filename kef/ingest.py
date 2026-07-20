"""
Ingest — turn ANY question/answer dataset into a usable knowledge base.

This is the inverse of export.py and arguably the more useful direction: take an
off-the-shelf dataset (trivia, domain FAQ, product docs as Q/A, ...) and make it
queryable knowledge for ANY model, with NO training.

Handles:
  - JSONL / CSV / list-of-dicts / HF datasets
  - flexible field names (auto-detects common prompt/answer column pairs)
  - batch encoding for speed
The result is a FactStore you can attach to a KEFModel or save into a bundle.
"""
import csv
import json
from typing import List, Dict, Optional

from kef.factstore import FactStore

# common column-name pairs seen in real datasets
PROMPT_KEYS = ["prompt", "question", "instruction", "query", "input", "q", "title"]
ANSWER_KEYS = ["answer", "output", "response", "completion", "target", "a", "text"]


def _autodetect(row: Dict) -> Optional[tuple]:
    """Pick (prompt_field, answer_field) from a sample row."""
    keys = {k.lower(): k for k in row.keys()}
    p = next((keys[k] for k in PROMPT_KEYS if k in keys), None)
    a = next((keys[k] for k in ANSWER_KEYS if k in keys), None)
    return (p, a) if p and a else None


def _read_rows(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if path.endswith(".csv"):
        with open(path) as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported file type: {path}")


def ingest_rows(rows: List[Dict], encoder, store: FactStore = None,
                prompt_field: str = None, answer_field: str = None,
                batch_size: int = 64) -> FactStore:
    """Build/extend a FactStore from in-memory rows."""
    store = store or FactStore()
    if not rows:
        return store
    if prompt_field is None or answer_field is None:
        det = _autodetect(rows[0])
        if det is None:
            raise ValueError(
                f"could not auto-detect prompt/answer fields from "
                f"{list(rows[0].keys())}; pass prompt_field/answer_field")
        prompt_field, answer_field = det
    # batch-encode prompts for speed
    prompts = [str(r[prompt_field]) for r in rows]
    answers = [r[answer_field] for r in rows]
    metas = [r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {} for r in rows]
    for idx, row in enumerate(rows):
        for key in ("subject", "id"):
            if key in row and key not in metas[idx]:
                metas[idx][key] = row[key]
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i:i + batch_size]
        vecs = encoder.encode_batch(chunk_p)
        for j, p in enumerate(chunk_p):
            store.add(vecs[j], value=answers[i + j], key_text=p,
                      meta=metas[i + j], check_conflict=False)
    return store, prompt_field, answer_field


def ingest_file(path: str, encoder, store: FactStore = None,
                prompt_field: str = None, answer_field: str = None) -> FactStore:
    rows = _read_rows(path)
    return ingest_rows(rows, encoder, store, prompt_field, answer_field)


def ingest_hf(dataset_name: str, encoder, split: str = "train",
              limit: int = None, prompt_field: str = None,
              answer_field: str = None, **load_kwargs):
    """Ingest directly from a HuggingFace dataset by name."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [dict(r) for r in ds]
    return ingest_rows(rows, encoder, store=None,
                       prompt_field=prompt_field, answer_field=answer_field)
