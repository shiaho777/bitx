"""
Export a FactStore as a standard DATASET.

A stored fact is just (key_text -> value) plus a regenerable embedding. So the
knowledge IS a question/answer table and exports cleanly to common formats:

  - JSONL  : one {"prompt","answer","meta"} per line (HF/datasets friendly)
  - CSV    : prompt,answer columns
  - HF     : a datasets.Dataset (if `datasets` is installed)
  - alpaca : [{"instruction","input","output"}] for instruction-tuning reuse

The embedding (key_vec) is NOT exported by default — it's derivable from the
prompt with the encoder, and keeping the dataset model-agnostic is the point.
"""
import csv
import json
from typing import List, Dict

from kef.factstore import FactStore


def _records_as_rows(store: FactStore) -> List[Dict]:
    rows = []
    for r in store._records:
        val = r.value
        # values may be token ids or strings; keep as-is but stringify ids note
        rows.append({"prompt": r.key_text, "answer": val, "meta": r.meta})
    return rows


def to_jsonl(store: FactStore, path: str) -> int:
    rows = _records_as_rows(store)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def to_csv(store: FactStore, path: str) -> int:
    rows = _records_as_rows(store)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "answer"])
        for row in rows:
            w.writerow([row["prompt"], row["answer"]])
    return len(rows)


def to_alpaca(store: FactStore, path: str) -> int:
    """Instruction-tuning format, so the extracted knowledge can be reused to
    FINETUNE another model if someone wants the parametric route."""
    rows = [{"instruction": r.key_text, "input": "",
             "output": str(r.value).strip()} for r in store._records]
    with open(path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return len(rows)


def to_hf_dataset(store: FactStore):
    """Return a datasets.Dataset (requires `datasets`)."""
    from datasets import Dataset
    rows = _records_as_rows(store)
    return Dataset.from_list([{"prompt": r["prompt"], "answer": r["answer"]}
                              for r in rows])


def from_jsonl(path: str, encoder, store: FactStore = None) -> FactStore:
    """Inverse: build a FactStore from a JSONL dataset, re-embedding prompts.
    Lets anyone turn a plain Q/A dataset INTO a knowledge base."""
    store = store or FactStore()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            store.add(encoder.encode(row["prompt"]), value=row["answer"],
                      key_text=row["prompt"], meta=row.get("meta", {}),
                      check_conflict=False)
    return store
