"""Spell-only pure-weight stage: faithful orthography expansion + hard stop."""

from __future__ import annotations

from kef.paths import default_model, repo_root, result_path

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

BANNED = {
    "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
    "boysenberry", "huckleberry", "elderberry", "gooseberry",
}

POOL = [
    "apple", "banana", "orange", "grape", "melon", "peach", "mango", "lemon",
    "cherry", "papaya", "coconut", "tomato", "potato", "carrot", "pepper",
    "onion", "garlic", "ginger", "table", "chair", "window", "house", "school",
    "bridge", "river", "forest", "ocean", "island", "python", "rust", "swift",
    "happy", "brave", "quiet", "yellow", "elephant", "penguin", "rabbit",
    "keyboard", "monitor", "server", "matrix", "vector", "buffer", "bicycle",
    "morning", "winter", "summer", "coffee", "butter", "cheese", "friend",
    "family", "teacher", "system", "process", "thread", "cache", "kernel",
    "quantum", "proton", "photon", "crystal", "language", "network", "socket",
    "diamond", "travel", "ticket", "alpha", "beta", "gamma", "delta", "omega",
    "success", "address", "balloon", "level", "civic", "radar", "rotor",
    "kayak", "refer", "trains", "letters", "hello", "world", "count", "letter",
    "character", "sense", "cat", "dog", "sun", "moon", "tree", "book", "pen",
    "cup", "box", "key", "red", "blue", "green", "black", "white", "small",
    "quick", "code", "data", "model", "token", "layer", "gate", "score",
    "parallel", "google", "pizza", "beekeeper", "mississippi", "bookkeeper",
]

POOL = [w for w in POOL if w.isalpha()]

TEMPLATES = [
    "Spell the word '{word}' one character at a time.",
    "Write '{word}' letter by letter with indices.",
    "Expand '{word}' into indexed characters. Use LEN first, then exactly LEN lines, then STOP.",
    "Orthography of '{word}': list every character with positions.",
    "Break '{word}' into characters. Format: LEN=n then n lines of i:char then STOP.",
    "Spell '{word}'. Output EXACTLY:\nLEN=n\n1:c\n...\nn:c\nSTOP\nNothing after STOP.",
    "List characters of '{word}' with positions. After the last character, write STOP and end.",
]

REHEARSAL = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is 17 + 25?", "42."),
    ("What is 9 times 6?", "54."),
    ("What is 12 + 8?", "20."),
]


@dataclass
class Sample:
    question: str
    answer: str
    word: str
    kind: str


def teacher_spell(word: str) -> str:
    lines = [f"LEN={len(word)}"]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append("STOP")
    return "\n".join(lines)


def teacher_partial(word: str, keep: int) -> str:
    lines = [f"LEN={len(word)}"]
    for i, c in enumerate(word[:keep]):
        lines.append(f"{i+1}:{c}")
    return "\n".join(lines)


def parse_spell(pred: str) -> Dict:
    m = re.search(r"LEN\s*=\s*(\d+)", pred, flags=re.I)
    claimed = int(m.group(1)) if m else None
    body = pred
    if re.search(r"(?m)^\s*STOP\s*$", pred, flags=re.I):
        body = re.split(r"(?m)^\s*STOP\s*$", pred, maxsplit=1, flags=re.I)[0]
    pairs = re.findall(r"(?m)^(\d+)\s*:\s*([A-Za-z])\s*$", body)
    if not pairs:
        pairs = re.findall(r"(?m)(\d+)\s*:\s*([A-Za-z])", body)
    ordered = []
    seen = set()
    for idx_s, ch in pairs:
        idx = int(idx_s)
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append((idx, ch))
    ordered.sort(key=lambda x: x[0])
    chars = "".join(c for _, c in ordered)
    n_lines = len(ordered)
    has_stop = bool(re.search(r"(?m)^\s*STOP\s*$", pred, flags=re.I))
    return {
        "claimed": claimed,
        "chars": chars,
        "n_lines": n_lines,
        "has_stop": has_stop,
        "pairs": ordered,
    }


def core_string(pairs: Sequence[Tuple[int, str]], word: str) -> str:
    n = len(word)
    m = {i: c for i, c in pairs if 1 <= i <= n}
    return "".join(m.get(i + 1, "") for i in range(n))


def score_spell(pred: str, word: str) -> Dict[str, float]:
    p = parse_spell(pred)
    gold = word.lower()
    full_chars = p["chars"].lower()
    core = core_string(p["pairs"], word).lower()
    exact = 1.0 if full_chars == gold else 0.0
    core_exact = 1.0 if core == gold else 0.0
    if p["claimed"] is None:
        cons = 0.0
    elif p["claimed"] == len(word) and p["n_lines"] == len(word) and core_exact == 1.0:
        cons = 1.0
    elif p["claimed"] == p["n_lines"] == len(word):
        cons = 0.75
    elif p["claimed"] == len(word):
        cons = 0.35
    elif p["n_lines"] == len(word):
        cons = 0.25
    else:
        cons = 0.0
    stop = 1.0 if p["has_stop"] else 0.0
    pref = 0.0
    for i in range(min(len(core), len(gold))):
        if core[i] == gold[i]:
            pref += 1
        else:
            break
    pref = pref / max(1, len(gold))
    overrun = 0.0
    if p["n_lines"] > len(word):
        overrun = 1.0
    elif full_chars and len(full_chars) > len(gold):
        overrun = 1.0
    clean = 1.0 if (core_exact == 1.0 and stop == 1.0 and overrun == 0.0) else 0.0
    score = (
        0.35 * core_exact
        + 0.20 * exact
        + 0.20 * cons
        + 0.15 * stop
        + 0.10 * pref
        - 0.10 * overrun
    )
    return {
        "exact": exact,
        "core_exact": core_exact,
        "consistency": cons,
        "stop": stop,
        "prefix": pref,
        "overrun": overrun,
        "clean": clean,
        "score": max(0.0, score),
    }


def build_dataset(n_train: int = 1200, n_heldout: int = 40, seed: int = 41) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    held_words = [
        "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
        "mississippi", "bookkeeper", "beekeeper", "parallel", "google",
        "pizza", "banana", "occurrence", "queueing", "xylophone", "jazz",
    ]
    held_words += ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9))) for _ in range(20)]
    held_words = list(dict.fromkeys(held_words))
    held_set = {w.lower() for w in held_words}

    train_words = []
    for _ in range(1000):
        if rng.random() < 0.55:
            n = rng.randint(2, 5)
        elif rng.random() < 0.85:
            n = rng.randint(6, 9)
        else:
            n = rng.randint(10, 12)
        train_words.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)))
    train_words += [w for w in POOL if w.lower() not in held_set and w.lower() not in BANNED]
    train_words = [w for w in train_words if w.lower() not in held_set]
    train_words = list(dict.fromkeys(train_words))
    rng.shuffle(train_words)

    def make_spell(w: str) -> Sample:
        q = rng.choice(TEMPLATES).format(word=w)
        return Sample(q, teacher_spell(w), w, "spell")

    def make_stop(w: str) -> Sample:
        keep = len(w)
        body = teacher_partial(w, keep)
        q = (
            f"Complete the orthography of '{w}'. The list below already has every character. "
            f"Output only STOP and nothing else.\n{body}"
        )
        return Sample(q, "STOP", w, "stop")

    def make_finish(w: str) -> Sample:
        if len(w) <= 2:
            return make_spell(w)
        keep = rng.randint(max(1, len(w) - 3), len(w) - 1)
        body = teacher_partial(w, keep)
        rest = []
        for i, c in enumerate(w[keep:], start=keep + 1):
            rest.append(f"{i}:{c}")
        rest.append("STOP")
        q = (
            f"Continue spelling '{w}' from the next missing index. "
            f"Do not repeat finished lines. End with STOP.\n{body}"
        )
        return Sample(q, "\n".join(rest), w, "finish")

    train: List[Sample] = []
    i = 0
    while len(train) < n_train:
        w = train_words[i % len(train_words)]
        i += 1
        r = rng.random()
        if r < 0.58:
            train.append(make_spell(w))
        elif r < 0.78:
            train.append(make_stop(w))
        else:
            train.append(make_finish(w))
    for q, a in REHEARSAL:
        train.append(Sample(q, a, "", "rehearsal"))
    rng.shuffle(train)

    held = [make_spell(w) for w in held_words[:n_heldout]]
    classic = [
        Sample(f"Spell the word '{w}' one character at a time.", teacher_spell(w), w, "spell")
        for w in ["strawberry", "blueberry", "mississippi", "banana", "beekeeper", "parallel", "google", "pizza"]
    ]
    held = classic + held
    return train, held


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 384):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        messages = [
            {"role": "user", "content": s.question},
            {"role": "assistant", "content": s.answer},
        ]
        full = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        eos = self.tok.eos_token_id
        if eos is not None and (not full_ids or full_ids[-1] != eos):
            full_ids = full_ids + [eos]
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]
        prompt_len = min(len(prompt_ids), max(1, len(full_ids) - 1))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        if len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))
        ids = torch.tensor(full_ids, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_id):
    m = max(x["input_ids"].size(0) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        n = m - x["input_ids"].size(0)
        out["input_ids"].append(torch.nn.functional.pad(x["input_ids"], (0, n), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(x["attention_mask"], (0, n), value=0))
        out["labels"].append(torch.nn.functional.pad(x["labels"], (0, n), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int, needles: Sequence[str]):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.needles = list(needles)

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0][self.start_len :]
        if gen.numel() == 0:
            return False
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        upper = text.upper()
        for n in self.needles:
            if n.upper() in upper:
                return True
        return False


def make_gen(model, tok, device, stop_on_stop: bool = True):
    def gen(prompt: str, max_new_tokens: int = 120):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        model.eval()
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if stop_on_stop:
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopOnSubstrings(tok, enc["input_ids"].shape[1], ["\nSTOP\n", "\nSTOP"])]
            )
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    return gen


def budget_for_word(word: str) -> int:
    return min(160, 10 + 5 * max(1, len(word)) + 12)


def evaluate(gen, samples: Sequence[Sample], limit: Optional[int] = None) -> Dict:
    rows = []
    selected = [s for s in samples if s.kind == "spell"]
    if limit is not None:
        selected = selected[:limit]
    keys = ["exact", "core_exact", "consistency", "stop", "prefix", "overrun", "clean", "score"]
    sums = {k: 0.0 for k in keys}
    for s in selected:
        pred = gen(s.question, budget_for_word(s.word))
        sc = score_spell(pred, s.word)
        for k in sums:
            sums[k] += sc[k]
        rows.append({"word": s.word, "question": s.question, "pred": pred, **sc})
    n = max(1, len(rows))
    return {**{k: v / n for k, v in sums.items()}, "n": len(rows), "rows": rows}


def evaluate_controls(gen) -> Dict:
    cases = [
        ("What is the capital of France?", "paris"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
    ]
    rows = []
    ok = 0
    for q, g in cases:
        pred = gen(q, 40)
        if g.isdigit():
            nums = re.findall(r"-?\d+", pred)
            hit = bool(nums) and nums[-1] == g
        else:
            hit = g.lower() in pred.lower()
        ok += int(hit)
        rows.append({"q": q, "gold": g, "pred": pred, "ok": hit})
    return {"accuracy": ok / len(cases), "rows": rows}


def evaluate_count_smoke(gen) -> Dict:
    cases = [
        ("How many r's are in the word strawberry?", "3"),
        ("How many a's are in banana?", "3"),
        ("How many characters are in the word 'strawberry'?", "10"),
    ]
    rows = []
    ok = 0
    for q, g in cases:
        pred = gen(q, 160)
        answers = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
        if answers:
            nums = re.findall(r"-?\d+", answers[-1])
            got = nums[-1] if nums else ""
        else:
            nums = re.findall(r"-?\d+", pred)
            got = nums[-1] if nums else ""
        hit = got == g
        ok += int(hit)
        rows.append({"q": q, "gold": g, "pred": pred, "ok": hit, "got": got})
    return {"accuracy": ok / len(cases), "rows": rows}


def train(
    model_path: str,
    out_dir: str,
    resume: Optional[str] = None,
    n_train: int = 1200,
    n_heldout: int = 36,
    epochs: int = 1,
    lr: float = 1.5e-5,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_len: int = 384,
    seed: int = 41,
    device: str = "mps",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    train_s, held_s = build_dataset(n_train, n_heldout, seed)
    leak = [s.word for s in train_s if s.word and s.word.lower() in BANNED]
    if leak:
        raise RuntimeError(f"train leak {leak[:5]}")
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_s:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    with open(out / "data" / "heldout.jsonl", "w", encoding="utf-8") as f:
        for s in held_s:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    model.to(device)
    if resume:
        print(f"resume {resume}", flush=True)
        model = load_causal_lm(resume or model_path, device=device, trainable=True)
    else:
        model = load_causal_lm(args.model, device=device, trainable=True)
    print_trainable(model)

    ds = ChatDS(train_s, tok, max_len=max_len)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = make_gen(model, tok, device, stop_on_stop=True)

    baseline = evaluate(gen, held_s, limit=14)
    base_ctrl = evaluate_controls(gen)
    with open(out / "baseline.json", "w", encoding="utf-8") as f:
        json.dump({"heldout": baseline, "ctrl": base_ctrl}, f, ensure_ascii=False, indent=2)
    print(
        f"BASELINE exact={baseline['exact']:.3f} core={baseline['core_exact']:.3f} "
        f"stop={baseline['stop']:.3f} prefix={baseline['prefix']:.3f} "
        f"overrun={baseline['overrun']:.3f} ctrl={base_ctrl['accuracy']:.3f}",
        flush=True,
    )

    health = []
    best = {"rank": -1.0, "epoch": 0, "path": str(out / "model_best")}
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out_m = model(**batch)
            loss = out_m.loss / grad_accum
            loss.backward()
            running += float(out_m.loss.detach().cpu())
            seen += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 60 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss={running/max(1,seen):.4f}", flush=True)
        if seen % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        held = evaluate(gen, held_s, limit=14)
        ctrl = evaluate_controls(gen)
        count = evaluate_count_smoke(gen)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "exact": held["exact"],
            "core_exact": held["core_exact"],
            "consistency": held["consistency"],
            "stop": held["stop"],
            "prefix": held["prefix"],
            "overrun": held["overrun"],
            "clean": held["clean"],
            "score": held["score"],
            "ctrl": ctrl["accuracy"],
            "count_smoke": count["accuracy"],
            "elapsed_s": time.perf_counter() - t0,
        }
        health.append(entry)
        print(
            f"EPOCH {epoch} exact={held['exact']:.3f} core={held['core_exact']:.3f} "
            f"stop={held['stop']:.3f} prefix={held['prefix']:.3f} overrun={held['overrun']:.3f} "
            f"ctrl={ctrl['accuracy']:.3f} count={count['accuracy']:.3f} loss={entry['train_loss']:.4f}",
            flush=True,
        )
        with open(out / f"heldout_epoch{epoch}.json", "w", encoding="utf-8") as f:
            json.dump({"held": held, "ctrl": ctrl, "count": count, "entry": entry}, f, ensure_ascii=False, indent=2)
        with open(out / "health_curve.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        rank = (
            held["core_exact"]
            + 0.35 * held["stop"]
            + 0.25 * held["consistency"]
            + 0.15 * held["exact"]
            + 0.15 * ctrl["accuracy"]
            - 0.20 * held["overrun"]
        )
        if rank > best["rank"]:
            best = {"rank": rank, "epoch": epoch, "path": str(out / "model_best")}
            save_checkpoint(model, tok, out / "model_best")
            print(f"saved best epoch={epoch} rank={rank:.3f}", flush=True)
        if held["core_exact"] >= 0.75 and held["stop"] >= 0.7 and ctrl["accuracy"] >= 0.66:
            print("early stop: target met", flush=True)
            break
        if epoch >= 2 and entry["train_loss"] < 0.01 and held["core_exact"] < health[0]["core_exact"] + 0.05:
            print("early stop: overfit without core gain", flush=True)
            break

    save_checkpoint(model, tok, out / "model_last")

    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    base.to(device)
    model = load_causal_lm(best["path"], device=device, trainable=False)
    model.to(device)
    gen = make_gen(model, tok, device, stop_on_stop=True)
    classic_words = ["strawberry", "blueberry", "mississippi", "banana", "beekeeper", "parallel", "google", "pizza", "cranberry"]
    classic_samples = [
        Sample(f"Spell the word '{w}' one character at a time.", teacher_spell(w), w, "spell") for w in classic_words
    ]
    classic = evaluate(gen, classic_samples)
    base_gen = make_gen(base, tok, device, stop_on_stop=True)
    classic_base = evaluate(base_gen, classic_samples)
    ctrl = evaluate_controls(gen)
    ctrl_base = evaluate_controls(base_gen)
    count = evaluate_count_smoke(gen)
    count_base = evaluate_count_smoke(base_gen)

    report = {
        "method": "spell_only_stage_v2_stop_curriculum",
        "resume": resume,
        "device": device,
        "n_train": len(train_s),
        "epochs_ran": len(health),
        "best_epoch": best["epoch"],
        "best_rank": best["rank"],
        "baseline_exact": baseline["exact"],
        "baseline_core": baseline["core_exact"],
        "baseline_stop": baseline["stop"],
        "classic_exact": classic["exact"],
        "classic_core_exact": classic["core_exact"],
        "classic_consistency": classic["consistency"],
        "classic_stop": classic["stop"],
        "classic_prefix": classic["prefix"],
        "classic_overrun": classic["overrun"],
        "classic_clean": classic["clean"],
        "classic_base_exact": classic_base["exact"],
        "classic_base_core": classic_base["core_exact"],
        "ctrl": ctrl["accuracy"],
        "ctrl_base": ctrl_base["accuracy"],
        "count_smoke": count["accuracy"],
        "count_smoke_base": count_base["accuracy"],
        "health": health,
        "wall_time_s": time.perf_counter() - t0,
        "classic": classic,
        "classic_base": classic_base,
        "count_detail": count,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    core_keys = [k for k in report if k not in ("classic", "classic_base", "count_detail", "health")]
    print("REPORT_CORE", json.dumps({k: report[k] for k in core_keys}, ensure_ascii=False), flush=True)
    print("CLASSIC_SPELL", flush=True)
    for r in classic["rows"]:
        print(
            f"exact={r['exact']} core={r['core_exact']} stop={r['stop']} overrun={r['overrun']} word={r['word']}",
            flush=True,
        )
        print(r["pred"][:240], flush=True)
        print("---", flush=True)
    print("COUNT_SMOKE", flush=True)
    for r in count["rows"]:
        print(f"ok={r['ok']} gold={r['gold']} got={r['got']} q={r['q']}", flush=True)
        print(r["pred"][:240], flush=True)
        print("---", flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument("--out", default=result_path('spell_sense_v2'))
    p.add_argument("--resume", default=result_path('spell_sense_v1', 'model_best'))
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-heldout", type=int, default=36)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)
    train(
        model_path=args.model,
        out_dir=args.out,
        resume=args.resume or None,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
