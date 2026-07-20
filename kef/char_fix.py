"""Surgical pure-weight fix using the original v3 natural CoT format only."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BANNED = {
    "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
    "boysenberry", "huckleberry", "elderberry", "gooseberry",
}

POOL = [
    "apple", "banana", "orange", "grape", "melon", "peach", "mango", "lemon",
    "cherry", "papaya", "coconut", "tomato", "potato", "carrot", "pepper",
    "onion", "garlic", "table", "chair", "window", "house", "school",
    "bridge", "river", "forest", "ocean", "island", "python", "rust", "swift",
    "happy", "brave", "quiet", "yellow", "elephant", "penguin", "rabbit",
    "keyboard", "monitor", "server", "matrix", "vector", "buffer", "bicycle",
    "morning", "winter", "summer", "coffee", "butter", "cheese", "friend",
    "family", "teacher", "system", "process", "thread", "cache", "kernel",
    "success", "address", "balloon", "level", "civic", "radar", "rotor",
    "kayak", "refer", "hello", "world", "letter", "character", "sense",
    "parallel", "google", "pizza", "beekeeper", "mississippi", "bookkeeper",
    "committee", "occurrence", "possession", "assessment", "queueing",
    "trains", "letters", "strings", "yellow", "balloon", "address",
]

REHEARSAL = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is the capital of Italy?", "Rome."),
    ("What is 17 + 25?", "42."),
    ("What is 9 times 6?", "54."),
    ("What is 12 + 8?", "20."),
    ("What is 7 times 7?", "49."),
    ("What is 6 times 7?", "42."),
    ("What is 11 + 14?", "25."),
    ("What is 3 times 9?", "27."),
    ("How many days are in a week?", "7."),
    ("How many months are in a year?", "12."),
    ("What planet do humans live on?", "Earth."),
    ("Is water H2O?", "Yes."),
]


@dataclass
class Sample:
    question: str
    answer: str
    kind: str
    word: str
    gold: str


def cot_count_v3(word: str, ch: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    matches = []
    for i, c in enumerate(word):
        if c == ch:
            lines.append(f"{i+1}:{c} MATCH")
            matches.append(str(i + 1))
        else:
            lines.append(f"{i+1}:{c}")
    mt = ",".join(matches) if matches else "none"
    n = len(matches)
    lines.append(f"Step2 collect matches for '{ch}': {mt}")
    lines.append(f"Step3 count matches: {n}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)


def cot_len_v3(word: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append(f"Step2 count characters: {len(word)}")
    lines.append(f"Answer: {len(word)}")
    return "\n".join(lines)


def cot_list_v3(word: str, ch: str) -> str:
    listing = ", ".join(list(word))
    lines = [f"Scan the given list:"]
    matches = []
    for i, c in enumerate(word):
        if c == ch:
            lines.append(f"item{i+1}={c} MATCH")
            matches.append(str(i + 1))
        else:
            lines.append(f"item{i+1}={c}")
    mt = ",".join(matches) if matches else "none"
    n = len(matches)
    lines.append(f"Matches for '{ch}': {mt}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines), listing


def cot_spell_v3(word: str) -> str:
    lines = ["I will read each character in order."]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append("Done.")
    return "\n".join(lines)


def random_word(rng: random.Random) -> str:
    n = rng.choices([3, 4, 5, 6, 7, 8, 9], weights=[15, 20, 22, 18, 12, 8, 5])[0]
    chars = [rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)]
    if rng.random() < 0.75:
        i = rng.randrange(n)
        j = rng.randrange(n)
        chars[j] = chars[i]
    if n >= 4 and rng.random() < 0.4:
        i = rng.randrange(n - 1)
        chars[i + 1] = chars[i]
    return "".join(chars)


def build_dataset(n_train: int, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    pool = [w for w in POOL if w.lower() not in BANNED and w.isalpha()]
    words = [random_word(rng) for _ in range(600)] + pool
    words = list(dict.fromkeys(words))
    rng.shuffle(words)

    count_t = [
        "How many '{ch}' characters are in '{word}'?",
        "How many {ch}'s are in {word}?",
        "Count the letter {ch} in the word {word}.",
        "How many letter {ch} appear in {word}?",
        "In the string \"{word}\", how many times does '{ch}' appear?",
    ]
    len_t = [
        "How many characters are in the word '{word}'?",
        "What is the length of the string \"{word}\"?",
        "Count every character in '{word}'.",
    ]
    spell_t = [
        "Spell the word '{word}' one character at a time.",
        "Write '{word}' letter by letter with positions.",
    ]
    list_t = [
        "Given letters {listing}, how many times does {ch} occur?",
        "In the character list [{listing}], how many '{ch}' are there?",
        "Count '{ch}' in this letter sequence: {listing}",
    ]

    train: List[Sample] = []
    i = 0
    while len(train) < n_train:
        w = words[i % len(words)]
        i += 1
        r = rng.random()
        if r < 0.50:
            ch = rng.choice(list(w)) if rng.random() < 0.85 else rng.choice("abcdefghijklmnopqrstuvwxyz")
            q = rng.choice(count_t).format(ch=ch, word=w)
            train.append(Sample(q, cot_count_v3(w, ch), "count", w, str(w.count(ch))))
        elif r < 0.68:
            ch = rng.choice(list(w)) if w else "a"
            ans, listing = cot_list_v3(w, ch)
            q = rng.choice(list_t).format(listing=listing, ch=ch)
            train.append(Sample(q, ans, "list_count", w, str(w.count(ch))))
        elif r < 0.82:
            q = rng.choice(len_t).format(word=w)
            train.append(Sample(q, cot_len_v3(w), "length", w, str(len(w))))
        elif r < 0.92:
            q = rng.choice(spell_t).format(word=w)
            train.append(Sample(q, cot_spell_v3(w), "spell", w, ", ".join(list(w))))
        else:
            q, a = rng.choice(REHEARSAL)
            train.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))

    anchors = ["banana", "google", "pizza", "parallel", "beekeeper", "success", "balloon", "mississippi", "committee"]
    for w in anchors:
        for ch in sorted(set(w)):
            if w.count(ch) >= 1:
                q = f"How many '{ch}' characters are in '{w}'?"
                train.append(Sample(q, cot_count_v3(w, ch), "count", w, str(w.count(ch))))
        train.append(Sample(f"How many characters are in the word '{w}'?", cot_len_v3(w), "length", w, str(len(w))))
        train.append(Sample(f"Spell the word '{w}' one character at a time.", cot_spell_v3(w), "spell", w, ", ".join(list(w))))

    for q, a in REHEARSAL:
        train.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))
        train.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))

    rng.shuffle(train)

    classic = [
        Sample("How many r's are in the word strawberry?", cot_count_v3("strawberry", "r"), "count", "strawberry", "3"),
        Sample("How many a's are in banana?", cot_count_v3("banana", "a"), "count", "banana", "3"),
        Sample("How many characters are in the word 'strawberry'?", cot_len_v3("strawberry"), "length", "strawberry", "10"),
        Sample("How many o's are in google?", cot_count_v3("google", "o"), "count", "google", "2"),
        Sample("How many l's are in parallel?", cot_count_v3("parallel", "l"), "count", "parallel", "3"),
        Sample("How many z's are in pizza?", cot_count_v3("pizza", "z"), "count", "pizza", "2"),
        Sample("How many e's in beekeeper?", cot_count_v3("beekeeper", "e"), "count", "beekeeper", str("beekeeper".count("e"))),
        Sample("Count the letter s in mississippi.", cot_count_v3("mississippi", "s"), "count", "mississippi", "4"),
        Sample("How many letter e appear in blueberry?", cot_count_v3("blueberry", "e"), "count", "blueberry", "3"),
        Sample("In the string \"cranberry\", how many times does 'c' appear?", cot_count_v3("cranberry", "c"), "count", "cranberry", "1"),
    ]
    return train, classic


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 560):
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


def make_gen(model, tok, device):
    def gen(prompt: str, max_new_tokens: int = 180):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    return gen


def extract_answer(pred: str) -> str:
    answers = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
    if answers:
        nums = re.findall(r"-?\d+", answers[0])
        if nums:
            return nums[0]
        return answers[0].strip()
    m = re.search(r"Step3 count matches:\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"Step2 count characters:\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+", pred)
    return nums[0] if nums else ""


def process_fid(pred: str, word: str) -> float:
    if not word:
        return 0.0
    body = pred
    m = re.search(r"Answer:|Step2|Step3", pred, flags=re.I)
    if m:
        body = pred[: m.start()]
    pairs = re.findall(r"(?m)^(\d+)\s*:\s*([A-Za-z])", body)
    got = {}
    for i_s, c in pairs:
        i = int(i_s)
        if i not in got:
            got[i] = c.lower()
    hits = 0
    for i, ch in enumerate(word.lower(), start=1):
        if got.get(i) == ch:
            hits += 1
        else:
            break
    return hits / max(1, len(word))


def evaluate(gen, samples: Sequence[Sample]) -> Dict:
    rows = []
    ok = 0
    fid = 0.0
    for s in samples:
        budget = min(200, 18 + 8 * max(1, len(s.word)) + 30) if s.word else 48
        pred = gen(s.question, budget)
        got = extract_answer(pred)
        hit = got == s.gold
        ok += int(hit)
        f = process_fid(pred, s.word)
        fid += f
        rows.append({"q": s.question, "gold": s.gold, "got": got, "ok": hit, "fid": f, "pred": pred})
    n = max(1, len(rows))
    return {"accuracy": ok / n, "fidelity": fid / n, "n": len(rows), "rows": rows}


def evaluate_controls(gen) -> Dict:
    cases = [
        ("What is the capital of France?", "paris"),
        ("What is the capital of Japan?", "tokyo"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
        ("What is 12 + 8?", "20"),
    ]
    rows = []
    ok = 0
    for q, g in cases:
        pred = gen(q, 40)
        if g.isdigit():
            answers = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
            if answers:
                nums = re.findall(r"-?\d+", answers[0])
                hit = bool(nums) and nums[0] == g
            else:
                m = re.search(r"=\s*(\d+)", pred)
                hit = bool(m) and m.group(1) == g
        else:
            hit = g.lower() in pred.lower()
        ok += int(hit)
        rows.append({"q": q, "gold": g, "pred": pred, "ok": hit})
    return {"accuracy": ok / len(cases), "rows": rows}


def train(
    model_path: str,
    out_dir: str,
    resume_adapter: str,
    n_train: int = 280,
    epochs: int = 1,
    lr: float = 4e-6,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_len: int = 560,
    seed: int = 83,
    device: str = "mps",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    train_s, classic = build_dataset(n_train, seed)
    leak = [s.word for s in train_s if s.word and s.word.lower() in BANNED]
    if leak:
        raise RuntimeError(f"train leak {leak[:5]}")
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_s:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    model.to(device)
    print(f"resume {resume_adapter}", flush=True)
    model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    model.print_trainable_parameters()

    ds = ChatDS(train_s, tok, max_len=max_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = make_gen(model, tok, device)

    baseline = evaluate(gen, classic)
    base_ctrl = evaluate_controls(gen)
    with open(out / "baseline.json", "w", encoding="utf-8") as f:
        json.dump({"classic": baseline, "ctrl": base_ctrl}, f, ensure_ascii=False, indent=2)
    print(
        f"BASELINE classic={baseline['accuracy']:.3f} fid={baseline['fidelity']:.3f} ctrl={base_ctrl['accuracy']:.3f}",
        flush=True,
    )
    for r in baseline["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} fid={r['fid']:.2f} | {r['q']}", flush=True)

    best_rank = baseline["accuracy"] + 0.25 * baseline["fidelity"] + 0.2 * base_ctrl["accuracy"]
    best = {"rank": best_rank, "epoch": 0, "from_resume": True}
    shutil.copytree(resume_adapter, out / "adapter_best", dirs_exist_ok=True)
    health = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(ds)))
        random.shuffle(order)
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        step = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            batch = collate([ds[i] for i in idxs], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += float(loss.detach().cpu())
            seen += 1
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 40 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss={running/max(1,seen):.4f}", flush=True)
        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        classic_e = evaluate(gen, classic)
        ctrl = evaluate_controls(gen)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "classic": classic_e["accuracy"],
            "fidelity": classic_e["fidelity"],
            "ctrl": ctrl["accuracy"],
            "elapsed_s": time.perf_counter() - t0,
        }
        health.append(entry)
        print(
            f"EPOCH {epoch} classic={classic_e['accuracy']:.3f} fid={classic_e['fidelity']:.3f} "
            f"ctrl={ctrl['accuracy']:.3f} loss={entry['train_loss']:.4f}",
            flush=True,
        )
        for r in classic_e["rows"]:
            print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} fid={r['fid']:.2f} | {r['q']}", flush=True)
            print("   ", r["pred"][:200].replace("\n", " | "), flush=True)

        with open(out / f"classic_epoch{epoch}.json", "w", encoding="utf-8") as f:
            json.dump({"classic": classic_e, "ctrl": ctrl, "entry": entry}, f, ensure_ascii=False, indent=2)
        with open(out / "health_curve.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        rank = classic_e["accuracy"] + 0.25 * classic_e["fidelity"] + 0.2 * ctrl["accuracy"]
        # require non-regression on the 3 core probes to promote
        core_qs = {
            "How many r's are in the word strawberry?",
            "How many a's are in banana?",
            "How many characters are in the word 'strawberry'?",
        }
        core_ok = sum(1 for r in classic_e["rows"] if r["q"] in core_qs and r["ok"])
        base_core = sum(1 for r in baseline["rows"] if r["q"] in core_qs and r["ok"])
        if rank > best["rank"] + 1e-6 and core_ok >= base_core and ctrl["accuracy"] >= 0.8:
            best = {"rank": rank, "epoch": epoch, "from_resume": False}
            model.save_pretrained(out / "adapter_best")
            tok.save_pretrained(out / "adapter_best")
            print(f"saved best epoch={epoch} rank={rank:.3f} core={core_ok}/{base_core}", flush=True)
        else:
            print(
                f"no promote epoch={epoch} rank={rank:.3f} best={best['rank']:.3f} core={core_ok}/{base_core}",
                flush=True,
            )

    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    report = {
        "method": "surgical_fix_v2_natural_v3_format",
        "resume_adapter": resume_adapter,
        "n_train": len(train_s),
        "epochs": epochs,
        "lr": lr,
        "baseline_classic": baseline["accuracy"],
        "baseline_fid": baseline["fidelity"],
        "baseline_ctrl": base_ctrl["accuracy"],
        "best_epoch": best["epoch"],
        "best_rank": best["rank"],
        "best_from_resume": best.get("from_resume", False),
        "health": health,
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("REPORT_CORE", json.dumps(report, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/char_fix_v2")
    p.add_argument("--resume-adapter", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    p.add_argument("--n-train", type=int, default=280)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-6)
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)
    train(
        model_path=args.model,
        out_dir=args.out,
        resume_adapter=args.resume_adapter,
        n_train=args.n_train,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
