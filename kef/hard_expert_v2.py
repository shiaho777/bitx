"""Round-2 hard expert: optimized CoT, small fast train, route-safe promotion."""

from __future__ import annotations

from kef.paths import default_model, repo_root, result_path

import argparse
import json
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.char_guardrails import HARD_PROBES, CORE_PROBES, validate_train_batch
from kef.char_router import is_hard_char_query

ANCHORS = [
    "google", "parallel", "pizza", "beekeeper", "mississippi", "bookkeeper",
    "success", "balloon", "committee", "address", "queueing", "possession",
    "yellow", "coffee", "pepper", "letter", "puppy", "kitten", "banana",
]


@dataclass
class Sample:
    question: str
    answer: str
    kind: str
    word: str
    gold: str


def cot_bound(word: str, ch: str) -> str:
    """Bound spelling to quoted word + running count + verify. No STOP/LEN scaffold."""
    lines = [
        f"Target word is exactly '{word}'.",
        f"Count character '{ch}' by reading left to right:",
    ]
    matches = []
    run = 0
    for i, c in enumerate(word):
        if c == ch:
            run += 1
            matches.append(str(i + 1))
            lines.append(f"{i+1}:{c} MATCH run={run}")
        else:
            lines.append(f"{i+1}:{c}")
    n = len(matches)
    mt = ",".join(matches) if matches else "none"
    lines.append(f"Match positions: {mt}")
    lines.append(f"Verify: number of matches is {n}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)


def cot_compact(word: str, ch: str) -> str:
    spell = " ".join(f"{i+1}:{c}" for i, c in enumerate(word))
    matches = [str(i + 1) for i, c in enumerate(word) if c == ch]
    n = len(matches)
    mt = ",".join(matches) if matches else "none"
    return (
        f"Spell '{word}': {spell}\n"
        f"'{ch}' at {mt}\n"
        f"Count={n}\n"
        f"Answer: {n}"
    )


def cot_list(word: str, ch: str) -> Tuple[str, str]:
    listing = ", ".join(list(word))
    matches = [str(i + 1) for i, c in enumerate(word) if c == ch]
    n = len(matches)
    mt = ",".join(matches) if matches else "none"
    body = "\n".join(
        [f"item{i+1}={c}{' MATCH' if c == ch else ''}" for i, c in enumerate(word)]
    )
    ans = f"Scan list [{listing}]:\n{body}\nMatches: {mt}\nAnswer: {n}"
    return ans, listing


def synth_word(rng: random.Random) -> str:
    n = rng.choice([5, 6, 7, 8, 9, 10])
    chars = [rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)]
    for _ in range(rng.randint(2, 4)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        chars[j] = chars[i]
    if rng.random() < 0.5:
        i = rng.randrange(n - 1)
        chars[i + 1] = chars[i]
    return "".join(chars)


def build_data(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    count_t = [
        "How many '{ch}' characters are in '{word}'?",
        "How many {ch}'s are in {word}?",
        "Count the letter {ch} in the word {word}.",
        "In the string \"{word}\", how many times does '{ch}' appear?",
    ]
    out: List[Sample] = []

    # Heavy anchor coverage: every char of every hard word, dual CoT styles
    for w in ANCHORS:
        for ch in sorted(set(w)):
            gold = str(w.count(ch))
            q1 = f"How many '{ch}' characters are in '{w}'?"
            out.append(Sample(q1, cot_bound(w, ch), "bound", w, gold))
            q2 = rng.choice(count_t).format(ch=ch, word=w)
            out.append(Sample(q2, cot_compact(w, ch), "compact", w, gold))
            ans, listing = cot_list(w, ch)
            out.append(Sample(f"Count '{ch}' in this letter sequence: {listing}", ans, "list", w, gold))

    # Focused failure set extra copies
    fails = [
        ("google", "o"), ("parallel", "l"), ("pizza", "z"),
        ("beekeeper", "e"), ("mississippi", "s"), ("bookkeeper", "e"),
        ("success", "s"), ("balloon", "l"), ("committee", "t"),
    ]
    for w, ch in fails:
        for _ in range(4):
            q = rng.choice(count_t).format(ch=ch, word=w)
            out.append(Sample(q, cot_bound(w, ch), "fail_focus", w, str(w.count(ch))))

    # synthetic double-letter words
    while len([s for s in out if s.kind == "synth"]) < max(40, n_train // 5):
        w = synth_word(rng)
        ch = rng.choice(list(w))
        out.append(Sample(
            f"How many '{ch}' characters are in '{w}'?",
            cot_bound(w, ch),
            "synth",
            w,
            str(w.count(ch)),
        ))

    rehearse = [
        ("What is the capital of France?", "Paris."),
        ("What is 17 + 25?", "42."),
        ("What is 9 times 6?", "54."),
        ("What is 12 + 8?", "20."),
    ]
    for q, a in rehearse:
        out.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))

    rng.shuffle(out)
    # keep size small for speed
    if len(out) > n_train:
        # keep all fail_focus + bound anchors first
        pri = [s for s in out if s.kind in ("fail_focus", "bound")]
        rest = [s for s in out if s.kind not in ("fail_focus", "bound")]
        rng.shuffle(pri)
        rng.shuffle(rest)
        out = (pri + rest)[:n_train]
    validate_train_batch([s.answer for s in out])
    return out


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 480):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}, {"role": "assistant", "content": s.answer}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        full_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]
        plen = min(len(prompt_ids), max(1, len(full_ids) - 1))
        labels = [-100] * plen + full_ids[plen:]
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
    def gen(prompt: str, max_new_tokens: int = 140):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def extract_answer(pred: str) -> str:
    answers = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
    if answers:
        nums = re.findall(r"-?\d+", answers[0])
        if nums:
            return nums[0]
    m = re.search(r"Count\s*=\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"Verify: number of matches is (\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+", pred)
    return nums[0] if nums else ""


def eval_probes(gen, probes: Sequence[Tuple[str, str]]) -> Dict:
    rows = []
    ok = 0
    for q, gold in probes:
        pred = gen(q, 150)
        got = extract_answer(pred)
        hit = got == gold
        ok += int(hit)
        rows.append({"q": q, "gold": gold, "got": got, "ok": hit, "pred": pred[:300]})
    return {"accuracy": ok / max(1, len(probes)), "rows": rows}


def eval_controls(gen) -> Dict:
    cases = [
        ("What is the capital of France?", "paris"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
    ]
    ok = 0
    rows = []
    for q, g in cases:
        pred = gen(q, 32)
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
        rows.append({"q": q, "ok": hit, "pred": pred[:80]})
    return {"accuracy": ok / len(cases), "rows": rows}


def eval_routed(core_gen, exp_gen, probes):
    def routed(q, max_new=150):
        return exp_gen(q, max_new) if is_hard_char_query(q) else core_gen(q, max_new)
    return eval_probes(routed, probes)


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_data(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    print(f"n_train={len(samples)}", flush=True)

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    base.to(device)
    model = load_causal_lm(args.resume or args.model, device=device, trainable=True)
    print_trainable(model)

    ds = ChatDS(samples, tok, max_len=args.max_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    # core model for route eval (frozen v3)
    core_base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    core_base.to(device)
    core_m = load_causal_lm(args.core, device=device, trainable=False)
    core_gen = make_gen(core_m, tok, device)

    hard0 = eval_probes(gen, HARD_PROBES)
    core0 = eval_probes(core_gen, CORE_PROBES)
    route0 = eval_routed(core_gen, gen, list(CORE_PROBES) + list(HARD_PROBES))
    ctrl0 = eval_controls(gen)
    print(
        f"BASELINE hard={hard0['accuracy']:.3f} route={route0['accuracy']:.3f} "
        f"core_v3={core0['accuracy']:.3f} ctrl={ctrl0['accuracy']:.3f}",
        flush=True,
    )
    for r in hard0["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} | {r['q']}", flush=True)

    shutil.copytree(args.resume, out / "model_best", dirs_exist_ok=True)
    best = {
        "hard": hard0["accuracy"],
        "route": route0["accuracy"],
        "core": core0["accuracy"],
        "ctrl": ctrl0["accuracy"],
        "from_resume": True,
    }

    t0 = time.perf_counter()
    model.train()
    order = list(range(len(ds)))
    random.shuffle(order)
    running = 0.0
    seen = 0
    step = 0
    ga = 8
    opt.zero_grad(set_to_none=True)
    for i in order:
        batch = collate([ds[i]], tok.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / ga
        loss.backward()
        running += float(loss.detach().cpu()) * ga
        seen += 1
        step += 1
        if step % ga == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 30 == 0:
            print(f"step {step}/{len(order)} loss={running/max(1,seen):.4f}", flush=True)
    if step % ga != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

    hard1 = eval_probes(gen, HARD_PROBES)
    route1 = eval_routed(core_gen, gen, list(CORE_PROBES) + list(HARD_PROBES))
    core1 = eval_probes(core_gen, CORE_PROBES)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER hard={hard1['accuracy']:.3f} route={route1['accuracy']:.3f} "
        f"core_v3={core1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} "
        f"loss={running/max(1,seen):.4f}",
        flush=True,
    )
    for r in hard1["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} | {r['q']}", flush=True)
        print("   ", r["pred"][:160].replace("\n", " | "), flush=True)

    # promote only if hard up, route not worse, core intact, ctrl ok
    promote = (
        hard1["accuracy"] > best["hard"] + 1e-9
        and route1["accuracy"] + 1e-9 >= best["route"]
        and core1["accuracy"] + 1e-9 >= 0.99
        and ctrl1["accuracy"] + 1e-9 >= min(0.66, best["ctrl"])
    )
    save_checkpoint(model, tok, out / "model_last")
    if promote:
        save_checkpoint(model, tok, out / "model_best")
        best.update({
            "hard": hard1["accuracy"],
            "route": route1["accuracy"],
            "core": core1["accuracy"],
            "ctrl": ctrl1["accuracy"],
            "from_resume": False,
        })
        print("PROMOTED expert v2", flush=True)
    else:
        print(
            f"NO_PROMOTE hard {hard1['accuracy']:.3f}<={best['hard']:.3f} or route/core/ctrl gate",
            flush=True,
        )

    report = {
        "method": "hard_expert_v2_bound_cot",
        "n_train": len(samples),
        "lr": args.lr,
        "resume": args.resume,
        "baseline": {
            "hard": hard0["accuracy"], "route": route0["accuracy"],
            "core": core0["accuracy"], "ctrl": ctrl0["accuracy"],
        },
        "after": {
            "hard": hard1["accuracy"], "route": route1["accuracy"],
            "core": core1["accuracy"], "ctrl": ctrl1["accuracy"],
        },
        "best": best,
        "promoted": promote,
        "hard_rows": hard1["rows"],
        "route_rows": route1["rows"],
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("REPORT", json.dumps({k: report[k] for k in report if k not in ("hard_rows", "route_rows")}, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument("--resume", default=result_path('char_advance', 'hard_expert', 'model_best'))
    p.add_argument("--core", default=result_path('char_sense_cot_v3', 'model_best'))
    p.add_argument("--out", default=result_path('char_advance', 'hard_expert_v2'))
    p.add_argument("--n-train", type=int, default=220)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--max-len", type=int, default=480)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
