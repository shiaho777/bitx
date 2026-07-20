"""Round-2c: ultra-gentle fail-only CoT polish on v2; anti-invent; route-safe."""

from __future__ import annotations

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
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kef.char_guardrails import HARD_PROBES, CORE_PROBES, validate_train_batch
from kef.char_router import is_hard_char_query

FAILS = [
    ("parallel", "l"),
    ("beekeeper", "e"),
    ("mississippi", "s"),
    ("bookkeeper", "e"),
    ("committee", "t"),
]

RETAIN = [
    ("google", "o"),
    ("pizza", "z"),
    ("banana", "a"),
    ("success", "s"),
    ("balloon", "l"),
]


@dataclass
class Sample:
    question: str
    answer: str
    kind: str
    word: str
    gold: str


def cot_strict(word: str, ch: str) -> str:
    n_len = len(word)
    lines = [
        f"Target word is exactly '{word}' with exactly {n_len} letters.",
        f"Do not invent letters past position {n_len}.",
        f"Count '{ch}' left to right:",
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
    lines.append(f"End of word at position {n_len}.")
    lines.append(f"Match positions: {mt}")
    lines.append(f"Verify count={n}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)


def cot_step(word: str, ch: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    matches = []
    for i, c in enumerate(word):
        tag = " MATCH" if c == ch else ""
        lines.append(f"{i+1}:{c}{tag}")
        if c == ch:
            matches.append(str(i + 1))
    n = len(matches)
    mt = ",".join(matches) if matches else "none"
    lines.append(f"Step2 collect matches for '{ch}': {mt}")
    lines.append(f"Step3 count matches: {n}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)


def build_data(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    templates = [
        "How many '{ch}' characters are in '{word}'?",
        "How many {ch}'s are in {word}?",
        "Count the letter {ch} in the word {word}.",
        "In the string \"{word}\", how many times does '{ch}' appear?",
        "How many {ch}'s in {word}?",
        "Count the letter {ch} in {word}.",
    ]
    out: List[Sample] = []

    for w, ch in FAILS:
        gold = str(w.count(ch))
        for _ in range(10):
            q = rng.choice(templates).format(ch=ch, word=w)
            out.append(Sample(q, cot_strict(w, ch), "fail", w, gold))
        for _ in range(4):
            q = rng.choice(templates).format(ch=ch, word=w)
            out.append(Sample(q, cot_step(w, ch), "fail_step", w, gold))

    for w, ch in RETAIN:
        gold = str(w.count(ch))
        for _ in range(3):
            q = rng.choice(templates).format(ch=ch, word=w)
            out.append(Sample(q, cot_step(w, ch), "retain", w, gold))

    for q, a in [
        ("What is the capital of France?", "Paris."),
        ("What is 17 + 25?", "42."),
        ("What is 9 times 6?", "54."),
    ]:
        out.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))

    rng.shuffle(out)
    if len(out) > n_train:
        fails = [s for s in out if s.kind.startswith("fail")]
        rest = [s for s in out if not s.kind.startswith("fail")]
        need = max(0, n_train - len(fails))
        out = fails + rest[:need]
        out = out[:n_train]
    validate_train_batch([s.answer for s in out])
    return out


class ChatDS(Dataset):
    def __init__(self, samples: List[Sample], tok, max_len: int = 480):
        self.rows = []
        for s in samples:
            text = tok.apply_chat_template(
                [
                    {"role": "user", "content": s.question},
                    {"role": "assistant", "content": s.answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            ids = tok(text, add_special_tokens=False)["input_ids"]
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": s.question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            pids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_len:
                ids = ids[:max_len]
            labels = [-100] * min(len(pids), len(ids)) + ids[min(len(pids), len(ids)):]
            if len(labels) < len(ids):
                labels = labels + [-100] * (len(ids) - len(labels))
            self.rows.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


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
    def gen(prompt: str, max_new_tokens: int = 160):
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
    m = re.search(r"Verify count\s*=\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"Step3 count matches:\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"Count\s*=\s*(\d+)", pred, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+", pred)
    return nums[0] if nums else ""


def eval_probes(gen, probes: Sequence[Tuple[str, str]]) -> Dict:
    rows = []
    ok = 0
    for q, gold in probes:
        pred = gen(q, 180)
        got = extract_answer(pred)
        hit = got == gold
        ok += int(hit)
        rows.append({"q": q, "gold": gold, "got": got, "ok": hit, "pred": pred[:360]})
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
                nums = re.findall(r"-?\d+", pred)
                hit = bool(nums) and g in nums
        else:
            hit = g.lower() in pred.lower()
        ok += int(hit)
        rows.append({"q": q, "ok": hit, "pred": pred[:80]})
    return {"accuracy": ok / len(cases), "rows": rows}


def eval_routed(core_gen, exp_gen, probes):
    def routed(q, max_new=180):
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
    model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    model.print_trainable_parameters()

    ds = ChatDS(samples, tok, max_len=args.max_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    core_base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    core_base.to(device)
    core_m = PeftModel.from_pretrained(core_base, args.core)
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

    shutil.copytree(args.resume, out / "adapter_best", dirs_exist_ok=True)
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
    ga = 4
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            print(f"step {step}/{len(order)} loss={running/max(1,seen):.4f}", flush=True)
    if step % ga != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5)
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

    promote = (
        hard1["accuracy"] > best["hard"] + 1e-9
        and route1["accuracy"] + 1e-9 >= best["route"]
        and core1["accuracy"] + 1e-9 >= 0.99
        and ctrl1["accuracy"] + 1e-9 >= min(0.66, best["ctrl"])
    )
    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    if promote:
        model.save_pretrained(out / "adapter_best")
        tok.save_pretrained(out / "adapter_best")
        best.update({
            "hard": hard1["accuracy"],
            "route": route1["accuracy"],
            "core": core1["accuracy"],
            "ctrl": ctrl1["accuracy"],
            "from_resume": False,
        })
        print("PROMOTED expert v4", flush=True)
    else:
        print(
            f"NO_PROMOTE hard {hard1['accuracy']:.3f}<={best['hard']:.3f} or route/core/ctrl gate",
            flush=True,
        )

    report = {
        "method": "hard_expert_v4_gentle_fail",
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
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--resume", default="/Users/shiaho/Desktop/bitx/kef_results/char_advance/hard_expert_v2/adapter_best")
    p.add_argument("--core", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/char_advance/hard_expert_v4")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--lr", type=float, default=6e-6)
    p.add_argument("--max-len", type=int, default=480)
    p.add_argument("--seed", type=int, default=117)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
