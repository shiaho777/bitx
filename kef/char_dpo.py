"""Minimal DPO on top of cot_v3. Preferred=gold natural CoT, rejected=wrong traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BANNED = {
    "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
}


@dataclass
class Pair:
    question: str
    chosen: str
    rejected: str
    word: str
    gold: str
    tag: str


def cot_count(word: str, ch: str) -> str:
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


def cot_len(word: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append(f"Step2 count characters: {len(word)}")
    lines.append(f"Answer: {len(word)}")
    return "\n".join(lines)


def wrong_count(word: str, ch: str, bad_n: int) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    for i, c in enumerate(word[: max(1, min(len(word), 6))]):
        lines.append(f"{i+1}:{c}")
    lines.append(f"Step2 collect matches for '{ch}': guess")
    lines.append(f"Step3 count matches: {bad_n}")
    lines.append(f"Answer: {bad_n}")
    return "\n".join(lines)


def build_pairs() -> List[Pair]:
    pairs: List[Pair] = []

    hard = [
        ("google", "o", "How many o's are in google?"),
        ("parallel", "l", "How many l's are in parallel?"),
        ("pizza", "z", "How many z's are in pizza?"),
        ("beekeeper", "e", "How many e's in beekeeper?"),
        ("mississippi", "s", "Count the letter s in mississippi."),
        ("banana", "a", "How many a's are in banana?"),
        ("success", "s", "How many s's are in success?"),
        ("balloon", "l", "How many l's are in balloon?"),
        ("committee", "t", "How many t's are in committee?"),
        ("address", "d", "How many d's are in address?"),
        ("bookkeeper", "e", "How many e's are in bookkeeper?"),
        ("queueing", "u", "How many u's are in queueing?"),
    ]
    for word, ch, q in hard:
        gold_n = word.count(ch)
        chosen = cot_count(word, ch)
        for delta in (-2, -1, 1, 2, 3):
            bad = gold_n + delta
            if bad < 0 or bad == gold_n:
                continue
            pairs.append(Pair(q, chosen, wrong_count(word, ch, bad), word, str(gold_n), "hard"))
        # common real failure modes
        pairs.append(Pair(q, chosen, f"Answer: {max(0, gold_n-1)}", word, str(gold_n), "short_reject"))
        pairs.append(Pair(q, chosen, f"1\nAnswer: 1", word, str(gold_n), "short_reject"))

    # reinforce classics
    classics = [
        ("strawberry", "r", "How many r's are in the word strawberry?", 3),
        ("banana", "a", "How many a's are in banana?", 3),
        ("blueberry", "e", "How many letter e appear in blueberry?", 3),
        ("cranberry", "c", "In the string \"cranberry\", how many times does 'c' appear?", 1),
    ]
    for word, ch, q, n in classics:
        chosen = cot_count(word, ch)
        pairs.append(Pair(q, chosen, wrong_count(word, ch, n + 1), word, str(n), "reinforce"))
        pairs.append(Pair(q, chosen, wrong_count(word, ch, max(0, n - 1)), word, str(n), "reinforce"))
        pairs.append(Pair(q, chosen, f"Answer: {n+1}", word, str(n), "reinforce"))

    # length
    for word, q in [
        ("strawberry", "How many characters are in the word 'strawberry'?"),
        ("banana", "How many characters are in the word 'banana'?"),
        ("google", "What is the length of the string \"google\"?"),
        ("mississippi", "How many characters are in the word 'mississippi'?"),
    ]:
        n = len(word)
        chosen = cot_len(word)
        pairs.append(Pair(q, chosen, f"Answer: {n+2}", word, str(n), "length"))
        pairs.append(Pair(q, chosen, wrong_count(word, word[0], n - 1).replace(f"'{word[0]}'", "length"), word, str(n), "length"))
        pairs.append(Pair(q, chosen, cot_len(word)[:-1] + str(n + 1), word, str(n), "length"))

    # random medium words
    rng = random.Random(11)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(40):
        n = rng.randint(4, 9)
        word = "".join(rng.choice(alphabet) for _ in range(n))
        if rng.random() < 0.7:
            i = rng.randrange(n)
            j = rng.randrange(n)
            word = list(word)
            word[j] = word[i]
            word = "".join(word)
        if word.lower() in BANNED:
            continue
        ch = rng.choice(list(word))
        gold_n = word.count(ch)
        q = f"How many '{ch}' characters are in '{word}'?"
        chosen = cot_count(word, ch)
        bad = gold_n + rng.choice([-1, 1, 2])
        if bad < 0:
            bad = gold_n + 1
        if bad == gold_n:
            bad = gold_n + 1
        pairs.append(Pair(q, chosen, wrong_count(word, ch, bad), word, str(gold_n), "rand"))

    random.Random(11).shuffle(pairs)
    return pairs


def chat_ids(tok, question: str, answer: str, max_len: int):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    if eos is not None and (not full_ids or full_ids[-1] != eos):
        full_ids = full_ids + [eos]
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    prompt_len = min(len(prompt_ids), max(1, len(full_ids) - 1))
    return full_ids, prompt_len


def sequence_logprob(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    # input_ids: [1, T]
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    # mask prompt tokens (positions < prompt_len-1 in labels index)
    resp_start = max(0, prompt_len - 1)
    mask = torch.zeros_like(token_logp)
    mask[:, resp_start:] = 1.0
    # avoid empty
    denom = mask.sum().clamp_min(1.0)
    return (token_logp * mask).sum() / denom


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


def evaluate_classic(gen) -> Dict:
    cases = [
        ("How many r's are in the word strawberry?", "3", "strawberry"),
        ("How many a's are in banana?", "3", "banana"),
        ("How many characters are in the word 'strawberry'?", "10", "strawberry"),
        ("How many o's are in google?", "2", "google"),
        ("How many l's are in parallel?", "3", "parallel"),
        ("How many z's are in pizza?", "2", "pizza"),
        ("How many e's in beekeeper?", "5", "beekeeper"),
        ("Count the letter s in mississippi.", "4", "mississippi"),
        ("How many letter e appear in blueberry?", "3", "blueberry"),
        ("In the string \"cranberry\", how many times does 'c' appear?", "1", "cranberry"),
    ]
    rows = []
    ok = 0
    fid = 0.0
    for q, gold, word in cases:
        pred = gen(q, min(200, 18 + 8 * len(word) + 30))
        got = extract_answer(pred)
        hit = got == gold
        ok += int(hit)
        f = process_fid(pred, word)
        fid += f
        rows.append({"q": q, "gold": gold, "got": got, "ok": hit, "fid": f, "pred": pred})
    n = len(cases)
    return {"accuracy": ok / n, "fidelity": fid / n, "rows": rows}


def evaluate_controls(gen) -> Dict:
    cases = [
        ("What is the capital of France?", "paris"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
    ]
    ok = 0
    rows = []
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
    steps: int = 120,
    lr: float = 2e-6,
    beta: float = 0.1,
    max_len: int = 512,
    seed: int = 19,
    device: str = "mps",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    pairs = build_pairs()
    with open(out / "data" / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"pairs={len(pairs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    model.print_trainable_parameters()
    model.train()

    gen = make_gen(model, tok, device)
    baseline = evaluate_classic(gen)
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
    shutil.copytree(resume_adapter, out / "adapter_best", dirs_exist_ok=True)
    best = {"rank": best_rank, "step": 0, "from_resume": True}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    t0 = time.perf_counter()
    running = 0.0

    for step in range(1, steps + 1):
        pair = random.choice(pairs)
        c_ids, c_plen = chat_ids(tok, pair.question, pair.chosen, max_len)
        r_ids, r_plen = chat_ids(tok, pair.question, pair.rejected, max_len)
        c = torch.tensor([c_ids], device=device)
        r = torch.tensor([r_ids], device=device)

        model.train()
        pol_c = sequence_logprob(model, c, c_plen)
        pol_r = sequence_logprob(model, r, r_plen)
        with torch.no_grad():
            with model.disable_adapter():
                ref_c = sequence_logprob(model, c, c_plen)
                ref_r = sequence_logprob(model, r, r_plen)

        pi = (pol_c - ref_c) - (pol_r - ref_r)
        loss = -F.logsigmoid(beta * pi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        running += float(loss.detach().cpu())

        if step % 20 == 0 or step == steps:
            avg = running / 20 if step % 20 == 0 else running / max(1, step % 20)
            running = 0.0
            print(f"step {step}/{steps} loss={avg:.4f} pi={float(pi.detach().cpu()):.3f}", flush=True)
            classic = evaluate_classic(gen)
            ctrl = evaluate_controls(gen)
            rank = classic["accuracy"] + 0.25 * classic["fidelity"] + 0.2 * ctrl["accuracy"]
            core_qs = {
                "How many r's are in the word strawberry?",
                "How many a's are in banana?",
                "How many characters are in the word 'strawberry'?",
            }
            core_ok = sum(1 for row in classic["rows"] if row["q"] in core_qs and row["ok"])
            base_core = sum(1 for row in baseline["rows"] if row["q"] in core_qs and row["ok"])
            print(
                f"  eval classic={classic['accuracy']:.3f} fid={classic['fidelity']:.3f} "
                f"ctrl={ctrl['accuracy']:.3f} core={core_ok}/{base_core} rank={rank:.3f}",
                flush=True,
            )
            for row in classic["rows"]:
                print(f"  {'OK' if row['ok'] else 'NO'} gold={row['gold']} got={row['got']} | {row['q']}", flush=True)
            with open(out / f"eval_step{step}.json", "w", encoding="utf-8") as f:
                json.dump({"classic": classic, "ctrl": ctrl, "rank": rank}, f, ensure_ascii=False, indent=2)
            if rank > best["rank"] + 1e-6 and core_ok >= base_core and ctrl["accuracy"] >= 0.66:
                best = {"rank": rank, "step": step, "from_resume": False}
                model.save_pretrained(out / "adapter_best")
                tok.save_pretrained(out / "adapter_best")
                print(f"  saved best step={step} rank={rank:.3f}", flush=True)
            else:
                print(f"  no promote best={best['rank']:.3f}", flush=True)

    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    report = {
        "method": "dpo_v1_on_v3",
        "resume_adapter": resume_adapter,
        "pairs": len(pairs),
        "steps": steps,
        "lr": lr,
        "beta": beta,
        "baseline_classic": baseline["accuracy"],
        "baseline_fid": baseline["fidelity"],
        "baseline_ctrl": base_ctrl["accuracy"],
        "best_step": best["step"],
        "best_rank": best["rank"],
        "best_from_resume": best.get("from_resume", False),
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
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/char_dpo_v1")
    p.add_argument("--resume-adapter", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)
    train(
        model_path=args.model,
        out_dir=args.out,
        resume_adapter=args.resume_adapter,
        steps=args.steps,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
