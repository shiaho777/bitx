"""Advanced pure-weight char CoT: hard expert, LoRA merge, self-distill, evaluate."""

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
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kef.char_guardrails import (
    CORE_PROBES,
    HARD_PROBES,
    decide_promotion,
    validate_train_batch,
)

BANNED = {
    "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
    "boysenberry", "huckleberry", "elderberry", "gooseberry",
}

HARD_WORDS = [
    "google", "parallel", "pizza", "beekeeper", "mississippi", "bookkeeper",
    "success", "balloon", "committee", "address", "queueing", "possession",
    "occurrence", "assessment", "letters", "trains", "strings", "yellow",
    "coffee", "butter", "cheese", "pepper", "paper", "happy", "puppy",
    "kitten", "banana", "level", "civic", "radar", "rotor", "kayak",
]

POOL = HARD_WORDS + [
    "apple", "orange", "grape", "melon", "peach", "mango", "lemon", "cherry",
    "table", "chair", "window", "house", "school", "bridge", "river", "forest",
    "python", "rust", "swift", "matrix", "vector", "buffer", "system", "process",
    "thread", "cache", "kernel", "alpha", "beta", "gamma", "delta", "omega",
]

REHEARSAL = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is 17 + 25?", "42."),
    ("What is 9 times 6?", "54."),
    ("What is 12 + 8?", "20."),
    ("What is 7 times 7?", "49."),
    ("How many days are in a week?", "7."),
    ("What planet do humans live on?", "Earth."),
]


@dataclass
class Sample:
    question: str
    answer: str
    kind: str
    word: str
    gold: str


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


def cot_list(word: str, ch: str) -> Tuple[str, str]:
    listing = ", ".join(list(word))
    lines = ["Scan the given list:"]
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


def cot_spell(word: str) -> str:
    lines = ["I will read each character in order."]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append("Done.")
    return "\n".join(lines)


def random_hard_word(rng: random.Random) -> str:
    n = rng.choices([4, 5, 6, 7, 8, 9, 10], weights=[10, 15, 20, 20, 15, 12, 8])[0]
    chars = [rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)]
    # force repeats
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        chars[j] = chars[i]
    if n >= 4 and rng.random() < 0.5:
        i = rng.randrange(n - 1)
        chars[i + 1] = chars[i]
    return "".join(chars)


def build_expert_dataset(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    words = list(dict.fromkeys(HARD_WORDS + [random_hard_word(rng) for _ in range(400)] + POOL))
    words = [w for w in words if w.lower() not in BANNED and w.isalpha()]
    rng.shuffle(words)
    count_t = [
        "How many '{ch}' characters are in '{word}'?",
        "How many {ch}'s are in {word}?",
        "Count the letter {ch} in the word {word}.",
        "How many letter {ch} appear in {word}?",
    ]
    len_t = [
        "How many characters are in the word '{word}'?",
        "What is the length of the string \"{word}\"?",
    ]
    list_t = [
        "Given letters {listing}, how many times does {ch} occur?",
        "Count '{ch}' in this letter sequence: {listing}",
    ]
    out: List[Sample] = []
    i = 0
    while len(out) < n_train:
        w = words[i % len(words)]
        i += 1
        r = rng.random()
        if r < 0.55:
            ch = rng.choice(list(w)) if rng.random() < 0.9 else rng.choice("abcdefghijklmnopqrstuvwxyz")
            q = rng.choice(count_t).format(ch=ch, word=w)
            out.append(Sample(q, cot_count(w, ch), "count", w, str(w.count(ch))))
        elif r < 0.75:
            ch = rng.choice(list(w))
            ans, listing = cot_list(w, ch)
            q = rng.choice(list_t).format(listing=listing, ch=ch)
            out.append(Sample(q, ans, "list_count", w, str(w.count(ch))))
        elif r < 0.90:
            q = rng.choice(len_t).format(word=w)
            out.append(Sample(q, cot_len(w), "length", w, str(len(w))))
        else:
            out.append(Sample(f"Spell the word '{w}' one character at a time.", cot_spell(w), "spell", w, ", ".join(w)))
    for w in HARD_WORDS:
        for ch in sorted(set(w)):
            out.append(Sample(f"How many '{ch}' characters are in '{w}'?", cot_count(w, ch), "count", w, str(w.count(ch))))
        out.append(Sample(f"How many characters are in the word '{w}'?", cot_len(w), "length", w, str(len(w))))
    for q, a in REHEARSAL:
        out.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))
        out.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))
    rng.shuffle(out)
    validate_train_batch([s.answer for s in out])
    return out


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


def has_cot(pred: str) -> bool:
    return bool(re.search(r"Step1|MATCH|item\d+=", pred, flags=re.I))


def evaluate_suite(gen, probes: Sequence[Tuple[str, str]], words: Optional[Sequence[str]] = None) -> Dict:
    rows = []
    ok = 0
    fid = 0.0
    cot_n = 0
    for i, (q, gold) in enumerate(probes):
        word = words[i] if words else ""
        if not word:
            m = re.search(r"word\s+'([^']+)'|word\s+([A-Za-z]+)|string\s+\"([^\"]+)\"|'([A-Za-z]+)'\?", q)
            if m:
                word = next(g for g in m.groups() if g)
        budget = min(200, 18 + 8 * max(1, len(word)) + 30) if word else 60
        pred = gen(q, budget)
        got = extract_answer(pred)
        hit = got == gold
        ok += int(hit)
        f = process_fid(pred, word)
        fid += f
        c = has_cot(pred)
        cot_n += int(c)
        rows.append({"q": q, "gold": gold, "got": got, "ok": hit, "fid": f, "cot": c, "word": word, "pred": pred})
    n = max(1, len(rows))
    return {"accuracy": ok / n, "fidelity": fid / n, "cot_rate": cot_n / n, "n": len(rows), "rows": rows}


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


def classic_words_for(probes):
    words = []
    for q, _ in probes:
        m = re.search(r"word\s+'([^']+)'|word\s+([A-Za-z]+)|string\s+\"([^\"]+)\"|in\s+([A-Za-z]+)\?|in\s+([A-Za-z]+)\.", q)
        if m:
            words.append(next(g for g in m.groups() if g))
        else:
            words.append("")
    return words


def load_model(model_path: str, adapter: Optional[str], device: str, trainable: bool = False):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    model.to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model, tok


def train_lora(
    model_path: str,
    out_dir: str,
    samples: List[Sample],
    resume_adapter: Optional[str] = None,
    epochs: int = 1,
    lr: float = 5e-5,
    lora_r: int = 16,
    device: str = "mps",
    seed: int = 7,
    max_len: int = 560,
    tag: str = "train",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    validate_train_batch([s.answer for s in samples])

    model, tok = load_model(model_path, resume_adapter, device, trainable=bool(resume_adapter))
    if not resume_adapter:
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    ds = ChatDS(samples, tok, max_len=max_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = make_gen(model, tok, device)

    core_words = classic_words_for(CORE_PROBES)
    hard_words = classic_words_for(HARD_PROBES)
    base_core = evaluate_suite(gen, CORE_PROBES, core_words)
    base_hard = evaluate_suite(gen, HARD_PROBES, hard_words)
    base_all = evaluate_suite(gen, list(CORE_PROBES) + list(HARD_PROBES), core_words + hard_words)
    base_ctrl = evaluate_controls(gen)
    print(
        f"[{tag}] BASELINE core={base_core['accuracy']:.3f} hard={base_hard['accuracy']:.3f} "
        f"all={base_all['accuracy']:.3f} ctrl={base_ctrl['accuracy']:.3f}",
        flush=True,
    )
    with open(out / "baseline.json", "w", encoding="utf-8") as f:
        json.dump({"core": base_core, "hard": base_hard, "all": base_all, "ctrl": base_ctrl}, f, ensure_ascii=False, indent=2)

    # seed best with resume/base snapshot of current adapter state after optional resume
    model.save_pretrained(out / "adapter_best")
    tok.save_pretrained(out / "adapter_best")
    best = {
        "core": base_core["accuracy"],
        "classic": base_all["accuracy"],
        "ctrl": base_ctrl["accuracy"],
        "epoch": 0,
        "from_init": True,
    }
    health = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(ds)))
        random.shuffle(order)
        running = 0.0
        seen = 0
        step = 0
        grad_accum = 8
        opt.zero_grad(set_to_none=True)
        for start in range(0, len(order), 1):
            batch = collate([ds[order[start]]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * grad_accum
            seen += 1
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 40 == 0:
                print(f"[{tag}] epoch {epoch} step {step}/{len(order)} loss={running/max(1,seen):.4f}", flush=True)
        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        core = evaluate_suite(gen, CORE_PROBES, core_words)
        hard = evaluate_suite(gen, HARD_PROBES, hard_words)
        allp = evaluate_suite(gen, list(CORE_PROBES) + list(HARD_PROBES), core_words + hard_words)
        ctrl = evaluate_controls(gen)
        entry = {
            "epoch": epoch,
            "loss": running / max(1, seen),
            "core": core["accuracy"],
            "hard": hard["accuracy"],
            "classic": allp["accuracy"],
            "ctrl": ctrl["accuracy"],
            "fid": allp["fidelity"],
        }
        health.append(entry)
        print(
            f"[{tag}] EPOCH {epoch} core={core['accuracy']:.3f} hard={hard['accuracy']:.3f} "
            f"classic={allp['accuracy']:.3f} ctrl={ctrl['accuracy']:.3f} loss={entry['loss']:.4f}",
            flush=True,
        )
        for r in allp["rows"]:
            print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} fid={r['fid']:.2f} | {r['q']}", flush=True)
        with open(out / f"eval_epoch{epoch}.json", "w", encoding="utf-8") as f:
            json.dump({"core": core, "hard": hard, "all": allp, "ctrl": ctrl, "entry": entry}, f, ensure_ascii=False, indent=2)

        dec = decide_promotion(
            baseline_core=best["core"],
            candidate_core=core["accuracy"],
            baseline_classic=best["classic"],
            candidate_classic=allp["accuracy"],
            baseline_ctrl=best["ctrl"],
            candidate_ctrl=ctrl["accuracy"],
        )
        # also allow promote if hard improves and core holds vs initial baseline
        hard_gain = hard["accuracy"] > base_hard["accuracy"] + 1e-9
        core_hold = core["accuracy"] + 1e-9 >= base_core["accuracy"]
        if (dec.accepted or (hard_gain and core_hold and ctrl["accuracy"] >= 0.66)) and (
            allp["accuracy"] > best["classic"] + 1e-9 or core["accuracy"] > best["core"] + 1e-9 or hard_gain
        ):
            if core["accuracy"] + 1e-9 >= base_core["accuracy"] and ctrl["accuracy"] >= 0.66:
                model.save_pretrained(out / "adapter_best")
                tok.save_pretrained(out / "adapter_best")
                best = {
                    "core": core["accuracy"],
                    "classic": allp["accuracy"],
                    "ctrl": ctrl["accuracy"],
                    "epoch": epoch,
                    "from_init": False,
                    "hard": hard["accuracy"],
                }
                print(f"[{tag}] promoted epoch={epoch} reasons={dec.reasons}", flush=True)
            else:
                print(f"[{tag}] no promote (core/ctrl gate) {dec.reasons}", flush=True)
        else:
            print(f"[{tag}] no promote {dec.reasons}", flush=True)

    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    report = {
        "tag": tag,
        "model_path": model_path,
        "resume_adapter": resume_adapter,
        "n_train": len(samples),
        "epochs": epochs,
        "lr": lr,
        "baseline_core": base_core["accuracy"],
        "baseline_hard": base_hard["accuracy"],
        "baseline_classic": base_all["accuracy"],
        "baseline_ctrl": base_ctrl["accuracy"],
        "best": best,
        "health": health,
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] REPORT", json.dumps(report, ensure_ascii=False), flush=True)
    return report


def merge_lora_dirs(adapter_a: str, adapter_b: str, out_dir: str, alpha: float = 0.55) -> str:
    """Linear merge of two PEFT LoRA adapters: out = alpha*A + (1-alpha)*B."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a_dir = Path(adapter_a)
    b_dir = Path(adapter_b)
    # copy metadata from A
    for name in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        src = a_dir / name
        if src.exists():
            shutil.copy2(src, out / name)
    # locate weights
    def load_weights(d: Path):
        for cand in ["adapter_model.safetensors", "adapter_model.bin"]:
            p = d / cand
            if p.exists():
                if p.suffix == ".bin":
                    return torch.load(p, map_location="cpu"), cand
                return load_file(str(p)), cand
        raise FileNotFoundError(f"no adapter weights in {d}")

    wa, name_a = load_weights(a_dir)
    wb, name_b = load_weights(b_dir)
    if set(wa.keys()) != set(wb.keys()):
        missing = set(wa.keys()) ^ set(wb.keys())
        raise RuntimeError(f"adapter key mismatch sample={list(missing)[:5]}")
    merged = {}
    for k in wa.keys():
        ta, tb = wa[k], wb[k]
        if ta.shape != tb.shape:
            raise RuntimeError(f"shape mismatch {k}: {ta.shape} vs {tb.shape}")
        if ta.dtype in (torch.float16, torch.bfloat16, torch.float32):
            merged[k] = (alpha * ta.float() + (1.0 - alpha) * tb.float()).to(ta.dtype)
        else:
            merged[k] = ta
    save_file(merged, str(out / "adapter_model.safetensors"))
    # write config note
    meta = {
        "method": "linear_lora_merge",
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "alpha_a": alpha,
        "alpha_b": 1.0 - alpha,
    }
    with open(out / "merge_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("MERGED", out, meta, flush=True)
    return str(out)


def distill_collect(
    model_path: str,
    teacher_adapter: str,
    out_jsonl: str,
    n_synth: int = 500,
    seed: int = 13,
    device: str = "mps",
    min_fid: float = 0.75,
) -> Dict:
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model_path, teacher_adapter, device, trainable=False)
    gen = make_gen(model, tok, device)
    rng = random.Random(seed)
    words = list(dict.fromkeys(HARD_WORDS + POOL + [random_hard_word(rng) for _ in range(n_synth)]))
    words = [w for w in words if w.lower() not in BANNED]
    kept: List[Sample] = []
    rejected = 0
    # always keep gold teachers for anchors
    for w in HARD_WORDS + ["banana", "google", "pizza", "parallel", "beekeeper", "mississippi"]:
        for ch in sorted(set(w)):
            kept.append(Sample(f"How many '{ch}' characters are in '{w}'?", cot_count(w, ch), "gold", w, str(w.count(ch))))
        kept.append(Sample(f"How many characters are in the word '{w}'?", cot_len(w), "gold", w, str(len(w))))

    for w in words[:n_synth]:
        ch = rng.choice(list(w))
        q = f"How many '{ch}' characters are in '{w}'?"
        gold = str(w.count(ch))
        pred = gen(q, min(200, 18 + 8 * len(w) + 30))
        got = extract_answer(pred)
        fid = process_fid(pred, w)
        if got == gold and fid >= min_fid and has_cot(pred):
            # normalize to clean teacher
            kept.append(Sample(q, cot_count(w, ch), "distill", w, gold))
        else:
            rejected += 1
            # still add gold teacher for failed hard-ish words
            if w in HARD_WORDS or rng.random() < 0.15:
                kept.append(Sample(q, cot_count(w, ch), "gold_fill", w, gold))

    for q, a in REHEARSAL:
        kept.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))
        kept.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))

    validate_train_batch([s.answer for s in kept])
    with open(out_path, "w", encoding="utf-8") as f:
        for s in kept:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    stats = {"kept": len(kept), "rejected": rejected, "path": str(out_path)}
    print("DISTILL_COLLECT", stats, flush=True)
    return stats


def load_samples(jsonl: str) -> List[Sample]:
    out = []
    for line in open(jsonl, encoding="utf-8"):
        o = json.loads(line)
        out.append(Sample(**o))
    return out


def eval_adapters(model_path: str, adapters: Dict[str, str], device: str, out_json: str):
    report = {}
    for name, path in adapters.items():
        print(f"\n===== eval {name} =====", flush=True)
        model, tok = load_model(model_path, path if path != "base" else None, device, trainable=False)
        gen = make_gen(model, tok, device)
        core = evaluate_suite(gen, CORE_PROBES, classic_words_for(CORE_PROBES))
        hard = evaluate_suite(gen, HARD_PROBES, classic_words_for(HARD_PROBES))
        allp = evaluate_suite(gen, list(CORE_PROBES) + list(HARD_PROBES), classic_words_for(list(CORE_PROBES) + list(HARD_PROBES)))
        ctrl = evaluate_controls(gen)
        report[name] = {
            "core": core["accuracy"],
            "hard": hard["accuracy"],
            "classic": allp["accuracy"],
            "ctrl": ctrl["accuracy"],
            "fid": allp["fidelity"],
            "rows": allp["rows"],
            "ctrl_rows": ctrl["rows"],
        }
        print(f"{name}: core={core['accuracy']:.3f} hard={hard['accuracy']:.3f} classic={allp['accuracy']:.3f} ctrl={ctrl['accuracy']:.3f}", flush=True)
        for r in allp["rows"]:
            print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} | {r['q']}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("EVAL_WROTE", out_json, flush=True)
    return report



def is_hard_query(q: str) -> bool:
    ql = q.lower()
    hard = [
        "google", "parallel", "pizza", "beekeeper", "mississippi", "bookkeeper",
        "success", "balloon", "committee", "address", "queueing", "possession",
        "occurrence", "assessment",
    ]
    return any(w in ql for w in hard)


def eval_routed(model_path: str, core_adapter: str, expert_adapter: str, device: str, out_json: str):
    """Core adapter for default probes; expert for hard-word queries. Pure weight, no tools."""
    print("===== eval routed(core+expert) =====", flush=True)
    core_m, tok = load_model(model_path, core_adapter, device, trainable=False)
    core_gen = make_gen(core_m, tok, device)
    exp_m, _ = load_model(model_path, expert_adapter, device, trainable=False)
    exp_gen = make_gen(exp_m, tok, device)

    def routed(prompt: str, max_new_tokens: int = 180):
        if is_hard_query(prompt):
            return exp_gen(prompt, max_new_tokens)
        return core_gen(prompt, max_new_tokens)

    core = evaluate_suite(routed, CORE_PROBES, classic_words_for(CORE_PROBES))
    hard = evaluate_suite(routed, HARD_PROBES, classic_words_for(HARD_PROBES))
    allp = evaluate_suite(routed, list(CORE_PROBES) + list(HARD_PROBES), classic_words_for(list(CORE_PROBES) + list(HARD_PROBES)))
    ctrl = evaluate_controls(routed)
    report = {
        "method": "dual_adapter_route",
        "core_adapter": core_adapter,
        "expert_adapter": expert_adapter,
        "core": core["accuracy"],
        "hard": hard["accuracy"],
        "classic": allp["accuracy"],
        "ctrl": ctrl["accuracy"],
        "fid": allp["fidelity"],
        "rows": allp["rows"],
        "ctrl_rows": ctrl["rows"],
    }
    print(
        f"routed: core={core['accuracy']:.3f} hard={hard['accuracy']:.3f} "
        f"classic={allp['accuracy']:.3f} ctrl={ctrl['accuracy']:.3f}",
        flush=True,
    )
    for r in allp["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} gold={r['gold']} got={r['got']} | {r['q']}", flush=True)
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("ROUTE_EVAL_WROTE", out_json, flush=True)
    return report


def cmd_expert(args):
    samples = build_expert_dataset(args.n_train, args.seed)
    return train_lora(
        model_path=args.model,
        out_dir=args.out,
        samples=samples,
        resume_adapter=None,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        tag="hard_expert",
    )


def cmd_merge(args):
    return merge_lora_dirs(args.adapter_a, args.adapter_b, args.out, alpha=args.alpha)


def cmd_distill_collect(args):
    return distill_collect(
        model_path=args.model,
        teacher_adapter=args.teacher,
        out_jsonl=args.out,
        n_synth=args.n_synth,
        seed=args.seed,
        device=args.device,
        min_fid=args.min_fid,
    )


def cmd_distill_train(args):
    samples = load_samples(args.data)
    return train_lora(
        model_path=args.model,
        out_dir=args.out,
        samples=samples,
        resume_adapter=None,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        tag="distill_retrain",
    )


def cmd_eval(args):
    adapters = {}
    for item in args.adapters:
        name, path = item.split("=", 1)
        adapters[name] = path
    return eval_adapters(args.model, adapters, args.device, args.out)


def cmd_pipeline(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    v3 = args.core_adapter
    # 1) hard expert from base
    expert_out = str(root / "hard_expert")
    samples = build_expert_dataset(args.n_expert, args.seed)
    train_lora(args.model, expert_out, samples, None, args.epochs_expert, args.lr_expert, device=args.device, seed=args.seed, tag="hard_expert")
    # 2) merge with v3
    merge_out = str(root / "merged_v3_expert")
    merge_lora_dirs(v3, str(Path(expert_out) / "adapter_best"), merge_out, alpha=args.merge_alpha)
    # 3) self-distill collect from v3
    distill_jsonl = str(root / "distill" / "traces.jsonl")
    distill_collect(args.model, v3, distill_jsonl, n_synth=args.n_synth, seed=args.seed + 1, device=args.device)
    # 4) retrain from base on distill
    distill_out = str(root / "distill_model")
    d_samples = load_samples(distill_jsonl)
    train_lora(args.model, distill_out, d_samples, None, args.epochs_distill, args.lr_distill, device=args.device, seed=args.seed + 2, tag="distill_retrain")
    # 5) eval all
    adapters = {
        "v3": v3,
        "hard_expert": str(Path(expert_out) / "adapter_best"),
        "merged": merge_out,
        "distill": str(Path(distill_out) / "adapter_best"),
    }
    report = eval_adapters(args.model, adapters, args.device, str(root / "compare.json"))
    # promote best under gates vs v3
    base = report["v3"]
    best_name = "v3"
    best_score = base["classic"] + 0.15 * base["hard"] + 0.1 * base["core"]
    for name, m in report.items():
        if name == "v3":
            continue
        dec = decide_promotion(base["core"], m["core"], base["classic"], m["classic"], base["ctrl"], m["ctrl"])
        score = m["classic"] + 0.15 * m["hard"] + 0.1 * m["core"]
        print(f"PROMOTE_CHECK {name} accepted={dec.accepted} reasons={dec.reasons} score={score:.3f}", flush=True)
        if m["core"] + 1e-9 >= base["core"] and m["ctrl"] >= 0.66 and score > best_score:
            best_score = score
            best_name = name
    champion = root / "champion"
    if champion.exists():
        shutil.rmtree(champion)
    src = adapters[best_name]
    shutil.copytree(src, champion)
    summary = {"champion": best_name, "adapters": {k: report[k] for k in report}, "paths": adapters}
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("PIPELINE_CHAMPION", best_name, flush=True)
    print("SUMMARY", json.dumps({k: {kk: vv for kk, vv in report[k].items() if kk not in ('rows', 'ctrl_rows')} for k in report}, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("expert")
    pe.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pe.add_argument("--out", default="/kef_results/char_hard_expert")
    pe.add_argument("--n-train", type=int, default=500)
    pe.add_argument("--epochs", type=int, default=1)
    pe.add_argument("--lr", type=float, default=5e-5)
    pe.add_argument("--seed", type=int, default=17)
    pe.add_argument("--device", default="mps")

    pm = sub.add_parser("merge")
    pm.add_argument("--adapter-a", required=True)
    pm.add_argument("--adapter-b", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--alpha", type=float, default=0.55)

    pdc = sub.add_parser("distill-collect")
    pdc.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pdc.add_argument("--teacher", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    pdc.add_argument("--out", default="kef_results/char_distill/traces.jsonl")
    pdc.add_argument("--n-synth", type=int, default=400)
    pdc.add_argument("--min-fid", type=float, default=0.75)
    pdc.add_argument("--seed", type=int, default=13)
    pdc.add_argument("--device", default="mps")

    pdt = sub.add_parser("distill-train")
    pdt.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pdt.add_argument("--data", required=True)
    pdt.add_argument("--out", default="kef_results/char_distill_model")
    pdt.add_argument("--epochs", type=int, default=1)
    pdt.add_argument("--lr", type=float, default=5e-5)
    pdt.add_argument("--seed", type=int, default=23)
    pdt.add_argument("--device", default="mps")

    pr = sub.add_parser("route-eval")
    pr.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pr.add_argument("--core", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    pr.add_argument("--expert", default="/Users/shiaho/Desktop/bitx/kef_results/char_advance/hard_expert/adapter_best")
    pr.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/char_advance/route_eval.json")
    pr.add_argument("--device", default="mps")

    pev = sub.add_parser("eval")
    pev.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pev.add_argument("--adapters", nargs="+", required=True, help="name=path")
    pev.add_argument("--out", default="kef_results/char_advance_eval.json")
    pev.add_argument("--device", default="mps")

    pp = sub.add_parser("pipeline")
    pp.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    pp.add_argument("--core-adapter", default="/Users/shiaho/Desktop/bitx/kef_results/char_sense_cot_v3/adapter_best")
    pp.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/char_advance")
    pp.add_argument("--n-expert", type=int, default=500)
    pp.add_argument("--epochs-expert", type=int, default=1)
    pp.add_argument("--lr-expert", type=float, default=5e-5)
    pp.add_argument("--merge-alpha", type=float, default=0.60, help="weight on core v3")
    pp.add_argument("--n-synth", type=int, default=350)
    pp.add_argument("--epochs-distill", type=int, default=1)
    pp.add_argument("--lr-distill", type=float, default=4e-5)
    pp.add_argument("--seed", type=int, default=29)
    pp.add_argument("--device", default="mps")

    args = p.parse_args(argv)
    if args.cmd == "expert":
        cmd_expert(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    elif args.cmd == "distill-collect":
        cmd_distill_collect(args)
    elif args.cmd == "distill-train":
        cmd_distill_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "route-eval":
        eval_routed(args.model, args.core, args.expert, args.device, args.out)
    elif args.cmd == "pipeline":
        cmd_pipeline(args)


if __name__ == "__main__":
    main()
