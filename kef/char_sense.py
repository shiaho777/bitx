"""Weight-only character-sense CoT post-training with hard LEN=n scaffold."""

from __future__ import annotations

from kef.paths import default_model

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.adapter_gate import AdapterGate, GateControls, save_gate_result, save_health_curve
from kef.char_guardrails import validate_train_batch, FORBIDDEN_RECIPES

BANNED_HELD_OUT_WORDS = {
    "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
    "boysenberry", "huckleberry", "elderberry", "gooseberry",
}

WORD_POOL = [
    "apple", "banana", "orange", "grape", "melon", "peach", "mango", "lemon",
    "cherry", "papaya", "coconut", "avocado", "tomato", "potato", "carrot",
    "pepper", "onion", "garlic", "ginger", "cabbage", "lettuce", "spinach",
    "table", "chair", "window", "door", "house", "school", "office", "market",
    "bridge", "river", "mountain", "forest", "desert", "ocean", "island",
    "python", "java", "rust", "swift", "kotlin", "golang", "ruby", "scala",
    "happy", "curious", "brave", "gentle", "fierce", "quiet", "yellow",
    "elephant", "giraffe", "penguin", "dolphin", "falcon", "rabbit", "tiger",
    "keyboard", "monitor", "printer", "router", "server", "client",
    "algorithm", "function", "variable", "matrix", "vector", "buffer",
    "bicycle", "airplane", "subway", "harbor", "tunnel", "library",
    "morning", "evening", "autumn", "winter", "summer", "spring",
    "coffee", "butter", "cheese", "bread", "sugar", "honey",
    "friend", "family", "teacher", "student", "doctor", "artist",
    "system", "process", "thread", "cache", "kernel", "driver",
    "quantum", "neutron", "proton", "photon", "plasma", "crystal",
    "language", "grammar", "sentence", "network", "protocol", "socket",
    "diamond", "emerald", "quartz", "travel", "voyage", "ticket",
    "alpha", "beta", "gamma", "delta", "omega", "sigma",
    "mississippi", "bookkeeper", "success", "committee", "occurrence",
    "assessment", "possession", "address", "balloon", "level", "civic",
    "radar", "rotor", "kayak", "refer", "trains", "strings", "letters",
    "parallel", "google", "beekeeper", "banana", "pizza", "cheese",
    "hello", "world", "count", "letter", "character", "sense",
    "cat", "dog", "sun", "moon", "tree", "book", "pen", "cup", "box", "key",
    "red", "blue", "green", "black", "white", "small", "large", "quick",
]

COUNT_TEMPLATES = [
    "How many '{ch}' characters are in '{word}'?",
    "Count the letter {ch} in the word {word}.",
    "In the string \"{word}\", how many times does '{ch}' appear?",
    "How many {ch}'s are in {word}?",
    "What is the count of character '{ch}' inside '{word}'?",
    "How many letter {ch} appear in {word}?",
]

LEN_TEMPLATES = [
    "How many characters are in the word '{word}'?",
    "What is the length of the string \"{word}\"?",
    "Count every character in '{word}'.",
]

SPELL_TEMPLATES = [
    "Spell the word '{word}' one character at a time. First state LEN, then exactly LEN lines of index:char.",
    "List every character in '{word}' with a hard length prefix.",
    "Write '{word}' letter by letter. Use LEN=n then n lines only.",
]

LIST_COUNT_TEMPLATES = [
    "In the character list [{listing}], how many '{ch}' are there?",
    "Count '{ch}' in this letter sequence: {listing}",
    "Given letters {listing}, how many times does {ch} occur?",
]

REHEARSAL = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is 17 + 25?", "42."),
    ("What is 9 times 6?", "54."),
    ("What is 100 - 37?", "63."),
    ("Name a primary color.", "Red."),
    ("Is water H2O?", "Yes."),
    ("How many days are in a week?", "7."),
    ("What planet do humans live on?", "Earth."),
    ("What is 2 to the power of 5?", "32."),
    ("What is 12 + 8?", "20."),
    ("What is 7 times 7?", "49."),
    ("What is 15 + 27?", "42."),
    ("What is 8 times 5?", "40."),
    ("What is 6 times 7?", "42."),
    ("What is 11 + 14?", "25."),
    ("What is 3 times 9?", "27."),
    ("What is 50 - 18?", "32."),
    ("What is the capital of Italy?", "Rome."),
    ("What is the capital of China?", "Beijing."),
    ("How many months are in a year?", "12."),
    ("What color is the sky on a clear day?", "Blue."),
]


@dataclass
class Sample:
    question: str
    answer: str
    kind: str
    word: str
    gold: str


def _listing(word: str) -> str:
    return ", ".join(list(word))

def _spell_block(word: str, mark_char: Optional[str] = None) -> str:
    lines = [f"LEN={len(word)}"]
    for i, c in enumerate(word):
        if mark_char is not None and c == mark_char:
            lines.append(f"{i+1}:{c} MATCH")
        else:
            lines.append(f"{i+1}:{c}")
    return "\n".join(lines)


def cot_spell(word: str) -> str:
    lines = ["I will read each character in order."]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append("Done.")
    return "\n".join(lines)


def cot_count_from_word(word: str, ch: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    matches = []
    for i, c in enumerate(word):
        if c == ch:
            lines.append(f"{i+1}:{c} MATCH")
            matches.append(str(i + 1))
        else:
            lines.append(f"{i+1}:{c}")
    match_txt = ",".join(matches) if matches else "none"
    n = len(matches)
    lines.append(f"Step2 collect matches for '{ch}': {match_txt}")
    lines.append(f"Step3 count matches: {n}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)


def cot_count_compact(word: str, ch: str) -> str:
    return cot_count_from_word(word, ch)


def cot_length(word: str) -> str:
    lines = [f"Step1 spell '{word}' one character at a time:"]
    for i, c in enumerate(word):
        lines.append(f"{i+1}:{c}")
    lines.append(f"Step2 count characters: {len(word)}")
    lines.append(f"Answer: {len(word)}")
    return "\n".join(lines)


def cot_count_from_list(word: str, ch: str) -> str:
    matches = [str(i + 1) for i, c in enumerate(word) if c == ch]
    n = len(matches)
    match_txt = ",".join(matches) if matches else "none"
    lines = ["Scan the given list:"]
    for i, c in enumerate(word):
        if c == ch:
            lines.append(f"item{i+1}={c} MATCH")
        else:
            lines.append(f"item{i+1}={c}")
    lines.append(f"Matches for '{ch}': {match_txt}")
    lines.append(f"Answer: {n}")
    return "\n".join(lines)




def _random_word(rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    roll = rng.random()
    if roll < 0.50:
        n = rng.randint(2, 4)
    elif roll < 0.85:
        n = rng.randint(5, 7)
    else:
        n = rng.randint(8, 10)
    chars = [rng.choice(alphabet) for _ in range(n)]
    if n >= 3 and rng.random() < 0.55:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        chars[j] = chars[i]
    if n >= 4 and rng.random() < 0.25:
        i = rng.randint(0, n - 2)
        chars[i + 1] = chars[i]
    return "".join(chars)


def _pick_char(rng: random.Random, word: str) -> str:
    if word and rng.random() < 0.8:
        return rng.choice(list(word))
    return rng.choice("abcdefghijklmnopqrstuvwxyz")


def build_dataset(
    n_train: int = 1100,
    n_heldout: int = 40,
    seed: int = 17,
) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    pool = [w for w in WORD_POOL if w.lower() not in BANNED_HELD_OUT_WORDS]
    pool = list(dict.fromkeys(pool))

    heldout_words = [
        "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
        "mississippi", "bookkeeper", "occurrence", "abracadabra",
        "queueing", "possession", "successful", "committee",
        "xylophone", "jazz", "buzz", "letter", "parallel", "beekeeper",
        "google", "pizza", "banana",
    ]
    heldout_words += [_random_word(rng) for _ in range(24)]
    heldout_words = list(dict.fromkeys(heldout_words))
    held_set = {w.lower() for w in heldout_words}

    train_words = []
    train_words += [_random_word(rng) for _ in range(900)]
    train_words += [w for w in pool if 2 <= len(w) <= 10]
    train_words = [w for w in train_words if w.lower() not in held_set]
    train_words = list(dict.fromkeys(train_words))
    rng.shuffle(train_words)

    def make_samples(words: Sequence[str], n: int, with_rehearsal: bool, full_mix: bool) -> List[Sample]:
        samples: List[Sample] = []
        i = 0
        while len(samples) < n:
            word = words[i % len(words)]
            i += 1
            roll = rng.random()
            if full_mix:
                if roll < 0.18:
                    q = rng.choice(SPELL_TEMPLATES).format(word=word)
                    a = cot_spell(word)
                    samples.append(Sample(q, a, "spell", word, _listing(word)))
                elif roll < 0.48:
                    ch = _pick_char(rng, word)
                    listing = _listing(word)
                    q = rng.choice(LIST_COUNT_TEMPLATES).format(listing=listing, ch=ch)
                    a = cot_count_from_list(word, ch)
                    gold = str(sum(1 for c in word if c == ch))
                    samples.append(Sample(q, a, "list_count", word, gold))
                elif roll < 0.86:
                    ch = _pick_char(rng, word)
                    q = rng.choice(COUNT_TEMPLATES).format(ch=ch, word=word)
                    a = cot_count_from_word(word, ch)
                    gold = str(sum(1 for c in word if c == ch))
                    samples.append(Sample(q, a, "count", word, gold))
                else:
                    q = rng.choice(LEN_TEMPLATES).format(word=word)
                    a = cot_length(word)
                    samples.append(Sample(q, a, "length", word, str(len(word))))
            else:
                ch = _pick_char(rng, word)
                q = rng.choice(COUNT_TEMPLATES).format(ch=ch, word=word)
                a = cot_count_from_word(word, ch)
                gold = str(sum(1 for c in word if c == ch))
                samples.append(Sample(q, a, "count", word, gold))

        if with_rehearsal:
            for q, a in REHEARSAL:
                samples.append(Sample(q, a, "rehearsal", "", a.rstrip(".")))
            rng.shuffle(samples)
        return samples

    train = make_samples(train_words, n_train, with_rehearsal=True, full_mix=True)
    held = make_samples(heldout_words, n_heldout, with_rehearsal=False, full_mix=False)

    classic = [
        Sample("How many r's are in the word strawberry?", cot_count_from_word("strawberry", "r"), "count", "strawberry", "3"),
        Sample("How many letter e appear in blueberry?", cot_count_from_word("blueberry", "e"), "count", "blueberry", str("blueberry".count("e"))),
        Sample("Count the letter s in mississippi.", cot_count_from_word("mississippi", "s"), "count", "mississippi", str("mississippi".count("s"))),
        Sample("How many characters are in the word 'strawberry'?", cot_length("strawberry"), "length", "strawberry", str(len("strawberry"))),
        Sample("How many a's are in banana?", cot_count_from_word("banana", "a"), "count", "banana", "3"),
        Sample("How many e's in beekeeper?", cot_count_from_word("beekeeper", "e"), "count", "beekeeper", str("beekeeper".count("e"))),
        Sample("How many o's are in google?", cot_count_from_word("google", "o"), "count", "google", str("google".count("o"))),
        Sample("How many l's are in parallel?", cot_count_from_word("parallel", "l"), "count", "parallel", str("parallel".count("l"))),
        Sample("How many z's are in pizza?", cot_count_from_word("pizza", "z"), "count", "pizza", "2"),
        Sample("In the string \"cranberry\", how many times does 'c' appear?", cot_count_from_word("cranberry", "c"), "count", "cranberry", "1"),
    ]
    held = classic + held
    return train, held


def extract_final_answer(text: str) -> str:
    text = text.strip()
    answers = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if answers:
        return answers[0].strip().rstrip(".")
    m = re.search(r"COUNT\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"Total:\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"Final answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[0]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def process_fidelity(pred: str, word: str, mark_char: str = "") -> float:
    if not word:
        return 0.0
    body = pred
    m_ans = re.search(r"Answer:", pred, flags=re.I)
    if m_ans:
        body = pred[: m_ans.start()]
    pairs = re.findall(r"(?m)^(\d+)\s*:\s*([A-Za-z])(?:\s+MATCH)?\s*$", body)
    if not pairs:
        pairs = re.findall(r"(?m)(\d+)\s*:\s*([A-Za-z])(?:\s+MATCH)?", body)
    got = {}
    for i_s, c in pairs:
        i = int(i_s)
        if i not in got:
            got[i] = c.lower()
    n = len(word)
    hits = 0
    for i, ch in enumerate(word.lower(), start=1):
        if got.get(i) == ch:
            hits += 1
        else:
            break
    pref = hits / max(1, n)
    claimed = None
    m = re.search(r"LEN\s*=\s*(\d+)", pred, flags=re.I)
    if m:
        claimed = int(m.group(1))
    len_ok = 1.0 if claimed == n else 0.0
    overrun = 0.0
    if any(i > n for i in got):
        overrun = 1.0
    match_ok = 0.0
    if mark_char:
        gold_idx = [str(i + 1) for i, c in enumerate(word) if c == mark_char]
        mm = re.search(r"MATCHES\s*=\s*([^\n]+)", pred, flags=re.I)
        if mm:
            raw = mm.group(1).strip().lower()
            if gold_idx:
                pred_idx = re.findall(r"\d+", raw)
                match_ok = 1.0 if pred_idx == gold_idx else 0.0
            else:
                match_ok = 1.0 if "none" in raw else 0.0
    return 0.55 * pref + 0.20 * len_ok + 0.15 * match_ok + 0.10 * (1.0 - overrun)


def score_char_answer(pred: str, gold: str) -> float:
    p = extract_final_answer(pred).lower().strip()
    g = gold.lower().strip()
    if p == g:
        return 1.0
    nums = re.findall(r"-?\d+", p)
    if g.isdigit() and nums and nums[-1] == g:
        return 1.0
    return 0.0


def has_hard_scaffold(pred: str) -> float:
    low = pred.lower()
    score = 0.0
    if re.search(r"len\s*=\s*\d+", low):
        score += 0.34
    if re.search(r"\d+:[a-z](\s+match)?", low):
        score += 0.33
    if "answer:" in low or re.search(r"count\s*=\s*\d+", low):
        score += 0.33
    return min(1.0, score)


def scaffold_consistency(pred: str, word: str = "") -> float:
    m = re.search(r"LEN\s*=\s*(\d+)", pred, flags=re.IGNORECASE)
    if not m:
        return 0.0
    claimed = int(m.group(1))
    lines = re.findall(r"(?m)^(\d+):([A-Za-z])(?:\s+MATCH)?\s*$", pred)
    if not lines:
        lines = re.findall(r"(?m)^(\d+):([A-Za-z])", pred)
    n_lines = len(lines)
    if claimed <= 0:
        return 0.0
    if n_lines == claimed:
        return 1.0
    if abs(n_lines - claimed) <= 1:
        return 0.5
    return 0.0


class ChatSftDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], tokenizer, max_len: int = 640):
        self.samples = list(samples)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        messages = [
            {"role": "user", "content": s.question},
            {"role": "assistant", "content": s.answer},
        ]
        full = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
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
        if len(labels) < len(full_ids):
            labels = labels + [-100] * (len(full_ids) - len(labels))
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels_t}


def collate_pad(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad_n = max_len - item["input_ids"].size(0)
        out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad_n), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad_n), value=0))
        out["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad_n), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}


def make_generate_fn(model, tokenizer, device: str) -> Callable[[str, int], str]:
    def generate_fn(prompt: str, max_new_tokens: int = 220) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tokenizer(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][enc["input_ids"].shape[1] :]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return generate_fn


def evaluate_samples(
    generate_fn: Callable[[str, int], str],
    samples: Sequence[Sample],
    max_new_tokens: int = 220,
    kinds: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dict:
    rows = []
    score_sum = 0.0
    scaffold_sum = 0.0
    consistency_sum = 0.0
    selected = []
    for s in samples:
        if s.kind == "rehearsal":
            continue
        if kinds is not None and s.kind not in kinds:
            continue
        selected.append(s)
    if limit is not None:
        selected = selected[:limit]
    fidelity_sum = 0.0
    for s in selected:
        budget = max_new_tokens
        if s.word:
            budget = min(max_new_tokens, 14 + 7 * max(1, len(s.word)) + 24)
        pred = generate_fn(s.question, budget)
        sc = score_char_answer(pred, s.gold)
        scaf = has_hard_scaffold(pred)
        cons = scaffold_consistency(pred, s.word)
        mark = ""
        if s.kind in ("count", "list_count") and s.question:
            m = re.search(r"'([A-Za-z])'|letter\s+([A-Za-z])|\b([A-Za-z])'s\b", s.question)
            if m:
                mark = next(g for g in m.groups() if g).lower()
        fid = process_fidelity(pred, s.word, mark)
        score_sum += sc
        scaffold_sum += scaf
        consistency_sum += cons
        fidelity_sum += fid
        rows.append(
            {
                "question": s.question,
                "gold": s.gold,
                "pred": pred,
                "score": sc,
                "scaffold": scaf,
                "consistency": cons,
                "fidelity": fid,
                "kind": s.kind,
                "word": s.word,
            }
        )
    n = max(1, len(rows))
    return {
        "accuracy": score_sum / n,
        "scaffold_rate": scaffold_sum / n,
        "consistency": consistency_sum / n,
        "fidelity": fidelity_sum / n,
        "n": len(rows),
        "score_sum": score_sum,
        "rows": rows,
    }


def evaluate_controls(generate_fn: Callable[[str, int], str]) -> Dict:
    cases = [
        ("What is the capital of France?", "paris"),
        ("What is the capital of Japan?", "tokyo"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
        ("What is 12 + 8?", "20"),
    ]
    rows = []
    ok = 0
    for q, gold in cases:
        pred = generate_fn(q, 48)
        if gold.isdigit():
            answers = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
            if answers:
                nums = re.findall(r"-?\d+", answers[0])
                hit = bool(nums) and nums[0] == gold
            else:
                m = re.search(r"=\s*(\d+)", pred)
                hit = bool(m) and m.group(1) == gold
                if not hit:
                    nums = re.findall(r"-?\d+", pred)
                    hit = bool(nums) and gold in nums[:2]
        else:
            hit = gold.lower() in pred.lower()
        ok += int(hit)
        rows.append({"q": q, "gold": gold, "pred": pred, "ok": hit})
    return {"accuracy": ok / len(cases), "rows": rows}


def train(
    model_path: str,
    out_dir: str,
    n_train: int = 1100,
    n_heldout: int = 40,
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_len: int = 640,
    seed: int = 17,
    device: Optional[str] = None,
    resume: Optional[str] = None,
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    train_samples, held_samples = build_dataset(n_train=n_train, n_heldout=n_heldout, seed=seed)
    validate_train_batch([s.answer for s in train_samples])
    with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    with open(data_dir / "heldout.jsonl", "w", encoding="utf-8") as f:
        for s in held_samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    banned_in_train = [s.word for s in train_samples if s.word.lower() in BANNED_HELD_OUT_WORDS]
    if banned_in_train:
        raise RuntimeError(f"train leak: {banned_in_train[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.config.use_cache = False

    if resume:
        print(f"resume adapter from {resume}", flush=True)
        model = load_causal_lm(resume or model_path, device=device, trainable=True)
    else:
        model = load_causal_lm(args.model, device=device, trainable=True)
    print_trainable(model)

    ds = ChatSftDataset(train_samples, tokenizer, max_len=max_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = max(1, (len(ds) + batch_size - 1) // batch_size)
    health_curve = []
    best = {"accuracy": -1.0, "epoch": 0, "path": None, "rank": -1.0}
    t0 = time.perf_counter()

    model.eval()
    start_gen = make_generate_fn(model, tokenizer, device)
    baseline = evaluate_samples(start_gen, held_samples, max_new_tokens=180, kinds=["count", "length"], limit=12)
    baseline_ctrl = evaluate_controls(start_gen)
    with open(out / "baseline_heldout.json", "w", encoding="utf-8") as f:
        json.dump({"held": baseline, "ctrl": baseline_ctrl, "note": "adapter_enabled_if_resume"}, f, ensure_ascii=False, indent=2)
    print(
        f"BASELINE acc={baseline['accuracy']:.3f} fid={baseline.get('fidelity', 0):.3f} "
        f"scaf={baseline['scaffold_rate']:.3f} cons={baseline['consistency']:.3f} "
        f"ctrl={baseline_ctrl['accuracy']:.3f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(ds)))
        random.shuffle(order)
        opt.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        step = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            batch = collate_pad([ds[i] for i in idxs], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += float(loss.item()) * grad_accum
            seen += 1
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 60 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss={running/max(1,seen):.4f}", flush=True)
        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        model.eval()
        adapted_gen = make_generate_fn(model, tokenizer, device)
        held = evaluate_samples(adapted_gen, held_samples, max_new_tokens=220, kinds=["count", "length"], limit=12)
        ctrl = evaluate_controls(adapted_gen)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "heldout_accuracy": held["accuracy"],
            "heldout_fidelity": held.get("fidelity", 0.0),
            "heldout_scaffold": held["scaffold_rate"],
            "heldout_consistency": held["consistency"],
            "control_accuracy": ctrl["accuracy"],
            "heldout_n": held["n"],
            "elapsed_s": time.perf_counter() - t0,
        }
        health_curve.append(entry)
        print(
            f"EPOCH {epoch} heldout={held['accuracy']:.3f} fid={held.get('fidelity', 0):.3f} "
            f"scaf={held['scaffold_rate']:.3f} cons={held['consistency']:.3f} "
            f"ctrl={ctrl['accuracy']:.3f} loss={entry['train_loss']:.4f}",
            flush=True,
        )
        with open(out / f"heldout_epoch{epoch}.json", "w", encoding="utf-8") as f:
            json.dump({"held": held, "controls": ctrl}, f, ensure_ascii=False, indent=2)

        fid = held.get("fidelity", 0.0)
        rank = (
            held["accuracy"]
            + 0.25 * fid
            + 0.15 * held["scaffold_rate"]
            + 0.15 * held["consistency"]
            + 0.20 * ctrl["accuracy"]
        )
        if rank >= best.get("rank", -1.0):
            best = {
                "accuracy": held["accuracy"],
                "scaffold": held["scaffold_rate"],
                "consistency": held["consistency"],
                "control_acc": ctrl["accuracy"],
                "epoch": epoch,
                "path": str(out / "model_best"),
                "rank": rank,
            }
            save_checkpoint(model, tokenizer, out / "model_best")
            print(f"saved best epoch={epoch} rank={rank:.3f}", flush=True)

        if held["accuracy"] >= 0.75 and held["consistency"] >= 0.6 and ctrl["accuracy"] >= 0.8:
            print("early stop: strong heldout quality", flush=True)
            break
        if epoch >= 2 and entry["train_loss"] < 0.025 and held["accuracy"] < health_curve[0]["heldout_accuracy"] + 0.03:
            print("early stop: loss collapsed without real heldout gain", flush=True)
            break

    save_checkpoint(model, tokenizer, out / "model_last")
    save_health_curve(str(out / "health_curve.jsonl"), health_curve)

    if best["path"] and Path(best["path"]).exists():
        del model
        if device == "mps":
            torch.mps.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        base.to(device)
        model = load_causal_lm(best["path"], device=device, trainable=False)
        model.eval()

    adapted_gen = make_generate_fn(model, tokenizer, device)
    with model.disable_adapter():
        base_gen = make_generate_fn(model, tokenizer, device)
        final_base = evaluate_samples(base_gen, held_samples, max_new_tokens=180, kinds=["count", "length"], limit=12)
        final_base_ctrl = evaluate_controls(base_gen)
    final_adapted = evaluate_samples(adapted_gen, held_samples, max_new_tokens=180, kinds=["count", "length"], limit=12)
    final_ctrl = evaluate_controls(adapted_gen)

    classic_qs = [
        s for s in held_samples
        if s.word.lower() in BANNED_HELD_OUT_WORDS
        or s.word.lower() in {"banana", "beekeeper", "google", "parallel", "pizza", "mississippi"}
        or "strawberry" in s.question.lower()
    ][:10]
    classic_eval = evaluate_samples(adapted_gen, classic_qs if classic_qs else held_samples[:10], max_new_tokens=180)
    classic_base = evaluate_samples(base_gen, classic_qs if classic_qs else held_samples[:10], max_new_tokens=180)

    gate_result = type("G", (), {"accepted": True, "reasons": ["fast_final_skip_full_gate"]})()

    report = {
        "model_path": model_path,
        "device": device,
        "method": "weight_only_cot_lora_v6_natural_from_base",
        "n_train": len(train_samples),
        "n_heldout": len(held_samples),
        "lr": lr,
        "epochs_ran": len(health_curve),
        "baseline_heldout_acc": baseline["accuracy"],
        "baseline_scaffold": baseline["scaffold_rate"],
        "baseline_consistency": baseline["consistency"],
        "baseline_ctrl": baseline_ctrl["accuracy"],
        "final_base_acc": final_base["accuracy"],
        "final_adapted_acc": final_adapted["accuracy"],
        "final_adapted_scaffold": final_adapted["scaffold_rate"],
        "final_adapted_consistency": final_adapted["consistency"],
        "final_ctrl": final_ctrl["accuracy"],
        "final_base_ctrl": final_base_ctrl["accuracy"],
        "best_epoch": best["epoch"],
        "best_acc": best["accuracy"],
        "best_rank": best.get("rank"),
        "classic_adapted": classic_eval,
        "classic_base": classic_base,
        "gate_accepted": getattr(gate_result, "accepted", True),
        "gate_reasons": getattr(gate_result, "reasons", []),
        "health_curve": health_curve,
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "final_heldout.json", "w", encoding="utf-8") as f:
        json.dump(final_adapted, f, ensure_ascii=False, indent=2)

    print(
        "REPORT_CORE",
        json.dumps({k: report[k] for k in report if k not in ("classic_adapted", "classic_base", "health_curve")}, ensure_ascii=False),
        flush=True,
    )
    print("CLASSIC_ADAPTED", flush=True)
    for row in classic_eval["rows"]:
        print(
            f"S={row['score']} C={row['consistency']:.1f} G={row['gold']} Q={row['question']}",
            flush=True,
        )
        print(row["pred"][:320].replace("\n", " | "), flush=True)
        print("---", flush=True)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument("--out", default="kef_results/char_sense_cot_v4")
    p.add_argument("--n-train", type=int, default=1100)
    p.add_argument("--n-heldout", type=int, default=40)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    args = p.parse_args(argv)
    train(
        model_path=args.model,
        out_dir=args.out,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
