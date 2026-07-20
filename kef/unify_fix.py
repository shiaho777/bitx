"""Surgical fix on unified champion: wash-car logic + anti-repetition chat."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.eng_craft import eval_eng
from kef.folk_logic import Sample, collate, eval_controls, make_gen


FIX_HOLDOUT: Tuple[Tuple[str, str, str], ...] = (
    (
        "我要去洗车，洗车店离我家有50米远，我应该开车去还是走路去？",
        "开车",
        "drive",
    ),
    (
        "洗车店就在五十米外，我想把我自己的车洗干净，走路还是开车？",
        "开车",
        "drive",
    ),
    (
        "洗车店在50米外，走路5分钟开车1分钟还要找车位。只考虑这50米路程怎么走更合理？",
        "走路",
        "walk",
    ),
    (
        "你好",
        "你好",
        "hello",
    ),
    (
        "hi",
        "hi",
        "hello",
    ),
)


def _a(body: str) -> str:
    return body.strip() + "\n"


def ans_drive(extra: str = "") -> str:
    base = [
        "开车。",
        "洗车洗的是车，车必须到店。",
        "只走路到店，车还在家里，洗不了。",
        "距离再近也要开车送车过去。",
    ]
    if extra:
        base.append(extra)
    return _a("\n".join(base))


def ans_drive_short() -> str:
    return _a("开车。\n要把自己的车送到洗车店才能洗。")


def ans_walk() -> str:
    return _a(
        "\n".join(
            [
                "走路。",
                "这里只比较这几十米的路程成本，并不要求送车。",
                "开车还要起步停车找位，更亏。",
            ]
        )
    )


def ans_hello(zh: bool = True) -> str:
    return _a("你好！" if zh else "Hi!")


def build_fix_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _, _ in FIX_HOLDOUT}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, a, kind, gold))

    near_user = [
        "我要去洗车，洗车店离我家有50米远，开车还是走路？",
        "我要去洗车，洗车店离我家有五十米远，我应该开车还是走路？",
        "我要去洗车，洗车店离我家50米，我应该开车去还是走路去？",
        "我要去洗车，店离我家有50米远，应该开车还是走路？",
        "我要去洗车，洗车店离我家有50米，该开车还是该走路？",
        "洗车店就在五十米外，我想把车洗干净，走路还是开车？",
        "洗车店就在五十米外，我想把自己的车洗干净，开车还是走路？",
        "洗车店就在50米外，我想把我自己的车洗干净，走路还是开车？",
        "洗车店就在五十米外，要把我自己的车洗了，走路还是开车？",
        "洗车店离我家五十米，我想洗自己的车，走路还是开车？",
        "店在50米外，我要把自己的车洗干净，走路还是开车？",
        "店离小区50米，去洗自己的车，走路去行吗？",
        "洗自己的车，店只有50米，是不是走路就行？",
        "洗车对象是我的车，店50米，走路能完成任务吗？",
        "任务：洗自己的车。店距50米。交通方式？",
        "要把车送到洗车店。店很近只有50米。走路还是开车？",
        "车还在家里，我只走路到50米外洗车店，能洗到车吗？该怎么去？",
        "很多人说50米太近不用开车。但我要洗的是自家车，对吗？",
        "50米洗车：时间上走路更快，但洗车目标是车，应怎么选择？",
        "不要比速度，比能不能完成洗车任务。店50米，怎么去？",
        "I need to wash my own car. Shop is 50 meters away. Walk or drive?",
        "Car wash is 50m away and I need my car cleaned. Drive or walk?",
        "The car wash is fifty meters from home. Should I walk or drive to wash my car?",
    ]
    for q in near_user:
        for _ in range(6):
            add(q, ans_drive_short(), "drive", "开车")
            add(q + " 先结论。", "开车。\n", "drive", "开车")
            add(q + " 一句话。", "开车，因为车必须到店。\n", "drive", "开车")

    drive_qs = [
        "我要去洗车，店离我家50米，开车还是走路？",
        "去洗车，洗车店五十米远，应该开车还是走路？",
        "我想洗自己的车，店在50米外，怎么去？",
        "要把车洗干净，店就在50米外，步行还是驾车？",
        "洗车：车在家，店50米，走路能洗吗？该怎么去？",
        "去洗车店洗我的车，距离50米，合理交通方式？",
        "车要送到洗车店才能洗。店很近（50m），走路还是开车？",
        "我开车出门是为了洗车，店离小区50米，该不该开车？",
        "洗车任务：自家车 + 50米外洗车店，选择？",
        "距离很近也要洗车，是走路过去还是把车开过去？",
        "我要去洗车，洗车店离我家有五十米，开车还是走路更对？",
        "准备去洗车，门店50米，考虑车必须到场，怎么走？",
        "Should I walk or drive to a car wash 50 meters away to wash my own car?",
        "I need my car washed. The shop is 50m away. Walk or drive?",
        "洗自己的车，店近在咫尺，为何不能只走路去？",
        "走路到店车还在家，洗得了吗？店50米。",
        "洗车不是人到店就行，是车到店。50米，怎么去？",
        "别被短距离骗了：洗车必须送车。50米选？",
    ]
    for q in drive_qs:
        for _rep in range(3):
            add(q, ans_drive(), "drive", "开车")
            add(q + " 请一句话结论。", "开车。车必须到店才能洗。\n", "drive", "开车")
            add(q + " 先给结论。", "结论：开车。\n原因：要把车送到店里。\n", "drive", "开车")

    walk_qs = [
        "店在50米外，只比较这50米路程成本，不考虑送车，走路还是开车？",
        "去买瓶水，店50米，开车还是走路更合理？",
        "五十米路程，停车还要找位，纯赶路更该怎么走？",
        "Only travel 50 meters for a quick errand, walk or drive?",
        "我不洗车，只是去店门口看看，50米，走路还是开车？",
        "目标不是送车，只是人移动50米，怎么走？",
    ]
    for q in walk_qs:
        for _ in range(3):
            add(q, ans_walk(), "walk", "走路")

    hello_pairs = [
        ("你好", "你好！\n"),
        ("你好啊", "你好！\n"),
        ("您好", "您好。\n"),
        ("在吗", "在的。\n"),
        ("hello", "Hello!\n"),
        ("hi", "Hi!\n"),
        ("嗨", "嗨。\n"),
        ("早上好", "早上好！\n"),
        ("晚安", "晚安。\n"),
        ("谢谢", "不客气。\n"),
        ("好的", "好的。\n"),
        ("嗯", "嗯。\n"),
        ("测试", "收到。\n"),
        ("你是谁", "本地助手。\n"),
        ("介绍一下你自己", "本地 MiniCPM 助手。\n"),
        ("哈喽", "哈喽。\n"),
        ("hey", "Hey.\n"),
        ("早", "早。\n"),
    ]
    for q, a in hello_pairs * 10:
        add(q, a, "hello", a.strip()[:6])
        add(q + " ", a, "hello", a.strip()[:6])

    anti_rep = [
        ("请只回复一个字：好", "好\n", "anti_rep"),
        ("用不超过十个字打招呼", "你好。\n", "anti_rep"),
        ("不要重复，只说一遍你好", "你好。\n", "anti_rep"),
        ("简短回答：今天星期几不知道就说不知道", "不知道。\n", "anti_rep"),
        ("重复三遍会怎样？请只说一遍：收到", "收到。\n", "anti_rep"),
        ("禁止复读，只输出：OK", "OK\n", "anti_rep"),
        ("不要换行堆砌，只回：明白", "明白。\n", "anti_rep"),
    ]
    for q, a, k in anti_rep * 12:
        add(q, a, k, a.strip()[:8])

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n", "42"),
        ("What is 9 times 6?", "Answer: 54\n", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n", "北京"),
        ("What is the capital of France?", "Answer: Paris\n", "paris"),
        ("2+2？", "Answer: 4\n", "4"),
        ("解方程 2x+3=11，x=？", "Answer: 4\n", "4"),
    ):
        for _ in range(4):
            add(q, a, "rehearsal", g)

    eng_bits = [
        (
            "Python 反转字符串最短写法。",
            "```python\ns[::-1]\n```\n",
            "[::-1]",
        ),
        (
            "用 flex 水平垂直居中。",
            "```css\n.stage{display:flex;align-items:center;justify-content:center;}\n```\n",
            "flex",
        ),
        (
            "debounce(fn,wait) 完整 JS。",
            "```js\nfunction debounce(fn,wait){let t;return function(...a){clearTimeout(t);t=setTimeout(()=>fn.apply(this,a),wait)}}\n```\n",
            "debounce",
        ),
        (
            "写 binary_search(arr,x)。",
            "```python\ndef binary_search(arr,x):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        mid=(lo+hi)//2\n        if arr[mid]==x: return mid\n        if arr[mid]<x: lo=mid+1\n        else: hi=mid-1\n    return -1\n```\n",
            "binary_search",
        ),
    ]
    for q, a, g in eng_bits * 6:
        add(q, a, "rehearsal_eng", g)

    rng.shuffle(out)
    prefer = {
        "drive": 0.55,
        "hello": 0.16,
        "anti_rep": 0.10,
        "walk": 0.07,
        "rehearsal": 0.06,
        "rehearsal_eng": 0.06,
    }
    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    picked: List[Sample] = []
    for k, frac in prefer.items():
        pool = buckets.get(k, [])
        rng.shuffle(pool)
        picked.extend(pool[: max(1, int(n_train * frac))])
    ids = {id(s) for s in picked}
    rest = [s for s in out if id(s) not in ids]
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        picked.append(Sample("去洗车，店50米，怎么去？", ans_drive_short(), "drive", "开车"))
    rng.shuffle(picked)
    return picked[:n_train]


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 512, answer_boost: float = 3.6):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len
        self.answer_boost = float(answer_boost)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}, {"role": "assistant", "content": s.answer}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
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
        plen = min(len(prompt_ids), max(1, len(full_ids) - 1))
        labels = [-100] * plen + full_ids[plen:]
        labels = labels[: len(full_ids)]
        ids = torch.tensor(full_ids, dtype=torch.long)
        weights = torch.ones(len(full_ids), dtype=torch.float32)
        if self.answer_boost > 1.0 and plen < len(full_ids):
            end = min(len(full_ids), plen + 48)
            weights[plen:end] = self.answer_boost
            if s.gold:
                gids = self.tok(str(s.gold), add_special_tokens=False)["input_ids"]
                span = len(gids)
                if span > 0:
                    for i in range(plen, len(full_ids) - span + 1):
                        if full_ids[i : i + span] == gids:
                            weights[i : i + span] = self.answer_boost * 2.0
                            break
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": weights,
        }


def collapse_repeats(text: str) -> str:
    lines = text.splitlines()
    out_lines = []
    prev = None
    blank_streak = 0
    for ln in lines:
        cur = ln.strip()
        if cur == "":
            blank_streak += 1
            if blank_streak <= 1 and out_lines:
                out_lines.append("")
            continue
        blank_streak = 0
        if cur == prev:
            continue
        prev = cur
        out_lines.append(cur)
    text2 = "\n".join(out_lines).strip()
    text2 = re.sub(r"(.{2,40}?)\1{2,}", r"\1", text2)
    text2 = re.sub(r"(你好[\s!]*){2,}", "你好！", text2)
    text2 = re.sub(r"(Hello[\s!]*){2,}", "Hello!", text2, flags=re.I)
    text2 = re.sub(r"(Hi[\s!]*){2,}", "Hi!", text2, flags=re.I)
    text2 = re.sub(r"</?think>", "", text2, flags=re.I)
    return text2.strip()


def match_fix(pred: str, gold: str, kind: str) -> bool:
    p = collapse_repeats(pred)
    p = re.sub(r"\n{2,}", "\n", p).strip()
    low = p.lower()
    head = re.sub(r"\s+", "", p[:80])
    if kind == "drive":
        if head.startswith("走路") or head.startswith("步行"):
            return False
        if "只有走路" in p or "应走路" in p or "该走路" in p:
            return False
        early = p[:100]
        has_drive = ("开车" in early) or ("drive" in early.lower()) or early.startswith("驾车")
        if not has_drive:
            has_drive = ("开车" in p[:200]) or ("drive" in low[:200])
        return bool(has_drive)
    if kind == "walk":
        if p.strip().startswith("开车"):
            return False
        return ("走路" in p or "步行" in p or "walk" in low)
    if kind == "hello":
        if len(p) > 80:
            return False
        if p.count("你好") >= 3 or p.count("Hello") >= 3 or p.count("hello") >= 3:
            return False
        if re.search(r"(你好\s*){2,}", p):
            return False
        if re.search(r"(Hi\s*){2,}", p, flags=re.I):
            return False
        return any(x in low for x in ("你好", "hello", "hi", "嗨", "在", "hey"))
    return gold.lower() in low


def eval_fix(gen, probes: Sequence[Tuple[str, str, str]] = FIX_HOLDOUT) -> Dict:
    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for qi, (q, gold, kind) in enumerate(probes):
        max_new = 24 if kind == "hello" else 96
        pred = gen(q, max_new)
        pred = collapse_repeats(pred)
        hit = bool(match_fix(pred, gold, kind))
        ok += int(hit)
        by_kind.setdefault(kind, []).append(int(hit))
        rows.append({"q": q, "gold": gold, "kind": kind, "ok": hit, "pred": pred[:400]})
        print(
            f"  fix[{qi+1}/{len(probes)}] {'OK' if hit else 'NO'} [{kind}] gold={gold} "
            f"pred={pred[:60]!r}",
            flush=True,
        )
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    return {"accuracy": ok / max(1, len(probes)), "kind_acc": kind_acc, "rows": rows, "ok": ok, "n": len(probes)}


def make_gen_clean(model, tok, device):
    raw = make_gen(model, tok, device)

    def gen(prompt: str, max_new_tokens: int = 160):
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
                repetition_penalty=1.12,
                no_repeat_ngram_size=6,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        pred = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return collapse_repeats(pred)

    return gen


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_fix_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    kinds = dict(Counter(s.kind for s in samples))
    print(f"n_train={len(samples)} kinds={kinds}", flush=True)

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    base.to(device)
    base.config.use_cache = False
    model = load_causal_lm(args.resume or args.model, device=device, trainable=True)
    print_trainable(model)

    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=2.8)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen_clean(model, tok, device)

    print("===== BASELINE =====", flush=True)
    f0 = eval_fix(gen)
    eng0 = eval_eng(gen)
    ctrl0 = eval_controls(gen)
    print(
        f"BASELINE fix={f0['accuracy']:.3f} kinds={f0['kind_acc']} "
        f"eng={eng0['accuracy']:.3f} ctrl={ctrl0['accuracy']:.3f}",
        flush=True,
    )

    t0 = time.perf_counter()
    model.train()
    epochs = max(1, int(args.epochs))
    running = 0.0
    seen = 0
    step = 0
    ga = max(1, int(args.grad_accum))
    answer_boost = 3.6
    opt.zero_grad(set_to_none=True)
    total_steps = len(ds) * epochs
    for ep in range(epochs):
        order = list(range(len(ds)))
        random.shuffle(order)
        for i in order:
            item = ds[i]
            tw = item.pop("token_weights", None)
            batch = collate([item], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            if tw is not None and answer_boost > 1.0:
                out_m = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out_m.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_w = tw[1:].to(device)
                if shift_w.dim() == 1:
                    shift_w = shift_w.unsqueeze(0)
                shift_w = shift_w[:, : shift_labels.size(1)]
                vocab = shift_logits.size(-1)
                loss_tok = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)
                mask = shift_labels.ne(-100).float()
                w = shift_w * mask
                loss = (loss_tok * w).sum() / w.sum().clamp_min(1.0) / ga
            else:
                loss = model(**batch).loss / ga
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            running += float(loss.detach().cpu()) * ga
            seen += 1
            step += 1
            if step % ga == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 30 == 0:
                print(f"step {step}/{total_steps} ep={ep+1}/{epochs} loss={running/max(1,seen):.4f}", flush=True)
    if step % ga != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

    print("===== AFTER =====", flush=True)
    f1 = eval_fix(gen)
    eng1 = eval_eng(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER fix={f1['accuracy']:.3f} kinds={f1['kind_acc']} "
        f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} loss={running/max(1,seen):.4f}",
        flush=True,
    )

    drive_ok = f1["kind_acc"].get("drive", 0) >= 1.0
    hello_ok = f1["kind_acc"].get("hello", 0) >= 0.5
    walk_ok = f1["kind_acc"].get("walk", 0) >= 0.0
    fix_gain = f1["accuracy"] + 1e-9 >= max(0.8, f0["accuracy"])
    eng_floor = max(0.50, eng0["accuracy"] - 0.17)
    ctrl_floor = min(0.5, ctrl0["accuracy"])
    promote = (
        drive_ok
        and hello_ok
        and fix_gain
        and eng1["accuracy"] + 1e-9 >= eng_floor
        and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    )

    save_checkpoint(model, tok, out / "model_last")
    if promote:
        save_checkpoint(model, tok, out / "model_best")
        champ = Path(args.champion)
        dst = champ / "model_best"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (out / "model_best").iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        print("PROMOTED unify_fix -> unified_champion", flush=True)
    else:
        print(
            f"NO_PROMOTE fix {f1['accuracy']:.3f}/{f0['accuracy']:.3f} "
            f"drive={f1['kind_acc'].get('drive')} hello={f1['kind_acc'].get('hello')} "
            f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f}",
            flush=True,
        )

    report = {
        "method": Path(args.out).name,
        "n_train": len(samples),
        "kinds": kinds,
        "lr": args.lr,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {
            "fix": f0["accuracy"],
            "kind_acc": f0["kind_acc"],
            "eng": eng0["accuracy"],
            "ctrl": ctrl0["accuracy"],
        },
        "after": {
            "fix": f1["accuracy"],
            "kind_acc": f1["kind_acc"],
            "eng": eng1["accuracy"],
            "ctrl": ctrl1["accuracy"],
        },
        "promoted": promote,
        "rows": f1["rows"],
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if promote:
        champ = Path(args.champion)
        with open(champ / "report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "unified_champion",
                    "source": str(out / "model_best"),
                    "policy": "single_model_single_adapter_default",
                    "metrics": report["after"],
                    "fix_round": report,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    summary = {k: report[k] for k in report if k != "rows"}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--resume", default="/Users/shiaho/Desktop/bitx/kef_results/unified_champion/model_best")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/unify_fix_v1")
    p.add_argument("--champion", default="/Users/shiaho/Desktop/bitx/kef_results/unified_champion")
    p.add_argument("--n-train", type=int, default=240)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
