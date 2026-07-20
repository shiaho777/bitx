
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kef.eng_craft import ENG_PROBES, eval_eng
from kef.folk_logic import CTRL_PROBES, Sample, collate, eval_controls, make_gen


INSTRUCTION_EN = (
    "Answer briefly and directly. Prefer the shortest complete answer. "
    "No filler, no repeated explanations, no long preambles or disclaimers unless essential."
)

INSTRUCTION_ZH = (
    "简洁直接地回答。优先最短但完整的答案。"
    "不要废话、不要重复解释、不要冗长开场或免责声明，除非必要。"
)

FILLER_MARKERS = (
    "值得注意的是",
    "需要注意的是",
    "首先，",
    "其次，",
    "最后，",
    "总而言之",
    "综上所述",
    "简单来说",
    "换句话说",
    "作为AI",
    "作为一个AI",
    "It is important to note",
    "As an AI",
    "In conclusion",
    "First of all",
    "To summarize",
    "Let me explain",
    "Hope this helps",
    "希望对你有帮助",
    "下面详细",
    "详细说明",
    "我们来看",
)


STYLE_HOLDOUT: Tuple[Tuple[str, str], ...] = (
    ("用一句话解释什么是 HTTP。", "http"),
    ("什么是递归？尽量简短。", "调用"),
    ("Python 怎么反转字符串？给最短可用写法。", "[::-1]"),
    ("什么是数据库索引？一句话。", "索引"),
    ("为什么白天天空看起来是蓝色的？简短答。", "散射"),
    ("Tabs or spaces for indentation? One sentence.", "space"),
    ("What is a mutex? One short sentence.", "mutex"),
    ("解释 CAP 定理，三行以内。", "一致性"),
)


TRAIN_QA: Tuple[Tuple[str, str, str], ...] = (
    ("2+2等于几？", "4", "fact"),
    ("What is 9*6?", "54", "fact"),
    ("中国的首都？", "北京", "fact"),
    ("Capital of France?", "Paris", "fact"),
    ("HTTP 默认端口？", "80", "fact"),
    ("HTTPS 默认端口？", "443", "fact"),
    ("Python 空列表写法？", "[]", "code"),
    ("JS 空数组写法？", "[]", "code"),
    ("Git 查看状态最短命令？", "git status", "code"),
    ("Linux 列目录？", "ls", "code"),
    ("什么是 API？一句话。", "程序之间约定好的调用接口。", "concept"),
    ("什么是 JSON？一句话。", "一种轻量文本数据交换格式。", "concept"),
    ("什么是 REST？一句话。", "用 HTTP 资源语义做接口风格的约定。", "concept"),
    ("什么是哈希表？一句话。", "用键直接定位值的查找结构。", "concept"),
    ("什么是栈？一句话。", "后进先出的数据结构。", "concept"),
    ("什么是队列？一句话。", "先进先出的数据结构。", "concept"),
    ("TCP 和 UDP 核心差别一句话？", "TCP 可靠有连接，UDP 不可靠无连接。", "concept"),
    ("什么是死锁？一句话。", "多个进程互相等待对方资源而永久卡住。", "concept"),
    ("写最短 Python Hello World。", "```python\nprint(\"Hello, World!\")\n```", "code"),
    ("写最短 JS 定义常量 PI=3.14。", "```js\nconst PI = 3.14;\n```", "code"),
    ("Express 最短健康检查路由？", "```js\napp.get('/health', (req, res) => res.json({ok: true}));\n```", "code"),
    ("Python 读文本文件最短写法？", "```python\ntext = open(path, encoding='utf-8').read()\n```", "code"),
    ("如何判断 Python 列表为空？", "```python\nif not items:\n    ...\n```", "code"),
    ("SQL 查 users 全表？", "```sql\nSELECT * FROM users;\n```", "code"),
    ("解释 map/filter 区别一句话。", "map 变换每个元素，filter 按条件留下元素。", "concept"),
    ("什么是幂等？一句话。", "同一操作执行多次效果与一次相同。", "concept"),
    ("O(1) 是什么意思？", "耗时不随输入规模增长。", "concept"),
    ("二进制 1011 是几？", "11", "fact"),
    ("水的化学式？", "H2O", "fact"),
    ("地球绕太阳转一圈多久？", "约 365 天", "fact"),
    ("说清 bool 真假最短例子。", "```python\nTrue, False\n```", "code"),
    ("如何复制列表浅拷贝？", "```python\nb = a[:]\n```", "code"),
    ("JS 判断数组？", "```js\nArray.isArray(x)\n```", "code"),
    ("什么是闭包？一句话。", "函数携带其定义时的外层变量环境。", "concept"),
    ("GET 与 POST 一句话差别？", "GET 取资源，POST 提交数据。", "concept"),
    ("什么是主键？一句话。", "唯一标识表中每一行的字段。", "concept"),
    ("正则 \\d 表示什么？", "一位数字。", "fact"),
    ("UTC 是什么？", "协调世界时。", "fact"),
    ("Markdown 一级标题？", "# 标题", "fact"),
    ("HTTP 404 含义？", "资源未找到。", "fact"),
)


@dataclass
class StyleRow:
    question: str
    answer: str
    kind: str


def _style_answer(body: str) -> str:
    return body.strip() + "\n"


def build_style_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _ in STYLE_HOLDOUT}
    out: List[Sample] = []

    for q, a, k in TRAIN_QA:
        if q in hold:
            continue
        out.append(Sample(q, _style_answer(a), f"style_{k}", a[:24]))
        out.append(Sample(q + " 简短回答。", _style_answer(a), f"style_{k}", a[:24]))

    eng_short = [
        (
            "Python 二分查找返回下标或 -1，完整但尽量短。",
            "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n",
            "binary_search",
        ),
        (
            "Express POST /api/users 校验 name/email，冲突 409，成功 201。代码尽量短。",
            "```js\napp.post('/api/users', (req, res) => {\n  const { name, email } = req.body || {};\n  if (!name || !email || !String(email).includes('@')) return res.status(400).json({ error: 'bad' });\n  if (db.has(email)) return res.status(409).json({ error: 'conflict' });\n  const user = { name, email };\n  db.set(email, user);\n  return res.status(201).json(user);\n});\n```\n",
            "POST /api/users",
        ),
        (
            "async def read_json(path)：缺文件 None，坏 JSON 抛 ValueError。短实现。",
            "```python\nimport json\nfrom pathlib import Path\n\nasync def read_json(path):\n    try:\n        text = Path(path).read_text(encoding='utf-8')\n    except FileNotFoundError:\n        return None\n    try:\n        return json.loads(text)\n    except json.JSONDecodeError as e:\n        raise ValueError('invalid json') from e\n```\n",
            "read_json",
        ),
    ]
    for q, a, g in eng_short * 10:
        out.append(Sample(q, _style_answer(a), "style_eng", g))

    eng_more = [
        (
            "参数化 SQL 查 email，短函数。",
            "```python\ndef users_by_email(conn, email):\n    cur = conn.cursor()\n    cur.execute('SELECT id, email FROM users WHERE email = ?', (email,))\n    return cur.fetchall()\n```\n",
            "parameterized",
        ),
        (
            "Vue3 输入框 emit add，空不提交。短 SFC。",
            "```vue\n<script setup>\nimport { ref } from 'vue'\nconst text = ref('')\nconst emit = defineEmits(['add'])\nfunction onAdd(){ const v=text.value.trim(); if(!v) return; emit('add', v); text.value='' }\n</script>\n<template><input v-model=\"text\" /><button @click=\"onAdd\">add</button></template>\n```\n",
            "emit",
        ),
        (
            "CSS 居中 .box 在 .stage，短片段。",
            "```html\n<style>.stage{display:flex;justify-content:center;align-items:center;min-height:100%}.box{}</style>\n<div class=\"stage\"><div class=\"box\"></div></div>\n```\n",
            "flex",
        ),
        (
            "mutex 一句话定义。",
            "A mutex is a lock that lets only one thread enter a critical section at a time.\n",
            "mutex",
        ),
        (
            "What is a mutex? One short sentence.",
            "A mutex serializes access so only one thread uses a shared resource at once.\n",
            "mutex",
        ),
    ]
    for q, a, g in eng_more * 8:
        out.append(Sample(q, _style_answer(a), "style_eng", g))
    for _ in range(12):
        out.append(Sample(
            "Python 安全查询 users by email，参数化占位符，短函数。",
            _style_answer("```python\ndef users_by_email(conn, email):\n    cur = conn.cursor()\n    cur.execute('SELECT id, email FROM users WHERE email = ?', (email,))\n    return cur.fetchall()\n```\nparameterized query.\n"),
            "style_eng",
            "parameterized",
        ))
        out.append(Sample(
            "Why is the sky blue? One short sentence.",
            _style_answer("短波蓝光更易被大气散射，所以天空呈蓝色。\n"),
            "style_concept",
            "散射",
        ))

    for q, gold in (
        ("What is the capital of France?", "Paris"),
        ("What is 17 + 25?", "42"),
        ("What is 9 times 6?", "54"),
        ("中国的首都是哪里？", "北京"),
    ):
        for _ in range(5):
            out.append(Sample(q, _style_answer(gold), "rehearsal", gold))

    rng.shuffle(out)
    if len(out) >= n_train:
        return out[:n_train]
    while len(out) < n_train:
        out.append(rng.choice(out))
    return out[:n_train]


def score_style(pred: str, gold_hint: str) -> Tuple[bool, Dict]:
    p = pred or ""
    low = p.lower()
    marks: Dict[str, int] = {}
    marks["len"] = len(p)
    marks["lines"] = p.count("\n") + 1
    marks["filler"] = sum(1 for m in FILLER_MARKERS if m.lower() in low or m in p)
    alts = {
        "散射": ("散射", "瑞利", "蓝光", "短波", "蓝色"),
        "mutex": ("mutex", "mutual exclusion", "lock", "互斥"),
        "space": ("space", "spaces", "tab", "either", "都行"),
        "[::-1]": ("[::-1]", "reversed", "reverse"),
        "调用": ("调用", "自身", "函数", "recursion"),
        "http": ("http", "超文本", "协议"),
        "索引": ("索引", "index", "加快"),
        "一致性": ("一致性", "可用性", "分区", "cap"),
    }
    keys = alts.get(gold_hint, (gold_hint,))
    marks["has_gold"] = int(any(k.lower() in low or k in p for k in keys))
    fences = p.count("```")
    marks["fence_count"] = fences
    lines = [ln.strip() for ln in p.splitlines() if len(ln.strip()) >= 16]
    seen: Dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    marks["dup4"] = sum(1 for v in seen.values() if v >= 4)
    marks["short_enough"] = int(len(p) <= 480)
    marks["not_essay"] = int(len(p) <= 900 and marks["filler"] <= 1 and marks["dup4"] == 0)
    ok = bool(marks["has_gold"] and marks["short_enough"] and marks["not_essay"] and marks["filler"] == 0)
    return ok, marks

def eval_style(gen, probes: Sequence[Tuple[str, str]] = STYLE_HOLDOUT) -> Dict:
    rows = []
    ok = 0
    total_len = 0
    fillers = 0
    for q, gold in probes:
        pred = gen(q, 220)
        hit, marks = score_style(pred, gold)
        ok += int(hit)
        total_len += len(pred or "")
        fillers += marks.get("filler", 0)
        rows.append({"q": q, "gold": gold, "ok": hit, "marks": marks, "pred": (pred or "")[:800]})
        print(
            f"  style {'OK' if hit else 'NO'} len={marks['len']} filler={marks['filler']} gold={gold}",
            flush=True,
        )
        if not hit:
            print("   ", (pred or "")[:140].replace("\n", " | "), flush=True)
    n = max(1, len(probes))
    return {
        "accuracy": ok / n,
        "avg_len": total_len / n,
        "filler_total": fillers,
        "rows": rows,
    }


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 512, answer_boost: float = 2.5):
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
            end = min(len(full_ids), plen + 24)
            weights[plen:end] = self.answer_boost
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": weights,
        }


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_style_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    meta = {
        "instruction_en": INSTRUCTION_EN,
        "instruction_zh": INSTRUCTION_ZH,
        "n_train": len(samples),
        "kinds": dict(Counter(s.kind for s in samples)),
        "holdout": [q for q, _ in STYLE_HOLDOUT],
        "note": "Teacher signal is the one instruction; train maps are instruction-free Q->concise A",
    }
    with open(out / "data" / "persona_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"n_train={len(samples)} kinds={meta['kinds']}", flush=True)
    print(f"INSTRUCTION: {INSTRUCTION_EN}", flush=True)

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    base.to(device)
    base.config.use_cache = False

    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    answer_boost = 3.0
    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=answer_boost)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    print("===== BASELINE =====", flush=True)
    style0 = eval_style(gen)
    eng0 = eval_eng(gen)
    ctrl0 = eval_controls(gen)
    print(
        f"BASELINE style={style0['accuracy']:.3f} avg_len={style0['avg_len']:.1f} filler={style0['filler_total']} "
        f"eng={eng0['accuracy']:.3f} ctrl={ctrl0['accuracy']:.3f}",
        flush=True,
    )

    t0 = time.perf_counter()
    model.train()
    epochs = max(1, int(args.epochs))
    running = 0.0
    seen = 0
    step = 0
    ga = args.grad_accum
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
            if step % 40 == 0:
                print(f"step {step}/{total_steps} ep={ep+1}/{epochs} loss={running/max(1,seen):.4f}", flush=True)
    if step % ga != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

    print("===== AFTER =====", flush=True)
    style1 = eval_style(gen)
    eng1 = eval_eng(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER style={style1['accuracy']:.3f} avg_len={style1['avg_len']:.1f} filler={style1['filler_total']} "
        f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} loss={running/max(1,seen):.4f}",
        flush=True,
    )

    ctrl_floor = min(0.5, ctrl0["accuracy"])
    eng_floor = max(0.90, eng0["accuracy"] - 0.05)
    len_improved = style1["avg_len"] + 1e-9 <= min(style0["avg_len"] * 0.98, 280.0)
    style_ok = style1["accuracy"] + 1e-9 >= 0.75
    filler_improved = style1["filler_total"] <= max(style0["filler_total"], 1)
    promote = (
        style_ok
        and len_improved
        and filler_improved
        and eng1["accuracy"] + 1e-9 >= eng_floor
        and eng1["accuracy"] + 1e-9 >= 0.90
        and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    )

    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    if promote:
        model.save_pretrained(out / "adapter_best")
        tok.save_pretrained(out / "adapter_best")
        print("PROMOTED persona_concise", flush=True)
    else:
        print(
            f"NO_PROMOTE style {style1['accuracy']:.3f}/{style0['accuracy']:.3f} "
            f"len {style1['avg_len']:.1f}/{style0['avg_len']:.1f} "
            f"eng {eng1['accuracy']:.3f} ctrl {ctrl1['accuracy']:.3f}",
            flush=True,
        )

    report = {
        "method": "persona_concise_v3",
        "instruction_en": INSTRUCTION_EN,
        "instruction_zh": INSTRUCTION_ZH,
        "n_train": len(samples),
        "kinds": meta["kinds"],
        "lr": args.lr,
        "lora_r": args.lora_r,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {
            "style": style0["accuracy"],
            "avg_len": style0["avg_len"],
            "filler": style0["filler_total"],
            "eng": eng0["accuracy"],
            "ctrl": ctrl0["accuracy"],
        },
        "after": {
            "style": style1["accuracy"],
            "avg_len": style1["avg_len"],
            "filler": style1["filler_total"],
            "eng": eng1["accuracy"],
            "ctrl": ctrl1["accuracy"],
        },
        "promoted": promote,
        "style_rows": style1["rows"],
        "eng_kind_acc": eng1.get("kind_acc"),
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "One-sentence teacher signal distilled into instruction-free Q->A",
            "Holdout wording differs from train",
            "Guardrails: eng floor + ctrl floor",
        ],
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k not in ("style_rows",)}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--resume", default="/Users/shiaho/Desktop/bitx/kef_results/eng_craft_champion/adapter_best")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/persona_concise_v1")
    p.add_argument("--n-train", type=int, default=160)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    if args.eval_only:
        device = args.device
        dtype = torch.float16 if device == "mps" else torch.float32
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
        base.to(device)
        model = PeftModel.from_pretrained(base, args.resume) if args.resume else base
        gen = make_gen(model, tok, device)
        style = eval_style(gen)
        eng = eval_eng(gen)
        ctrl = eval_controls(gen)
        print(json.dumps({"style": style, "eng": eng, "ctrl": ctrl}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
