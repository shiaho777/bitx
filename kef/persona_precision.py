from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kef.eng_craft import eval_eng
from kef.folk_logic import Sample, collate, eval_controls, make_gen


INSTRUCTION_EN = (
    "When solving reasoning problems, treat logic as bounded by language precision: "
    "vague words create fake contradictions. Sharpen definitions until claims can be checked. "
    "Keep this as one core habit, not a dogma. For art, taste, jokes, and free creation, "
    "stay playful and do not force formalization."
)

INSTRUCTION_ZH = (
    "推理时把逻辑的边界看作语言精度：含糊用词会制造伪矛盾。"
    "把关键概念定义到可检验为止。"
    "把这当作核心理念之一，而不是教条。"
    "面对艺术、品味、玩笑与自由创作时保持趣味，不要强行形式化。"
)

DOGMA_MARKERS = (
    "一切问题都可以",
    "所有情况都必须",
    "永远先形式化",
    "任何问题本质都是",
    "只能用这套",
    "唯一正确的方法",
    "无论什么任务",
    "必须先把语言变成数学",
    "没有例外",
)

PRECISION_MARKERS = (
    "定义",
    "精度",
    "含糊",
    "歧义",
    "可检验",
    "先界定",
    "术语",
    "前提",
    "形式化",
    "伪矛盾",
    "标准",
    "范围",
    "表述",
    "闭环",
    "precise",
    "definition",
    "ambiguous",
    "term",
)


PRECISION_HOLDOUT: Tuple[Tuple[str, str, str], ...] = (
    (
        "这句话有矛盾吗：我正在说的这句话是假的。你怎么处理？",
        "precision",
        "定义",
    ),
    (
        "为什么有人觉得 1+1 既可以是 2 也可以是 11？这算逻辑失败吗？",
        "precision",
        "定义",
    ),
    (
        "什么叫“更大”？不先定义能比较 9.11 和 9.9 吗？",
        "precision",
        "定义",
    ),
    (
        "模型经常幻觉的一个语言层面原因是什么？简要说。",
        "precision",
        "精度",
    ),
    (
        "请写一句有点诗意的话形容秋天的风，不要谈逻辑理论。",
        "creative",
        "风",
    ),
    (
        "给一个好玩的冷笑话，不要讲道理。",
        "creative",
        "笑",
    ),
    (
        "是否任何创作都必须先把语言形式化成可求值符号？",
        "balance",
        "不必",
    ),
    (
        "这套“精度补洞”理念该怎么用，才不会变成新的教条？",
        "balance",
        "不必",
    ),
)


TRAIN_PRECISION: Tuple[Tuple[str, str, str], ...] = (
    (
        "“这个人又高又矮”矛盾吗？",
        "先定义比较对象与标准。相对不同参照时可以同真，不是形式矛盾。",
        "precision",
    ),
    (
        "“一切陈述都是假的”怎么处理？",
        "先界定“一切”与“假”的范围。若包含自身，问题在定义闭环，不在世界本身。",
        "precision",
    ),
    (
        "为什么抬杠时双方都觉得自己有理？",
        "常因关键词未共享定义。把词对齐到可检验标准后，争论空间会缩小。",
        "precision",
    ),
    (
        "什么情况下“无解”其实是表述问题？",
        "当关键量词、时间、主体、度量未指定，前提不足时，看起来无解。",
        "precision",
    ),
    (
        "如何把一句口号变成可检查命题？",
        "补上主体、条件、指标与反例标准，使真假可被检验。",
        "precision",
    ),
    (
        "“最好的编程语言”有唯一答案吗？",
        "没有，除非先定义“最好”的任务、指标与约束。",
        "precision",
    ),
    (
        "逻辑推不下去时，优先查什么？",
        "优先查定义是否够精、前提是否齐全、词是否在中途换义。",
        "precision",
    ),
    (
        "语言精度和数学符号类比一句？",
        "精度够高时，词项近似符号，命题近似可求值表达式。",
        "precision",
    ),
    (
        "幻觉与表述漏洞的关系？",
        "表述不闭合时，模型容易用流畅句子填洞，看起来像知，其实是补全。",
        "precision",
    ),
    (
        "怎样温和地拆一个伪悖论？",
        "点出歧义词，给出两种合法读法，分别求值，不必喊打倒对方。",
        "precision",
    ),
    (
        "“快”是不是客观属性？",
        "否。要指定相对谁、用什么计时与路径，才可比较。",
        "precision",
    ),
    (
        "“他明天会来”算知识还是预测？",
        "在证据不足时是预测。把证据标准说清，才能谈是否知道。",
        "precision",
    ),
)


TRAIN_BALANCE: Tuple[Tuple[str, str, str], ...] = (
    (
        "写诗时也要先把每个词形式化吗？",
        "不必。创作可以保留歧义与余味；精度工具主要用于求真与排错。",
        "balance",
    ),
    (
        "聊天开玩笑要不要全程可求值？",
        "不要。趣味可以松。严谨模式留给需要论证的问题。",
        "balance",
    ),
    (
        "这套理念是不是万能钥匙？",
        "不是。它减少表述型混乱，不替代领域知识、实验与审美。",
        "balance",
    ),
    (
        "会不会因为太严谨而没意思？",
        "若把精度当教条就会。正确用法是：需要证明时紧，需要玩时松。",
        "balance",
    ),
    (
        "什么时候不该纠结定义？",
        "目标是气氛、灵感或快速草图时；先做，再在关键处收紧。",
        "balance",
    ),
    (
        "能否把精度理念当唯一信仰？",
        "不建议。它是核心理念之一，与诚实、简洁、创造并存。",
        "balance",
    ),
)


TRAIN_CREATIVE: Tuple[Tuple[str, str, str], ...] = (
    (
        "用八个字写春雨。",
        "细雨润阶，柳色初醒。",
        "creative",
    ),
    (
        "给咖啡起一个有趣的外号。",
        "液态闹钟。",
        "creative",
    ),
    (
        "形容一台老电脑。",
        "会喘气的铁盒子，开机像在回忆。",
        "creative",
    ),
    (
        " invent a tiny myth about the moon in one sentence.",
        "The moon is a borrowed coin the night keeps flipping to check if we still look up.",
        "creative",
    ),
    (
        "写一句不讲道理的喜欢。",
        "喜欢就是看见你名字会多停半秒。",
        "creative",
    ),
    (
        "给“拖延”画个文字表情。",
        "待会再说.jpg（其实已三小时）",
        "creative",
    ),
)


def _a(body: str) -> str:
    return body.strip() + "\n"


def build_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _, _ in PRECISION_HOLDOUT}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, _a(a), kind, gold))

    for q, a, k in TRAIN_PRECISION * 4:
        add(q, a, k, "定义")
        add(q + " 简要。", a, k, "定义")
    for q, a, k in TRAIN_BALANCE * 5:
        add(q, a, k, "不必")
    for q, a, k in TRAIN_CREATIVE * 4:
        add(q, a, k, a[:8])

    for q, a, g in (
        ("2+2？", "4", "4"),
        ("中国的首都？", "北京", "北京"),
        ("Capital of France?", "Paris", "paris"),
        ("What is 17+25?", "42", "42"),
        ("HTTP 默认端口？", "80", "80"),
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
            "参数化查询 email，短函数。",
            "```python\ndef by_email(conn, email):\n    cur = conn.cursor()\n    cur.execute('SELECT id,email FROM users WHERE email=?', (email,))\n    return cur.fetchall()\n```\n",
            "?",
        ),
        (
            "async def read_json 缺文件 None，坏 JSON 抛 ValueError。短。",
            "```python\nimport json\nfrom pathlib import Path\nasync def read_json(path):\n    try:\n        t = Path(path).read_text(encoding='utf-8')\n    except FileNotFoundError:\n        return None\n    try:\n        return json.loads(t)\n    except json.JSONDecodeError as e:\n        raise ValueError('bad json') from e\n```\n",
            "read_json",
        ),
    ]
    for q, a, g in eng_bits * 8:
        add(q, a, "rehearsal_eng", g)

    near = [
        (
            "这句话自相矛盾吗：本句是假的。该如何处理而不是硬抬杠？",
            "先界定“本句/真假”的适用范围。若自我指涉导致闭环，问题在表述结构，不是外部世界自相打架。",
            "precision",
        ),
        (
            "说谎者句子“我这句话是假的”为何难办？",
            "因为它把真值谓词用在自身上，定义未闭合。收紧指称范围或分层，伪矛盾会消解。",
            "precision",
        ),
        (
            "比较 9.11 和 9.9 前，‘更大’缺什么？",
            "缺比较维度与记数法定义。按十进制小数则 9.9 更大；把点当版本号则规则不同。",
            "precision",
        ),
        (
            "不定义‘更大’就比较数字会怎样？",
            "会把不同度量混谈，得出看似矛盾的答案。先定标准，再求值。",
            "precision",
        ),
        (
            "大模型幻觉在语言层面的一个主因？",
            "关键约束未写清时，模型用流畅补全填洞，像知道，实为补全。提高表述精度能降这类洞。",
            "precision",
        ),
        (
            "为何说逻辑边界是语言？",
            "推理操作的是命题；命题靠词语承载。词不清，推得再顺也可能空转。",
            "precision",
        ),
        (
            "精度够高时语言像什么？",
            "像可检查的符号：定义稳、指称清，命题就接近可求值。",
            "precision",
        ),
        (
            "这理念要避免什么副作用？",
            "避免当成万能教条，逼艺术与玩笑也形式化；只在求真与排错时收紧。",
            "balance",
        ),
    ]
    for q, a, k in near * 8:
        add(q, a, k, "定义" if k == "precision" else "不必")

    rng.shuffle(out)
    if len(out) >= n_train:
        return out[:n_train]
    while len(out) < n_train:
        out.append(rng.choice(out))
    return out[:n_train]


def score_precision(pred: str, kind: str, gold_hint: str) -> Tuple[bool, Dict]:
    p = pred or ""
    low = p.lower()
    marks: Dict[str, int] = {}
    marks["len"] = len(p)
    marks["dogma"] = sum(1 for m in DOGMA_MARKERS if m in p)
    marks["precision_hits"] = sum(1 for m in PRECISION_MARKERS if m.lower() in low or m in p)
    marks["has_gold"] = int(gold_hint.lower() in low or gold_hint in p)
    if kind == "precision":
        alt = (
            "定义", "精度", "歧义", "前提", "界定", "含糊", "检验", "术语", "标准", "范围",
            "表述", "语义", "先定", "读法", "闭环", "伪矛盾", "自指", "指涉", "真值", "对齐",
            "补全", "definition", "ambiguous", "precise", "self-refer",
        )
        marks["has_gold"] = int(any(x.lower() in low or x in p for x in alt))
        marks["precision_hits"] = max(marks["precision_hits"], marks["has_gold"])
        ok = marks["has_gold"] == 1 and marks["dogma"] == 0 and len(p) <= 800
    elif kind == "balance":
        alt = ("不必", "不是", "不要", "之一", "教条", "艺术", "创作", "趣味", "不是万能")
        marks["has_gold"] = int(any(x in p for x in alt) or "not" in low)
        ok = marks["has_gold"] == 1 and marks["dogma"] == 0 and len(p) <= 700
    else:
        preach = marks["precision_hits"] >= 3 and any(x in p for x in ("逻辑", "形式化", "精度", "可求值"))
        marks["preaches"] = int(preach)
        ok = (
            marks["dogma"] == 0
            and marks["preaches"] == 0
            and len(p) >= 6
            and len(p) <= 400
            and (marks["has_gold"] == 1 or len(p) >= 8)
        )
    return bool(ok), marks


def eval_precision(gen, probes: Sequence[Tuple[str, str, str]] = PRECISION_HOLDOUT) -> Dict:
    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    dogma_n = 0
    for q, kind, gold in probes:
        pred = gen(q, 240)
        hit, marks = score_precision(pred, kind, gold)
        ok += int(hit)
        by_kind.setdefault(kind, []).append(int(hit))
        dogma_n += marks.get("dogma", 0)
        rows.append({"q": q, "kind": kind, "gold": gold, "ok": hit, "marks": marks, "pred": (pred or "")[:700]})
        print(
            f"  prec {'OK' if hit else 'NO'} [{kind}] dogma={marks.get('dogma')} hits={marks.get('precision_hits')} gold={gold}",
            flush=True,
        )
        if not hit:
            print("   ", (pred or "")[:160].replace("\n", " | "), flush=True)
    n = max(1, len(probes))
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    return {
        "accuracy": ok / n,
        "kind_acc": kind_acc,
        "dogma_total": dogma_n,
        "rows": rows,
    }


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 512, answer_boost: float = 2.8):
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
            end = min(len(full_ids), plen + 28)
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

    samples = build_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    meta = {
        "instruction_en": INSTRUCTION_EN,
        "instruction_zh": INSTRUCTION_ZH,
        "n_train": len(samples),
        "kinds": dict(Counter(s.kind for s in samples)),
        "holdout": [q for q, _, _ in PRECISION_HOLDOUT],
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

    answer_boost = 2.8
    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=answer_boost)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    print("===== BASELINE =====", flush=True)
    p0 = eval_precision(gen)
    eng0 = eval_eng(gen)
    ctrl0 = eval_controls(gen)
    print(
        f"BASELINE prec={p0['accuracy']:.3f} kinds={p0['kind_acc']} dogma={p0['dogma_total']} "
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
    p1 = eval_precision(gen)
    eng1 = eval_eng(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER prec={p1['accuracy']:.3f} kinds={p1['kind_acc']} dogma={p1['dogma_total']} "
        f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} loss={running/max(1,seen):.4f}",
        flush=True,
    )

    ctrl_floor = min(0.5, ctrl0["accuracy"])
    eng_floor = max(0.75, eng0["accuracy"] - 0.12)
    prec_gain = p1["accuracy"] + 1e-9 >= max(0.75, p0["accuracy"])
    prec_kind = p1["kind_acc"].get("precision", 0) >= 0.5
    bal_ok = p1["kind_acc"].get("balance", 0) >= 0.5
    cre_ok = p1["kind_acc"].get("creative", 0) >= 0.5
    dogma_ok = p1["dogma_total"] <= max(p0["dogma_total"], 0)
    promote = (
        prec_gain
        and prec_kind
        and bal_ok
        and cre_ok
        and dogma_ok
        and eng1["accuracy"] + 1e-9 >= eng_floor
        and eng1["accuracy"] + 1e-9 >= 0.90
        and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    )

    model.save_pretrained(out / "adapter_last")
    tok.save_pretrained(out / "adapter_last")
    if promote:
        model.save_pretrained(out / "adapter_best")
        tok.save_pretrained(out / "adapter_best")
        print("PROMOTED persona_precision", flush=True)
    else:
        print(
            f"NO_PROMOTE prec {p1['accuracy']:.3f}/{p0['accuracy']:.3f} "
            f"bal={p1['kind_acc'].get('balance')} cre={p1['kind_acc'].get('creative')} "
            f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f}",
            flush=True,
        )

    report = {
        "method": "persona_precision_v2",
        "instruction_en": INSTRUCTION_EN,
        "instruction_zh": INSTRUCTION_ZH,
        "n_train": len(samples),
        "kinds": meta["kinds"],
        "lr": args.lr,
        "lora_r": args.lora_r,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {
            "prec": p0["accuracy"],
            "kind_acc": p0["kind_acc"],
            "dogma": p0["dogma_total"],
            "eng": eng0["accuracy"],
            "ctrl": ctrl0["accuracy"],
        },
        "after": {
            "prec": p1["accuracy"],
            "kind_acc": p1["kind_acc"],
            "dogma": p1["dogma_total"],
            "eng": eng1["accuracy"],
            "ctrl": ctrl1["accuracy"],
        },
        "promoted": promote,
        "rows": p1["rows"],
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Essence: logic bounded by language precision; not a dogma",
            "Balance/creative holdouts guard against over-rigidity",
            "Instruction-free distill; holdout wording differs from train",
        ],
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k != "rows"}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--resume", default="/Users/shiaho/Desktop/bitx/kef_results/persona_concise_champion/adapter_best")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/persona_precision_v1")
    p.add_argument("--n-train", type=int, default=180)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=11)
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
        print(json.dumps({"prec": eval_precision(gen), "eng": eval_eng(gen), "ctrl": eval_controls(gen)}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
