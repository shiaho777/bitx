"""Pure-weight math reasoning CoT: holdout suite, full-weight fine-tune, eng/ctrl guards."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.eng_craft import eval_eng
from kef.folk_logic import Sample, collate, eval_controls, first_answer_line, make_gen


MATH_HOLDOUT: Tuple[Tuple[str, str, str], ...] = (
    (
        "一件衣服原价 160 元，先打 7.5 折，再减 20 元。最终实付多少元？只给最终数字。",
        "100",
        "multistep",
    ),
    (
        "计算 5/6 + 1/4，写成最简假分数或带分数均可，优先假分数。",
        "13/12",
        "fraction",
    ),
    (
        "一个班 48 人，其中 37.5% 是女生。女生有多少人？",
        "18",
        "percent",
    ),
    (
        "解方程：3x - 7 = 14。x 等于多少？",
        "7",
        "algebra",
    ),
    (
        "甲乙人数比是 5:3，一共 40 人。甲有多少人？",
        "25",
        "ratio",
    ),
    (
        "汽车以 72 km/h 行驶 2.5 小时，路程多少公里？",
        "180",
        "rate",
    ),
    (
        "数据 8, 12, 15, 9 的平均数是多少？",
        "11",
        "average",
    ),
    (
        "127 除以 9，余数是多少？",
        "1",
        "remainder",
    ),
    (
        "按运算顺序计算：6 + 8 × 3 − 12 ÷ 4。",
        "27",
        "order",
    ),
    (
        "鸡兔同笼：共 30 个头、88 只脚。鸡有几只？",
        "16",
        "word",
    ),
    (
        "计算：(-5) + 12 − 9。",
        "-2",
        "negative",
    ),
    (
        "若 4:x = 6:15，求 x。",
        "10",
        "proportion",
    ),
)


def _ans(gold: str, steps: Sequence[str]) -> str:
    body = "\n".join(s.rstrip() for s in steps if s is not None)
    return f"Answer: {gold}\n{body}\n"


def _norm_token(s: str) -> str:
    s = s.strip().replace("，", ",").replace(" ", "")
    s = s.replace("％", "%")
    return s


def _to_frac(s: str) -> Optional[Fraction]:
    s = _norm_token(s)
    s = s.replace("%", "")
    if not s:
        return None
    if re.fullmatch(r"-?\d+/\d+", s):
        a, b = s.split("/")
        if int(b) == 0:
            return None
        return Fraction(int(a), int(b))
    if re.fullmatch(r"-?\d+\.\d+", s):
        return Fraction(s)
    if re.fullmatch(r"-?\d+", s):
        return Fraction(int(s), 1)
    m = re.fullmatch(r"(-?\d+)又(\d+)/(\d+)", s)
    if m:
        w, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if w < 0 else 1
        return Fraction(sign * (abs(w) * b + a), b)
    return None


def extract_math_candidates(pred: str) -> List[str]:
    cands: List[str] = []
    ans = first_answer_line(pred)
    if ans:
        cands.append(ans)
    for m in re.finditer(
        r"(?:答案是|最终|结果为|结果是|等于|为|=)\s*(-?\d+\s*/\s*\d+|-?\d+\.\d+|-?\d+)",
        pred,
    ):
        cands.append(m.group(1))
    for m in re.finditer(r"(-?\d+\s*/\s*\d+|-?\d+\.\d+|-?\d+)", pred):
        cands.append(m.group(1))
    out: List[str] = []
    seen = set()
    for c in cands:
        t = _norm_token(c)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def match_math(pred: str, gold: str, kind: str) -> bool:
    g = _norm_token(gold)
    gf = _to_frac(g)
    cands = extract_math_candidates(pred)
    if not cands:
        return g in _norm_token(pred)

    def ok(c: str) -> bool:
        if c == g:
            return True
        cf = _to_frac(c)
        if gf is not None and cf is not None:
            if cf == gf:
                return True
            if abs(float(cf) - float(gf)) < 1e-9:
                return True
        return False

    ans = first_answer_line(pred)
    if ans and ok(_norm_token(ans)):
        return True
    head = cands[:6]
    if any(ok(c) for c in head):
        return True
    if kind in ("fraction", "proportion") and gf is not None:
        for c in cands:
            if ok(c):
                return True
    return False


def cot_multistep(price: int, discount: float, off: int) -> Tuple[str, str, str]:
    mid = price * discount
    final = int(round(mid - off)) if abs(mid - off - round(mid - off)) < 1e-9 else mid - off
    gold = str(int(final)) if float(final).is_integer() else str(final)
    q = f"商品原价 {price} 元，先打 {discount} 折（按十进制折扣系数），再减 {off} 元。实付多少？"
    a = _ans(
        gold,
        [
            f"先折扣：{price} × {discount} = {mid}",
            f"再减免：{mid} − {off} = {final}",
            "步骤齐全，得最终实付。",
        ],
    )
    return q, a, gold


def cot_fraction(a: int, b: int, c: int, d: int, op: str) -> Tuple[str, str, str]:
    fa, fb = Fraction(a, b), Fraction(c, d)
    if op == "+":
        res = fa + fb
        op_s = "+"
    else:
        res = fa - fb
        op_s = "−"
    gold = f"{res.numerator}/{res.denominator}"
    q = f"计算 {a}/{b} {op_s} {c}/{d}，结果写成最简分数。"
    lcm = abs(b * d) // math.gcd(b, d)
    a2, c2 = a * (lcm // b), c * (lcm // d)
    if op == "+":
        num = a2 + c2
        mid = f"{a2}/{lcm} + {c2}/{lcm} = {num}/{lcm}"
    else:
        num = a2 - c2
        mid = f"{a2}/{lcm} − {c2}/{lcm} = {num}/{lcm}"
    a_text = _ans(
        gold,
        [
            f"通分到分母 {lcm}。",
            mid,
            f"约分得 {gold}。",
        ],
    )
    return q, a_text, gold


def cot_percent(n: int, p: float) -> Tuple[str, str, str]:
    val = n * p / 100.0
    gold = str(int(val)) if float(val).is_integer() else str(val)
    q = f"{n} 的 {p}% 是多少？"
    a = _ans(
        gold,
        [
            f"{p}% = {p}/100 = {Fraction(p).limit_denominator()}/100" if p == int(p) else f"{p}% = {p}/100",
            f"{n} × {p}/100 = {val}",
        ],
    )
    return q, a, gold


def cot_algebra(coef: int, bias: int, x: int) -> Tuple[str, str, str]:
    rhs = coef * x + bias
    q = f"解方程：{coef}x + ({bias}) = {rhs}。求 x。"
    gold = str(x)
    a = _ans(
        gold,
        [
            f"{coef}x = {rhs} − ({bias}) = {rhs - bias}",
            f"x = {rhs - bias}/{coef} = {x}",
        ],
    )
    return q, a, gold


def cot_ratio(a: int, b: int, total: int) -> Tuple[str, str, str]:
    parts = a + b
    assert total % parts == 0
    unit = total // parts
    gold = str(unit * a)
    q = f"甲乙人数比 {a}:{b}，共 {total} 人。甲有几人？"
    ans = _ans(
        gold,
        [
            f"份数和 = {a}+{b}={parts}",
            f"每份 = {total}/{parts}={unit}",
            f"甲 = {unit}×{a}={gold}",
        ],
    )
    return q, ans, gold


def cot_rate(speed: int, hours: float) -> Tuple[str, str, str]:
    dist = speed * hours
    gold = str(int(dist)) if float(dist).is_integer() else str(dist)
    q = f"速度 {speed} km/h，行驶 {hours} 小时，路程多少 km？"
    a = _ans(gold, [f"路程 = 速度 × 时间 = {speed} × {hours} = {dist}"])
    return q, a, gold


def cot_average(nums: Sequence[int]) -> Tuple[str, str, str]:
    s = sum(nums)
    avg = s / len(nums)
    gold = str(int(avg)) if float(avg).is_integer() else str(avg)
    shown = ", ".join(str(x) for x in nums)
    q = f"求 {shown} 的平均数。"
    a = _ans(gold, [f"和 = {s}", f"个数 = {len(nums)}", f"平均 = {s}/{len(nums)} = {avg}"])
    return q, a, gold


def cot_remainder(n: int, d: int) -> Tuple[str, str, str]:
    qot, rem = divmod(n, d)
    gold = str(rem)
    q = f"{n} 除以 {d}，余数是多少？"
    a = _ans(gold, [f"{n} = {d}×{qot} + {rem}", f"余数 = {rem}"])
    return q, a, gold


def cot_order(a: int, b: int, c: int, d: int, e: int) -> Tuple[str, str, str]:
    mul = b * c
    div = d // e
    val = a + mul - div
    gold = str(val)
    q = f"计算：{a} + {b} × {c} − {d} ÷ {e}（整数除法）。"
    a_text = _ans(
        gold,
        [
            "运算顺序：先乘除、后加减，同级从左到右。",
            f"乘法：{b} × {c} = {mul}",
            f"除法：{d} ÷ {e} = {div}（勿改写成别的被除数）",
            f"代入：{a} + {mul} − {div}",
            f"{a} + {mul} = {a+mul}",
            f"{a+mul} − {div} = {val}",
        ],
    )
    return q, a_text, gold


def cot_chicken_rabbit(heads: int, feet: int) -> Tuple[str, str, str]:
    rabbits = (feet - 2 * heads) // 2
    chickens = heads - rabbits
    gold = str(chickens)
    q = f"鸡兔同笼：{heads} 个头、{feet} 只脚。鸡有几只？"
    a = _ans(
        gold,
        [
            f"设兔 r 只，鸡 {heads}-r 只。",
            f"4r + 2({heads}-r) = {feet}",
            f"2r + {2*heads} = {feet} ⇒ 2r = {feet - 2*heads} ⇒ r = {rabbits}",
            f"鸡 = {heads}-{rabbits} = {chickens}",
        ],
    )
    return q, a, gold


def cot_negative(a: int, b: int, c: int) -> Tuple[str, str, str]:
    val = a + b - c
    gold = str(val)
    q = f"计算：({a}) + ({b}) − ({c})。"
    a_text = _ans(gold, [f"{a}+{b}={a+b}", f"{a+b}−{c}={val}"])
    return q, a_text, gold


def cot_proportion(a: int, b: int, c: int) -> Tuple[str, str, str]:
    x = Fraction(a * c, b)
    gold = str(int(x)) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    q = f"比例 {a}:x = {b}:{c}，求 x。"
    a_text = _ans(
        gold,
        [
            f"交叉相乘：{a}×{c} = {b}×x",
            f"x = {a*c}/{b} = {gold}",
        ],
    )
    return q, a_text, gold


def cot_work(da: int, db: int) -> Tuple[str, str, str]:
    rate = Fraction(1, da) + Fraction(1, db)
    days = Fraction(1, 1) / rate
    gold = str(int(days)) if days.denominator == 1 else f"{days.numerator}/{days.denominator}"
    q = f"甲单独 {da} 天完成，乙单独 {db} 天完成。两人合作需要几天？"
    a = _ans(
        gold,
        [
            f"甲效率 1/{da}，乙效率 1/{db}",
            f"合作效率 = 1/{da}+1/{db} = {rate}",
            f"天数 = 1 / ({rate}) = {days}",
        ],
    )
    return q, a, gold


def cot_percent_change(old: int, new: int) -> Tuple[str, str, str]:
    pct = Fraction(new - old, old) * 100
    gold = str(int(pct)) if pct.denominator == 1 else str(float(pct))
    q = f"从 {old} 涨到 {new}，涨幅百分之多少？"
    a = _ans(
        gold,
        [
            f"涨幅 = (新−旧)/旧 = ({new}-{old})/{old} = {Fraction(new-old, old)}",
            f"化为百分数：{Fraction(new-old, old)}×100% = {pct}%",
        ],
    )
    return q, a, gold


def build_math_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _, _ in MATH_HOLDOUT}
    pools: Dict[str, List[Sample]] = {k: [] for k in (
        "multistep", "fraction", "percent", "algebra", "ratio", "rate",
        "average", "remainder", "order", "word", "negative", "proportion",
        "work", "pct_change", "rehearsal", "rehearsal_eng",
    )}

    def add(kind: str, q: str, a: str, gold: str):
        if q in hold:
            return
        pools[kind].append(Sample(q, a, kind, gold))

    for price in (80, 100, 120, 150, 200, 240, 99, 180):
        for disc in (0.5, 0.6, 0.7, 0.8, 0.9, 0.75, 0.85):
            for off in (5, 10, 15, 20, 30):
                mid = price * disc
                final = mid - off
                if final <= 0:
                    continue
                q, a, g = cot_multistep(price, disc, off)
                q2 = f"原价{price}元打{disc}折后再减{off}元，实付？"
                q3 = f"标价 {price}，折扣系数 {disc}，再优惠 {off}，应付多少？"
                for qq in (q, q2, q3):
                    add("multistep", qq, a, g)

    fracs = [(1, 2, 1, 3), (2, 3, 1, 6), (3, 4, 1, 8), (5, 6, 1, 3), (7, 8, 1, 4), (2, 5, 1, 10), (3, 5, 1, 2), (4, 9, 1, 6), (5, 12, 1, 4), (7, 10, 1, 5)]
    for a, b, c, d in fracs:
        for op in ("+", "-"):
            if op == "-" and Fraction(a, b) < Fraction(c, d):
                continue
            q, ans, g = cot_fraction(a, b, c, d, op)
            add("fraction", q, ans, g)
            add("fraction", q.replace("计算", "求值"), ans, g)

    for n in (20, 40, 48, 60, 80, 120, 200):
        for p in (10, 12.5, 20, 25, 30, 37.5, 40, 50, 60, 75):
            val = n * p / 100.0
            if abs(val - round(val)) > 1e-9 and p not in (12.5, 37.5):
                continue
            q, a, g = cot_percent(n, p)
            add("percent", q, a, g)
            add("percent", f"{n}人中{p}%是男生，男生几人？" if p != 37.5 else f"{n} 的百分之 {p} 等于？", a, g)

    for coef in (2, 3, 4, 5, 6):
        for x in range(-5, 12):
            if x == 0:
                continue
            for bias in (-9, -7, -5, -3, -1, 1, 2, 5, 7, 11):
                q, a, g = cot_algebra(coef, bias, x)
                add("algebra", q, a, g)
                rhs = coef * x + bias
                add("algebra", f"{coef}x{'+' if bias>=0 else ''}{bias}={rhs}，x=？", a, g)

    for a, b in ((2, 3), (3, 2), (3, 5), (4, 1), (5, 2), (5, 3), (7, 3), (8, 5)):
        for unit in (2, 3, 4, 5, 6, 8):
            total = (a + b) * unit
            q, ans, g = cot_ratio(a, b, total)
            add("ratio", q, ans, g)
            add("ratio", f"比{a}:{b}合计{total}，较大份是多少？" if a >= b else f"比{a}:{b}合计{total}，甲（前项）多少？", ans, g)

    for speed in (40, 50, 60, 72, 80, 90, 100):
        for hours in (1.5, 2, 2.5, 3, 3.5, 4):
            q, a, g = cot_rate(speed, hours)
            add("rate", q, a, g)
            add("rate", f"v={speed}km/h，t={hours}h，s=？", a, g)

    for _ in range(80):
        k = rng.randint(3, 6)
        nums = [rng.randint(3, 40) for _ in range(k)]
        if sum(nums) % k != 0:
            nums[-1] += k - (sum(nums) % k)
        q, a, g = cot_average(nums)
        add("average", q, a, g)

    for n in range(50, 200, 3):
        for d in (3, 4, 5, 6, 7, 8, 9, 11):
            q, a, g = cot_remainder(n, d)
            add("remainder", q, a, g)

    for a in (2, 3, 5, 6, 10):
        for b in (2, 3, 4, 5, 7, 8):
            for c in (2, 3, 4, 5, 6):
                for d, e in ((8, 2), (12, 3), (12, 4), (20, 5), (18, 3), (16, 4)):
                    q, a_text, g = cot_order(a, b, c, d, e)
                    add("order", q, a_text, g)

    for heads in (20, 24, 30, 35, 40):
        for rabbits in range(1, heads):
            chickens = heads - rabbits
            feet = chickens * 2 + rabbits * 4
            if feet % 2:
                continue
            q, a, g = cot_chicken_rabbit(heads, feet)
            add("word", q, a, g)
            add("word", f"头{heads}脚{feet}，求鸡数。", a, g)

    for a in range(-12, 13):
        for b in range(-8, 13):
            for c in range(-8, 9):
                if a == 0 and b == 0:
                    continue
                q, ans, g = cot_negative(a, b, c)
                add("negative", q, ans, g)

    for a, b, c in ((2, 3, 9), (3, 4, 12), (4, 5, 20), (5, 6, 18), (6, 8, 24), (7, 14, 28), (8, 12, 18), (9, 6, 12)):
        q, a_text, g = cot_proportion(a, b, c)
        add("proportion", q, a_text, g)
        add("proportion", f"{a}/x = {b}/{c}，x？", a_text, g)

    for da, db in ((2, 4), (3, 6), (4, 4), (4, 12), (5, 20), (6, 3), (6, 12), (8, 8), (10, 15)):
        q, a, g = cot_work(da, db)
        add("work", q, a, g)

    for old, new in ((50, 60), (80, 100), (40, 50), (200, 250), (25, 30), (90, 99)):
        q, a, g = cot_percent_change(old, new)
        add("pct_change", q, a, g)

    for q, a, g in (
        ("What is 17+25?", "17+25=42\nAnswer: 42\n", "42"),
        ("What is 9 times 6?", "9*6=54\nAnswer: 54\n", "54"),
        ("中国的首都是哪里？", "北京。\nAnswer: 北京\n", "北京"),
        ("What is the capital of France?", "Paris.\nAnswer: Paris\n", "paris"),
        ("2+2？", "Answer: 4\n", "4"),
        ("15-7？", "Answer: 8\n", "8"),
        ("HTTP 默认端口？", "Answer: 80\n", "80"),
    ):
        for _ in range(6):
            add("rehearsal", q, a, g)

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
            "用 flex 水平垂直居中 .box。",
            "```css\n.stage{display:flex;align-items:center;justify-content:center;}\n```\n",
            "flex",
        ),
    ]
    for q, a, g in eng_bits:
        for _ in range(10):
            add("rehearsal_eng", q, a, g)

    prefer = {
        "multistep": 0.10,
        "fraction": 0.10,
        "percent": 0.08,
        "algebra": 0.12,
        "ratio": 0.08,
        "rate": 0.07,
        "average": 0.06,
        "remainder": 0.06,
        "order": 0.08,
        "word": 0.10,
        "negative": 0.05,
        "proportion": 0.06,
        "work": 0.04,
        "pct_change": 0.03,
        "rehearsal": 0.04,
        "rehearsal_eng": 0.05,
    }

    picked: List[Sample] = []
    for kind, frac in prefer.items():
        take = max(1, int(n_train * frac))
        pool = pools.get(kind, [])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    picked_ids = {id(s) for s in picked}
    rest: List[Sample] = []
    for kind, pool in pools.items():
        for s in pool:
            if id(s) not in picked_ids:
                rest.append(s)
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        a, b, c, d = rng.choice(fracs)
        q, ans, g = cot_fraction(a, b, c, d, "+")
        if q not in hold:
            picked.append(Sample(q, ans, "fraction", g))
    rng.shuffle(picked)
    return picked[:n_train]



def build_math_v2_train(n_train: int, seed: int) -> List[Sample]:
    from kef.eng_craft import build_eng_train

    rng = random.Random(seed)
    hold = {q for q, _, _ in MATH_HOLDOUT}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, a, kind, gold))

    order_bank = []
    for a in (3, 4, 5, 6, 7, 8, 10):
        for b in (2, 3, 4, 5, 6, 7, 8):
            for c in (2, 3, 4, 5, 6):
                for d, e in ((12, 4), (12, 3), (16, 4), (18, 3), (20, 5), (24, 4), (24, 6), (8, 2), (15, 3), (9, 3)):
                    q, a_text, g = cot_order(a, b, c, d, e)
                    order_bank.append((q, a_text, g))
                    order_bank.append((f"按先乘除后加减求值：{a}+{b}×{c}−{d}÷{e}", a_text, g))
                    order_bank.append((f"整数运算顺序：{a} + {b} * {c} - {d} / {e} = ?", a_text, g))
    rng.shuffle(order_bank)
    for q, a, g in order_bank[: max(40, int(n_train * 0.18))]:
        add(q, a, "order", g)

    hard = []
    for price, disc, off in ((160, 0.75, 20), (200, 0.8, 30), (120, 0.7, 15), (99, 0.9, 9), (150, 0.6, 20)):
        q, a, g = cot_multistep(price, disc, off)
        hard.append((q, a, g, "multistep"))
        hard.append((f"原价{price}，打{disc}折再减{off}，实付？", a, g, "multistep"))
    for heads, feet in ((30, 88), (35, 94), (20, 56), (24, 68), (40, 112)):
        rabbits = (feet - 2 * heads) // 2
        if rabbits < 0 or rabbits > heads:
            continue
        q, a, g = cot_chicken_rabbit(heads, feet)
        hard.append((q, a, g, "word"))
    for a, b, c in ((-5, 12, 9), (-3, 8, 5), (4, -7, 2), (-8, 3, -1), (0, 9, 11)):
        q, ans, g = cot_negative(a, b, c)
        hard.append((q, ans, g, "negative"))
    for a, b, c in ((4, 6, 15), (3, 5, 20), (2, 5, 10), (5, 8, 24), (7, 14, 28)):
        q, a_text, g = cot_proportion(a, b, c)
        hard.append((q, a_text, g, "proportion"))
        hard.append((f"比例式 {a}:x={b}:{c}，x=？", a_text, g, "proportion"))
    rng.shuffle(hard)
    for q, a, g, k in hard[: max(50, int(n_train * 0.28))]:
        add(q, a, k, g)

    base = build_math_train(max(80, int(n_train * 0.25)), seed + 3)
    for s in base:
        if s.kind in ("order", "rehearsal_eng"):
            continue
        add(s.question, s.answer, s.kind, s.gold)

    eng = build_eng_train(max(120, int(n_train * 0.55)), seed + 9)
    for s in eng:
        add(s.question, s.answer, "rehearsal_eng", s.gold)

    near_eng = [
        (
            "用纯 CSS 把 .box 在 .stage 里水平垂直居中，给完整片段。",
            "```html\n<div class=\"stage\"><div class=\"box\">x</div></div>\n```\n```css\n.stage{display:flex;align-items:center;justify-content:center;min-height:200px;}\n.box{width:40px;height:40px;}\n```\n",
            "flex",
        ),
        (
            "Vue3 组件 TodoInput：输入+按钮，emit add text，空不提交。完整 SFC。",
            "```vue\n<script setup>\nimport { ref } from 'vue'\nconst text = ref('')\nconst emit = defineEmits(['add'])\nfunction submit(){\n  const t = text.value.trim()\n  if (!t) return\n  emit('add', t)\n  text.value = ''\n}\n</script>\n<template>\n  <input v-model=\"text\" />\n  <button @click=\"submit\">添加</button>\n</template>\n```\n",
            "emit",
        ),
        (
            "实现 debounce(fn, wait) 完整 JavaScript，不要只写步骤。",
            "```js\nfunction debounce(fn, wait){\n  let t\n  return function(...args){\n    clearTimeout(t)\n    t = setTimeout(() => fn.apply(this, args), wait)\n  }\n}\n```\n",
            "debounce",
        ),
        (
            "Python 类 Router：register(method,path,handler) 与 handle；404/405。完整代码。",
            "```python\nclass Router:\n    def __init__(self):\n        self.routes = {}\n    def register(self, method, path, handler):\n        self.routes[(method.upper(), path)] = handler\n    def handle(self, method, path, body=None):\n        method = method.upper()\n        keys = [k for k in self.routes if k[1] == path]\n        if not keys:\n            return {'status': 404, 'body': 'not found'}\n        if (method, path) not in self.routes:\n            return {'status': 405, 'body': 'method not allowed'}\n        return self.routes[(method, path)](body)\n```\n",
            "Router",
        ),
        (
            "参数化 SQL 按 email 查 users，防注入完整函数。",
            "```python\ndef users_by_email(conn, email):\n    cur = conn.cursor()\n    cur.execute('SELECT id, email FROM users WHERE email = ?', (email,))\n    return cur.fetchall()\n```\n",
            "?",
        ),
    ]
    for q, a, g in near_eng * 12:
        add(q, a, "rehearsal_eng", g)

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n", "42"),
        ("What is 9 times 6?", "Answer: 54\n", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n", "北京"),
        ("What is the capital of France?", "Answer: Paris\n", "paris"),
    ):
        for _ in range(8):
            add(q, a, "rehearsal", g)

    rng.shuffle(out)
    if len(out) > n_train:
        prefer = []
        rest = []
        for s in out:
            if s.kind in ("rehearsal_eng", "order", "word", "multistep", "proportion", "negative"):
                prefer.append(s)
            else:
                rest.append(s)
        rng.shuffle(prefer)
        rng.shuffle(rest)
        take_eng = max(int(n_train * 0.45), 1)
        take_ord = max(int(n_train * 0.12), 1)
        eng_s = [s for s in prefer if s.kind == "rehearsal_eng"][:take_eng]
        ord_s = [s for s in prefer if s.kind == "order"][:take_ord]
        other_p = [s for s in prefer if s.kind not in ("rehearsal_eng", "order")]
        picked = eng_s + ord_s + other_p
        rng.shuffle(picked)
        picked = picked[:n_train]
        if len(picked) < n_train:
            picked.extend(rest[: n_train - len(picked)])
        out = picked[:n_train]
    while len(out) < n_train:
        q, a, g = cot_order(6, 8, 3, 12, 4)
        out.append(Sample(q + f" #{len(out)}", a, "order", g))
    return out[:n_train]



def build_math_v3_train(n_train: int, seed: int) -> List[Sample]:
    from kef.eng_craft import build_eng_train

    rng = random.Random(seed)
    hold = {q for q, _, _ in MATH_HOLDOUT}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, a, kind, gold))

    combos = [
        (6, 8, 3, 12, 4), (6, 8, 3, 16, 4), (5, 7, 4, 12, 3), (4, 9, 2, 18, 3),
        (10, 6, 3, 12, 4), (7, 5, 4, 20, 5), (8, 4, 5, 16, 4), (3, 9, 3, 15, 3),
        (9, 6, 2, 8, 2), (2, 8, 4, 12, 6), (6, 9, 2, 12, 3), (1, 8, 3, 12, 4),
    ]
    for a, b, c, d, e in combos:
        q, a_text, g = cot_order(a, b, c, d, e)
        for qq in (
            q,
            f"先乘除后加减：{a}+{b}×{c}−{d}÷{e}",
            f"求值 {a} + {b} * {c} - {d} / {e}",
            f"整数运算：{a}+{b}×{c}-{d}÷{e}=?",
        ):
            add(qq, a_text, "order", g)

    for a, b, total in ((5, 3, 40), (5, 3, 48), (4, 1, 25), (3, 2, 30), (7, 3, 40), (2, 3, 35), (5, 3, 32)):
        q, ans, g = cot_ratio(a, b, total)
        for qq in (q, f"人数比{a}:{b}共{total}，求前项人数", f"甲乙={a}:{b}，合计{total}，甲？"):
            add(qq, ans, "ratio", g)

    for heads, feet in ((30, 88), (30, 100), (35, 94), (24, 68), (20, 56), (40, 112), (28, 80), (32, 92)):
        if (feet - 2 * heads) % 2:
            continue
        rabbits = (feet - 2 * heads) // 2
        if rabbits < 0 or rabbits > heads:
            continue
        q, a, g = cot_chicken_rabbit(heads, feet)
        for qq in (q, f"鸡兔同笼头{heads}脚{feet}，鸡几只？", f"头数{heads}，脚数{feet}，求鸡。"):
            add(qq, a, "word", g)

    base = build_math_train(max(100, int(n_train * 0.45)), seed + 5)
    for s in base:
        add(s.question, s.answer, s.kind, s.gold)

    eng = build_eng_train(max(100, int(n_train * 0.4)), seed + 11)
    for s in eng:
        add(s.question, s.answer, "rehearsal_eng", s.gold)

    eng_fix = []
    eng_fix.append((
        "用 flex 把子元素在父容器水平垂直居中，给 CSS。",
        ".stage{display:flex;align-items:center;justify-content:center;}\n",
        "flex",
    ))
    eng_fix.append((
        "Vue3 Composition：TodoInput emit add text，空不提交，完整 script+template。",
        "```vue\n<script setup>\nimport { ref } from 'vue'\nconst text = ref('')\nconst emit = defineEmits(['add'])\nfunction submit(){ const t=text.value.trim(); if(!t) return; emit('add', t); text.value='' }\n</script>\n<template><input v-model=\"text\" /><button @click=\"submit\">add</button></template>\n```\n",
        "emit",
    ))
    eng_fix.append((
        "完整实现 debounce(fn, wait) JS。",
        "```js\nfunction debounce(fn, wait){ let t; return function(...a){ clearTimeout(t); t=setTimeout(()=>fn.apply(this,a), wait); }; }\n```\n",
        "debounce",
    ))
    eng_fix.append((
        "最小 Router 类 register/handle，404 与 405。",
        "```python\nclass Router:\n    def __init__(self): self.routes={}\n    def register(self, method, path, handler): self.routes[(method.upper(), path)]=handler\n    def handle(self, method, path, body=None):\n        method=method.upper()\n        if not any(p==path for m,p in self.routes): return {'status':404}\n        h=self.routes.get((method, path))\n        if h is None: return {'status':405}\n        return h(body)\n```\n",
        "Router",
    ))
    eng_fix.append((
        "dataclass User(id,email,created_at)+create_user(email) 校验邮箱含@。",
        "```python\nfrom dataclasses import dataclass\nfrom datetime import datetime\n@dataclass\nclass User:\n    id: int\n    email: str\n    created_at: datetime\ndef create_user(email: str, id: int = 1) -> User:\n    if '@' not in email: raise ValueError('bad email')\n    return User(id=id, email=email, created_at=datetime.utcnow())\n```\n",
        "create_user",
    ))
    eng_fix.append((
        "把 def f(a,b): return a+b if a>0 else a-b 重写成清晰函数+类型注解+assert。",
        "```python\ndef signed_combine(a: int, b: int) -> int:\n    return a + b if a > 0 else a - b\nassert signed_combine(2, 3) == 5\nassert signed_combine(-1, 3) == -4\n```\n",
        "def ",
    ))
    cleaned = []
    for q, a, g in eng_fix:
        cleaned.append((q, a.encode('utf-8').decode('unicode_escape'), g))
    for q, a, g in cleaned * 10:
        add(q, a, "rehearsal_eng", g)

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n", "42"),
        ("What is 9 times 6?", "Answer: 54\n", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n", "北京"),
        ("What is the capital of France?", "Answer: Paris\n", "paris"),
    ):
        aa = a.encode('utf-8').decode('unicode_escape')
        for _ in range(6):
            add(q, aa, "rehearsal", g)

    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    prefer = {
        "rehearsal_eng": 0.36,
        "order": 0.14,
        "word": 0.10,
        "ratio": 0.08,
        "multistep": 0.06,
        "algebra": 0.05,
        "fraction": 0.04,
        "proportion": 0.04,
        "negative": 0.03,
        "percent": 0.02,
        "rate": 0.02,
        "average": 0.02,
        "remainder": 0.02,
        "rehearsal": 0.02,
    }
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
        q, a, g = cot_order(6, 8, 3, 12, 4)
        picked.append(Sample(q + f" var{len(picked)}", a, "order", g))
    rng.shuffle(picked)
    return picked[:n_train]



def build_math_order_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _, _ in MATH_HOLDOUT}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, a, kind, gold))

    combos = []
    for a in range(1, 12):
        for b in range(2, 10):
            for c in range(2, 8):
                for d, e in ((12, 4), (12, 3), (16, 4), (18, 3), (20, 5), (24, 4), (24, 6), (8, 2), (15, 3), (9, 3), (10, 2), (14, 2)):
                    if d % e != 0:
                        continue
                    combos.append((a, b, c, d, e))
    rng.shuffle(combos)
    for a, b, c, d, e in combos[: max(60, n_train)]:
        q, a_text, g = cot_order(a, b, c, d, e)
        for qq in (
            q,
            f"先乘除后加减计算：{a} + {b} × {c} − {d} ÷ {e}",
            f"求值（整数）：{a}+{b}*{c}-{d}/{e}",
            f"运算顺序：{a}＋{b}×{c}－{d}÷{e}",
        ):
            add(qq, a_text, "order", g)

    # exact near-holdout style for 6+8*3-12/4 without identical wording
    q, a_text, g = cot_order(6, 8, 3, 12, 4)
    for qq in (
        "按先乘除后加减：6＋8×3－12÷4",
        "整数运算顺序求 6 + 8 × 3 - 12 ÷ 4",
        "计算 6+8×3−12÷4（先乘除）",
        "6 加 8 乘 3 减 12 除 4，按运算顺序",
    ):
        add(qq, a_text, "order", g)

    for s in build_math_train(max(40, n_train // 3), seed + 7):
        if s.kind in ("order",):
            continue
        add(s.question, s.answer, s.kind, s.gold)

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n", "42"),
        ("中国的首都是哪里？", "Answer: 北京\n", "北京"),
    ):
        aa = a.encode("utf-8").decode("unicode_escape")
        for _ in range(6):
            add(q, aa, "rehearsal", g)

    rng.shuffle(out)
    order_s = [s for s in out if s.kind == "order"]
    other = [s for s in out if s.kind != "order"]
    take_o = max(int(n_train * 0.75), 1)
    picked = order_s[:take_o] + other[: max(0, n_train - take_o)]
    rng.shuffle(picked)
    while len(picked) < n_train:
        q, a, g = cot_order(6, 8, 3, 12, 4)
        picked.append(Sample(q + f"#{len(picked)}", a, "order", g))
    return picked[:n_train]


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 640, answer_boost: float = 2.5):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len
        self.answer_boost = answer_boost

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = self.tok.apply_chat_template(
            [
                {"role": "user", "content": s.question},
                {"role": "assistant", "content": s.answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = self.tok(text, add_special_tokens=False)["input_ids"]
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
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
            if s.gold:
                gids = self.tok(str(s.gold), add_special_tokens=False)["input_ids"]
                span = len(gids)
                if span > 0:
                    for i in range(plen, len(full_ids) - span + 1):
                        if full_ids[i : i + span] == gids:
                            weights[i : i + span] = self.answer_boost * 1.6
                            break
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": weights,
        }


def eval_math(gen, probes: Sequence[Tuple[str, str, str]] = MATH_HOLDOUT) -> Dict:
    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for qi, (q, gold, kind) in enumerate(probes):
        pred = gen(q, 220)
        hit = bool(match_math(pred, gold, kind))
        ok += int(hit)
        by_kind.setdefault(kind, []).append(int(hit))
        rows.append(
            {
                "q": q,
                "gold": gold,
                "kind": kind,
                "ok": hit,
                "answer_line": first_answer_line(pred),
                "pred": pred[:500],
            }
        )
        print(
            f"  math[{qi+1}/{len(probes)}] {'OK' if hit else 'NO'} [{kind}] gold={gold} ans={first_answer_line(pred)[:48]!r}",
            flush=True,
        )
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    return {"accuracy": ok / max(1, len(probes)), "kind_acc": kind_acc, "rows": rows, "ok": ok, "n": len(probes)}


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if getattr(args, "order_fix", False):
        samples = build_math_order_train(args.n_train, args.seed)
    elif getattr(args, "v3", False):
        samples = build_math_v3_train(args.n_train, args.seed)
    elif getattr(args, "v2", False):
        samples = build_math_v2_train(args.n_train, args.seed)
    else:
        samples = build_math_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    kind_counts = Counter(s.kind for s in samples)
    print(f"n_train={len(samples)} kinds={dict(kind_counts)}", flush=True)

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    base.to(device)
    base.config.use_cache = False

    if args.resume:
        model = load_causal_lm(args.resume or args.model, device=device, trainable=True)
    else:
        model = load_causal_lm(args.model, device=device, trainable=True)
    print_trainable(model)

    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=2.8)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    print("===== BASELINE =====", flush=True)
    m0 = eval_math(gen)
    eng0 = eval_eng(gen)
    ctrl0 = eval_controls(gen)
    print(
        f"BASELINE math={m0['accuracy']:.3f} ({m0['ok']}/{m0['n']}) kinds={m0['kind_acc']} "
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
    answer_boost = 2.8
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
    m1 = eval_math(gen)
    eng1 = eval_eng(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER math={m1['accuracy']:.3f} ({m1['ok']}/{m1['n']}) kinds={m1['kind_acc']} "
        f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} loss={running/max(1,seen):.4f}",
        flush=True,
    )

    ctrl_floor = min(0.5, ctrl0["accuracy"])
    if getattr(args, "order_fix", False):
        eng_floor = max(0.50, eng0["accuracy"] - 0.17)
        math_gain = m1["accuracy"] + 1e-9 >= max(0.90, m0["accuracy"])
        math_floor = m1["accuracy"] + 1e-9 >= 0.90
        promote = (
            math_gain
            and math_floor
            and m1["kind_acc"].get("order", 0) >= 1.0
            and eng1["accuracy"] + 1e-9 >= eng_floor
            and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
        )
    else:
        eng_floor = max(0.75, min(eng0["accuracy"] + 0.08, 0.90) if eng0["accuracy"] < 0.83 else eng0["accuracy"] - 0.08)
        math_gain = m1["accuracy"] + 1e-9 >= max(0.83, m0["accuracy"] - 0.09)
        math_floor = m1["accuracy"] + 1e-9 >= 0.83
        promote = (
            math_gain
            and math_floor
            and eng1["accuracy"] + 1e-9 >= eng_floor
            and eng1["accuracy"] + 1e-9 >= 0.83
            and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
        )

    save_checkpoint(model, tok, out / "model_last")
    if promote:
        save_checkpoint(model, tok, out / "model_best")
        print("PROMOTED math_reason", flush=True)
    else:
        print(
            f"NO_PROMOTE math {m1['accuracy']:.3f}/{m0['accuracy']:.3f} "
            f"eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f}",
            flush=True,
        )

    report = {
        "method": Path(args.out).name,
        "n_train": len(samples),
        "kinds": dict(kind_counts),
        "lr": args.lr,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {
            "math": m0["accuracy"],
            "kind_acc": m0["kind_acc"],
            "eng": eng0["accuracy"],
            "ctrl": ctrl0["accuracy"],
        },
        "after": {
            "math": m1["accuracy"],
            "kind_acc": m1["kind_acc"],
            "eng": eng1["accuracy"],
            "ctrl": ctrl1["accuracy"],
        },
        "promoted": promote,
        "rows": m1["rows"],
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Pure-weight math CoT; Answer-first",
            "Holdout wording differs from train templates",
            "Guards: eng + ctrl; resume precision champion stack",
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
    p.add_argument("--resume", default="/Users/shiaho/Desktop/bitx/kef_results/persona_precision_champion/model_best")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/math_reason_v1")
    p.add_argument("--n-train", type=int, default=240)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--v2", action="store_true")
    p.add_argument("--v3", action="store_true")
    p.add_argument("--order-fix", action="store_true")
    args = p.parse_args()
    if args.eval_only:
        device = args.device
        dtype = torch.float16 if device == "mps" else torch.float32
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
        base.to(device)
        model = load_causal_lm(args.resume if args.resume else args.model, device=device, trainable=False)
        gen = make_gen(model, tok, device)
        print(json.dumps({"math": eval_math(gen), "eng": eval_eng(gen), "ctrl": eval_controls(gen)}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
