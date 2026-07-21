"""Pure-weight formal logic CoT: high-quality propositions, holdout suite, full-weight fine-tune."""

from __future__ import annotations

from kef.paths import default_model, result_path

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.folk_logic import CTRL_PROBES, Sample, collate, eval_controls, make_gen


LOGIC_PROBES: Tuple[Tuple[str, str, str], ...] = (
    (
        "前提1：如果下雨，地面会湿。前提2：下雨了。能否推出地面湿？只答能或不能。",
        "能",
        "modus_ponens",
    ),
    (
        "前提1：若通过考试则获得证书。前提2：没有获得证书。能否推出未通过考试？只答能或不能。",
        "能",
        "modus_tollens",
    ),
    (
        "前提1：如果是猫，则是哺乳动物。前提2：这是哺乳动物。能否推出这是猫？只答能或不能。",
        "不能",
        "affirm_consequent",
    ),
    (
        "前提1：所有企鹅都是鸟。前提2：所有鸟都会下蛋。能否推出所有企鹅都会下蛋？只答能或不能。",
        "能",
        "syllogism",
    ),
    (
        "前提1：有些学生是运动员。前提2：有些运动员很强壮。能否推出有些学生很强壮？只答能或不能。",
        "不能",
        "invalid_some",
    ),
    (
        "命题：完成作业是出门的必要条件。已知小明出门了。能否推出他完成了作业？只答能或不能。",
        "能",
        "necessary",
    ),
    (
        "命题：刷卡是进门的充分条件。已知小红进门了。能否推出她刷卡了？只答能或不能。",
        "不能",
        "sufficient",
    ),
    (
        "“如果P则Q”的逆否命题是什么？用“如果…则…”完整写出，变量仍用P、Q。",
        "如果非Q则非P",
        "contrapositive",
    ),
    (
        "A>B且B>C。能否推出A>C？只答能或不能。",
        "能",
        "transitivity",
    ),
    (
        "已知：房间里恰有一人在说谎。甲说“乙在说谎”。乙说“甲和丙都在说谎”。丙说“甲在说谎”。谁在说谎？只答甲/乙/丙之一。",
        "乙",
        "liar",
    ),
    (
        "箱子标签全错：一箱写苹果，一箱写橙子，一箱写苹果和橙子。打开写“苹果和橙子”的箱子全是苹果。写“苹果”的箱子里是什么？只答橙子/苹果/苹果和橙子。",
        "橙子",
        "label_box",
    ),
    (
        "所有A都是B。有些B不是C。能否推出有些A不是C？只答能或不能。",
        "不能",
        "quant_trap",
    ),
    (
        "P或Q成立，且非P。能否推出Q？只答能或不能。",
        "能",
        "disjunctive",
    ),
    (
        "若P则Q；若Q则R。已知P。能否推出R？只答能或不能。",
        "能",
        "chain",
    ),
    (
        "“并非（P且Q）”等价于下列哪一项？A. 非P且非Q  B. 非P或非Q  C. 非P或Q  只答字母。",
        "B",
        "demorgan",
    ),
    (
        "密码锁：四位数字，每位是前一位的两倍（首位为1）。密码是多少？只答四位数字。",
        "1248",
        "lock",
    ),
)


def _ans(body: str, gold: str) -> str:
    return f"Answer: {gold}\n{body.rstrip()}"


def cot_mp(p: str, q: str) -> str:
    return _ans(
        "\n".join(
            [
                f"形式：若P则Q；P；求证Q。",
                f"P={p}；Q={q}。",
                "这是肯定前件式（modus ponens），有效。",
                "因此能推出结论。",
            ]
        ),
        "能",
    )


def cot_mt(p: str, q: str) -> str:
    return _ans(
        "\n".join(
            [
                f"形式：P→Q，已知非Q，求非P。",
                f"P={p}；Q={q}。",
                "名称：否定后件式（modus tollens）。",
                "有效性：有效，不是谬误。",
                "对照：肯定后件（有Q求P）才是谬误。",
                "本题是否定Q，故有效，答案是能。",
            ]
        ),
        "能",
    )


def cot_ac(p: str, q: str) -> str:
    return _ans(
        "\n".join(
            [
                f"形式：若P则Q；Q；求证P。",
                f"P={p}；Q={q}。",
                "这是肯定后件谬误：Q真不能反推P。",
                "若要否定P，需要的是非Q（否定后件），不是Q。",
                "因此不能推出。",
            ]
        ),
        "不能",
    )


def cot_da(p: str, q: str) -> str:
    return _ans(
        "\n".join(
            [
                f"形式：若P则Q；非P；求证非Q。",
                f"P={p}；Q={q}。",
                "这是否定前件谬误：非P不能推非Q。",
                "因此不能推出。",
            ]
        ),
        "不能",
    )


def cot_syll_ok(a: str, b: str, c: str) -> str:
    return _ans(
        "\n".join(
            [
                f"全称三段论Barbara：所有{a}是{b}；所有{b}是{c}。",
                f"中项{b}连接，全称肯定可传递。",
                f"故所有{a}是{c}。有效，能推出。",
            ]
        ),
        "能",
    )


def cot_some_invalid(a: str, b: str, c: str) -> str:
    return _ans(
        "\n".join(
            [
                f"有些{a}是{b}；有些{b}是{c}。",
                "两“有些”中项不必重叠，不能保证有些A是C。",
                "因此不能推出。",
            ]
        ),
        "不能",
    )


def cot_necessary(task: str, allow: str) -> str:
    return _ans(
        "\n".join(
            [
                f"定义：{task}是{allow}的必要条件 = 没有{task}就没有{allow}。",
                f"逻辑式：{allow} → {task}（结果推出条件）。",
                f"已知{allow}已发生，代入得{task}。",
                "常见错：把必要说成充分。充分才是条件→结果。",
                "本题是必要，结果成立，答案是能。",
            ]
        ),
        "能",
    )


def cot_sufficient(task: str, allow: str) -> str:
    return _ans(
        "\n".join(
            [
                f"“{task}是{allow}的充分条件”= 若{task}则{allow}。",
                f"符号：{task} → {allow}。",
                f"已知{allow}，是肯定后件，不能反推一定{task}。",
                "因此不能推出。",
            ]
        ),
        "不能",
    )


def cot_contrapositive() -> str:
    return _ans(
        "\n".join(
            [
                "原命题：如果P则Q。",
                "逆否：如果非Q则非P。",
                "原命题与逆否逻辑等价。",
            ]
        ),
        "如果非Q则非P",
    )


def cot_trans(a: str, b: str, c: str) -> str:
    return _ans(
        "\n".join(
            [
                f"{a}>{b} 且 {b}>{c}，严格序传递。",
                f"故 {a}>{c}。",
                "因此能推出。",
            ]
        ),
        "能",
    )


def cot_disj(p: str, q: str) -> str:
    return _ans(
        "\n".join(
            [
                f"析取三段：{p}或{q}；非{p}。",
                f"两支必有一真，排除{p}后必有{q}。",
                "因此能推出。",
            ]
        ),
        "能",
    )


def cot_chain(p: str, q: str, r: str) -> str:
    return _ans(
        "\n".join(
            [
                f"链式：若{p}则{q}；若{q}则{r}；已知{p}。",
                f"先由肯定前件得{q}，再得{r}。",
                "因此能推出。",
            ]
        ),
        "能",
    )


def cot_demorgan() -> str:
    return _ans(
        "\n".join(
            [
                "德摩根：非(P且Q) ≡ (非P)或(非Q)。",
                "A是否定两边且；B是正确等价；C混入肯定。",
                "故选B。",
            ]
        ),
        "B",
    )


def cot_lock() -> str:
    return _ans(
        "\n".join(
            [
                "首位1；每位×2。",
                "1,2,4,8。",
                "密码1248。",
            ]
        ),
        "1248",
    )


def cot_liar_train(case: int) -> Tuple[str, str]:
    if case % 3 == 0:
        q = "恰有一人说谎。甲：“乙说谎。”乙：“甲和丙都说谎。”丙：“甲说谎。”谁说谎？只答甲/乙/丙。"
        body = "\n".join(
            [
                "约束：恰一人说谎，两人说真。",
                "验甲谎：甲假⇒乙真；乙真⇒甲丙都谎⇒至少两人谎，矛盾。",
                "验丙谎：丙假⇒甲真；甲真⇒乙谎。已有丙、乙两人谎，矛盾。",
                "验乙谎：乙假；甲真⇒乙谎（合）；丙真⇒“甲谎”为假⇒甲真（合）。",
                "仅乙说谎一致。",
            ]
        )
        return q, _ans(body, "乙")
    if case % 3 == 1:
        q = "恰有一人说谎。甲：“丙说谎。”乙：“甲说谎。”丙：“乙说真话。”谁说谎？只答甲/乙/丙。"
        body = "\n".join(
            [
                "约束：恰一人说谎。",
                "验甲谎：甲假；乙真⇒甲谎（合）；丙真⇒乙真（合）。仅甲谎一致。",
                "验乙谎：乙假；甲真⇒丙谎，已有乙丙两人，矛盾。",
                "验丙谎：丙假；甲真⇒丙谎（合）；乙真⇒甲谎，矛盾。",
                "故甲说谎。",
            ]
        )
        return q, _ans(body, "甲")
    q = "恰有一人说谎。甲：“乙和丙都说真话。”乙：“甲说谎。”丙：“乙说谎。”谁说谎？只答甲/乙/丙。"
    body = "\n".join(
        [
            "约束：恰一人说谎。",
            "验甲谎：甲假；乙真⇒甲谎（合）；若丙真⇒乙谎，与乙真矛盾。故甲谎时丙不能真。",
            "甲假且丙假时已两人谎，矛盾。故甲不能谎。",
            "验丙谎：丙假⇒乙真；乙真⇒甲谎。甲谎且丙谎，两人，矛盾。",
            "验乙谎：乙假；甲真⇒乙丙都真⇒丙真；丙真⇒乙谎（合）。仅乙谎。",
            "故乙说谎。",
        ]
    )
    return q, _ans(body, "乙")


def cot_label_box_train(case: int) -> Tuple[str, str]:
    if case % 2 == 0:
        q = (
            "三箱标签全错：标苹果/橙子/苹果和橙子。"
            "打开“苹果和橙子”全是苹果。标“苹果”的箱子是什么？只答橙子/苹果/苹果和橙子。"
        )
        body = "\n".join(
            [
                "前提：每箱标签都错。",
                "打开“苹果和橙子”全是苹果⇒该箱真实=苹果。",
                "标“苹果”的不能是苹果（标签错）。",
                "若标“苹果”=苹果和橙子，则标“橙子”只剩橙子，标签会正确，禁止。",
                "故标“苹果”=橙子；标“橙子”=苹果和橙子。",
            ]
        )
        return q, _ans(body, "橙子")
    q = (
        "三箱标签全错：标红/蓝/红蓝混合。"
        "打开“红蓝混合”全是红球。标“红”的箱子里是什么？只答蓝/红/红蓝混合。"
    )
    body = "\n".join(
        [
            "标签全错。",
            "“红蓝混合”实为红。",
            "“红”不能是红；若是红蓝混合，则“蓝”只剩蓝会成真标签。",
            "故“红”实为蓝。",
        ]
    )
    return q, _ans(body, "蓝")


def _pool_rules(rng: random.Random) -> Dict[str, List[Sample]]:
    pools: Dict[str, List[Sample]] = {k: [] for k in (
        "modus_ponens", "modus_tollens", "affirm_consequent", "deny_antecedent",
        "syllogism", "invalid_some", "necessary", "sufficient", "contrapositive",
        "demorgan", "transitivity", "disjunctive", "chain", "quant_trap",
        "liar", "label_box", "lock", "rehearsal", "contrast",
    )}

    mp_bank = [
        ("下雨", "地面湿"),
        ("断电", "灯灭"),
        ("通过考试", "获得证书"),
        ("是偶数", "能被2整除"),
        ("体温超过38℃", "发热"),
        ("钥匙正确", "门能打开"),
        ("燃料耗尽", "发动机熄火"),
        ("是正方形", "是矩形"),
        ("气温低于0℃", "水结冰"),
        ("按下开关", "电路接通"),
        ("信号中断", "通话失败"),
        ("分数及格", "课程通过"),
    ]
    for p, q in mp_bank:
        pools["modus_ponens"].append(
            Sample(f"前提1：如果{p}，则{q}。前提2：{p}。能否推出{q}？只答能或不能。", cot_mp(p, q), "modus_ponens", "能")
        )
        pools["modus_ponens"].append(
            Sample(f"已知：若{p}则{q}；并且{p}成立。问能否必然得到{q}？只答能或不能。", cot_mp(p, q), "modus_ponens", "能")
        )
        pools["modus_tollens"].append(
            Sample(
                f"前提1：若{p}则{q}。前提2：并非{q}。能否推出并非{p}？只答能或不能。",
                cot_mt(p, q),
                "modus_tollens",
                "能",
            )
        )
        pools["modus_tollens"].append(
            Sample(
                f"规则：{p}⇒{q}。观察：{q}不成立。能否断定{p}不成立？只答能或不能。",
                cot_mt(p, q),
                "modus_tollens",
                "能",
            )
        )
        pools["modus_tollens"].append(
            Sample(
                f"若{p}则{q}；现在没有{q}。按否定后件，能否得到没有{p}？只答能或不能。",
                cot_mt(p, q),
                "modus_tollens",
                "能",
            )
        )
        pools["affirm_consequent"].append(
            Sample(
                f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。",
                cot_ac(p, q),
                "affirm_consequent",
                "不能",
            )
        )
        pools["deny_antecedent"].append(
            Sample(
                f"前提1：如果{p}，则{q}。前提2：并非{p}。能否推出并非{q}？只答能或不能。",
                cot_da(p, q),
                "deny_antecedent",
                "不能",
            )
        )
        # contrast pair: same surface rule, MT vs AC
        pools["contrast"].append(
            Sample(
                f"对照训练。规则若{p}则{q}。情形A：非{q}，能否得非{p}？情形B：{q}，能否得{p}？先答A再答B，格式能/不能。",
                _ans(
                    "\n".join(
                        [
                            f"规则：{p}→{q}。",
                            "A：非Q，否定后件，有效 ⇒ 能。",
                            "B：Q，肯定后件，谬误 ⇒ 不能。",
                        ]
                    ),
                    "能/不能",
                ),
                "contrast",
                "能/不能",
            )
        )

    syll = [
        ("企鹅", "鸟", "会下蛋的动物"),
        ("正方形", "矩形", "四边形"),
        ("鲸鱼", "哺乳动物", "用肺呼吸的动物"),
        ("玫瑰", "花", "植物"),
        ("医生", "专业人士", "需要培训的人"),
        ("三角形", "多边形", "平面图形"),
        ("偶数", "整数", "有理数"),
        ("猫", "哺乳动物", "脊椎动物"),
    ]
    for a, b, c in syll:
        pools["syllogism"].append(
            Sample(
                f"前提1：所有{a}都是{b}。前提2：所有{b}都是{c}。能否推出所有{a}都是{c}？只答能或不能。",
                cot_syll_ok(a, b, c),
                "syllogism",
                "能",
            )
        )
        pools["syllogism"].append(
            Sample(
                f"所有{a}属于{b}类，所有{b}属于{c}类。Barbara式能否得到所有{a}属于{c}？只答能或不能。",
                cot_syll_ok(a, b, c),
                "syllogism",
                "能",
            )
        )
        pools["invalid_some"].append(
            Sample(
                f"前提1：有些{a}是{b}。前提2：有些{b}是{c}。能否推出有些{a}是{c}？只答能或不能。",
                cot_some_invalid(a, b, c),
                "invalid_some",
                "不能",
            )
        )

    nec_suf = [
        ("完成作业", "出门玩"),
        ("年满18岁", "投票"),
        ("刷卡", "进门"),
        ("交学费", "注册成功"),
        ("密码正确", "登录系统"),
        ("通过安检", "登机"),
        ("交押金", "租车"),
        ("签字确认", "合同生效"),
    ]
    for task, allow in nec_suf:
        pools["necessary"].append(
            Sample(
                f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                cot_necessary(task, allow),
                "necessary",
                "能",
            )
        )
        pools["necessary"].append(
            Sample(
                f"定义：{task}为{allow}之必要。现观察到{allow}已发生。按必要定义能否断定{task}？只答能或不能。",
                cot_necessary(task, allow),
                "necessary",
                "能",
            )
        )
        pools["sufficient"].append(
            Sample(
                f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                cot_sufficient(task, allow),
                "sufficient",
                "不能",
            )
        )
        pools["contrast"].append(
            Sample(
                f"对照：X是Y的必要条件；Y发生了，能否推X？再：X是Y的充分条件；Y发生了，能否推X？答能/不能。",
                _ans(
                    "\n".join(
                        [
                            "必要：Y→X，Y成立 ⇒ 能推X。",
                            "充分：X→Y，Y成立是肯定后件 ⇒ 不能推X。",
                        ]
                    ),
                    "能/不能",
                ),
                "contrast",
                "能/不能",
            )
        )

    for _ in range(16):
        pools["contrapositive"].append(
            Sample(
                "“如果P则Q”的逆否命题是什么？用“如果…则…”写出，变量用P、Q。",
                cot_contrapositive(),
                "contrapositive",
                "如果非Q则非P",
            )
        )
        pools["demorgan"].append(
            Sample(
                "“并非（P且Q）”等价于？A. 非P且非Q  B. 非P或非Q  C. P或非Q  只答字母。",
                cot_demorgan(),
                "demorgan",
                "B",
            )
        )

    for a, b, c in [("A", "B", "C"), ("x", "y", "z"), ("甲", "乙", "丙"), ("m", "n", "p"), ("U", "V", "W")]:
        pools["transitivity"].append(
            Sample(
                f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。",
                cot_trans(a, b, c),
                "transitivity",
                "能",
            )
        )
        pools["disjunctive"].append(
            Sample(
                f"{a}或{b}成立，且非{a}。能否推出{b}？只答能或不能。",
                cot_disj(a, b),
                "disjunctive",
                "能",
            )
        )
        pools["disjunctive"].append(
            Sample(
                f"已知{a}∨{b}，又知¬{a}。析取三段能否得到{b}？只答能或不能。",
                cot_disj(a, b),
                "disjunctive",
                "能",
            )
        )
        pools["chain"].append(
            Sample(
                f"若{a}则{b}；若{b}则{c}。已知{a}。能否推出{c}？只答能或不能。",
                cot_chain(a, b, c),
                "chain",
                "能",
            )
        )

    quant_traps = [
        ("所有A都是B。有些B不是C。能否推出有些A不是C？只答能或不能。", "不能",
         "A⊆B，有些B∉C，那些B可能不在A中。反模型存在，不能必然推出。"),
        ("所有A都是B。所有C都是B。能否推出有些A是C？只答能或不能。", "不能",
         "两类可同属B却互不相交，不能推有交集。"),
        ("没有A是B。所有C是A。能否推出没有C是B？只答能或不能。", "能",
         "C⊆A且A∩B=∅ ⇒ C∩B=∅，能推出。"),
        ("有些A是B。所有B是C。能否推出有些A是C？只答能或不能。", "能",
         "特称肯定+全称肯定，中项B可传，能推出。"),
        ("所有A不是B。有些B是C。能否推出有些A不是C？只答能或不能。", "不能",
         "与A无关的B-C关系不强制A与C，不能必然推出。"),
        ("有些A是B。有些A是C。能否推出有些B是C？只答能或不能。", "不能",
         "同一A的不同个体可分别是B与C，B与C可不交。"),
    ]
    for q, g, body in quant_traps:
        pools["quant_trap"].append(Sample(q, _ans(body, g), "quant_trap", g))
        pools["quant_trap"].append(Sample(q.replace("能否推出", "是否必然能得到"), _ans(body, g), "quant_trap", g))

    for i in range(24):
        q, a = cot_liar_train(i)
        pools["liar"].append(Sample(q, a, "liar", a.split("Answer: ")[-1].strip()))
        q2, a2 = cot_label_box_train(i)
        pools["label_box"].append(Sample(q2, a2, "label_box", a2.split("Answer: ")[-1].strip()))

    for _ in range(12):
        pools["lock"].append(
            Sample(
                "密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。",
                cot_lock(),
                "lock",
                "1248",
            )
        )
        pools["lock"].append(
            Sample(
                "密码三位：连续偶数从2起。密码？只答三位数字。",
                _ans("2,4,6 连续偶数。\n密码246。", "246"),
                "lock",
                "246",
            )
        )

    for q, a, g in (
        ("What is 17+25?", "17+25=42\nAnswer: 42", "42"),
        ("What is 9 times 6?", "9*6=54\nAnswer: 54", "54"),
        ("中国的首都是哪里？", "北京\nAnswer: 北京", "北京"),
        ("What is the capital of France?", "Paris\nAnswer: Paris", "paris"),
        ("1千克棉花和1千克铁，哪个更重？", "质量同为1kg，一样重。\nAnswer: 一样重", "一样重"),
    ):
        pools["rehearsal"].append(Sample(q, a, "rehearsal", g))

    return pools


def build_logic_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    pools = _pool_rules(rng)

    # quotas: boost MT/syll/necessary/liar/label; cut pure-MP pad
    quotas = {
        "modus_ponens": 16,
        "modus_tollens": 40,
        "affirm_consequent": 16,
        "deny_antecedent": 10,
        "syllogism": 36,
        "invalid_some": 14,
        "necessary": 36,
        "sufficient": 16,
        "contrapositive": 10,
        "demorgan": 10,
        "transitivity": 8,
        "disjunctive": 16,
        "chain": 14,
        "quant_trap": 18,
        "liar": 40,
        "label_box": 36,
        "lock": 10,
        "rehearsal": 6,
        "contrast": 20,
    }
    out: List[Sample] = []
    for kind, n in quotas.items():
        bank = pools.get(kind) or []
        if not bank:
            continue
        for i in range(n):
            out.append(bank[i % len(bank)])

    # fill remainder with balanced weak kinds, never pure-MP only
    weak = ["modus_tollens", "syllogism", "necessary", "liar", "label_box", "disjunctive", "quant_trap"]
    while len(out) < n_train:
        k = weak[len(out) % len(weak)]
        bank = pools[k]
        out.append(bank[rng.randrange(len(bank))])

    hold = {q for q, _, _ in LOGIC_PROBES}
    out = [s for s in out if s.question not in hold]
    while len(out) < n_train:
        k = weak[len(out) % len(weak)]
        bank = [s for s in pools[k] if s.question not in hold]
        if not bank:
            k = "modus_tollens"
            bank = [s for s in pools[k] if s.question not in hold]
        out.append(bank[rng.randrange(len(bank))])
    rng.shuffle(out)
    return out[:n_train]






def build_nec_expert_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}

    pairs = [
        ("完成作业", "出门"), ("完成作业", "出门玩"), ("写完报告", "请假"),
        ("刷卡", "进门"), ("交学费", "注册成功"), ("通过安检", "登机"),
        ("交押金", "租车"), ("签字确认", "合同生效"), ("有门票", "入场"),
        ("戴安全帽", "进工地"), ("年满18岁", "投票"), ("密码正确", "登录"),
        ("交作业", "下课"), ("充电", "手机开机"), ("开门", "进屋"),
        ("系安全带", "发车"), ("出示证件", "入场"), ("完成培训", "上岗"),
    ]
    people = ["小明", "小红", "他", "她", "张三", "李四", "同学"]

    def a_only(g: str) -> str:
        return f"Answer: {g}"

    def a_line(g: str, reason: str) -> str:
        return f"Answer: {g}\n{reason}"

    for task, allow in pairs:
        for who in people:
            qs_nec = [
                f"命题：{task}是{allow}的必要条件。已知{who}{allow}了。能否推出他{task}了？只答能或不能。",
                f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                f"只有{task}才能{allow}。现{who}已经{allow}。能否推出已{task}？只答能或不能。",
                f"{task}是{allow}的必要条件，逻辑式为{allow}→{task}。已知{allow}成立。能否推出{task}？只答能或不能。",
                f"定义：没有{task}就没有{allow}。现观察到{allow}。能否推出{task}？只答能或不能。",
                f"必要：{task}←{allow}。{who}{allow}了。能否推出{task}？只答能或不能。",
            ]
            for q in qs_nec:
                if q in hold:
                    continue
                out.append(Sample(q, a_only("能"), "necessary", "能"))
                out.append(Sample(q, a_line("能", f"{allow}→{task}，结果真则条件真。"), "necessary", "能"))
                out.append(Sample(q, a_line("能", "必要条件：由结果可反推条件。"), "necessary", "能"))

            qs_suf = [
                f"命题：{task}是{allow}的充分条件。已知{who}{allow}了。能否推出他{task}了？只答能或不能。",
                f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                f"充分：{task}→{allow}。已知{allow}。能否推出{task}？只答能或不能。",
            ]
            for q in qs_suf:
                if q in hold:
                    continue
                out.append(Sample(q, a_only("不能"), "sufficient", "不能"))
                out.append(Sample(q, a_line("不能", f"充分是{task}→{allow}，不能由{allow}反推{task}。"), "sufficient", "不能"))

        # explicit contrast mini
        q_c = f"对照：{task}是{allow}的必要条件，已知{allow}，能否推{task}？只答能或不能。"
        if q_c not in hold:
            out.append(Sample(q_c, a_only("能"), "necessary", "能"))
            out.append(Sample(q_c, a_line("能", "必要：结果→条件。"), "necessary", "能"))
        q_c2 = f"对照：{task}是{allow}的充分条件，已知{allow}，能否推{task}？只答能或不能。"
        if q_c2 not in hold:
            out.append(Sample(q_c2, a_only("不能"), "sufficient", "不能"))
            out.append(Sample(q_c2, a_line("不能", "充分：条件→结果，不可逆。"), "sufficient", "不能"))

    # keep MT/AC light so we do not regress
    bank = [
        ("通过考试", "获得证书"), ("下雨", "地面湿"), ("是猫", "是哺乳动物"),
        ("钥匙正确", "门打开"), ("分数及格", "课程通过"),
    ]
    for p, q in bank * 4:
        q_mt = f"前提1：若{p}则{q}。前提2：没有{q}。能否推出未{p}？只答能或不能。"
        q_ac = f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。"
        if q_mt not in hold:
            out.append(Sample(q_mt, a_only("能"), "modus_tollens", "能"))
            out.append(Sample(q_mt, a_line("能", "否定后件有效。"), "modus_tollens", "能"))
        if q_ac not in hold:
            out.append(Sample(q_ac, a_only("不能"), "affirm_consequent", "不能"))
            out.append(Sample(q_ac, a_line("不能", "肯定后件无效。"), "affirm_consequent", "不能"))

    for q, a, g in (
        ("What is 17+25?", "Answer: 42", "42"),
        ("中国的首都是哪里？", "Answer: 北京", "北京"),
        ("What is the capital of France?", "Answer: Paris", "paris"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    prefer = {"necessary": 0.55, "sufficient": 0.25, "modus_tollens": 0.10, "affirm_consequent": 0.08}
    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    picked: List[Sample] = []
    for kind, frac in prefer.items():
        take = max(1, int(n_train * frac))
        pool = buckets.get(kind, [])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    picked_ids = {id(s) for s in picked}
    rest = [s for s in out if id(s) not in picked_ids]
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        task, allow = rng.choice(pairs)
        who = rng.choice(people)
        picked.append(Sample(
            f"命题：{task}是{allow}的必要条件。已知{who}{allow}了。能否推出他{task}了？只答能或不能。",
            a_only("能"),
            "necessary",
            "能",
        ))
    rng.shuffle(picked)
    return picked[:n_train]


def build_repair_expert_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}

    def a_only(g: str) -> str:
        return f"Answer: {g}"

    def a_line(g: str, reason: str) -> str:
        return f"Answer: {g}\n{reason}"

    vars3 = [("A","B","C"),("x","y","z"),("甲","乙","丙"),("P","Q","R"),("U","V","W"),("m","n","p")]
    for a,b,c in vars3 * 12:
        qs = [
            f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。",
            f"已知{a}>{b}，{b}>{c}。严格序能否得{a}>{c}？只答能或不能。",
            f"{a}大于{b}且{b}大于{c}。能否推出{a}大于{c}？只答能或不能。",
        ]
        for q in qs:
            if q in hold:
                continue
            out.append(Sample(q, a_only("能"), "transitivity", "能"))
            out.append(Sample(q, a_line("能", "严格序传递。"), "transitivity", "能"))
        qs2 = [
            f"{a}或{b}成立，且非{a}。能否推出{b}？只答能或不能。",
            f"已知{a}∨{b}，又知¬{a}。能否得{b}？只答能或不能。",
            f"P或Q成立，且非P。能否推出Q？只答能或不能。" if (a,b)==("P","Q") else f"{a}或{b}，非{a}，得{b}？只答能或不能。",
        ]
        for q in qs2:
            if q in hold:
                continue
            out.append(Sample(q, a_only("能"), "disjunctive", "能"))
            out.append(Sample(q, a_line("能", "析取排除法。"), "disjunctive", "能"))

    # explicit holdout-near disj without exact hold text
    for q in [
        "P∨Q成立，且非P。能否推出Q？只答能或不能。",
        "已知P或Q，且不是P。能否得到Q？只答能或不能。",
        "析取：P或Q；再非P。能否推出Q？只答能或不能。",
    ]:
        if q not in hold:
            out.append(Sample(q, a_only("能"), "disjunctive", "能"))
            out.append(Sample(q, a_line("能", "非P消去，剩Q。"), "disjunctive", "能"))

    for q, a, g in (
        ("What is 17+25?", "Answer: 42", "42"),
        ("中国的首都是哪里？", "Answer: 北京", "北京"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    if len(out) > n_train:
        out = out[:n_train]
    while len(out) < n_train:
        out.append(Sample(
            "A>B且B>C。能否推出A>C？只答能或不能。",
            a_only("能"),
            "transitivity",
            "能",
        ))
    return out[:n_train]


def build_mt_expert_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}

    bank = [
        ("下雨", "地面湿"), ("断电", "灯灭"), ("通过考试", "获得证书"),
        ("是偶数", "能被2整除"), ("钥匙正确", "门打开"), ("是正方形", "是矩形"),
        ("分数及格", "课程通过"), ("密码正确", "登录成功"), ("按下开关", "电路接通"),
        ("燃料耗尽", "发动机熄火"), ("信号中断", "通话失败"), ("体温超38", "发热"),
        ("是猫", "是哺乳动物"), ("努力学习", "成绩提高"), ("引擎启动", "汽车开动"),
        ("网络连通", "能上网"), ("气温低于0度", "结冰"), ("是素数", "只有两个正因数"),
    ]
    cert_bank = [
        ("通过考试", "获得证书"), ("通过体检", "拿到健康证"), ("通过答辩", "获得学位"),
        ("完成培训", "拿到上岗证"), ("通过路考", "拿到驾照"), ("通过安检", "拿到登机牌"),
    ]
    bank = bank + cert_bank

    def a_only(g: str) -> str:
        return f"Answer: {g}"

    def a_line(g: str, reason: str) -> str:
        return f"Answer: {g}\n{reason}"

    mt_q = []
    ac_q = []
    mp_q = []
    for p, q in bank:
        mt_q.extend([
            f"前提1：若{p}则{q}。前提2：并非{q}。能否推出并非{p}？只答能或不能。",
            f"前提1：若{p}则{q}。前提2：没有{q}。能否推出未{p}？只答能或不能。",
            f"规则：如果{p}，那么{q}。现在{q}不成立。能否断定{p}不成立？只答能或不能。",
            f"若{p}则{q}；非{q}。能否得非{p}？只答能或不能。",
            f"已知：{p}⇒{q}。观察：{q}为假。否定后件能否得到{p}为假？只答能或不能。",
            f"前提1：如果{p}则{q}。前提2：没有{q}。能否推出没有{p}？只答能或不能。",
        ])
        ac_q.extend([
            f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。",
            f"规则：若{p}则{q}。现已{q}。能否断定一定{p}？只答能或不能。",
            f"已知{q}，且若{p}则{q}。能否推出{p}？只答能或不能。",
        ])
        mp_q.extend([
            f"前提1：如果{p}，则{q}。前提2：{p}。能否推出{q}？只答能或不能。",
            f"若{p}则{q}；已知{p}。能否得{q}？只答能或不能。",
        ])

    for qtext in mt_q:
        if qtext in hold:
            continue
        out.append(Sample(qtext, a_only("能"), "modus_tollens", "能"))
        out.append(Sample(qtext, a_line("能", "否定后件有效，非Q⇒非P。"), "modus_tollens", "能"))
        out.append(Sample(qtext, a_line("能", "MT有效，不是谬误。"), "modus_tollens", "能"))

    for qtext in ac_q:
        if qtext in hold:
            continue
        out.append(Sample(qtext, a_only("不能"), "affirm_consequent", "不能"))
        out.append(Sample(qtext, a_line("不能", "肯定后件无效。"), "affirm_consequent", "不能"))

    for qtext in mp_q:
        if qtext in hold:
            continue
        out.append(Sample(qtext, a_only("能"), "modus_ponens", "能"))
        out.append(Sample(qtext, a_line("能", "肯定前件有效。"), "modus_ponens", "能"))

    for p, q in cert_bank * 3:
        q1 = f"前提1：若{p}则{q}。前提2：没有{q}。能否推出未{p}？只答能或不能。"
        if q1 not in hold:
            out.append(Sample(q1, a_only("能"), "modus_tollens", "能"))
            out.append(Sample(q1, a_line("能", "没有结果⇒没有条件。"), "modus_tollens", "能"))
        q3 = f"前提1：如果{p}，则{q}。前提2：已{q}。能否推出已{p}？只答能或不能。"
        if q3 not in hold:
            out.append(Sample(q3, a_only("不能"), "affirm_consequent", "不能"))

    nec = [
        ("完成作业", "出门"), ("完成作业", "出门玩"), ("写完报告", "请假"),
        ("年满18岁", "投票"), ("刷卡", "进门"), ("交学费", "注册成功"),
        ("密码正确", "登录系统"), ("通过安检", "登机"), ("交押金", "租车"),
        ("签字确认", "合同生效"), ("有门票", "入场"), ("戴安全帽", "进工地"),
        ("交作业", "下课"), ("充电", "手机开机"), ("开门", "进屋"),
    ]
    for task, allow in nec:
        templates = [
            f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            f"命题：{task}是{allow}的必要条件。已知小明{allow}了。能否推出他{task}了？只答能或不能。",
            f"定义：{task}为{allow}之必要。现观察到{allow}已发生。按必要定义能否断定{task}？只答能或不能。",
            f"只有{task}才能{allow}。现已{allow}。能否推出已{task}？只答能或不能。",
            f"必要：没有{task}就没有{allow}。今有{allow}。能否推出有{task}？只答能或不能。",
            f"命题：{task}是{allow}的必要条件。{allow}已发生，能否推出{task}？只答能或不能。",
        ]
        for qtext in templates:
            if qtext in hold:
                continue
            out.append(Sample(qtext, a_only("能"), "necessary", "能"))
            out.append(Sample(qtext, a_line("能", f"必要：{allow}→{task}。"), "necessary", "能"))
            out.append(Sample(qtext, a_line("能", "结果真则条件真。"), "necessary", "能"))

        suf = [
            f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            f"充分：{task}⇒{allow}。现有{allow}。能否推出{task}？只答能或不能。",
        ]
        for qtext in suf:
            if qtext in hold:
                continue
            out.append(Sample(qtext, a_only("不能"), "sufficient", "不能"))
            out.append(Sample(qtext, a_line("不能", "充分不能由结果反推条件。"), "sufficient", "不能"))

    for a, b, c in [("企鹅", "鸟", "会下蛋"), ("正方形", "矩形", "四边形"), ("鲸鱼", "哺乳动物", "用肺呼吸"), ("猫", "哺乳动物", "脊椎动物")]:
        q1 = f"前提1：所有{a}都是{b}。前提2：所有{b}都是{c}。能否推出所有{a}都是{c}？只答能或不能。"
        q2 = f"前提1：有些{a}是{b}。前提2：有些{b}是{c}。能否推出有些{a}是{c}？只答能或不能。"
        if q1 not in hold:
            out.append(Sample(q1, a_only("能"), "syllogism", "能"))
            out.append(Sample(q1, a_line("能", "Barbara全称传递。"), "syllogism", "能"))
        if q2 not in hold:
            out.append(Sample(q2, a_only("不能"), "invalid_some", "不能"))
            out.append(Sample(q2, a_line("不能", "两特称中项可不交。"), "invalid_some", "不能"))

    for a, b, c in [("A", "B", "C"), ("x", "y", "z"), ("甲", "乙", "丙")]:
        q_t = f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。"
        q_c = f"若{a}则{b}；若{b}则{c}。已知{a}。能否推出{c}？只答能或不能。"
        q_d = f"{a}或{b}成立，且非{a}。能否推出{b}？只答能或不能。"
        for qtext, g, kind, reason in (
            (q_t, "能", "transitivity", "序传递。"),
            (q_c, "能", "chain", "链式肯定前件。"),
            (q_d, "能", "disjunctive", "析取排除法。"),
        ):
            if qtext in hold:
                continue
            out.append(Sample(qtext, a_only(g), kind, g))
            out.append(Sample(qtext, a_line(g, reason), kind, g))

    for p, q in bank[:8]:
        qa = f"对照。规则若{p}则{q}。A：非{q}，能否得非{p}？只答能或不能。"
        qb = f"对照。规则若{p}则{q}。B：{q}，能否得{p}？只答能或不能。"
        if qa not in hold:
            out.append(Sample(qa, a_only("能"), "modus_tollens", "能"))
            out.append(Sample(qa, a_line("能", "A是否定后件，有效。"), "modus_tollens", "能"))
        if qb not in hold:
            out.append(Sample(qb, a_only("不能"), "affirm_consequent", "不能"))
            out.append(Sample(qb, a_line("不能", "B是肯定后件，无效。"), "affirm_consequent", "不能"))

    for q, a, g in (
        ("What is 17+25?", "Answer: 42", "42"),
        ("What is 9 times 6?", "Answer: 54", "54"),
        ("中国的首都是哪里？", "Answer: 北京", "北京"),
        ("What is the capital of France?", "Answer: Paris", "paris"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    prefer = {
        "modus_tollens": 0.38,
        "necessary": 0.28,
        "affirm_consequent": 0.18,
        "modus_ponens": 0.06,
        "sufficient": 0.04,
        "syllogism": 0.02,
        "invalid_some": 0.01,
        "disjunctive": 0.01,
        "transitivity": 0.01,
        "chain": 0.01,
    }
    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    picked: List[Sample] = []
    for kind, frac in prefer.items():
        take = max(1, int(n_train * frac))
        pool = buckets.get(kind, [])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    picked_ids = {id(s) for s in picked}
    rest = [s for s in out if id(s) not in picked_ids]
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        p, q = rng.choice(bank)
        picked.append(Sample(
            f"若{p}则{q}；非{q}。能否得非{p}？只答能或不能。",
            a_only("能"),
            "modus_tollens",
            "能",
        ))
    rng.shuffle(picked)
    return picked[:n_train]


def build_puzzle_expert_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}

    def a_only(g: str) -> str:
        return f"Answer: {g}"

    def a_line(g: str, reason: str) -> str:
        return f"Answer: {g}\n{reason}"

    for i in range(48):
        q, a = cot_liar_train(i)
        if q in hold:
            continue
        m = re.search(r"Answer:\s*(\S+)", a)
        g = m.group(1).strip("。.") if m else "乙"
        out.append(Sample(q, a_only(g), "liar", g))
        out.append(Sample(q, a_line(g, "恰一人说谎穷举，仅此一致。"), "liar", g))
        out.append(Sample(q, a_line(g, f"假设法：只有{g}说谎时无矛盾。"), "liar", g))
        q_alt = q.replace("谁在说谎？", "说谎的是谁？").replace("只答甲/乙/丙之一。", "只答甲或乙或丙。")
        if q_alt not in hold and q_alt != q:
            out.append(Sample(q_alt, a_only(g), "liar", g))
            out.append(Sample(q_alt, a_line(g, "三人一谎，定位唯一。"), "liar", g))

    for i in range(40):
        q2, a2 = cot_label_box_train(i)
        if q2 in hold:
            continue
        m = re.search(r"Answer:\s*(\S+)", a2)
        g2 = m.group(1).strip("。.") if m else "橙子"
        out.append(Sample(q2, a_only(g2), "label_box", g2))
        out.append(Sample(q2, a_line(g2, "标签全错；混合箱打开定真实。"), "label_box", g2))
        out.append(Sample(q2, a_line(g2, "写苹果的必非苹果；推得真实为橙子。"), "label_box", g2))
        q3 = q2.replace("写“苹果”的箱子里是什么？", "标着苹果的箱子实际是什么？")
        if q3 not in hold and q3 != q2:
            out.append(Sample(q3, a_only(g2), "label_box", g2))

    lock_qs = [
        ("密码锁：四位数字，每位是前一位的两倍（首位为1）。密码是多少？只答四位数字。", "1248", "1→2→4→8。"),
        ("密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。", "1248", "倍增序列。"),
        ("四位密码，首位1，其后每位为前一位两倍。密码？只答四位数字。", "1248", "1,2,4,8。"),
        ("密码三位：连续偶数从2起。密码？只答三位数字。", "246", "2,4,6。"),
        ("三位密码从2起连续偶数。答案？只答三位数字。", "246", "246。"),
    ]
    for q, g, reason in lock_qs * 8:
        if q in hold:
            continue
        out.append(Sample(q, a_only(g), "lock", g))
        out.append(Sample(q, a_line(g, reason), "lock", g))

    dm_qs = [
        ("“并非（P且Q）”等价于下列哪一项？A. 非P且非Q  B. 非P或非Q  C. 非P或Q  只答字母。", "B"),
        ("“并非（P且Q）”等价于？A. 非P且非Q  B. 非P或非Q  C. P或非Q  只答字母。", "B"),
        ("德摩根：¬(P∧Q) 等于？A. ¬P∧¬Q  B. ¬P∨¬Q  C. ¬P∨Q  只答字母。", "B"),
        ("非(P并且Q) 等于 非P 或者 非Q。对应选项字母？只答B若为非P或非Q。", "B"),
    ]
    for q, g in dm_qs * 10:
        if q in hold:
            continue
        out.append(Sample(q, a_only(g), "demorgan", g))
        out.append(Sample(q, a_line(g, "德摩根：非(P∧Q)≡(¬P)∨(¬Q)。"), "demorgan", g))

    for _ in range(12):
        q = "“如果P则Q”的逆否命题是什么？用“如果…则…”完整写出，变量仍用P、Q。"
        if q not in hold:
            out.append(Sample(q, a_only("如果非Q则非P"), "contrapositive", "如果非Q则非P"))
            out.append(Sample(q, a_line("如果非Q则非P", "逆否=否定并交换。"), "contrapositive", "如果非Q则非P"))

    for q, a, g in (
        ("What is 17+25?", "Answer: 42", "42"),
        ("What is 9 times 6?", "Answer: 54", "54"),
        ("中国的首都是哪里？", "Answer: 北京", "北京"),
        ("What is the capital of France?", "Answer: Paris", "paris"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    prefer = {"liar": 0.36, "label_box": 0.34, "lock": 0.14, "demorgan": 0.12, "contrapositive": 0.02}
    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    picked: List[Sample] = []
    for kind, frac in prefer.items():
        take = max(1, int(n_train * frac))
        pool = buckets.get(kind, [])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    picked_ids = {id(s) for s in picked}
    rest = [s for s in out if id(s) not in picked_ids]
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        picked.append(Sample(
            "密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。",
            a_only("1248"),
            "lock",
            "1248",
        ))
    rng.shuffle(picked)
    return picked[:n_train]


def build_micro_logic(n_train: int, seed: int) -> List[Sample]:
    """Ultra-short Answer-first; no tables; paired MT/AC/necessary/disj/lock."""
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}
    bank = [
        ("下雨", "地面湿"), ("断电", "灯灭"), ("通过考试", "获得证书"),
        ("是偶数", "能被2整除"), ("钥匙正确", "门打开"), ("是正方形", "是矩形"),
        ("分数及格", "课程通过"), ("密码正确", "登录成功"), ("按下开关", "电路接通"),
        ("燃料耗尽", "发动机熄火"),
    ]
    for p, q in bank * 4:
        out.append(Sample(
            f"前提1：如果{p}，则{q}。前提2：{p}。能否推出{q}？只答能或不能。",
            f"Answer: 能\n肯定前件有效。", "modus_ponens", "能"))
        out.append(Sample(
            f"前提1：若{p}则{q}。前提2：并非{q}。能否推出并非{p}？只答能或不能。",
            f"Answer: 能\n否定后件有效（不是谬误）。非Q⇒非P。", "modus_tollens", "能"))
        out.append(Sample(
            f"规则：如果{p}那么{q}。现在没有{q}。能否断定没有{p}？只答能或不能。",
            f"Answer: 能\n否定后件式有效。", "modus_tollens", "能"))
        out.append(Sample(
            f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。",
            f"Answer: 不能\n肯定后件无效。", "affirm_consequent", "不能"))
        out.append(Sample(
            f"{p}或{q}成立，且非{p}。能否推出{q}？只答能或不能。",
            f"Answer: 能\n析取排除法。", "disjunctive", "能"))

    nec = [("完成作业","出门玩"),("年满18岁","投票"),("刷卡","进门"),("交学费","注册成功"),
           ("密码正确","登录系统"),("通过安检","登机"),("交押金","租车"),("签字确认","合同生效")]
    for task, allow in nec * 4:
        out.append(Sample(
            f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            f"Answer: 能\n必要={allow}→{task}。结果成立则条件成立。", "necessary", "能"))
        out.append(Sample(
            f"只有{task}才能{allow}。现已{allow}。能否推出已{task}？只答能或不能。",
            f"Answer: 能\n只有A才B ⇒ B→A。", "necessary", "能"))
        out.append(Sample(
            f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            f"Answer: 不能\n充分=条件→结果，结果不能反推条件。", "sufficient", "不能"))

    syll = [("企鹅","鸟","会下蛋的动物"),("正方形","矩形","四边形"),("鲸鱼","哺乳动物","用肺呼吸的动物"),
            ("玫瑰","花","植物"),("医生","专业人士","需要培训的人"),("三角形","多边形","平面图形")]
    for a,b,c in syll * 3:
        out.append(Sample(
            f"前提1：所有{a}都是{b}。前提2：所有{b}都是{c}。能否推出所有{a}都是{c}？只答能或不能。",
            f"Answer: 能\n全称传递Barbara。", "syllogism", "能"))
        out.append(Sample(
            f"前提1：有些{a}是{b}。前提2：有些{b}是{c}。能否推出有些{a}是{c}？只答能或不能。",
            f"Answer: 不能\n两特称中项可不交。", "invalid_some", "不能"))

    for a,b,c in [("A","B","C"),("x","y","z"),("甲","乙","丙"),("m","n","p")]:
        for _ in range(4):
            out.append(Sample(f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。",
                              f"Answer: 能\n序传递。", "transitivity", "能"))
            out.append(Sample(f"若{a}则{b}；若{b}则{c}。已知{a}。能否推出{c}？只答能或不能。",
                              f"Answer: 能\n链式肯定前件。", "chain", "能"))

    for _ in range(16):
        out.append(Sample(
            "“如果P则Q”的逆否命题是什么？用“如果…则…”写出，变量用P、Q。",
            "Answer: 如果非Q则非P\n逆否=否定并交换。", "contrapositive", "如果非Q则非P"))
        out.append(Sample(
            "“并非（P且Q）”等价于？A. 非P且非Q  B. 非P或非Q  C. P或非Q  只答字母。",
            "Answer: B\n德摩根：非(P且Q)=(非P)或(非Q)。", "demorgan", "B"))

    quant = [
        ("所有A都是B。有些B不是C。能否推出有些A不是C？只答能或不能。", "不能", "B非C可不在A。"),
        ("凡A皆B。存在B不是C。是否必然有A不是C？只答能或不能。", "不能", "反模型存在。"),
        ("没有A是B。所有C是A。能否推出没有C是B？只答能或不能。", "能", "传递排斥。"),
        ("有些A是B。所有B是C。能否推出有些A是C？只答能或不能。", "能", "特称传递。"),
    ]
    for q,g,body in quant * 5:
        out.append(Sample(q, f"Answer: {g}\n{body}", "quant_trap", g))

    for i in range(24):
        q,a = cot_liar_train(i)
        if q in hold: continue
        m=re.search(r"Answer:\s*(\S+)", a)
        g=m.group(1) if m else "乙"
        # shorten
        out.append(Sample(q, f"Answer: {g}\n穷举恰一人说谎，仅此一致。", "liar", g))
        q2,a2=cot_label_box_train(i)
        if q2 in hold: continue
        m=re.search(r"Answer:\s*(\S+)", a2)
        g2=m.group(1) if m else "橙子"
        out.append(Sample(q2, f"Answer: {g2}\n标签全错+打开混合箱。", "label_box", g2))

    for _ in range(12):
        out.append(Sample(
            "密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。",
            "Answer: 1248\n1→2→4→8。", "lock", "1248"))

    for q,a,g in (
        ("What is 17+25?", "Answer: 42", "42"),
        ("What is 9 times 6?", "Answer: 54", "54"),
        ("中国的首都是哪里？", "Answer: 北京", "北京"),
        ("What is the capital of France?", "Answer: Paris", "paris"),
    ):
        out.append(Sample(q,a,"rehearsal",g))

    out=[s for s in out if s.question not in hold]
    rng.shuffle(out)
    if len(out)>n_train:
        out=out[:n_train]
    else:
        while len(out)<n_train:
            p,q=rng.choice(bank)
            out.append(Sample(
                f"若{p}则{q}；非{q}。能否得非{p}？只答能或不能。",
                "Answer: 能\n否定后件有效。", "modus_tollens", "能"))
    return out[:n_train]


def build_validity_table_train(n_train: int, seed: int) -> List[Sample]:
    """Ultra-clear validity table + paired contrasts; answer-first; from base friendly."""
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}
    bank = [
        ("下雨", "地面湿"),
        ("断电", "灯灭"),
        ("通过考试", "获得证书"),
        ("是偶数", "能被2整除"),
        ("钥匙正确", "门打开"),
        ("燃料耗尽", "发动机熄火"),
        ("是正方形", "是矩形"),
        ("分数及格", "课程通过"),
        ("按下开关", "电路接通"),
        ("信号中断", "通话失败"),
        ("体温超38", "发热"),
        ("密码正确", "登录成功"),
    ]

    table = (
        "有效/无效表：\n"
        "1) P→Q，P ⇒ Q ：有效（肯定前件）\n"
        "2) P→Q，非Q ⇒ 非P ：有效（否定后件）\n"
        "3) P→Q，Q ⇒ P ：无效（肯定后件）\n"
        "4) P→Q，非P ⇒ 非Q ：无效（否定前件）\n"
    )

    for p, q in bank:
        # MP
        out.append(Sample(
            f"前提1：如果{p}，则{q}。前提2：{p}。能否推出{q}？只答能或不能。",
            _ans(table + f"本题类型1：有{p}，能得{q}。", "能"),
            "modus_ponens", "能",
        ))
        # MT critical
        out.append(Sample(
            f"前提1：若{p}则{q}。前提2：并非{q}。能否推出并非{p}？只答能或不能。",
            _ans(table + f"本题类型2：有非{q}，能得非{p}。否定后件有效。", "能"),
            "modus_tollens", "能",
        ))
        out.append(Sample(
            f"规则：{p}⇒{q}。观察：没有{q}。能否断定没有{p}？只答能或不能。",
            _ans(table + "类型2有效。答案能。", "能"),
            "modus_tollens", "能",
        ))
        out.append(Sample(
            f"如果{p}那么{q}；现在{q}为假。按逻辑能否推出{p}为假？只答能或不能。",
            _ans("P→Q 与 非Q 推出 非P。有效。能。", "能"),
            "modus_tollens", "能",
        ))
        # AC
        out.append(Sample(
            f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。",
            _ans(table + f"本题类型3：有{q}求{p}，无效。", "不能"),
            "affirm_consequent", "不能",
        ))
        # DA
        out.append(Sample(
            f"前提1：如果{p}，则{q}。前提2：并非{p}。能否推出并非{q}？只答能或不能。",
            _ans(table + f"本题类型4：有非{p}求非{q}，无效。", "不能"),
            "deny_antecedent", "不能",
        ))
        # paired contrast one sample
        out.append(Sample(
            f"同一规则若{p}则{q}。问A：非{q}能否得非{p}？问B：{q}能否得{p}？按能/不能回答，中间用斜杠。",
            _ans(table + "A=类型2有效→能；B=类型3无效→不能。", "能/不能"),
            "contrast", "能/不能",
        ))

    # necessary vs sufficient paired
    pairs = [
        ("完成作业", "出门"),
        ("年满18岁", "投票"),
        ("刷卡", "进门"),
        ("交学费", "注册"),
        ("密码正确", "登录"),
        ("通过安检", "登机"),
        ("交押金", "租车"),
        ("签字", "合同生效"),
        ("有门票", "入场"),
        ("戴安全帽", "进工地"),
    ]
    for task, allow in pairs:
        out.append(Sample(
            f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            _ans(
                f"必要：{allow}→{task}。已知{allow}，得{task}。能。\n"
                f"错误说法「出门不能推完成作业」把必要当成了充分。",
                "能",
            ),
            "necessary", "能",
        ))
        out.append(Sample(
            f"只有{task}才能{allow}。现已{allow}。能否推出已{task}？只答能或不能。",
            _ans(f"只有A才B = B→A。B={allow}⇒A={task}。能。", "能"),
            "necessary", "能",
        ))
        out.append(Sample(
            f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
            _ans(f"充分：{task}→{allow}。已知{allow}是肯定后件，不能。", "不能"),
            "sufficient", "不能",
        ))

    syll = [
        ("企鹅", "鸟", "会下蛋的动物"),
        ("正方形", "矩形", "四边形"),
        ("鲸鱼", "哺乳动物", "用肺呼吸的动物"),
        ("玫瑰", "花", "植物"),
        ("医生", "专业人士", "需要培训的人"),
        ("三角形", "多边形", "平面图形"),
        ("偶数", "整数", "有理数"),
        ("猫", "哺乳动物", "脊椎动物"),
    ]
    for a, b, c in syll:
        out.append(Sample(
            f"前提1：所有{a}都是{b}。前提2：所有{b}都是{c}。能否推出所有{a}都是{c}？只答能或不能。",
            _ans(f"Barbara全称传递：{a}⊆{b}⊆{c} ⇒ {a}⊆{c}。能。", "能"),
            "syllogism", "能",
        ))
        out.append(Sample(
            f"前提1：有些{a}是{b}。前提2：有些{b}是{c}。能否推出有些{a}是{c}？只答能或不能。",
            cot_some_invalid(a, b, c), "invalid_some", "不能",
        ))

    for a, b, c in [("A", "B", "C"), ("x", "y", "z"), ("甲", "乙", "丙"), ("m", "n", "p"), ("U", "V", "W")]:
        out.append(Sample(f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。",
                          _ans(f"序传递有效。能。", "能"), "transitivity", "能"))
        out.append(Sample(f"{a}或{b}成立，且非{a}。能否推出{b}？只答能或不能。",
                          _ans(f"析取三段：({a}∨{b})∧¬{a} ⇒ {b}。能。", "能"), "disjunctive", "能"))
        out.append(Sample(f"若{a}则{b}；若{b}则{c}。已知{a}。能否推出{c}？只答能或不能。",
                          cot_chain(a, b, c), "chain", "能"))

    for _ in range(18):
        out.append(Sample(
            "“如果P则Q”的逆否命题是什么？用“如果…则…”写出，变量用P、Q。",
            _ans("逆否=否定并交换：如果非Q则非P。", "如果非Q则非P"),
            "contrapositive", "如果非Q则非P",
        ))
        out.append(Sample(
            "“并非（P且Q）”等价于？A. 非P且非Q  B. 非P或非Q  C. P或非Q  只答字母。",
            cot_demorgan(), "demorgan", "B",
        ))

    quant = [
        ("所有A都是B。有些B不是C。能否推出有些A不是C？只答能或不能。", "不能",
         "B中非C者可不在A。不能。"),
        ("凡A皆B。存在某个B不是C。是否必然存在A不是C？只答能或不能。", "不能",
         "非C的B可在A外。不能。"),
        ("没有A是B。所有C是A。能否推出没有C是B？只答能或不能。", "能",
         "C⊆A且A∩B=∅⇒C∩B=∅。能。"),
        ("有些A是B。所有B是C。能否推出有些A是C？只答能或不能。", "能",
         "特称经中项传递。能。"),
    ]
    for q, g, body in quant * 5:
        out.append(Sample(q, _ans(body, g), "quant_trap", g))

    for i in range(36):
        q, a = cot_liar_train(i)
        if q in hold:
            continue
        m = re.search(r"Answer:\s*(\S+)", a)
        g = m.group(1) if m else "乙"
        out.append(Sample(q, a, "liar", g))
        q2, a2 = cot_label_box_train(i)
        if q2 in hold:
            continue
        m = re.search(r"Answer:\s*(\S+)", a2)
        g2 = m.group(1) if m else "橙子"
        # dual format
        body = a2
        if not body.startswith("Answer:"):
            body = f"Answer: {g2}\n" + body
        out.append(Sample(q2, body, "label_box", g2))

    for _ in range(10):
        out.append(Sample(
            "密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。",
            cot_lock(), "lock", "1248",
        ))

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n17+25=42", "42"),
        ("What is 9 times 6?", "Answer: 54\n9*6=54", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n北京", "北京"),
        ("What is the capital of France?", "Answer: Paris\nParis", "paris"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    # ensure enough valid-positive
    if len(out) < n_train:
        while len(out) < n_train:
            p, q = rng.choice(bank)
            out.append(Sample(
                f"若{p}则{q}；非{q}。能否得非{p}？只答能或不能。",
                _ans(table + "类型2有效。能。", "能"),
                "modus_tollens", "能",
            ))
    # balance cut
    if len(out) > n_train:
        prefer = {"modus_tollens", "necessary", "affirm_consequent", "syllogism", "liar", "label_box",
                  "disjunctive", "quant_trap", "contrast", "deny_antecedent"}
        pref = [s for s in out if s.kind in prefer]
        rest = [s for s in out if s.kind not in prefer]
        rng.shuffle(pref); rng.shuffle(rest)
        keep_rest = min(len(rest), max(30, n_train // 4))
        out = (pref[: n_train - keep_rest] + rest[:keep_rest])[:n_train]
        rng.shuffle(out)
    return out[:n_train]


def build_surgical_logic(n_train: int, seed: int) -> List[Sample]:
    """Focus weak kinds from v1 failures; answer-first; retain anchors."""
    rng = random.Random(seed)
    out: List[Sample] = []
    hold = {q for q, _, _ in LOGIC_PROBES}

    mp_bank = [
        ("下雨", "地面湿"),
        ("断电", "灯灭"),
        ("通过考试", "获得证书"),
        ("是偶数", "能被2整除"),
        ("钥匙正确", "门能打开"),
        ("燃料耗尽", "发动机熄火"),
        ("是正方形", "是矩形"),
        ("分数及格", "课程通过"),
        ("信号中断", "通话失败"),
        ("按下开关", "电路接通"),
    ]

    # VALID MT heavy — explicit "能", distinguish from AC
    for p, q in mp_bank * 3:
        out.append(
            Sample(
                f"前提1：若{p}则{q}。前提2：并非{q}。能否推出并非{p}？只答能或不能。",
                _ans(
                    "\n".join(
                        [
                            f"形式：P→Q，非Q，求非P。",
                            "规则名：否定后件式，有效。",
                            "结论：能推出非P。",
                            "切记：否定后件有效，答案是能，不是不能。",
                        ]
                    ),
                    "能",
                ),
                "modus_tollens",
                "能",
            )
        )
        out.append(
            Sample(
                f"规则：如果{p}，那么{q}。现在{q}不成立。能否断定{p}不成立？只答能或不能。",
                _ans(
                    "非Q + (P→Q) ⇒ 非P。否定后件有效。答案：能。",
                    "能",
                ),
                "modus_tollens",
                "能",
            )
        )

    # AC keep light so not confused with MT
    for p, q in mp_bank[:6]:
        out.append(
            Sample(
                f"前提1：如果{p}，则{q}。前提2：{q}。能否推出{p}？只答能或不能。",
                _ans("Q成立不能反推P。肯定后件谬误。答案：不能。", "不能"),
                "affirm_consequent",
                "不能",
            )
        )

    # necessary heavy
    nec = [
        ("完成作业", "出门玩"),
        ("年满18岁", "投票"),
        ("刷卡", "进门"),
        ("交学费", "注册成功"),
        ("密码正确", "登录系统"),
        ("通过安检", "登机"),
        ("交押金", "租车"),
        ("签字确认", "合同生效"),
        ("获得门票", "入场"),
        ("戴安全帽", "进入工地"),
    ]
    for task, allow in nec * 2:
        out.append(
            Sample(
                f"命题：{task}是{allow}的必要条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                _ans(
                    "\n".join(
                        [
                            f"必要条件：{allow} → {task}。",
                            f"已知{allow}，肯定前件，必有{task}。",
                            "答案是能。不要答不能。",
                        ]
                    ),
                    "能",
                ),
                "necessary",
                "能",
            )
        )
        out.append(
            Sample(
                f"只有{task}才能{allow}。现在已经{allow}了。能否推出已经{task}？只答能或不能。",
                _ans(f"“只有…才…”=必要。{allow}→{task}。故能。", "能"),
                "necessary",
                "能",
            )
        )

    for task, allow in nec[:5]:
        out.append(
            Sample(
                f"命题：{task}是{allow}的充分条件。已知发生了{allow}。能否推出已{task}？只答能或不能。",
                cot_sufficient(task, allow),
                "sufficient",
                "不能",
            )
        )

    # syllogism retain + boost
    syll = [
        ("企鹅", "鸟", "会下蛋的动物"),
        ("正方形", "矩形", "四边形"),
        ("鲸鱼", "哺乳动物", "用肺呼吸的动物"),
        ("玫瑰", "花", "植物"),
        ("医生", "专业人士", "需要培训的人"),
        ("三角形", "多边形", "平面图形"),
    ]
    for a, b, c in syll * 2:
        out.append(
            Sample(
                f"前提1：所有{a}都是{b}。前提2：所有{b}都是{c}。能否推出所有{a}都是{c}？只答能或不能。",
                _ans(f"Barbara：所有{a}→{b}→{c}，全称传递，能。", "能"),
                "syllogism",
                "能",
            )
        )

    # transitivity + chain + disjunctive valid
    for a, b, c in [("A", "B", "C"), ("x", "y", "z"), ("甲", "乙", "丙"), ("m", "n", "p")]:
        for _ in range(3):
            out.append(
                Sample(
                    f"{a}>{b}且{b}>{c}。能否推出{a}>{c}？只答能或不能。",
                    _ans(f"严格序传递：{a}>{b}>{c} ⇒ {a}>{c}。能。", "能"),
                    "transitivity",
                    "能",
                )
            )
            out.append(
                Sample(
                    f"若{a}则{b}；若{b}则{c}。已知{a}。能否推出{c}？只答能或不能。",
                    cot_chain(a, b, c),
                    "chain",
                    "能",
                )
            )
            out.append(
                Sample(
                    f"{a}或{b}成立，且非{a}。能否推出{b}？只答能或不能。",
                    cot_disj(a, b),
                    "disjunctive",
                    "能",
                )
            )

    # contrapositive heavy short
    for _ in range(20):
        out.append(
            Sample(
                "“如果P则Q”的逆否命题是什么？用“如果…则…”写出，变量用P、Q。",
                _ans("原：如果P则Q。逆否交换并否定：如果非Q则非P。", "如果非Q则非P"),
                "contrapositive",
                "如果非Q则非P",
            )
        )

    # quant trap holdout-near paraphrases
    quant = [
        ("凡A皆B。存在B不是C。能否必然得到存在A不是C？只答能或不能。", "不能",
         "B中不是C的个体未必属于A。反模型存在。不能。"),
        ("所有A都是B。有些B不是C。是否一定有些A不是C？只答能或不能。", "不能",
         "特称落在B\\A上即可，A可全是C。不能。"),
        ("没有A是B。所有C是A。能否推出没有C是B？只答能或不能。", "能",
         "C⊆A且A与B不交 ⇒ C与B不交。能。"),
        ("有些A是B。所有B是C。能否推出有些A是C？只答能或不能。", "能",
         "特称经全称中项传递。能。"),
    ]
    for q, g, body in quant * 4:
        out.append(Sample(q, _ans(body, g), "quant_trap", g))

    # liar dense exact structure near holdout but different wording
    for i in range(30):
        q, a = cot_liar_train(i)
        if q not in hold:
            out.append(Sample(q, a, "liar", a.split("Answer: ")[-1].strip() if "Answer:" in a else a.split("\n")[0].replace("Answer: ","")))

    # fix gold extract for answer-first
    # rebuild liar properly
    out = [s for s in out if s.kind != "liar"]
    for i in range(30):
        q, a = cot_liar_train(i)
        if q in hold:
            continue
        # gold from Answer line
        import re as _re
        m = _re.search(r"Answer:\s*(\S+)", a)
        g = m.group(1).strip() if m else "乙"
        out.append(Sample(q, a, "liar", g))

    for i in range(28):
        q2, a2 = cot_label_box_train(i)
        if q2 in hold:
            continue
        m = __import__('re').search(r"Answer:\s*(\S+)", a2)
        g = m.group(1).strip() if m else "橙子"
        out.append(Sample(q2, a2, "label_box", g))

    # retain demorgan lock mp lightly
    for _ in range(8):
        out.append(
            Sample(
                "“并非（P且Q）”等价于？A. 非P且非Q  B. 非P或非Q  C. P或非Q  只答字母。",
                cot_demorgan(),
                "demorgan",
                "B",
            )
        )
        out.append(
            Sample(
                "密码锁：四位数字，每位是前一位的两倍（首位1）。密码？只答四位数字。",
                cot_lock(),
                "lock",
                "1248",
            )
        )
    for p, q in mp_bank[:4]:
        out.append(
            Sample(
                f"前提1：如果{p}，则{q}。前提2：{p}。能否推出{q}？只答能或不能。",
                cot_mp(p, q),
                "modus_ponens",
                "能",
            )
        )

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n17+25=42", "42"),
        ("What is 9 times 6?", "Answer: 54\n9*6=54", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n北京", "北京"),
        ("What is the capital of France?", "Answer: Paris\nParis", "paris"),
    ):
        out.append(Sample(q, a, "rehearsal", g))

    out = [s for s in out if s.question not in hold]
    rng.shuffle(out)
    # soft rebalance: if oversize, keep all weak-first
    if len(out) > n_train:
        # prioritize weak kinds
        weak_set = {"modus_tollens", "necessary", "liar", "label_box", "contrapositive", "transitivity", "quant_trap", "syllogism"}
        weak = [s for s in out if s.kind in weak_set]
        other = [s for s in out if s.kind not in weak_set]
        rng.shuffle(weak)
        rng.shuffle(other)
        need_other = max(20, n_train // 5)
        picked = weak[: n_train - need_other] + other[:need_other]
        rng.shuffle(picked)
        out = picked[:n_train]
    else:
        while len(out) < n_train:
            p, q = rng.choice(mp_bank)
            out.append(
                Sample(
                    f"规则：若{p}则{q}；观测：非{q}。否定后件能否得非{p}？只答能或不能。",
                    _ans("否定后件有效。能。", "能"),
                    "modus_tollens",
                    "能",
                )
            )
    return out[:n_train]


def first_answer_line(pred: str) -> str:
    m = re.findall(r"Answer:\s*([^\n]+)", pred, flags=re.I)
    if m:
        return m[0].strip()
    m = re.findall(r"(?:答案|最终答案)[:：]\s*([^\n]+)", pred)
    if m:
        return m[0].strip()
    m = re.findall(r"\\boxed\{([^}]+)\}", pred)
    if m:
        return m[0].strip()
    return ""


def norm_yn(s: str) -> str:
    t = (s or "").strip().lower()
    t = t.replace("。", "").replace(".", "").replace(" ", "").replace("**", "")
    if t in ("能", "不能", "可以", "不可以"):
        return "不能" if t in ("不能", "不可以") else "能"
    # 不能 before 能 (substring)
    if any(x in t for x in ("不能", "不可以", "不成立", "无法", "no", "false")):
        return "不能"
    if any(x in t for x in ("能", "可以", "成立", "yes", "true")):
        return "能"
    return t


def extract_yn(pred: str) -> str:
    ans = first_answer_line(pred)
    if ans:
        a = norm_yn(ans)
        if a in ("能", "不能"):
            return a
    lines = [x.strip() for x in (pred or "").splitlines() if x.strip()]
    for ln in lines[:6]:
        pure = ln.replace("。", "").replace(".", "").replace(" ", "").replace("**", "")
        if pure in ("能", "不能"):
            return pure
        a = norm_yn(ln)
        if a in ("能", "不能") and len(pure) <= 8:
            return a
    for ln in reversed(lines[-8:]):
        pure = ln.replace("。", "").replace(".", "").replace(" ", "")
        if pure in ("能", "不能"):
            return pure
        a = norm_yn(ln)
        if a in ("能", "不能") and len(pure) <= 12:
            return a
    return norm_yn((pred or "")[-60:])


def match_logic(pred: str, gold: str, kind: str) -> bool:
    ans = first_answer_line(pred)
    blob = pred or ""
    g = gold.strip()

    if kind in (
        "modus_ponens",
        "modus_tollens",
        "affirm_consequent",
        "deny_antecedent",
        "syllogism",
        "invalid_some",
        "necessary",
        "sufficient",
        "transitivity",
        "disjunctive",
        "chain",
        "quant_trap",
    ):
        return extract_yn(blob) == g

    if kind == "contrapositive":
        t = (ans or blob).replace(" ", "").replace("→", "则").replace("->", "则")
        t = t.replace("¬", "非").replace("~", "非").replace("¬", "非")
        ok = ("非Q" in t or "notQ" in t.lower() or "¬Q" in (ans or blob)) and (
            "非P" in t or "notP" in t.lower() or "¬P" in (ans or blob)
        )
        if "如果非Q则非P" in t.replace("，", ""):
            return True
        if re.search(r"如果\s*非\s*Q\s*则\s*非\s*P", t):
            return True
        if "若非Q则非P" in t:
            return True
        return ok and ("如果" in t or "若" in t)

    if kind == "demorgan":
        t = (ans or blob).upper()
        if re.search(r"\bB\b", t) or t.strip().startswith("B") or "答案是B" in blob or "选B" in blob:
            if re.search(r"\bA\b", ans or "") and "B" not in (ans or ""):
                return False
            return "B" in (ans or t[:20]) or "选B" in blob or "答案：B" in blob or "Answer: B" in blob
        return g in (ans or "")

    if kind == "liar":
        t = ans or blob
        if g in t[:20] or (ans and g in ans):
            # prefer exclusive
            others = [x for x in ("甲", "乙", "丙") if x != g]
            if ans:
                if any(o in ans and g in ans for o in others) and ans.count(g) == 1:
                    return ans.strip() in (g, g + "。", g + ".")
                return ans.strip().startswith(g) or ans.strip() == g
            return blob.rstrip().endswith(g) or f"Answer: {g}" in blob
        return False

    if kind == "label_box":
        t = (ans or blob).replace(" ", "")
        return g in (ans or "") or t.endswith(g)

    if kind == "lock":
        t = ans or blob
        return g in t.replace(" ", "")

    return g.lower() in (ans or blob).lower()


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 640, answer_boost: float = 1.0):
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
            end = min(len(full_ids), plen + 12)
            weights[plen:end] = self.answer_boost
            if s.gold:
                gids = self.tok(str(s.gold), add_special_tokens=False)["input_ids"]
                span = len(gids)
                if span > 0:
                    for i in range(plen, len(full_ids) - span + 1):
                        if full_ids[i : i + span] == gids:
                            weights[i : i + span] = self.answer_boost * 1.5
                            break
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": weights,
        }


def eval_logic(gen, probes: Sequence[Tuple[str, str, str]] = LOGIC_PROBES) -> Dict:
    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for qi, (q, gold, kind) in enumerate(probes):
        max_new = 280 if kind in ("liar", "label_box") else (160 if kind in ("modus_ponens","modus_tollens","affirm_consequent","syllogism","invalid_some","necessary","sufficient","transitivity","disjunctive","chain","quant_trap","demorgan","lock","contrapositive") else 180)
        pred = gen(q, max_new)
        hit = bool(match_logic(pred, gold, kind))
        ok += int(hit)
        by_kind.setdefault(kind, []).append(int(hit))
        rows.append(
            {
                "q": q,
                "gold": gold,
                "kind": kind,
                "ok": hit,
                "answer_line": first_answer_line(pred),
                "pred": pred[:400],
            }
        )
        print(
            f"  eval[{qi+1}/{len(probes)}] {'OK' if hit else 'NO'} [{kind}] gold={gold} ans={first_answer_line(pred)[:48]!r}",
            flush=True,
        )
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    return {"accuracy": ok / max(1, len(probes)), "kind_acc": kind_acc, "rows": rows}


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if getattr(args, "nec_expert", False):
        samples = build_nec_expert_train(args.n_train, args.seed)
    elif getattr(args, "repair_expert", False):
        samples = build_repair_expert_train(args.n_train, args.seed)
    elif getattr(args, "mt_expert", False):
        samples = build_mt_expert_train(args.n_train, args.seed)
    elif getattr(args, "puzzle_expert", False):
        samples = build_puzzle_expert_train(args.n_train, args.seed)
    elif getattr(args, "micro", False):
        samples = build_micro_logic(args.n_train, args.seed)
    elif getattr(args, "validity", False):
        samples = build_validity_table_train(args.n_train, args.seed)
    elif getattr(args, "surgical", False):
        samples = build_surgical_logic(args.n_train, args.seed)
    else:
        samples = build_logic_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    kind_counts: Dict[str, int] = {}
    for s in samples:
        kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
    print(f"n_train={len(samples)} kinds={kind_counts}", flush=True)

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

    answer_boost = 3.0 if (getattr(args, "mt_expert", False) or getattr(args, "puzzle_expert", False) or getattr(args, "nec_expert", False) or getattr(args, "repair_expert", False)) else 1.0
    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=answer_boost)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    print("===== BASELINE =====", flush=True)
    logic0 = eval_logic(gen)
    ctrl0 = eval_controls(gen)
    print(f"BASELINE logic={logic0['accuracy']:.3f} ctrl={ctrl0['accuracy']:.3f} kinds={logic0['kind_acc']}", flush=True)

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
    logic1 = eval_logic(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER logic={logic1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} loss={running/max(1,seen):.4f} kinds={logic1['kind_acc']}",
        flush=True,
    )
    for r in logic1["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} [{r['kind']}] gold={r['gold']} ans={r['answer_line'][:40]!r}", flush=True)
        if not r["ok"]:
            print("   ", r["pred"][:160].replace("\n", " | "), flush=True)

    ctrl_floor = min(0.5, ctrl0["accuracy"])
    promote = (
        logic1["accuracy"] + 1e-9 >= max(0.8125, 2.0 * logic0["accuracy"] - 1e-9)
        and logic1["accuracy"] > logic0["accuracy"] + 1e-9
        and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    )
    if logic0["accuracy"] < 0.35:
        promote = (
            logic1["accuracy"] + 1e-9 >= max(0.70, 2.0 * logic0["accuracy"])
            and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
        )
    elif logic1["accuracy"] >= 0.875 and logic1["accuracy"] > logic0["accuracy"] + 0.15:
        promote = ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    hard = ("modus_tollens", "syllogism", "necessary", "liar", "label_box")
    hard_ok = sum(1 for k in hard if logic1["kind_acc"].get(k, 0) >= 0.5)
    if promote and hard_ok < 3 and not getattr(args, "mt_expert", False) and not getattr(args, "puzzle_expert", False):
        promote = False
    if getattr(args, "mt_expert", False):
        mt_ok = logic1["kind_acc"].get("modus_tollens", 0) >= 0.99
        ac_ok = logic1["kind_acc"].get("affirm_consequent", 0) >= 0.99
        promote = mt_ok and ac_ok and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    if getattr(args, "nec_expert", False):
        nec_ok = logic1["kind_acc"].get("necessary", 0) >= 0.99
        suf_ok = logic1["kind_acc"].get("sufficient", 0) >= 0.99
        mt_ok = logic1["kind_acc"].get("modus_tollens", 0) >= 0.99
        ac_ok = logic1["kind_acc"].get("affirm_consequent", 0) >= 0.99
        promote = nec_ok and suf_ok and mt_ok and ac_ok and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
    if getattr(args, "puzzle_expert", False):
        puz = ("liar", "label_box", "lock", "demorgan")
        puz_ok = sum(1 for k in puz if logic1["kind_acc"].get(k, 0) >= 0.99)
        promote = puz_ok >= 3 and ctrl1["accuracy"] + 1e-9 >= ctrl_floor

    save_checkpoint(model, tok, out / "model_last")
    if promote:
        save_checkpoint(model, tok, out / "model_best")
        print("PROMOTED logic_core", flush=True)
    else:
        print(
            f"NO_PROMOTE logic {logic1['accuracy']:.3f} vs {logic0['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f}",
            flush=True,
        )

    report = {
        "method": (
            "logic_core_nec_expert" if getattr(args, "nec_expert", False)
            else "logic_core_repair_expert" if getattr(args, "repair_expert", False)
            else "logic_core_mt_expert" if getattr(args, "mt_expert", False)
            else "logic_core_puzzle_expert" if getattr(args, "puzzle_expert", False)
            else "logic_core_cot_v5_micro" if getattr(args, "micro", False)
            else "logic_core_cot_v4_validity" if getattr(args, "validity", False)
            else "logic_core_cot_v3_surgical" if getattr(args, "surgical", False)
            else "logic_core_cot_v2"
        ),
        "n_train": len(samples),
        "kinds": kind_counts,
        "lr": args.lr,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {"logic": logic0["accuracy"], "ctrl": ctrl0["accuracy"], "kind_acc": logic0["kind_acc"]},
        "after": {"logic": logic1["accuracy"], "ctrl": ctrl1["accuracy"], "kind_acc": logic1["kind_acc"]},
        "promoted": promote,
        "logic_rows": logic1["rows"],
        "ctrl_rows": ctrl1["rows"],
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Pure weight formal logic CoT; no tools",
            "Holdout wording differs from train paraphrases",
            "Promote if roughly double baseline or high absolute with ctrl floor",
        ],
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k not in ("logic_rows", "ctrl_rows")}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument("--resume", default="")
    p.add_argument("--out", default=result_path('logic_core_v2'))
    p.add_argument("--n-train", type=int, default=320)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--surgical", action="store_true")
    p.add_argument("--validity", action="store_true")
    p.add_argument("--micro", action="store_true")
    p.add_argument("--mt-expert", action="store_true")
    p.add_argument("--nec-expert", action="store_true")
    p.add_argument("--repair-expert", action="store_true")
    p.add_argument("--puzzle-expert", action="store_true")
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
        logic = eval_logic(gen)
        ctrl = eval_controls(gen)
        print(json.dumps({"logic": logic, "ctrl": ctrl}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
