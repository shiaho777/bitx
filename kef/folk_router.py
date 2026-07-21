"""Folk multi-adapter pure-weight router: core + decimal + riddle + yi experts."""

from __future__ import annotations

from kef.paths import default_model, result_path

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.folk_logic import CTRL_PROBES, FOLK_PROBES, eval_controls, eval_folk, make_gen


DEC_COMPARE_HINTS = (
    "哪个更大",
    "哪个大",
    "谁更大",
    "谁大",
    "比较",
    "更大",
    "更小",
    "大小",
    "vs",
    "VS",
    "小数",
    "十进制",
    "补零",
    "版本号",
)

LIMIT_HINTS = (
    "0.999",
    "0.999...",
    "无限循环",
    "等于1",
    "是否等于",
)

RIDDLE_HINTS = (
    "大象",
    "冰箱",
    "elephant",
    "fridge",
    "脑筋急转弯",
)


def extract_decimals(text: str) -> List[str]:
    return re.findall(r"-?\d+\.\d+", text or "")


def is_limit_query(text: str) -> bool:
    t = text or ""
    return any(h in t for h in LIMIT_HINTS)


def is_decimal_folk_query(text: str) -> bool:
    t = text or ""
    if is_limit_query(t):
        return False
    if is_riddle_folk_query(t):
        return False
    if is_yi_to_shi_query(t):
        return False
    decs = extract_decimals(t)
    if len(decs) < 2:
        if len(decs) == 1 and any(h in t for h in DEC_COMPARE_HINTS):
            return True
        return False
    if any(h in t for h in DEC_COMPARE_HINTS):
        return True
    if re.search(r"\d+\.\d+.+(和|与|或|还是).+\d+\.\d+", t) and ("大" in t or "小" in t or "比较" in t):
        return True
    return False


def is_riddle_folk_query(text: str) -> bool:
    t = (text or "").lower()
    if "大象" in (text or "") and ("冰箱" in (text or "") or "fridge" in t):
        return True
    if "elephant" in t and ("fridge" in t or "refrigerator" in t):
        return True
    if "脑筋急转弯" in (text or "") and ("大象" in (text or "") or "冰箱" in (text or "") or "几步" in (text or "")):
        return True
    return False


def is_yi_to_shi_query(text: str) -> bool:
    t = text or ""
    compact = (
        t.replace("、", "")
        .replace("，", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("．", "")
        .replace(".", "")
    )
    if "一到十" in t or "一至十" in t:
        return True
    if "一二三四五六七八九十" in compact:
        return True
    if ("结尾" in t or "末字" in t or "句末" in t or "收尾" in t) and ("一" in t and "十" in t) and (
        "句" in t or "句子" in t or "通顺" in t
    ):
        return True
    if "分别以" in t and "一" in t and "十" in t:
        return True
    if ("写10" in t or "写十" in t or "10个" in t or "十个" in t) and ("结尾" in t or "末字" in t) and "一" in t:
        return True
    return False


def route_name(prompt: str) -> str:
    if is_decimal_folk_query(prompt):
        return "decimal_expert"
    if is_riddle_folk_query(prompt):
        return "riddle_expert"
    if is_yi_to_shi_query(prompt):
        return "yi_expert"
    return "core"


class FolkMultiRouter:
    def __init__(
        self,
        core_gen: Callable[[str, int], str],
        decimal_gen: Callable[[str, int], str],
        riddle_gen: Optional[Callable[[str, int], str]] = None,
        yi_gen: Optional[Callable[[str, int], str]] = None,
    ):
        self.core_gen = core_gen
        self.decimal_gen = decimal_gen
        self.riddle_gen = riddle_gen or core_gen
        self.yi_gen = yi_gen or core_gen

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        name = route_name(prompt)
        if name == "decimal_expert":
            return self.decimal_gen(prompt, max_new_tokens)
        if name == "riddle_expert":
            return self.riddle_gen(prompt, max_new_tokens)
        if name == "yi_expert":
            return self.yi_gen(prompt, max_new_tokens)
        return self.core_gen(prompt, max_new_tokens)


def is_hard_char_query(text: str) -> bool:
    return is_decimal_folk_query(text)


class FolkDualRouter(FolkMultiRouter):
    def __init__(self, core_gen, expert_gen):
        super().__init__(core_gen, expert_gen, None, None)


def load_gen(model_path: str, adapter: str, device: str):
    path = resolve_checkpoint(model_path, adapter)
    model, tok = load_model_and_tokenizer(path, device=device, trainable=False)
    return make_gen(model, tok, device), model, tok



def eval_route_decisions(probes: Tuple = FOLK_PROBES) -> List[Dict]:
    rows = []
    for q, gold, kind in probes:
        rows.append({"q": q, "gold": gold, "kind": kind, "route": route_name(q)})
    return rows


def _free_mps():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def adapter_for_route(args, name: str) -> str:
    if name == "decimal_expert":
        return args.expert
    if name == "riddle_expert":
        return args.riddle_expert or args.core
    if name == "yi_expert":
        return getattr(args, "yi_expert", "") or args.core
    return args.core


def eval_with_adapter(args, adapter: str, probes=None, controls: bool = False) -> Dict:
    gen, model, tok = load_gen(args.model, adapter, args.device)
    try:
        if controls:
            return eval_controls(gen)
        return eval_folk(gen, probes or FOLK_PROBES)
    finally:
        del model
        del gen
        del tok
        _free_mps()


def run_route_eval(args) -> Dict:
    t0 = time.perf_counter()
    decisions = eval_route_decisions()
    print("ROUTE DECISIONS", flush=True)
    for d in decisions:
        print(f"  {d['route']:14} [{d['kind']}] {d['q'][:48]}", flush=True)

    print("===== eval core (single load) =====", flush=True)
    print("load core", args.core, flush=True)
    folk_core = eval_with_adapter(args, args.core)
    ctrl_core = eval_with_adapter(args, args.core, controls=True)

    # sequential routed: group probes by adapter to minimize reloads
    print("===== eval routed (sequential adapters) =====", flush=True)
    by_adapter: Dict[str, List[Tuple[str, str, str]]] = {}
    order = []
    for q, gold, kind in FOLK_PROBES:
        ad = adapter_for_route(args, route_name(q))
        if ad not in by_adapter:
            by_adapter[ad] = []
            order.append(ad)
        by_adapter[ad].append((q, gold, kind))

    pred_map: Dict[str, Dict] = {}
    for ad in order:
        probes = by_adapter[ad]
        print(f"load adapter for {len(probes)} probes: {ad}", flush=True)
        gen, model, tok = load_gen(args.model, ad, args.device)
        try:
            for qi, (q, gold, kind) in enumerate(probes):
                max_new = 360 if kind == "yi_to_shi" else 160
                pred = gen(q, max_new)
                from kef.folk_logic import match_gold, first_answer_line
                hit = bool(match_gold(pred, gold, kind))
                pred_map[q] = {
                    "q": q,
                    "gold": gold,
                    "kind": kind,
                    "ok": hit,
                    "answer_line": first_answer_line(pred),
                    "pred": pred[:400],
                    "route": route_name(q),
                }
                print(
                    f"  routed[{qi+1}/{len(probes)}] {'OK' if hit else 'NO'} via={route_name(q)} [{kind}] gold={gold}",
                    flush=True,
                )
        finally:
            del model
            del gen
            del tok
            _free_mps()

    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for q, gold, kind in FOLK_PROBES:
        r = pred_map[q]
        ok += int(r["ok"])
        by_kind.setdefault(kind, []).append(int(r["ok"]))
        rows.append(r)
    folk_r = {
        "accuracy": ok / max(1, len(FOLK_PROBES)),
        "kind_acc": {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()},
        "rows": rows,
    }

    # ctrl via core adapter (retain floor)
    print("===== eval routed ctrl via core =====", flush=True)
    ctrl_r = eval_with_adapter(args, args.core, controls=True)

    core_keep_kinds = ("distance", "sequence", "commonsense", "limit")
    core_keep = [r for r in folk_core["rows"] if r["kind"] in core_keep_kinds]
    route_keep = [r for r in folk_r["rows"] if r["kind"] in core_keep_kinds]
    core_kd = sum(1 for r in core_keep if r["ok"]) / max(1, len(core_keep))
    route_kd = sum(1 for r in route_keep if r["ok"]) / max(1, len(route_keep))

    promote = (
        folk_r["accuracy"] > folk_core["accuracy"] + 1e-9
        and ctrl_r["accuracy"] + 1e-9 >= min(0.75, ctrl_core["accuracy"])
        and route_kd + 1e-9 >= core_kd - 1e-9
    )

    report = {
        "method": "folk_multi_adapter_route",
        "model": args.model,
        "core": args.core,
        "expert": args.expert,
        "riddle_expert": args.riddle_expert or "",
        "yi_expert": getattr(args, "yi_expert", "") or "",
        "core_folk": folk_core["accuracy"],
        "routed_folk": folk_r["accuracy"],
        "core_ctrl": ctrl_core["accuracy"],
        "routed_ctrl": ctrl_r["accuracy"],
        "core_kind_acc": folk_core["kind_acc"],
        "routed_kind_acc": folk_r["kind_acc"],
        "core_keep_acc": core_kd,
        "routed_keep_acc": route_kd,
        "promoted": promote,
        "decisions": decisions,
        "rows": rows,
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Pure weight multi adapter; no tools",
            "Sequential adapter load to avoid MPS OOM",
            "Decimal compare -> decimal expert",
            "Elephant fridge riddle -> riddle expert if set else core",
            "yi_to_shi ending sentences -> yi expert if set else core",
            "Else core; limit stays core",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k not in ("rows", "decisions")}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    if promote:
        print("PROMOTED folk multi route", flush=True)
    else:
        print(
            f"NO_PROMOTE routed {folk_r['accuracy']:.3f} vs core {folk_core['accuracy']:.3f}",
            flush=True,
        )
    print("ROUTE_EVAL_OK", flush=True)
    return report



def install_champion_bundle(args, report: Optional[Dict] = None) -> None:
    root = Path(args.bundle)
    root.mkdir(parents=True, exist_ok=True)
    for name, src in (
        ("core_adapter", args.core),
        ("expert_adapter", args.expert),
    ):
        if not src:
            continue
        sp = Path(src)
        if not sp.exists():
            continue
        dp = root / name
        if dp.resolve() != sp.resolve():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
    if args.riddle_expert and Path(args.riddle_expert).exists():
        dp = root / "riddle_adapter"
        sp = Path(args.riddle_expert)
        if dp.resolve() != sp.resolve():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
    if getattr(args, "yi_expert", "") and Path(args.yi_expert).exists():
        dp = root / "yi_adapter"
        sp = Path(args.yi_expert)
        if dp.resolve() != sp.resolve():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
    meta = {
        "name": "folk_logic_multi_route",
        "core": str(root / "core_adapter"),
        "expert": str(root / "expert_adapter"),
        "riddle_expert": str(root / "riddle_adapter") if args.riddle_expert else "",
        "yi_expert": str(root / "yi_adapter") if getattr(args, "yi_expert", "") else "",
        "base_model": args.model,
        "router": "kef/folk_router.py",
        "report": args.out,
        "metrics": {
            "routed_folk": None if not report else report.get("routed_folk"),
            "core_folk": None if not report else report.get("core_folk"),
        },
    }
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BUNDLE", root, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument(
        "--core",
        default=result_path('folk_logic_dual', 'core_adapter'),
    )
    p.add_argument(
        "--expert",
        default=result_path('folk_logic_dual', 'expert_adapter'),
    )
    p.add_argument(
        "--riddle-expert",
        default="",
        help="optional riddle specialist adapter",
    )
    p.add_argument(
        "--yi-expert",
        default="",
        help="optional yi_to_shi specialist adapter",
    )
    p.add_argument(
        "--out",
        default=result_path('folk_logic_dual', 'route_eval_multi.json'),
    )
    p.add_argument(
        "--bundle",
        default=result_path('folk_logic_dual'),
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--install-bundle", action="store_true")
    args = p.parse_args()
    if args.install_bundle:
        install_champion_bundle(args)
        return
    report = run_route_eval(args)
    if report.get("promoted"):
        install_champion_bundle(args, report)


if __name__ == "__main__":
    main()
