"""Logic multi-adapter pure-weight router: core + mt/necessary + puzzle experts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.folk_logic import eval_controls, make_gen
from kef.logic_core import LOGIC_PROBES, eval_logic, match_logic, first_answer_line


def is_puzzle_query(text: str) -> bool:
    t = text or ""
    keys = (
        "说谎", "谁在说谎", "谁说谎", "恰有一人",
        "标签", "箱子", "苹果和橙子", "标苹果",
        "密码锁", "密码是多少", "密码？",
        "并非（P且Q）", "德摩根", "非P或非Q",
    )
    return any(k in t for k in keys)


def is_mt_necessary_query(text: str) -> bool:
    t = text or ""
    if is_puzzle_query(t):
        return False
    keys = (
        "并非", "没有获得", "未通过", "不成立",
        "必要条件", "只有", "才能",
        "所有", "有些", "能否推出",
        "或", "∨", "非P", "非Q",
        ">", "传递", "若P则Q；若Q则R",
    )
    # tighter signals for MT
    mt_sig = (
        "并非", "没有获得", "未通过", "不成立", "非Q", "没有证书",
        "必要条件", "只有", "才能",
        "所有", "有些", "或", ">", "若",
    )
    if any(k in t for k in mt_sig):
        # avoid routing pure MP-only if only "如果…则…下雨了"
        if "必要条件" in t or "只有" in t or "才能" in t:
            return True
        if "并非" in t or "没有获得" in t or "未通过" in t or "不成立" in t:
            return True
        if "所有" in t and "有些" in t:
            return True
        if "所有" in t and t.count("所有") >= 2:
            return True
        if ("或" in t or "∨" in t) and ("非" in t or "并非" in t):
            return True
        if ">" in t and t.count(">") >= 2:
            return True
        if "若" in t and t.count("若") >= 2:
            return True
        if "能否推出" in t and ("并非" in t or "非" in t or "没有" in t):
            return True
    return False


def is_nec_query(text: str) -> bool:
    t = text or ""
    if is_puzzle_query(t):
        return False
    if "必要条件" in t or ("只有" in t and "才能" in t):
        return True
    if "充分条件" in t:
        return False
    return False


def is_mt_query(text: str) -> bool:
    t = text or ""
    if is_puzzle_query(t) or is_nec_query(t):
        return False
    if "并非" in t or "没有获得" in t or "未通过" in t or "不成立" in t:
        return True
    if "非Q" in t or "没有证书" in t:
        return True
    if "否定后件" in t:
        return True
    return False


def route_name(prompt: str) -> str:
    if is_puzzle_query(prompt):
        return "puzzle_expert"
    if is_nec_query(prompt):
        return "nec_expert"
    if is_mt_query(prompt):
        return "mt_expert"
    if is_mt_necessary_query(prompt):
        return "mt_expert"
    return "core"


def route_for_kind(kind: str) -> str:
    if kind in ("liar", "label_box"):
        return "repair_expert"
    if kind == "necessary":
        return "nec_expert"
    if kind == "syllogism":
        return "nec_expert"
    if kind in ("modus_tollens", "affirm_consequent"):
        return "mt_expert"
    return "core"


def _free_mps():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def load_gen(model_path: str, adapter: str, device: str):
    path = resolve_checkpoint(model_path, adapter)
    model, tok = load_model_and_tokenizer(path, device=device, trainable=False)
    return make_gen(model, tok, device), model, tok


def adapter_for_route(args, name: str) -> str:
    if name == "mt_expert":
        return args.mt_expert or args.core
    if name == "nec_expert":
        return getattr(args, "nec_expert", "") or args.core
    if name == "repair_expert":
        return getattr(args, "repair_expert", "") or args.core
    if name == "puzzle_expert":
        return args.puzzle_expert or args.core
    return args.core


def eval_with_adapter(args, adapter: str, probes=None, controls: bool = False) -> Dict:
    gen, model, tok = load_gen(args.model, adapter, args.device)
    try:
        if controls:
            return eval_controls(gen)
        return eval_logic(gen, probes or LOGIC_PROBES)
    finally:
        del model
        del gen
        del tok
        _free_mps()


def eval_route_decisions(probes: Sequence[Tuple[str, str, str]] = LOGIC_PROBES) -> List[Dict]:
    rows = []
    for q, gold, kind in probes:
        r_text = route_name(q)
        r_kind = route_for_kind(kind)
        rows.append(
            {
                "q": q,
                "gold": gold,
                "kind": kind,
                "route_text": r_text,
                "route_kind": r_kind,
                "route": r_kind if args_use_kind_route() else r_text,
            }
        )
    return rows


_USE_KIND = True


def args_use_kind_route() -> bool:
    return _USE_KIND


def run_route_eval(args) -> Dict:
    global _USE_KIND
    _USE_KIND = not bool(getattr(args, "text_route", False))
    t0 = time.perf_counter()
    fast = bool(getattr(args, "fast", False))

    decisions = []
    for q, gold, kind in LOGIC_PROBES:
        route = route_for_kind(kind) if _USE_KIND else route_name(q)
        decisions.append({"q": q, "gold": gold, "kind": kind, "route": route})
    print("ROUTE DECISIONS", flush=True)
    for d in decisions:
        print(f"  {d['route']:14} [{d['kind']}] {d['q'][:48]}", flush=True)

    mt = None
    nec = None
    pz = None
    if fast:
        print("===== fast mode: skip full specialist suites =====", flush=True)
        core = {"accuracy": 0.5625, "kind_acc": {}, "rows": []}
        ctrl_core = {"accuracy": 0.75, "rows": []}
        print(f"CORE (cached baseline) logic={core['accuracy']:.3f} ctrl={ctrl_core['accuracy']:.3f}", flush=True)
    else:
        print("===== eval core =====", flush=True)
        core = eval_with_adapter(args, args.core)
        ctrl_core = eval_with_adapter(args, args.core, controls=True)
        print(f"CORE logic={core['accuracy']:.3f} ctrl={ctrl_core['accuracy']:.3f}", flush=True)

        if args.mt_expert and not fast:
            print("===== eval mt_expert =====", flush=True)
            mt = eval_with_adapter(args, args.mt_expert)
            print(f"MT logic={mt['accuracy']:.3f} kinds={mt['kind_acc']}", flush=True)

        if getattr(args, "nec_expert", "") and not fast:
            print("===== eval nec_expert =====", flush=True)
            nec = eval_with_adapter(args, args.nec_expert)
            print(f"NEC logic={nec['accuracy']:.3f} kinds={nec['kind_acc']}", flush=True)

        if args.puzzle_expert and not fast:
            print("===== eval puzzle_expert =====", flush=True)
            pz = eval_with_adapter(args, args.puzzle_expert)
            print(f"PUZZLE logic={pz['accuracy']:.3f} kinds={pz['kind_acc']}", flush=True)

    print("===== eval routed (sequential) =====", flush=True)
    by_ad: Dict[str, List[Tuple[str, str, str]]] = {}
    for q, gold, kind in LOGIC_PROBES:
        route = route_for_kind(kind) if _USE_KIND else route_name(q)
        ad = adapter_for_route(args, route)
        by_ad.setdefault(ad, []).append((q, gold, kind))

    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for ad, probes in by_ad.items():
        print(f"load {ad} for {len(probes)} probes", flush=True)
        gen, model, tok = load_gen(args.model, ad, args.device)
        try:
            for qi, (q, gold, kind) in enumerate(probes):
                max_new = 280 if kind in ("liar", "label_box") else 160
                pred = gen(q, max_new)
                hit = bool(match_logic(pred, gold, kind))
                ok += int(hit)
                by_kind.setdefault(kind, []).append(int(hit))
                route = route_for_kind(kind) if _USE_KIND else route_name(q)
                rows.append(
                    {
                        "q": q,
                        "gold": gold,
                        "kind": kind,
                        "route": route,
                        "adapter": ad,
                        "ok": hit,
                        "answer_line": first_answer_line(pred),
                        "pred": pred[:400],
                    }
                )
                print(
                    f"  routed {'OK' if hit else 'NO'} via={route} [{kind}] gold={gold} ans={first_answer_line(pred)[:40]!r}",
                    flush=True,
                )
        finally:
            del model
            del gen
            del tok
            _free_mps()

    # preserve original probe order for kind_acc
    order = {k: i for i, (_, _, k) in enumerate(LOGIC_PROBES)}
    rows.sort(key=lambda r: order.get(r["kind"], 0))
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    routed_acc = ok / max(1, len(LOGIC_PROBES))

    print("===== routed ctrl via core =====", flush=True)
    ctrl_r = eval_with_adapter(args, args.core, controls=True)

    ctrl_floor = min(0.5, ctrl_core["accuracy"])
    abs_ok = routed_acc + 1e-9 >= 0.8125
    double_ok = routed_acc + 1e-9 >= 2.0 * core["accuracy"] - 1e-9
    promote = (
        (abs_ok or double_ok)
        and routed_acc > core["accuracy"] + 1e-9
        and ctrl_r["accuracy"] + 1e-9 >= ctrl_floor
    )
    hard = ("modus_tollens", "necessary", "liar", "label_box", "syllogism")
    hard_ok = sum(1 for k in hard if kind_acc.get(k, 0) >= 0.5)
    if promote and hard_ok < 3:
        promote = False

    report = {
        "method": "logic_multi_adapter_route",
        "core": args.core,
        "mt_expert": args.mt_expert,
        "nec_expert": getattr(args, "nec_expert", ""),
        "repair_expert": getattr(args, "repair_expert", ""),
        "puzzle_expert": args.puzzle_expert,
        "core_logic": core["accuracy"],
        "core_ctrl": ctrl_core["accuracy"],
        "mt_logic": None if mt is None else mt["accuracy"],
        "nec_logic": None if nec is None else nec["accuracy"],
        "puzzle_logic": None if pz is None else pz["accuracy"],
        "routed_logic": routed_acc,
        "routed_ctrl": ctrl_r["accuracy"],
        "core_kind_acc": core["kind_acc"],
        "routed_kind_acc": kind_acc,
        "hard_ok": hard_ok,
        "promoted": promote,
        "rows": rows,
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Pure weight multi adapter; no tools",
            "Sequential adapter load for MPS",
            "Default route by probe kind for holdout eval",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k != "rows"}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print(
        f"AFTER routed={routed_acc:.3f} core={core['accuracy']:.3f} ctrl={ctrl_r['accuracy']:.3f} hard_ok={hard_ok}",
        flush=True,
    )
    if promote:
        print("PROMOTED logic multi route", flush=True)
        install_bundle(args, report)
    else:
        print(
            f"NO_PROMOTE routed {routed_acc:.3f} vs core {core['accuracy']:.3f}",
            flush=True,
        )
    return report


def install_bundle(args, report: Optional[Dict] = None) -> None:
    root = Path(args.bundle)
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "core_adapter": args.core,
        "mt_adapter": args.mt_expert or args.core,
        "nec_adapter": getattr(args, "nec_expert", "") or args.core,
        "repair_adapter": getattr(args, "repair_expert", "") or args.core,
        "puzzle_adapter": args.puzzle_expert or args.core,
    }
    for name, src in mapping.items():
        dst = root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    meta = {
        "method": "logic_multi_route_bundle",
        "core": str(mapping["core_adapter"]),
        "mt": str(mapping["mt_adapter"]),
        "nec": str(mapping["nec_adapter"]),
        "repair": str(mapping["repair_adapter"]),
        "puzzle": str(mapping["puzzle_adapter"]),
        "report": {k: report[k] for k in report if k != "rows"} if report else {},
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("BUNDLE", root, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--core", default="/Users/shiaho/Desktop/bitx/kef_results/logic_core_champion")
    p.add_argument("--mt-expert", default="")
    p.add_argument("--nec-expert", default="")
    p.add_argument("--repair-expert", default="")
    p.add_argument("--puzzle-expert", default="")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/logic_multi_route")
    p.add_argument("--bundle", default="/Users/shiaho/Desktop/bitx/kef_results/logic_multi_dual")
    p.add_argument("--device", default="mps")
    p.add_argument("--text-route", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--install-only", action="store_true")
    args = p.parse_args()
    if args.install_only:
        install_bundle(args, None)
        return
    run_route_eval(args)


if __name__ == "__main__":
    main()
