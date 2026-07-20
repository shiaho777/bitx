"""EngCraft multi-adapter pure-weight router: core + FE + BE experts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kef.eng_craft import ENG_HARD_PROBES, ENG_PROBES, eval_eng, score_eng
from kef.folk_logic import eval_controls, make_gen


def _free_mps():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def route_for_kind(kind: str) -> str:
    if kind in ("frontend_complete", "frontend_vue", "frontend_css", "anti_lazy_js", "frontend_hard"):
        return "fe_expert"
    if kind in (
        "backend_complete",
        "backend_security",
        "backend_router",
        "error_handling",
        "backend_hard",
    ):
        return "be_expert"
    if kind in ("algo_complete", "anti_lazy", "refactor_standards", "data_model", "algo_hard"):
        return "core"
    return "core"


def route_name(prompt: str) -> str:
    t = (prompt or "").lower()
    fe_keys = (
        "react", "vue", "css", "tsx", "jsx", "useState", "组件", "前端",
        "debounce", "html", "flex", "grid", "dom",
    )
    be_keys = (
        "express", "flask", "fastapi", "sql", "路由", "api", "后端",
        "http", "jwt", "middleware", "数据库", "parameterized", "async def read",
    )
    if any(k.lower() in t if isinstance(k, str) else k in t for k in fe_keys):
        if any(k in (prompt or "") for k in ("React", "Vue", "CSS", "组件", "debounce", "flex", "grid", "前端", "useState", "tsx", "jsx", "HTML")) or any(
            k in t for k in ("react", "vue", "css", "debounce", "flex", "grid", "usestate", "tsx", "jsx")
        ):
            return "fe_expert"
    if any(k in (prompt or "") for k in ("Express", "Flask", "SQL", "路由", "API", "后端", "JWT", "middleware", "数据库")) or any(
        k in t for k in ("express", "flask", "fastapi", "sql", "api", "jwt", "parameterized", "async def")
    ):
        return "be_expert"
    return "core"


def load_gen(model_path: str, adapter: str, device: str):
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    base.to(device)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    gen = make_gen(model, tok, device)
    return gen, model, tok


def adapter_for_route(args, name: str) -> str:
    if name == "fe_expert":
        return args.fe_expert or args.core
    if name == "be_expert":
        return args.be_expert or args.core
    return args.core


def eval_with_adapter(args, adapter: str, probes=None, controls: bool = False) -> Dict:
    gen, model, tok = load_gen(args.model, adapter, args.device)
    try:
        if controls:
            return eval_controls(gen)
        return eval_eng(gen, probes=probes)
    finally:
        del model
        del gen
        del tok
        _free_mps()


def run_route_eval(args) -> Dict:
    t0 = time.perf_counter()
    probes = list(ENG_PROBES) + (list(ENG_HARD_PROBES) if args.with_hard else [])
    print("===== core suite =====", flush=True)
    core = eval_with_adapter(args, args.core, probes=probes)
    ctrl_core = eval_with_adapter(args, args.core, controls=True)
    print(f"CORE eng={core['accuracy']:.3f} ctrl={ctrl_core['accuracy']:.3f}", flush=True)

    if args.fast:
        print("===== fast mode: skip specialist solo suites =====", flush=True)
        fe = be = None
    else:
        fe = eval_with_adapter(args, args.fe_expert or args.core, probes=probes) if args.fe_expert else None
        be = eval_with_adapter(args, args.be_expert or args.core, probes=probes) if args.be_expert else None

    # routed eval: group by route, load each adapter once
    by_route: Dict[str, List[Tuple[str, str, str]]] = {}
    for q, gold, kind in probes:
        rname = route_for_kind(kind) if not args.text_route else route_name(q)
        by_route.setdefault(rname, []).append((q, gold, kind))

    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for route, items in by_route.items():
        ad = adapter_for_route(args, route)
        print(f"===== route {route} via {ad} n={len(items)} =====", flush=True)
        gen, model, tok = load_gen(args.model, ad, args.device)
        try:
            for q, gold, kind in items:
                max_new = 420 if kind.startswith("frontend") or kind.startswith("backend") else 360
                pred = gen(q, max_new)
                hit, marks = score_eng(pred, kind, gold)
                ok += int(hit)
                by_kind.setdefault(kind, []).append(int(hit))
                rows.append(
                    {
                        "q": q,
                        "gold": gold,
                        "kind": kind,
                        "route": route,
                        "adapter": ad,
                        "ok": hit,
                        "marks": marks,
                        "pred": pred[:1200],
                    }
                )
                print(
                    f"  routed {'OK' if hit else 'NO'} via={route} [{kind}] gold={gold}",
                    flush=True,
                )
                if not hit:
                    print("   ", pred[:160].replace("\n", " | "), flush=True)
        finally:
            del model
            del gen
            del tok
            _free_mps()

    order = {k: i for i, (_, _, k) in enumerate(probes)}
    rows.sort(key=lambda r: order.get(r["kind"], 0))
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    routed_acc = ok / max(1, len(probes))
    ctrl_r = eval_with_adapter(args, args.core, controls=True)

    core_main = eval_with_adapter(args, args.core, probes=list(ENG_PROBES))
    promote = (
        routed_acc + 1e-9 >= max(0.9, core["accuracy"])
        and routed_acc + 1e-9 >= core_main["accuracy"]
        and ctrl_r["accuracy"] + 1e-9 >= min(0.5, ctrl_core["accuracy"])
    )
    report = {
        "method": "eng_multi_adapter_route",
        "core": args.core,
        "fe_expert": args.fe_expert,
        "be_expert": args.be_expert,
        "core_eng": core["accuracy"],
        "core_ctrl": ctrl_core["accuracy"],
        "core_main12": core_main["accuracy"],
        "fe_eng": None if fe is None else fe["accuracy"],
        "be_eng": None if be is None else be["accuracy"],
        "routed_eng": routed_acc,
        "routed_ctrl": ctrl_r["accuracy"],
        "kind_acc": kind_acc,
        "promoted": promote,
        "with_hard": bool(args.with_hard),
        "n_probes": len(probes),
        "wall_time_s": time.perf_counter() - t0,
        "rows": rows,
        "notes": [
            "Pure weight multi adapter; no tools",
            "Sequential adapter load for MPS",
            "Default route by probe kind",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        f"ROUTED eng={routed_acc:.3f} core={core['accuracy']:.3f} main12={core_main['accuracy']:.3f} ctrl={ctrl_r['accuracy']:.3f} promote={promote}",
        flush=True,
    )
    if promote:
        install_bundle(args, report)
        print("PROMOTED eng_multi_route", flush=True)
    else:
        print("NO_PROMOTE eng multi route", flush=True)
    print("TRAIN_OK", flush=True)
    return report


def install_bundle(args, report: Optional[Dict]):
    root = Path(args.bundle)
    if root.exists():
        for p in root.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    root.mkdir(parents=True, exist_ok=True)

    def copy_ad(src: str, name: str):
        s = Path(src)
        if not s.exists():
            return ""
        # accept either adapter dir or parent with adapter_best/last
        if (s / "adapter_model.safetensors").exists():
            src_dir = s
        elif (s / "adapter_best").exists():
            src_dir = s / "adapter_best"
        elif (s / "adapter_last").exists():
            src_dir = s / "adapter_last"
        else:
            src_dir = s
        dst = root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
        return str(dst)

    mapping = {
        "core_adapter": copy_ad(args.core, "core_adapter"),
        "fe_adapter": copy_ad(args.fe_expert or args.core, "fe_adapter"),
        "be_adapter": copy_ad(args.be_expert or args.core, "be_adapter"),
    }
    meta = {
        "method": "eng_multi_route_bundle",
        "core": mapping["core_adapter"],
        "fe": mapping["fe_adapter"],
        "be": mapping["be_adapter"],
        "report": {k: report[k] for k in report if k != "rows"} if report else {},
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("BUNDLE", root, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/shiaho/Desktop/MiniCPM5-1B")
    p.add_argument("--core", default="/Users/shiaho/Desktop/bitx/kef_results/eng_craft_champion/adapter_best")
    p.add_argument("--fe-expert", default="")
    p.add_argument("--be-expert", default="")
    p.add_argument("--out", default="/Users/shiaho/Desktop/bitx/kef_results/eng_multi_route")
    p.add_argument("--bundle", default="/Users/shiaho/Desktop/bitx/kef_results/eng_multi_dual")
    p.add_argument("--device", default="mps")
    p.add_argument("--text-route", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--with-hard", action="store_true")
    p.add_argument("--install-only", action="store_true")
    args = p.parse_args()
    if args.install_only:
        install_bundle(args, None)
        return
    run_route_eval(args)


if __name__ == "__main__":
    main()
