"""KEF command-line interface.

A real, usable entry point for the knowledge-externalization framework. The
FactStore is persisted to disk so knowledge survives between commands; each
command lazily loads only the model(s) it needs (memory-friendly).

Examples:
  python3 -m kef ask    "The capital of France is"
  python3 -m kef edit   "The capital of France is" "Lyon"
  python3 -m kef ask    "The capital of France is"      # -> Lyon (recall)
  python3 -m kef teach  "The CEO of Acme is" "Alice"
  python3 -m kef forget "The capital of France is"
  python3 -m kef list
  python3 -m kef bytes
  python3 -m kef diagnose [--tiny]
"""
import argparse
import os
import sys

from kef.config import Config

DEFAULT_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "kef_results", "store.pt")


def _load_store(path):
    from kef.factstore import FactStore
    if os.path.exists(path):
        return FactStore.load(path)
    return FactStore()


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cmd_ask(args):
    """encode -> gated lookup -> recall (store) or core (frozen LM)."""
    from kef.encoder import RetrievalEncoder
    from kef.core import ReasoningCore
    cfg = Config(profile="tiny" if args.tiny else "full")
    store = _load_store(args.store)
    enc = RetrievalEncoder(config=cfg)
    # try recall first (no big model needed if it hits)
    if len(store) > 0:
        vec = enc.encode(args.prompt)
        if args.subject is not None:
            hit, source = store.lookup(vec,
                                       threshold=cfg.sim_threshold,
                                       subject=args.subject,
                                       min_margin=args.min_margin)
        else:
            hit, source, _ = store.gated_lookup_with_text_policy(
                vec,
                threshold=cfg.sim_threshold,
                query_text=args.prompt,
                min_margin=args.min_margin,
                rerank_on_ambiguous=args.rerank == "lexical",
            )
        if hit is not None:
            print(f"{hit[2]}    [source: {source} | sim={hit[1]:.3f} | id={hit[0]}]")
            return
    # fall back to the frozen core
    core = ReasoningCore(cfg.core_name())
    tok_id = core.answer_token(args.prompt)
    print(f"{core.decode(tok_id).strip()}    [source: core | model={cfg.core_name()}]")


def cmd_teach(args):
    from kef.encoder import RetrievalEncoder
    cfg = Config()
    store = _load_store(args.store)
    enc = RetrievalEncoder(config=cfg)
    meta = {"subject": args.subject} if args.subject is not None else None
    rid = store.add(enc.encode(args.prompt), value=args.value, key_text=args.prompt, meta=meta)
    _ensure_parent(args.store); store.save(args.store)
    print(f"taught: {args.prompt!r} -> {args.value!r} (id={rid}, total={len(store)})")


def cmd_edit(args):
    from kef.encoder import RetrievalEncoder
    cfg = Config()
    store = _load_store(args.store)
    enc = RetrievalEncoder(config=cfg)
    vec = enc.encode(args.prompt)
    hit, source = store.lookup(vec, threshold=cfg.sim_threshold, subject=args.subject)
    if hit is None:
        meta = {"subject": args.subject} if args.subject is not None else None
        rid = store.add(vec, value=args.value, key_text=args.prompt, meta=meta)
        action = f"inserted (id={rid})"
    else:
        store.edit(hit[0], args.value)
        action = f"updated (id={hit[0]}, source={source}, sim={hit[1]:.3f})"
    _ensure_parent(args.store); store.save(args.store)
    print(f"edit: {args.prompt!r} -> {args.value!r} [{action}], no weights changed")


def cmd_forget(args):
    from kef.encoder import RetrievalEncoder
    cfg = Config()
    store = _load_store(args.store)
    enc = RetrievalEncoder(config=cfg)
    hit, _ = store.lookup(enc.encode(args.prompt),
                          threshold=cfg.sim_threshold,
                          subject=args.subject)
    if hit is None:
        print(f"no stored fact matches {args.prompt!r} (nothing to forget)")
        return
    store.delete(hit[0])
    _ensure_parent(args.store); store.save(args.store)
    print(f"forgot id={hit[0]} ({args.prompt!r}); total={len(store)}")


def cmd_list(args):
    store = _load_store(args.store)
    if len(store) == 0:
        print("(empty store)  add facts with: python3 -m kef teach/edit")
        return
    print(f"{len(store)} stored fact(s):")
    for r in store._records:
        print(f"  id={r.id:<3} {r.key_text!r:50s} -> {r.value!r}")


def cmd_bytes(args):
    store = _load_store(args.store)
    from kef.config import fmt_bytes
    nb = store.nbytes()
    print(f"external store: {nb['n']} facts | keys {fmt_bytes(nb['keys'])} "
          f"| values {fmt_bytes(nb['values'])} | total {fmt_bytes(nb['total'])}")
    print("(reasoning core bytes are constant and independent of this count)")


def cmd_diagnose(args):
    from kef.diagnose import report
    report(profile="tiny" if args.tiny else "full")


def _split_csv(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_floats(value):
    if value is None or value == "":
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def cmd_compose(args):
    from kef import weight_compose as wc
    method = args.compose_command
    if method == "linear":
        models = _split_csv(args.models)
        if len(models) < 2:
            raise SystemExit("linear needs at least two --models")
        out = wc.compose_linear_paths(models, _split_floats(args.weights), args.out, template_dir=(args.template or None))
        print(out)
        return
    if method == "task-vector":
        models = _split_csv(args.models)
        if not args.base or not models:
            raise SystemExit("task-vector needs --base and --models")
        out = wc.compose_task_vector_paths(args.base, models, _split_floats(args.lambdas), args.out)
        print(out)
        return
    if method == "ties":
        models = _split_csv(args.models)
        if not args.base or not models:
            raise SystemExit("ties needs --base and --models")
        out = wc.compose_ties_paths(args.base, models, _split_floats(args.lambdas), args.out, density=args.density)
        print(out)
        return
    if method == "stitch":
        import json
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)
        out = wc.compose_stitch_paths(spec, args.out)
        print(out)
        return
    raise SystemExit(f"unknown compose method: {method}")



def build_parser():
    p = argparse.ArgumentParser(
        prog="kef", description="Knowledge-Externalization Framework CLI")
    p.add_argument("--store", default=DEFAULT_STORE,
                   help="path to the persistent fact store")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="answer a prompt (recall or core)")
    a.add_argument("prompt"); a.add_argument("--tiny", action="store_true")
    a.add_argument("--subject")
    a.add_argument("--min-margin", type=float)
    a.add_argument("--rerank", choices=["lexical"])
    a.set_defaults(func=cmd_ask)

    t = sub.add_parser("teach", help="insert a new fact")
    t.add_argument("prompt"); t.add_argument("value")
    t.add_argument("--subject")
    t.set_defaults(func=cmd_teach)

    e = sub.add_parser("edit", help="change a fact (no retraining)")
    e.add_argument("prompt"); e.add_argument("value")
    e.add_argument("--subject")
    e.set_defaults(func=cmd_edit)

    f = sub.add_parser("forget", help="delete a fact")
    f.add_argument("prompt"); f.add_argument("--subject"); f.set_defaults(func=cmd_forget)

    sub.add_parser("list", help="list stored facts").set_defaults(func=cmd_list)
    sub.add_parser("bytes", help="store byte accounting").set_defaults(func=cmd_bytes)

    dg = sub.add_parser("diagnose", help="facts-vs-rules capacity probe")
    dg.add_argument("--tiny", action="store_true"); dg.set_defaults(func=cmd_diagnose)

    comp = sub.add_parser("compose", help="full-weight merge / stitch (no LoRA)")
    comp_sub = comp.add_subparsers(dest="compose_command", required=True)

    lin = comp_sub.add_parser("linear", help="weighted average of full checkpoints")
    lin.add_argument("--models", required=True, help="comma-separated checkpoint dirs")
    lin.add_argument("--weights", default="", help="comma-separated weights, default equal")
    lin.add_argument("--out", required=True)
    lin.add_argument("--template", default="", help="copy tokenizer/config from this dir")
    lin.set_defaults(func=cmd_compose)

    tv = comp_sub.add_parser("task-vector", help="W_base + sum lambda*(W_i-W_base)")
    tv.add_argument("--base", required=True)
    tv.add_argument("--models", required=True, help="comma-separated variant dirs")
    tv.add_argument("--lambdas", default="", help="comma-separated scales, default 1")
    tv.add_argument("--out", required=True)
    tv.set_defaults(func=cmd_compose)

    ties = comp_sub.add_parser("ties", help="TIES-style trim + sign election merge")
    ties.add_argument("--base", required=True)
    ties.add_argument("--models", required=True)
    ties.add_argument("--lambdas", default="")
    ties.add_argument("--density", type=float, default=0.5)
    ties.add_argument("--out", required=True)
    ties.set_defaults(func=cmd_compose)

    st = comp_sub.add_parser("stitch", help="layer/prefix stitch from a JSON spec")
    st.add_argument("--spec", required=True, help="JSON with sources/rules/default")
    st.add_argument("--out", required=True)
    st.set_defaults(func=cmd_compose)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
