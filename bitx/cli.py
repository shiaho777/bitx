import argparse
import json
import os

from bitx.bench import append_records, run_benchmark
from bitx.report import load_records, scale_report_markdown, summarize_records
from bitx.suite import make_ambiguity_suite, make_heldout_ambiguity_suite, make_keyed_suite, make_suite, write_suite


DEFAULT_OUTPUT_DIR = os.path.join("kef_results", "bitx_bench")
DEFAULT_RESULTS = os.path.join(DEFAULT_OUTPUT_DIR, "results.jsonl")


def cmd_bench(args):
    records = run_benchmark(
        args.task,
        args.output_dir,
        cwd=os.getcwd(),
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        suite_path=args.suite,
        suite_sizes=parse_sizes(args.sizes),
        encoder_batch_size=args.encoder_batch_size,
        index_probe=args.index_probe,
        lookup_min_margin=args.lookup_min_margin,
        lookup_rerank=args.lookup_rerank,
        core_prompt_strategy=args.core_prompt_strategy,
        core_output_policy=args.core_output_policy,
    )
    if not isinstance(records, list):
        records = [records]
    append_records(args.results, records)
    for record in records:
        print(record.to_json())


def cmd_summarize(args):
    print(summarize_records(load_records(args.results)))


def cmd_report(args):
    text = scale_report_markdown(load_records(args.results))
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(json.dumps({"out": args.out}, sort_keys=True))


def cmd_suite_make(args):
    if args.kind == "keyed":
        rows = make_keyed_suite(args.size)
    elif args.kind == "ambiguity":
        rows = make_ambiguity_suite(args.size)
    elif args.kind == "heldout-ambiguity":
        rows = make_heldout_ambiguity_suite(args.size)
    elif args.kind == "multitoken":
        from bitx.bench import make_multitoken_suite
        rows = make_multitoken_suite(args.size)
    else:
        rows = make_suite(args.size)
    write_suite(args.out, rows)
    print(json.dumps({
        "out": args.out,
        "kind": args.kind,
        "facts": len(rows),
        "edited": sum(1 for r in rows if r.get("edit")),
        "deleted": sum(1 for r in rows if r.get("delete")),
    }, sort_keys=True))


def build_parser():
    p = argparse.ArgumentParser(prog="bitx", description="BitX command line tools")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run a benchmark and append a JSONL record")
    b.add_argument("--task", default="smoke", choices=[
        "smoke",
        "kef-edit-smoke",
        "edit-comparison-smoke",
        "edit-mini",
        "edit-core-mini",
        "edit-trace-mini",
        "edit-suite-mini",
        "edit-suite-data-mini",
        "edit-suite-encoder-mini",
        "ambiguity-fallback-smoke",
        "semantic-rerank-smoke",
        "native-ambiguity-core-smoke",
        "suite-scale",
        "suite-index-scale",
        "suite-large-scale",
        "suite-100k-smoke",
        "suite-encoder-scale",
        "suite-encoder-keyed-scale",
        "suite-encoder-jsonl-scale",
        "suite-encoder-jsonl-exact",
        "suite-encoder-jsonl-keyed",
        "core-smoke",
        "native-smoke",
        "native-resident-smoke",
        "native-prompt-cache-smoke",
        "native-kv-cache-smoke",
        "native-quant-damage-smoke",
        "native-quant-damage-suite",
        "kef-edit-multitoken",
        "heldout-ambiguity-core",
        "adapter-gate-smoke",
        "native-kef-smoke",
        "native-kef-suite-smoke",
    ])
    b.add_argument("--model-id", default=None)
    b.add_argument("--max-new-tokens", type=int, default=8)
    b.add_argument("--suite", default=None)
    b.add_argument("--sizes", default=None)
    b.add_argument("--encoder-batch-size", type=int, default=32)
    b.add_argument("--index-probe", type=int, default=None)
    b.add_argument("--lookup-min-margin", type=float, default=None)
    b.add_argument("--lookup-rerank", choices=["lexical"], default=None)
    b.add_argument("--core-prompt-strategy", choices=["raw", "fewshot-domain-question"], default="fewshot-domain-question")
    b.add_argument("--core-output-policy", choices=["raw", "strict-domain-repair"], default="raw")
    b.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    b.add_argument("--results", default=DEFAULT_RESULTS)
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("summarize", help="summarize benchmark JSONL records")
    s.add_argument("--results", default=DEFAULT_RESULTS)
    s.set_defaults(func=cmd_summarize)

    r = sub.add_parser("report", help="write a Markdown scale evidence report")
    r.add_argument("--results", default=DEFAULT_RESULTS)
    r.add_argument("--out", default=os.path.join(DEFAULT_OUTPUT_DIR, "SCALE_REPORT.md"))
    r.set_defaults(func=cmd_report)

    sp = sub.add_parser("suite", help="suite data tools")
    suite_sub = sp.add_subparsers(dest="suite_command", required=True)
    mk = suite_sub.add_parser("make", help="generate a deterministic JSONL fact suite")
    mk.add_argument("--size", type=int, default=100)
    mk.add_argument("--kind", choices=["standard", "keyed", "ambiguity", "heldout-ambiguity", "multitoken"], default="standard")
    mk.add_argument("--out", required=True)
    mk.set_defaults(func=cmd_suite_make)
    return p


def parse_sizes(value):
    if not value:
        return None
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if part:
            size = int(part)
            if size < 6:
                raise ValueError("suite scale sizes must be at least 6")
            sizes.append(size)
    return sizes


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
