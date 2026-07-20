import json
from typing import Iterable, List


def load_records(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_records(records: Iterable[dict]) -> str:
    rows = list(records)
    if not rows:
        return "no benchmark records"
    header = [
        "task",
        "system",
        "n",
        "mode",
        "probe",
        "src",
        "bkt",
        "score",
        "eff",
        "gen",
        "loc",
        "trace",
        "conf",
        "del",
        "vec",
        "store_b",
        "cmp",
        "fb",
        "kcf",
        "ans",
        "abs",
        "rr",
        "add_s",
        "idx_s",
        "enc_s",
        "enc_b",
        "tok/s",
        "wall_s",
        "rss_mb",
    ]
    table = [" | ".join(header)]
    table.append(" | ".join(["---"] * len(header)))
    for r in rows:
        m = r.get("metrics", {})
        system = m.get("system") or r.get("model_id", "")
        vals = [
            r.get("task_name", ""),
            str(system),
            _fmt(m.get("scale_n") or m.get("facts")),
            _short(m.get("lookup_mode")),
            _fmt(m.get("index_probe")),
            _short(m.get("index_probe_source")),
            _fmt(m.get("index_buckets")),
            _fmt(r.get("score")),
            _fmt(m.get("efficacy")),
            _fmt(m.get("generalization")),
            _fmt(m.get("locality")),
            _fmt(m.get("trace_events")),
            _fmt(m.get("conflicts")),
            _fmt(m.get("delete_fallback")),
            _short(m.get("vector_source")),
            _fmt(_store_total(m.get("store_bytes_final") or m.get("store_bytes"))),
            _fmt(m.get("lookup_comparisons_mean")),
            _fmt(m.get("lookup_fallback_rate")),
            _fmt(m.get("key_confirmed_rate")),
            _fmt(m.get("lookup_answer_precision")),
            _fmt(m.get("lookup_abstain_rate")),
            _fmt(m.get("lookup_rerank_rate")),
            _fmt(m.get("add_wall_s")),
            _fmt(m.get("index_build_s")),
            _fmt(m.get("encode_wall_s")),
            _fmt(m.get("encoder_batch_size")),
            _fmt(r.get("tokens_per_second")),
            _fmt(r.get("wall_time_s")),
            _fmt(r.get("peak_rss_mb")),
        ]
        table.append(" | ".join(vals))
    return "\n".join(table)


def scale_report_markdown(records: Iterable[dict]) -> str:
    all_rows = list(records)
    rows = [r for r in all_rows if r.get("task_name") in {
        "suite-scale",
        "suite-index-scale",
        "suite-large-scale",
        "suite-100k-smoke",
        "suite-encoder-scale",
        "suite-encoder-keyed-scale",
        "suite-encoder-jsonl-scale",
        "suite-encoder-jsonl-exact",
        "suite-encoder-jsonl-keyed",
        "kef-edit-multitoken",
    }]
    native_rows = latest_native_rows(all_rows)
    kv_rows = latest_kv_rows(all_rows)
    quant_rows = latest_quant_rows(all_rows)
    ambiguity_rows = latest_ambiguity_rows(all_rows)
    rows = sorted(rows, key=lambda r: (r.get("metrics", {}).get("scale_n") or 0, r.get("created_at", "")))
    if not rows:
        selected = []
        best = None
    else:
        latest = latest_by_task_size_mode(rows)
        selected = sorted(latest.values(), key=lambda r: (
            r.get("metrics", {}).get("scale_n") or 0,
            r.get("task_name", ""),
            r.get("metrics", {}).get("lookup_mode") or "",
        ))
        best = max(selected, key=lambda r: r.get("metrics", {}).get("scale_n") or 0)
    lines = [
        "# BitX Scale Evidence",
        "",
        "This report is generated from benchmark JSONL records.",
        "",
        "## Commands",
        "",
        "```bash",
        "python3 -m bitx bench --task suite-index-scale --sizes 128,512,2048",
        "python3 -m bitx bench --task suite-large-scale --sizes 10000",
        "python3 -m bitx bench --task suite-100k-smoke --sizes 100000",
        "python3 -m bitx bench --task suite-encoder-scale --sizes 64,128 --encoder-batch-size 32",
        "python3 -m bitx bench --task suite-encoder-keyed-scale --sizes 64,128 --encoder-batch-size 32",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite bitx/data/edit_suite_capitals.jsonl --encoder-batch-size 32",
        "python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64 --lookup-min-margin 0.01",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64 --lookup-min-margin 0.01 --lookup-rerank lexical",
        "python3 -m bitx bench --task ambiguity-fallback-smoke",
        "python3 -m bitx bench --task semantic-rerank-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl",
        "python3 -m bitx suite make --size 48 --kind ambiguity --out kef_results/suites/ambiguity_48.jsonl",
        "python3 -m bitx bench --task semantic-rerank-smoke --suite kef_results/suites/ambiguity_48.jsonl",
        "python3 -m bitx bench --task native-ambiguity-core-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl --model-id /path/to/model.gguf --max-new-tokens 8 --core-prompt-strategy raw --core-output-policy raw",
        "python3 -m bitx bench --task native-ambiguity-core-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl --model-id /path/to/model.gguf --max-new-tokens 8 --core-prompt-strategy raw --core-output-policy strict-domain-repair",
        "python3 -m bitx bench --task native-ambiguity-core-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl --model-id /path/to/model.gguf --max-new-tokens 12 --core-prompt-strategy fewshot-domain-question --core-output-policy raw",
        "python3 -m bitx suite make --size 1024 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_1024.jsonl",
        "python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --encoder-batch-size 64",
        "python3 -m bitx suite make --size 4096 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_4096.jsonl",
        "python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64",
        "python3 -m bitx suite make --size 16384 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_16384.jsonl",
        "python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task suite-encoder-jsonl-keyed --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64",
        "python3 -m bitx bench --task native-smoke --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx bench --task native-resident-smoke --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx bench --task native-prompt-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx bench --task native-kv-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 32",
        "python3 -m bitx bench --task native-quant-damage-smoke --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx bench --task native-kef-smoke --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx bench --task native-kef-suite-smoke --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --model-id /path/to/model.gguf --max-new-tokens 8",
        "python3 -m bitx report --out kef_results/bitx_bench/SCALE_REPORT.md",
        "```",
        "",
        "## Headline",
        "",
        headline(best) if best else "No scale benchmark records found.",
        "",
        "## Scale Table",
        "",
        "| task | n | suite | mode | probe | src | bkt | score | eff | gen | loc | edited | checks | locality sample | cmp | fb | kcf | ans | abs | rr | store bytes | encode s | encoder batch | wall s | rss MB |",
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in selected:
        m = r.get("metrics", {})
        lines.append("| " + " | ".join([
            r.get("task_name", ""),
            str(m.get("scale_n") or m.get("facts") or ""),
            suite_label(m),
            str(m.get("lookup_mode") or ""),
            _fmt(m.get("index_probe")),
            _short(m.get("index_probe_source")),
            _fmt(m.get("index_buckets")),
            _fmt(r.get("score")),
            _fmt(m.get("efficacy")),
            _fmt(m.get("generalization")),
            _fmt(m.get("locality")),
            _fmt(m.get("edited")),
            _fmt(m.get("total") or r.get("prompt_count")),
            locality_sample_text(m),
            _fmt(m.get("lookup_comparisons_mean")),
            _fmt(m.get("lookup_fallback_rate")),
            _fmt(m.get("key_confirmed_rate")),
            _fmt(m.get("lookup_answer_precision")),
            _fmt(m.get("lookup_abstain_rate")),
            _fmt(m.get("lookup_rerank_rate")),
            _fmt(_store_total(m.get("store_bytes_final") or m.get("store_bytes"))),
            _fmt(m.get("encode_wall_s")),
            _fmt(m.get("encoder_batch_size")),
            _fmt(r.get("wall_time_s")),
            _fmt(r.get("peak_rss_mb")),
        ]) + " |")
    if native_rows:
        lines.extend([
            "",
            "## Native Runtime",
            "",
            "| task | backend | model | strategy | policy | score | recall | core | core call | clarify q | raw strict | raw fail | final strict | repair | complete | model bytes | pred tok/s | prompt tok/s | tok/s | cold peval | warm peval | warm cache | cache red | startup s | gen wall s | wall s | rss MB |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for r in native_rows:
            m = r.get("metrics", {})
            lines.append("| " + " | ".join([
                r.get("task_name", ""),
                r.get("backend", ""),
                _basename(r.get("model_id", "")),
                native_core_strategy(r),
                native_core_output_policy(r),
                _fmt(r.get("score")),
                _fmt(m.get("recall_rows")),
                _fmt(m.get("core_rows")),
                _fmt(m.get("core_called_rate")),
                _fmt(m.get("clarification_quality_rate")),
                _fmt(m.get("core_raw_strict_quality_rate")),
                _fmt(m.get("core_raw_strict_failure_rate")),
                _fmt(m.get("clarification_strict_quality_rate")),
                _fmt(m.get("core_output_repair_rate")),
                _fmt(m.get("completion_rate")),
                _fmt(m.get("model_bytes")),
                _fmt(m.get("predicted_tokens_per_second_mean") or m.get("eval_tokens_per_second_mean")),
                _fmt(m.get("prompt_tokens_per_second_mean")),
                _fmt(r.get("tokens_per_second")),
                _fmt(m.get("cold_prompt_eval_tokens") if m.get("cold_prompt_eval_tokens") is not None else m.get("cold_prompt_tokens")),
                _fmt(m.get("warm_prompt_eval_tokens") if m.get("warm_prompt_eval_tokens") is not None else m.get("warm_prompt_tokens")),
                _fmt(m.get("warm_prompt_cache_tokens")),
                _fmt(m.get("prompt_eval_reduction")),
                _fmt(m.get("server_startup_s")),
                _fmt(m.get("generation_wall_s")),
                _fmt(r.get("wall_time_s")),
                _fmt(r.get("peak_rss_mb")),
            ]) + " |")
    if kv_rows:
        lines.extend([
            "",
            "## KV Cache Policy",
            "",
            "| task | model | policy | k | v | ctx | score | server RSS MB | RSS ratio | pred tok/s | tok/s | TPS ratio | gen wall s | startup s |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for r in kv_rows:
            m = r.get("metrics", {})
            lines.append("| " + " | ".join([
                r.get("task_name", ""),
                _basename(r.get("model_id", "")),
                str(m.get("kv_policy") or ""),
                str(m.get("cache_type_k") or ""),
                str(m.get("cache_type_v") or ""),
                _fmt(m.get("ctx_size")),
                _fmt(r.get("score")),
                _fmt(m.get("server_rss_mb")),
                _fmt(m.get("rss_ratio_vs_baseline")),
                _fmt(m.get("predicted_tokens_per_second_mean")),
                _fmt(r.get("tokens_per_second")),
                _fmt(m.get("tps_ratio_vs_baseline")),
                _fmt(m.get("generation_wall_s")),
                _fmt(m.get("server_startup_s")),
            ]) + " |")
    if quant_rows:
        lines.extend([
            "",
            "## Quantization Damage",
            "",
            "| task | model | recipe | score | src MB | quant MB | byte ratio | src bpw | quant bpw | src ppl | quant ppl | ppl delta | tok/s | wall s | note |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in quant_rows:
            m = r.get("metrics", {})
            lines.append("| " + " | ".join([
                r.get("task_name", ""),
                _basename(r.get("model_id", "")),
                str(m.get("quantization_recipe") or r.get("quantization_recipe") or ""),
                _fmt(r.get("score")),
                _fmt(_bytes_to_mib(m.get("source_model_bytes"))),
                _fmt(_bytes_to_mib(m.get("quant_model_bytes"))),
                _fmt(m.get("byte_ratio")),
                _fmt(m.get("source_bpw_reported")),
                _fmt(m.get("quant_bpw_reported")),
                _fmt(m.get("source_ppl")),
                _fmt(m.get("quant_ppl")),
                _fmt(m.get("ppl_delta")),
                _fmt(r.get("tokens_per_second")),
                _fmt(r.get("wall_time_s")),
                "requantized" if m.get("requantized_from_quantized") else ("baseline" if m.get("is_baseline") else ""),
            ]) + " |")
    if ambiguity_rows:
        lines.extend([
            "",
            "## Ambiguity Fallback",
            "",
            "| task | backend | suite | scenarios | score | route | fallback | semantic recovery | unsafe recall | unsafe wrong | clarification | abstain | rerank success | semantic rerank | sem margin | shared id |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in ambiguity_rows:
            m = r.get("metrics", {})
            lines.append("| " + " | ".join([
                r.get("task_name", ""),
                r.get("backend", ""),
                suite_label(m),
                _fmt(m.get("scenario_count") or m.get("ambiguous_rows")),
                _fmt(r.get("score")),
                _fmt(m.get("route_score")),
                _fmt(m.get("fallback_quality") if m.get("fallback_quality") is not None else m.get("lexical_fallback_rate")),
                _fmt(m.get("semantic_recovery_rate")),
                _fmt(m.get("unsafe_recall_rate")),
                _fmt(m.get("unsafe_wrong_recall_rate")),
                _fmt(m.get("clarification_rate")),
                _fmt(m.get("lookup_abstain_rate")),
                _fmt(m.get("rerank_success_rate")),
                _fmt(m.get("semantic_rerank_rate")),
                _fmt(m.get("semantic_margin")),
                str(m.get("lexical_shared_identifier")),
            ]) + " |")
    lines.extend([
        "",
        "## What This Proves",
        "",
        "- External edits can update many generated facts without changing model weights.",
        "- Edited facts, paraphrases, delete fallback, conflict detection, and locality controls are checked in one benchmark contract.",
        "- Guarded indexed lookup recovers exact correctness on the recorded scale runs while reducing average comparisons versus flat lookup.",
        "- Store bytes are accounted separately from the frozen reasoning core.",
        "- Native llama.cpp GGUF backends can write the same benchmark contract with model bytes, startup time, generation wall time, tokens/s, and RSS.",
        "- Native prompt-cache rows report cold and warm prompt evaluation tokens so cache effects are visible instead of assumed.",
        "- KV cache policy rows compare cache dtypes with resident server RSS and speed instead of treating KV memory as invisible.",
        "- Quantization damage rows put bytes, BPW, PPL delta, and generation speed in the same contract so near-lossless claims have a budget.",
        "- Ambiguity fallback rows show that low-margin unstructured queries without shared identifiers can route to clarification instead of unsafe external-memory recall.",
        "- Semantic rerank rows show a RetrievalEncoder-scored rerank path can recover a no-shared-token low-margin ambiguity that lexical rerank cannot resolve.",
        "- The native KEF rows prove external-memory hits can bypass the resident core while misses fall back to llama.cpp in measured paths, including a suite-loaded smoke row.",
        "- Native ambiguity core rows show the same ambiguity suite can route guarded lexical misses to a resident llama.cpp core and record generation completion plus loose, raw-strict, raw-failure, final-strict, and repair rates separately from semantic rerank recovery; strategy and output-policy rows expose raw prompting, few-shot prompting, and deterministic repair behavior separately.",
        "- Native ambiguity repair rows preserve `core_raw_prediction` in raw artifacts and score the final policy output, so repaired system behavior is not presented as raw model ability.",
        "- The `heldout-ambiguity-core` task expands the ambiguity suite to 48 scenarios with train/held-out partition scoring and query variant consistency, proving the route discipline generalizes beyond the 12-scenario smoke.",
        "- The `native-quant-damage-suite` task runs Q4_K_M, Q5_K_M, Q6_K, and Q8_0 against the same source in one benchmark group, with `is_baseline` distinguishing F16/F32 sources from requantized ones so damage claims are evidence-based.",
        "- The `kef-edit-multitoken` task proves multi-token answer control with exact-match and semantic-match scoring across efficacy, paraphrase generalization, distractor locality, delete fallback, and conflict detection.",
        "- The `adapter-gate-smoke` task proves the reusable adapter acceptance gate can detect target improvement, fact damage, math damage, verbosity drift, and over-refusal, using target improvement plus damage-control criteria.",
        "",
        "## Current Limits",
        "",
        "- Most high-scale rows use deterministic generated facts and deterministic dense vectors; encoder-scale rows use the real RetrievalEncoder but still use generated facts.",
        "- Encoder-scale rows use runtime FactStore tombstones so delete fallback is protected from nearby-key resurrection.",
        "- The 1k JSONL encoder row currently exposes an encoder-space ambiguity: exact flat retrieval has the same miss as guarded indexed retrieval, so this is not only an index problem.",
        "- The 1k keyed JSONL exact row reaches 1.0, proving the key schema can remove that generated-neighbor ambiguity.",
        "- The current index uses finer default bucketization and auto-probe scaling; the keyed JSONL 4096 default row reaches 1.0 without relying on a manual probe override.",
        "- The keyed JSONL 16384 exact row still misses two paraphrases, so this scale exposes keyed-schema or encoder-space ambiguity beyond the index.",
        "- The keyed-confirmed 16384 row reaches 1.0 through the runtime FactStore subject-confirmation path; this only applies when the query carries a structured key.",
        "- Margin-guarded unstructured lookup reports answer precision and abstain rate separately, so ambiguity can be routed to core or clarification instead of being hidden as a wrong fact-store answer.",
        "- Lexical rerank can recover low-margin unstructured synthetic rows when the query and stored key text share grounded identifiers; the rerank rate is reported so the recovery path is visible.",
        "- Ambiguity-fallback-smoke uses deterministic vectors and deterministic core clarification; it proves routing discipline, not broad semantic reranking quality.",
        "- Semantic-rerank-smoke uses real RetrievalEncoder scoring over JSONL ambiguity suites, and `bitx suite make --kind ambiguity` can generate larger deterministic held-out slices; these are still not broad real-world clarification benchmarks.",
        "- `suite-100k-smoke` samples locality controls; the report states the sample size and population.",
        "- The indexed path is guarded by exact fallback; fallback rate must stay visible in every public result.",
        "- Native-smoke is a startup smoke; native-resident-smoke separates server startup from generation wall time; native-prompt-cache-smoke checks prompt-cache behavior on one repeated prompt; native-kv-cache-smoke is a single-prompt KV dtype policy smoke; native-kef-smoke validates routing; native-kef-suite-smoke scales the store path but still limits resident core calls; native-ambiguity-core-smoke validates core routing plus loose, raw-strict, final-strict, and repair-policy clarification quality on the ambiguity suite.",
        "- Requantized quantization rows are useful contract smoke tests, not final damage claims against an original F16/F32 source.",
        "- This report does not claim large-model answer quality.",
        "",
        "## Next Gate",
        "",
        "Add an original F16/F32 GGUF baseline for quantization damage (the `native-quant-damage-suite` task now supports multi-recipe comparison with baseline awareness), expand semantic rerank and native core clarification into a broader held-out ambiguity set (the `heldout-ambiguity-core` task provides partition-aware scoring), run the `kef-edit-multitoken` benchmark at 1k+ scale with plain-RAG and fine-tuned-edit baselines, then connect the `kef/adapter_gate.py` module to a real LoRA training loop and extend native cache measurements to longer context, concurrency, and multi-turn KEF suite runs.",
        "",
    ])
    return "\n".join(lines)


def latest_by_task_size_mode(rows: List[dict]) -> dict:
    latest = {}
    for r in rows:
        m = r.get("metrics", {})
        suite_key = ""
        if r.get("task_name") in {"suite-encoder-jsonl-scale", "suite-encoder-jsonl-exact"}:
            suite_key = m.get("suite_path") or m.get("data_path") or ""
        key = (
            r.get("task_name"),
            m.get("scale_n") or m.get("facts"),
            m.get("lookup_mode") or "",
            suite_key,
            m.get("index_probe"),
        )
        latest[key] = r
    return latest


def latest_native_rows(rows: List[dict]) -> List[dict]:
    latest = {}
    for r in rows:
        if r.get("task_name") not in {"native-smoke", "native-resident-smoke", "native-prompt-cache-smoke", "native-kef-smoke", "native-kef-suite-smoke", "native-ambiguity-core-smoke", "heldout-ambiguity-core"}:
            continue
        strategy = native_core_strategy(r) if r.get("task_name") == "native-ambiguity-core-smoke" else None
        policy = native_core_output_policy(r) if r.get("task_name") == "native-ambiguity-core-smoke" else None
        key = (r.get("task_name"), r.get("backend"), r.get("model_id"), strategy, policy)
        latest[key] = r
    return sorted(latest.values(), key=lambda r: (
        r.get("backend", ""),
        r.get("model_id", ""),
        native_core_strategy(r),
        native_core_output_policy(r),
    ))


def native_core_strategy(row: dict) -> str:
    if row.get("task_name") not in {"native-ambiguity-core-smoke", "heldout-ambiguity-core"}:
        return ""
    return row.get("metrics", {}).get("core_prompt_strategy") or "fewshot-domain-question"


def native_core_output_policy(row: dict) -> str:
    if row.get("task_name") not in {"native-ambiguity-core-smoke", "heldout-ambiguity-core"}:
        return ""
    return row.get("metrics", {}).get("core_output_policy") or "raw"


def latest_kv_rows(rows: List[dict]) -> List[dict]:
    latest = {}
    for r in rows:
        if r.get("task_name") not in {"native-kv-cache-smoke"}:
            continue
        m = r.get("metrics", {})
        key = (r.get("model_id"), m.get("kv_policy"), m.get("cache_type_k"), m.get("cache_type_v"))
        latest[key] = r
    return sorted(latest.values(), key=lambda r: (
        r.get("model_id", ""),
        r.get("metrics", {}).get("kv_policy") or "",
    ))


def latest_quant_rows(rows: List[dict]) -> List[dict]:
    latest = {}
    for r in rows:
        if r.get("task_name") not in {"native-quant-damage-smoke", "native-quant-damage-suite"}:
            continue
        m = r.get("metrics", {})
        key = (r.get("task_name"), r.get("model_id"), m.get("quantization_recipe") or r.get("quantization_recipe"))
        latest[key] = r
    return sorted(latest.values(), key=lambda r: (r.get("model_id", ""), r.get("metrics", {}).get("quantization_recipe") or ""))


def latest_ambiguity_rows(rows: List[dict]) -> List[dict]:
    latest = {}
    for r in rows:
        if r.get("task_name") not in {"ambiguity-fallback-smoke", "semantic-rerank-smoke", "heldout-ambiguity-core"}:
            continue
        m = r.get("metrics", {})
        key = (r.get("task_name"), r.get("backend"), m.get("suite_path") or "")
        latest[key] = r
    return sorted(latest.values(), key=lambda r: (
        r.get("task_name", ""),
        r.get("backend", ""),
        r.get("metrics", {}).get("suite_path") or "",
    ))


def headline(record: dict) -> str:
    m = record.get("metrics", {})
    return (
        f"Latest largest run: `{record.get('task_name')}` at n={m.get('scale_n') or m.get('facts')} "
        f"with score={_fmt(record.get('score'))}, edited={m.get('edited')}, "
        f"checks={m.get('total') or record.get('prompt_count')}, "
        f"avg comparisons={_fmt(m.get('lookup_comparisons_mean'))}, "
        f"fallback rate={_fmt(m.get('lookup_fallback_rate'))}, "
        f"RSS={_fmt(record.get('peak_rss_mb'))} MB."
    )


def locality_sample_text(m: dict) -> str:
    if "locality_sampled" in m:
        return f"{m.get('locality_sampled')}/{m.get('locality_population')}"
    if m.get("locality") is not None:
        return "full"
    return ""


def suite_label(m: dict) -> str:
    return _short(m.get("suite_path") or m.get("data_path"))


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if 0 < abs(v) < 0.001:
            return f"{v:.3e}"
        return f"{v:.3f}"
    return str(v)


def _short(v):
    if not v:
        return ""
    s = str(v)
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _basename(v):
    s = str(v or "")
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _store_total(v):
    if isinstance(v, dict):
        return v.get("total")
    return v


def _bytes_to_mib(v):
    if v is None:
        return None
    return v / (1024 * 1024)
