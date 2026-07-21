import json
import os
import tempfile

import torch

from bitx.bench import append_record, append_records, load_semantic_ambiguity_scenarios, load_suite_facts, routed_rows, run_benchmark, run_edit_suite_with_vectors, run_semantic_rerank_smoke, semantic_ambiguity_suite_path, suite_data_path
from bitx.backends import DeterministicBackend, LlamaCppBackend, LlamaCppServerBackend, LlamaCppTools
from bitx.report import load_records, scale_report_markdown, summarize_records
from bitx.suite import make_ambiguity_suite, make_keyed_suite, make_suite, read_suite, write_suite


def test_smoke_benchmark_writes_contract():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("smoke", d)
        results = os.path.join(d, "results.jsonl")
        append_record(results, record)
        assert record.task_name == "smoke"
        assert record.prompt_count == 3
        assert record.score == 1.0
        assert record.tokens_per_second > 0
        assert record.peak_rss_mb > 0
        assert os.path.exists(record.raw_predictions_path)
        row = json.loads(open(results, encoding="utf-8").readline())
        required = {
            "run_id",
            "created_at",
            "git_commit",
            "model_id",
            "backend",
            "quantization_recipe",
            "adapter_id",
            "task_name",
            "prompt_count",
            "raw_predictions_path",
            "score",
            "tokens_per_second",
            "first_token_latency_s",
            "peak_rss_mb",
            "wall_time_s",
            "notes",
            "metrics",
        }
        assert required <= set(row)


def test_kef_edit_smoke_reports_edit_metrics():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("kef-edit-smoke", d)
        assert record.task_name == "kef-edit-smoke"
        assert record.score == 1.0
        assert record.metrics["efficacy"] == 1
        assert record.metrics["generalization"] == 1
        assert record.metrics["locality"] == 1.0
        assert record.metrics["store_records"] == 3
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        assert [r["id"] for r in rows] == [
            "efficacy",
            "generalization",
            "locality_japan",
            "locality_italy",
        ]


def test_edit_comparison_smoke_returns_four_system_rows():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("edit-comparison-smoke", d)
        results = os.path.join(d, "results.jsonl")
        append_records(results, records)
        assert len(records) == 4
        systems = [r.metrics["system"] for r in records]
        assert systems == ["base", "plain-rag", "finetune-edit", "kef-edit"]
        by_system = {r.metrics["system"]: r for r in records}
        assert by_system["base"].metrics["efficacy"] == 0
        assert by_system["plain-rag"].metrics["locality"] == 1.0
        assert by_system["finetune-edit"].metrics["locality"] == 0.5
        assert by_system["kef-edit"].metrics["efficacy"] == 1
        assert by_system["kef-edit"].metrics["generalization"] == 1
        assert by_system["kef-edit"].metrics["locality"] == 1.0
        assert len(open(results, encoding="utf-8").readlines()) == 4


def test_summary_includes_edit_metrics():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("edit-comparison-smoke", d)
        results = os.path.join(d, "results.jsonl")
        append_records(results, records)
        text = summarize_records(load_records(results))
        assert "edit-comparison-smoke" in text
        assert "kef-edit" in text
        assert "finetune-edit" in text
        assert "0.500" in text


def test_summary_includes_trace_metrics():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-trace-mini", d)
        results = os.path.join(d, "results.jsonl")
        append_record(results, record)
        text = summarize_records(load_records(results))
        assert "edit-trace-mini" in text
        assert "12" in text
        assert "1" in text


def test_summary_includes_vector_source():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-suite-mini", d)
        results = os.path.join(d, "results.jsonl")
        append_record(results, record)
        text = summarize_records(load_records(results))
        assert "one-hot" in text


def test_edit_mini_uses_factstore_backends():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("edit-mini", d)
        assert len(records) == 4
        by_system = {r.metrics["system"]: r for r in records}
        assert by_system["plain-rag"].backend == "factstore-retrieval"
        assert by_system["kef-edit"].backend == "factstore-edit"
        assert by_system["plain-rag"].metrics["store_records"] == 3
        assert by_system["kef-edit"].metrics["store_records"] == 3
        assert by_system["kef-edit"].metrics["store_bytes"]["total"] > 0
        assert by_system["base"].metrics["efficacy"] == 0
        assert by_system["finetune-edit"].metrics["locality"] == 0.5
        assert by_system["kef-edit"].metrics["efficacy"] == 1.0
        assert by_system["kef-edit"].metrics["generalization"] == 1.0
        assert by_system["kef-edit"].metrics["locality"] == 1.0


def test_deterministic_backend_contract():
    backend = DeterministicBackend({"a": "b c"})
    result = backend.generate("a")
    assert result.text == "b c"
    assert result.token_count == 2
    assert result.wall_time_s >= 0


def test_llamacpp_backend_parses_perf_output():
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    output = """
0.00.498.731 I generate: n_ctx = 2048, n_batch = 2048, n_predict = 8, n_keep = 1
 John Smith. I am a software engineer
0.00.731.153 I common_perf_print: prompt eval time =     139.98 ms /     6 tokens (   23.33 ms per token,    42.86 tokens per second)
0.00.731.154 I common_perf_print:        eval time =      89.08 ms /     7 runs   (   12.73 ms per token,    78.58 tokens per second)
"""
    assert backend._extract_eval_tokens(output) == 7
    assert backend._extract_eval_tps(output) == 78.58
    assert "John Smith" in backend._extract_text(output)


def test_llamacpp_tools_parse_perplexity_and_quantize_output():
    tools = LlamaCppTools.__new__(LlamaCppTools)
    ppl_output = "0.01 I Final estimate: PPL = 1.7226 +/- 0.31511"
    assert tools.parse_perplexity(ppl_output) == (1.7226, 0.31511)
    quant_output = """
llama_model_quantize_impl: model size  =   636.18 MiB (4.85 BPW)
llama_model_quantize_impl: quant size  =   745.11 MiB (5.68 BPW)
"""
    assert tools.parse_quantize_sizes(quant_output) == (636.18, 4.85, 745.11, 5.68)


def test_llamacpp_server_backend_base_url():
    backend = LlamaCppServerBackend.__new__(LlamaCppServerBackend)
    backend.host = "127.0.0.1"
    backend.port = 18080
    assert backend.base_url == "http://127.0.0.1:18080"


def test_routed_rows_uses_recall_before_core():
    import torch
    from kef.factstore import FactStore

    backend = DeterministicBackend({"miss": "fallback"})
    store = FactStore()
    vectors = {
        "hit": torch.tensor([1.0, 0.0]),
        "miss": torch.tensor([0.0, 1.0]),
    }
    store.add(vectors["hit"], "stored")
    cases = [
        {"id": "hit_case", "prompt": "hit", "expected": "stored", "kind": "efficacy", "vec": "hit"},
        {"id": "miss_case", "prompt": "miss", "expected": "fallback", "kind": "locality", "vec": "miss"},
    ]
    rows = routed_rows("x", backend, store, vectors, cases, 2)
    assert rows[0]["source"] == "recall"
    assert rows[0]["prediction"] == "stored"
    assert rows[1]["source"] == "core"
    assert rows[1]["prediction"] == "fallback"


def test_native_smoke_writes_contract_with_llamacpp_backend():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeLlama:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-gguf"
            self.binary = "/tmp/llama-completion"
            self.last_metrics = {"model_bytes": 123, "eval_tokens_per_second": 50.0}

        def generate(self, prompt, max_new_tokens=8):
            self.last_metrics = {"model_bytes": 123, "eval_tokens_per_second": 50.0}
            return backends.GenerationResult("native text", 2, 0.02, 0.04)

    old = backends.LlamaCppBackend
    backends.LlamaCppBackend = FakeLlama
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark("native-smoke", d, model_id="/tmp/model.gguf", max_new_tokens=2)
            assert record.task_name == "native-smoke"
            assert record.backend == "llama.cpp-gguf"
            assert record.score == 1.0
            assert record.metrics["native_binary"] == "/tmp/llama-completion"
            assert record.metrics["model_bytes"] == 123
            assert record.metrics["eval_tokens_per_second_mean"] == 50.0
            assert os.path.exists(record.raw_predictions_path)
    finally:
        backends.LlamaCppBackend = old


def test_native_resident_smoke_writes_contract_with_server_backend():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            return backends.GenerationResult("resident text", 2, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark("native-resident-smoke", d, model_id="/tmp/model.gguf", max_new_tokens=2)
            assert record.task_name == "native-resident-smoke"
            assert record.backend == "llama.cpp-server-gguf"
            assert record.score == 1.0
            assert record.metrics["native_binary"] == "/tmp/llama-server"
            assert record.metrics["server_startup_s"] == 0.5
            assert record.metrics["generation_wall_s"] == 0.05
            assert record.metrics["predicted_tokens_per_second_mean"] == 80.0
            assert record.tokens_per_second == 80.0
            assert os.path.exists(record.raw_predictions_path)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_prompt_cache_smoke_records_cold_and_warm_prompt_eval():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}
            self.calls = []
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8, cache_prompt=False):
            self.calls.append(cache_prompt)
            prompt_eval_tokens = 32 if len(self.calls) == 1 else 1
            prompt_cache_tokens = 0 if len(self.calls) == 1 else 31
            self.last_metrics = {
                "model_bytes": 123,
                "prompt_tokens": 32,
                "tokens_evaluated": 32,
                "tokens_cached": 32,
                "predicted_tokens": 2,
                "prompt_eval_tokens": prompt_eval_tokens,
                "prompt_cache_tokens": prompt_cache_tokens,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0 if len(self.calls) == 1 else 1000.0,
                "cache_prompt": cache_prompt,
            }
            return backends.GenerationResult("cached text", 2, 0.0125, 0.04 if len(self.calls) == 1 else 0.01)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark("native-prompt-cache-smoke", d, model_id="/tmp/model.gguf", max_new_tokens=2)
            assert record.task_name == "native-prompt-cache-smoke"
            assert record.backend == "llama.cpp-server-gguf"
            assert record.score == 1.0
            assert record.metrics["cache_prompt"] is True
            assert record.metrics["cache_effective"] is True
            assert record.metrics["cold_prompt_tokens"] == 32
            assert record.metrics["warm_prompt_tokens"] == 32
            assert record.metrics["cold_prompt_eval_tokens"] == 32
            assert record.metrics["warm_prompt_eval_tokens"] == 1
            assert record.metrics["warm_prompt_cache_tokens"] == 31
            assert record.metrics["prompt_eval_reduction"] == 0.96875
            assert record.metrics["latency_speedup"] == 4.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert [r["id"] for r in rows] == ["cold", "warm"]
            assert all(r["backend_metrics"]["cache_prompt"] for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_kv_cache_smoke_compares_cache_policies():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.kwargs = kwargs
            self.last_metrics = {"model_bytes": 123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def rss_mb(self):
            return 100.0 if self.kwargs.get("cache_type_k") == "f16" else 80.0

        def generate(self, prompt, max_new_tokens=8, cache_prompt=False):
            wall = 0.04 if self.kwargs.get("cache_type_k") == "f16" else 0.05
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 100.0 if self.kwargs.get("cache_type_k") == "f16" else 80.0,
                "prompt_tokens_per_second": 200.0,
                "prompt_eval_tokens": 16,
                "prompt_cache_tokens": 0,
                "server_rss_mb": self.rss_mb(),
            }
            return backends.GenerationResult("kv text", 4, 0.01, wall)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            records = bench.run_benchmark("native-kv-cache-smoke", d, model_id="/tmp/model.gguf", max_new_tokens=4)
            assert len(records) == 2
            by_policy = {r.metrics["kv_policy"]: r for r in records}
            assert sorted(by_policy) == ["kv-f16", "kv-q8_0"]
            assert by_policy["kv-f16"].metrics["server_rss_mb"] == 100.0
            assert by_policy["kv-q8_0"].metrics["server_rss_mb"] == 80.0
            assert by_policy["kv-q8_0"].metrics["rss_ratio_vs_baseline"] == 0.8
            assert by_policy["kv-q8_0"].metrics["tps_ratio_vs_baseline"] == 0.8
            assert by_policy["kv-q8_0"].metrics["cache_type_k"] == "q8_0"
            assert by_policy["kv-q8_0"].metrics["cache_type_v"] == "q8_0"
            assert os.path.exists(by_policy["kv-f16"].raw_predictions_path)
            assert os.path.exists(by_policy["kv-q8_0"].raw_predictions_path)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_quant_damage_smoke_writes_damage_contract():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeTools:
        def __init__(self):
            self.quantize_binary = "/tmp/llama-quantize"
            self.perplexity_binary = "/tmp/llama-perplexity"
            self.calls = 0

        def perplexity(self, model_path, text_path, ctx_size=128, chunks=2):
            self.calls += 1
            ppl = 1.0 if self.calls == 1 else 1.02
            return backends.PerplexityResult(ppl, 0.01, 0.1, "ppl")

        def quantize(self, source_path, output_path, recipe, allow_requantize=False):
            with open(output_path, "wb") as f:
                f.write(b"x" * 50)
            return backends.QuantizeResult(output_path, 100.0, 50.0, 4.0, 2.0, 0.2, "quant")

    class FakeLlama:
        def __init__(self, model_id):
            self.model_id = model_id
            self.binary = "/tmp/llama-completion"
            self.backend = "llama.cpp-gguf"
            self.last_metrics = {"eval_tokens_per_second": 70.0}

        def generate(self, prompt, max_new_tokens=8):
            return backends.GenerationResult("quant text", 2, 0.01, 0.02)

    old_tools = backends.LlamaCppTools
    old_llama = backends.LlamaCppBackend
    backends.LlamaCppTools = FakeTools
    backends.LlamaCppBackend = FakeLlama
    try:
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "source.gguf")
            with open(model, "wb") as f:
                f.write(b"x" * 100)
            record = bench.run_benchmark("native-quant-damage-smoke", d, model_id=model, max_new_tokens=2)
            assert record.task_name == "native-quant-damage-smoke"
            assert record.backend == "llama.cpp-quantize+perplexity"
            assert record.quantization_recipe == "Q5_K_M"
            assert record.score == 1.0
            assert record.metrics["source_ppl"] == 1.0
            assert record.metrics["quant_ppl"] == 1.02
            assert round(record.metrics["ppl_delta"], 6) == 0.02
            assert record.metrics["byte_ratio"] == 0.5
            assert record.metrics["source_bpw_reported"] == 4.0
            assert record.metrics["quant_bpw_reported"] == 2.0
            assert record.metrics["requantized_from_quantized"] is True
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert [r["id"] for r in rows] == ["source_perplexity", "quantize", "quantized_perplexity", "quantized_generation"]
    finally:
        backends.LlamaCppTools = old_tools
        backends.LlamaCppBackend = old_llama


def test_native_kef_smoke_routes_recall_before_resident_core():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.calls += 1
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            return backends.GenerationResult("resident core", 2, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark("native-kef-smoke", d, model_id="/tmp/model.gguf", max_new_tokens=2)
            assert record.task_name == "native-kef-smoke"
            assert record.backend == "factstore+llama.cpp-server-gguf"
            assert record.score == 1.0
            assert record.metrics["recall_rows"] == 2
            assert record.metrics["core_rows"] == 2
            assert record.metrics["core_called_rate"] == 0.5
            assert record.metrics["predicted_tokens_per_second_mean"] == 80.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert [r["source"] for r in rows].count("recall") == 2
            assert [r["source"] for r in rows].count("core") == 2
            assert all(r["route_ok"] for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_kef_suite_smoke_loads_jsonl_and_limits_core_calls():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.calls += 1
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            return backends.GenerationResult("resident suite core", 2, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "suite.jsonl")
            write_suite(path, make_keyed_suite(12))
            record = bench.run_benchmark("native-kef-suite-smoke", d, suite_path=path, model_id="/tmp/model.gguf", max_new_tokens=2)
            assert record.task_name == "native-kef-suite-smoke"
            assert record.backend == "factstore+llama.cpp-server-gguf"
            assert record.score == 1.0
            assert record.metrics["facts"] == 12
            assert record.metrics["edited"] == 4
            assert record.metrics["deleted"] == 1
            assert record.metrics["recall_rows"] == 11
            assert record.metrics["core_rows"] == 2
            assert record.metrics["delete_core_rows"] == 1
            assert record.metrics["route_score"] == 1.0
            assert record.metrics["recall_value_score"] == 1.0
            assert record.metrics["predicted_tokens_per_second_mean"] == 80.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert [r["source"] for r in rows].count("recall") == 11
            assert [r["source"] for r in rows].count("core") == 2
            assert all(r["route_ok"] for r in rows)
            assert all(r["value_ok"] for r in rows if r["source"] == "recall")
            assert any(r["lookup_source"] == "key-tombstone" for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_ambiguity_core_smoke_routes_lexical_misses_to_resident_core():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            if prompt.startswith("clarify:"):
                return backends.GenerationResult("handled internally", 2, 0.0125, 0.025)
            domain = prompt.rsplit("Ambiguous domain: ", 1)[-1].split(". Good clarification:", 1)[0]
            return backends.GenerationResult(f"Which domain do you mean, {domain}?", 6, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark(
                "native-ambiguity-core-smoke",
                d,
                suite_path=semantic_ambiguity_suite_path(),
                model_id="/tmp/model.gguf",
                max_new_tokens=2,
            )
            assert record.task_name == "native-ambiguity-core-smoke"
            assert record.backend == "factstore+llama.cpp-server-gguf+ambiguity-core"
            assert record.score == 1.0
            assert record.metrics["scenario_count"] == 12
            assert record.metrics["route_score"] == 1.0
            assert record.metrics["completion_rate"] == 1.0
            assert record.metrics["clarification_quality_rate"] == 1.0
            assert record.metrics["clarification_strict_quality_rate"] == 1.0
            assert record.metrics["core_raw_strict_quality_rate"] == 1.0
            assert record.metrics["core_raw_strict_failure_rate"] == 0.0
            assert record.metrics["core_rows"] == 12
            assert record.metrics["core_called_rate"] == 1.0
            assert record.metrics["core_prompt_strategy"] == "fewshot-domain-question"
            assert record.metrics["core_output_policy"] == "raw"
            assert record.metrics["core_output_repair_rate"] == 0.0
            assert record.metrics["predicted_tokens_per_second_mean"] == 80.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert len(rows) == 12
            assert all(r["route"] == "core" for r in rows)
            assert all(r["lookup_source"] == "ambiguous" for r in rows)
            assert all(r["core_completed"] for r in rows)
            assert all(r["clarification_quality"] for r in rows)
            assert all(r["clarification_strict_quality"] for r in rows)
            assert all(r["core_raw_strict_quality"] for r in rows)
            assert all(r["core_prompt_strategy"] == "fewshot-domain-question" for r in rows)
            assert all(r["core_output_policy"] == "raw" for r in rows)
            assert not any(r["core_output_repaired"] for r in rows)
            assert all("Which domain do you mean" in r["core_prompt"] for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_ambiguity_core_smoke_raw_strategy_records_unconstrained_prompt():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            return backends.GenerationResult("handled internally", 2, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark(
                "native-ambiguity-core-smoke",
                d,
                suite_path=semantic_ambiguity_suite_path(),
                model_id="/tmp/model.gguf",
                max_new_tokens=2,
                core_prompt_strategy="raw",
            )
            assert record.task_name == "native-ambiguity-core-smoke"
            assert abs(record.score - (2 / 3)) < 1e-9
            assert record.metrics["core_prompt_strategy"] == "raw"
            assert record.metrics["route_score"] == 1.0
            assert record.metrics["completion_rate"] == 1.0
            assert record.metrics["clarification_quality_rate"] == 0.0
            assert record.metrics["clarification_strict_quality_rate"] == 0.0
            assert record.metrics["core_raw_strict_quality_rate"] == 0.0
            assert record.metrics["core_raw_strict_failure_rate"] == 1.0
            assert record.metrics["core_output_policy"] == "raw"
            assert record.metrics["core_output_repair_rate"] == 0.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert len(rows) == 12
            assert all(r["core_prompt_strategy"] == "raw" for r in rows)
            assert all(r["core_prompt"].startswith("clarify:") for r in rows)
            assert not any(r["clarification_quality"] for r in rows)
            assert not any(r["clarification_strict_quality"] for r in rows)
            assert not any(r["core_raw_strict_quality"] for r in rows)
            assert all(r["core_raw_prediction"] == "handled internally" for r in rows)
            assert not any(r["core_output_repaired"] for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_native_ambiguity_core_smoke_output_policy_repairs_failed_clarification():
    import bitx.bench as bench
    import bitx.backends as backends

    class FakeServer:
        def __init__(self, model_id):
            self.model_id = model_id
            self.backend = "llama.cpp-server-gguf"
            self.binary = "/tmp/llama-server"
            self.startup_s = 0.5
            self.last_metrics = {"model_bytes": 123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def generate(self, prompt, max_new_tokens=8):
            self.last_metrics = {
                "model_bytes": 123,
                "predicted_tokens_per_second": 80.0,
                "prompt_tokens_per_second": 100.0,
            }
            return backends.GenerationResult("handled internally", 2, 0.0125, 0.025)

    old = backends.LlamaCppServerBackend
    backends.LlamaCppServerBackend = FakeServer
    try:
        with tempfile.TemporaryDirectory() as d:
            record = bench.run_benchmark(
                "native-ambiguity-core-smoke",
                d,
                suite_path=semantic_ambiguity_suite_path(),
                model_id="/tmp/model.gguf",
                max_new_tokens=2,
                core_prompt_strategy="raw",
                core_output_policy="strict-domain-repair",
            )
            assert record.task_name == "native-ambiguity-core-smoke"
            assert record.score == 1.0
            assert record.metrics["core_prompt_strategy"] == "raw"
            assert record.metrics["core_output_policy"] == "strict-domain-repair"
            assert record.metrics["core_output_repair_rate"] == 1.0
            assert record.metrics["clarification_quality_rate"] == 1.0
            assert record.metrics["clarification_strict_quality_rate"] == 1.0
            assert record.metrics["core_raw_strict_quality_rate"] == 0.0
            assert record.metrics["core_raw_strict_failure_rate"] == 1.0
            rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
            assert len(rows) == 12
            assert all(r["core_output_repaired"] for r in rows)
            assert all(r["core_output_repair_reason"] == "strict_clarification_failed" for r in rows)
            assert all(r["core_raw_prediction"] == "handled internally" for r in rows)
            assert not any(r["core_raw_strict_quality"] for r in rows)
            assert all(r["prediction"].startswith("Which domain do you mean, ") for r in rows)
            assert all(r["clarification_strict_quality"] for r in rows)
    finally:
        backends.LlamaCppServerBackend = old


def test_clarification_strict_quality_requires_question_and_both_domains():
    import bitx.bench as bench

    prompt = "clarify: need leasing or billing domain"
    good = bench.clarification_strict_quality("Which domain do you mean, leasing or billings?", prompt)
    missing = bench.clarification_strict_quality("Which domain do you mean, leasing?", prompt)
    statement = bench.clarification_strict_quality("leasing or billing", prompt)
    assert good["ok"]
    assert "both_domains" in good["reasons"]
    assert not missing["ok"]
    assert missing["missing_domains"] == ["billing"]
    assert not statement["ok"]


def test_bitx_cli_accepts_native_smoke_task():
    from bitx.cli import build_parser

    args = build_parser().parse_args(["bench", "--task", "native-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-smoke"
    assert args.model_id == "/tmp/model.gguf"
    args = build_parser().parse_args(["bench", "--task", "native-resident-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-resident-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-prompt-cache-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-prompt-cache-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-kv-cache-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-kv-cache-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-quant-damage-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-quant-damage-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-kef-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-kef-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-kef-suite-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-kef-suite-smoke"
    args = build_parser().parse_args(["bench", "--task", "ambiguity-fallback-smoke"])
    assert args.task == "ambiguity-fallback-smoke"
    args = build_parser().parse_args(["bench", "--task", "semantic-rerank-smoke"])
    assert args.task == "semantic-rerank-smoke"
    args = build_parser().parse_args(["bench", "--task", "native-ambiguity-core-smoke", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-ambiguity-core-smoke"
    assert args.core_prompt_strategy == "fewshot-domain-question"
    assert args.core_output_policy == "raw"
    args = build_parser().parse_args(["bench", "--task", "native-ambiguity-core-smoke", "--model-id", "/tmp/model.gguf", "--core-prompt-strategy", "raw"])
    assert args.task == "native-ambiguity-core-smoke"
    assert args.core_prompt_strategy == "raw"
    args = build_parser().parse_args(["bench", "--task", "native-ambiguity-core-smoke", "--model-id", "/tmp/model.gguf", "--core-output-policy", "strict-domain-repair"])
    assert args.task == "native-ambiguity-core-smoke"
    assert args.core_output_policy == "strict-domain-repair"
    args = build_parser().parse_args(["suite", "make", "--size", "16", "--kind", "ambiguity", "--out", "/tmp/ambiguity.jsonl"])
    assert args.kind == "ambiguity"


def test_edit_trace_mini_records_audit_events():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-trace-mini", d)
        assert record.task_name == "edit-trace-mini"
        assert record.score == 1.0
        assert record.metrics["conflicts"] == 1
        assert record.metrics["delete_fallback"] == 1
        assert record.metrics["post_delete_locality"] == 1
        assert record.metrics["store_records_final"] == 2
        events = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        names = [e["event"] for e in events]
        assert "edit" in names
        assert "conflict_add" in names
        assert "delete" in names
        assert "lookup_after_delete" in names


def test_edit_suite_mini_scores_batch_edits():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-suite-mini", d)
        assert record.task_name == "edit-suite-mini"
        assert record.score == 1.0
        assert record.metrics["facts"] == 8
        assert record.metrics["edited"] == 4
        assert record.metrics["deleted"] == 1
        assert record.metrics["conflicts"] == 1
        assert record.metrics["efficacy"] == 1.0
        assert record.metrics["generalization"] == 1.0
        assert record.metrics["locality"] == 1.0
        assert record.metrics["delete_fallback"] == 1.0
        assert record.metrics["store_records_final"] == 7
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        assert len(rows) == record.metrics["trace_events"]
        assert sum(1 for r in rows if r.get("kind") == "efficacy") == 3
        assert sum(1 for r in rows if r.get("kind") == "locality") == 4


def test_edit_suite_mini_records_vector_source():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-suite-mini", d)
        assert record.metrics["vector_source"] == "one-hot"


def test_edit_suite_data_mini_loads_jsonl_facts():
    facts = load_suite_facts(suite_data_path())
    assert len(facts) == 8
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("edit-suite-data-mini", d)
        assert record.task_name == "edit-suite-data-mini"
        assert record.score == 1.0
        assert record.metrics["facts"] == 8
        assert record.metrics["edited"] == 4
        assert record.metrics["deleted"] == 1
        assert record.metrics["data_path"].endswith("edit_suite_capitals.jsonl")
        assert record.metrics["efficacy"] == 1.0
        assert record.metrics["generalization"] == 1.0
        assert record.metrics["locality"] == 1.0


def test_edit_suite_data_mini_accepts_custom_suite_path():
    custom = [
        {"id": "alpha", "prompt": "alpha key", "paraphrase": "alpha alternate", "old": "A0", "new": "A1", "edit": True, "delete": False},
        {"id": "bravo", "prompt": "bravo key", "paraphrase": "bravo alternate", "old": "B0", "new": "B1", "edit": True, "delete": False},
        {"id": "charlie", "prompt": "charlie key", "paraphrase": "charlie alternate", "old": "C0", "new": "C1", "edit": True, "delete": False},
        {"id": "delta", "prompt": "delta key", "paraphrase": "delta alternate", "old": "D0", "new": "D1", "edit": True, "delete": True},
        {"id": "echo", "prompt": "echo key", "paraphrase": "echo alternate", "old": "E0", "new": None, "edit": False, "delete": False},
        {"id": "foxtrot", "prompt": "foxtrot key", "paraphrase": "foxtrot alternate", "old": "F0", "new": None, "edit": False, "delete": False},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "suite.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in custom:
                f.write(json.dumps(row) + "\n")
        record = run_benchmark("edit-suite-data-mini", d, suite_path=path)
        assert record.score == 1.0
        assert record.metrics["facts"] == 6
        assert record.metrics["edited"] == 4
        assert record.metrics["deleted"] == 1
        assert record.metrics["data_path"] == path
        assert record.metrics["store_records_final"] == 5


def test_generated_suite_runs_data_benchmark():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "generated.jsonl")
        rows = make_suite(12)
        write_suite(path, rows)
        loaded = read_suite(path)
        assert len(loaded) == 12
        assert sum(1 for r in loaded if r["edit"]) == 4
        assert sum(1 for r in loaded if r["delete"]) == 1
        record = run_benchmark("edit-suite-data-mini", d, suite_path=path)
        assert record.score == 1.0
        assert record.metrics["facts"] == 12
        assert record.metrics["edited"] == 4
        assert record.metrics["deleted"] == 1


def test_keyed_suite_has_structured_prompts():
    rows = make_keyed_suite(8)
    assert len(rows) == 8
    assert rows[0]["id"].startswith("bx-")
    assert "registry key" in rows[0]["prompt"]
    assert rows[0]["id"] in rows[0]["paraphrase"]
    assert sum(1 for r in rows if r["edit"]) == 4
    assert sum(1 for r in rows if r["delete"]) == 1


def test_suite_scale_returns_one_record_per_size():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-scale", d, suite_sizes=[8, 12])
        assert len(records) == 2
        assert [r.task_name for r in records] == ["suite-scale", "suite-scale"]
        assert [r.metrics["scale_n"] for r in records] == [8, 12]
        assert [r.metrics["facts"] for r in records] == [8, 12]
        assert all(r.score == 1.0 for r in records)
        assert all(r.metrics["efficacy"] == 1.0 for r in records)
        assert all(r.metrics["generalization"] == 1.0 for r in records)
        assert all(r.metrics["locality"] == 1.0 for r in records)
        assert all(r.metrics["store_bytes_final"]["total"] > 0 for r in records)
        assert len({r.metrics["scale_group"] for r in records}) == 1


def test_summary_includes_scale_fields():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-scale", d, suite_sizes=[8])
        results = os.path.join(d, "results.jsonl")
        append_records(results, records)
        text = summarize_records(load_records(results))
        assert "suite-scale" in text
        assert "n" in text
        assert "store_b" in text
        assert "8" in text


def test_suite_index_scale_compares_flat_and_indexed():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-index-scale", d, suite_sizes=[32])
        assert len(records) == 3
        by_mode = {r.metrics["lookup_mode"]: r for r in records}
        assert sorted(by_mode) == ["flat", "indexed", "indexed-guarded"]
        assert by_mode["flat"].score == 1.0
        assert by_mode["indexed"].score == 1.0
        assert by_mode["indexed-guarded"].score == 1.0
        assert by_mode["flat"].metrics["lookup_comparisons_mean"] <= 31
        assert by_mode["flat"].metrics["tombstone_blocks"] == 1
        assert by_mode["indexed"].metrics["lookup_comparisons_mean"] < 31
        assert by_mode["indexed-guarded"].metrics["lookup_comparisons_mean"] < 31
        assert by_mode["indexed"].metrics["index_build_s"] > 0
        assert by_mode["indexed-guarded"].metrics["index_build_s"] > 0
        assert by_mode["indexed"].metrics["index_buckets"] > 0
        assert by_mode["indexed-guarded"].metrics["index_buckets"] > 0
        assert len({r.metrics["scale_group"] for r in records}) == 1


def test_summary_includes_index_fields():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-index-scale", d, suite_sizes=[32])
        results = os.path.join(d, "results.jsonl")
        append_records(results, records)
        text = summarize_records(load_records(results))
        assert "mode" in text
        assert "cmp" in text
        assert "bkt" in text
        assert "src" in text
        assert "fb" in text
        assert "flat" in text
        assert "indexed" in text


def test_lookup_min_margin_records_ambiguity_policy():
    facts = [
        {"id": "alpha", "prompt": "attribute of entity 00067", "paraphrase": "entity 00067 attribute", "old": "A0", "new": "A1", "edit": True, "delete": False},
        {"id": "beta", "prompt": "attribute of entity 00076", "paraphrase": "entity 00076 attribute", "old": "B0", "new": "B1", "edit": True, "delete": False},
        {"id": "clear", "prompt": "attribute of entity 99999", "paraphrase": "entity 99999 attribute", "old": "C0", "new": None, "edit": False, "delete": False},
    ]
    vectors = {
        "alpha": torch.tensor([1.0, 0.0]),
        "beta": torch.tensor([0.99, 0.1]),
        "clear": torch.tensor([0.0, 1.0]),
    }
    para_vectors = {
        "alpha": torch.tensor([0.995, 0.05]),
        "beta": torch.tensor([0.995, 0.05]),
        "clear": torch.tensor([0.0, 1.0]),
    }
    with tempfile.TemporaryDirectory() as d:
        record = run_edit_suite_with_vectors(
            d,
            None,
            "margin-policy-mini",
            "factstore-suite+margin-policy",
            "margin policy test",
            vectors,
            para_vectors,
            facts=facts,
            lookup_mode="flat",
            lookup_min_margin=0.01,
        )
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        ambiguous = [r for r in rows if r.get("ambiguous")]
        assert record.metrics["lookup_min_margin"] == 0.01
        assert record.metrics["lookup_ambiguous"] == len(ambiguous)
        assert record.metrics["lookup_ambiguous"] > 0
        assert record.metrics["lookup_abstain_rate"] > 0
        assert record.metrics["lookup_answer_precision"] == 1.0
        assert any(r["kind"] == "generalization" for r in ambiguous)
        reranked = run_edit_suite_with_vectors(
            d,
            None,
            "margin-rerank-mini",
            "factstore-suite+margin-rerank",
            "margin rerank test",
            vectors,
            para_vectors,
            facts=facts,
            lookup_mode="flat",
            lookup_min_margin=0.01,
            lookup_rerank="lexical",
        )
        assert reranked.metrics["lookup_reranked"] > 0
        assert reranked.metrics["lookup_abstain_rate"] < record.metrics["lookup_abstain_rate"]
        assert reranked.metrics["lookup_answer_precision"] == 1.0


def test_ambiguity_fallback_smoke_routes_unshared_identifier_ambiguity_to_core():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("ambiguity-fallback-smoke", d)
        assert record.task_name == "ambiguity-fallback-smoke"
        assert record.backend == "factstore+margin-policy+deterministic-core"
        assert record.score == 1.0
        assert record.metrics["route_score"] == 1.0
        assert record.metrics["fallback_quality"] == 1.0
        assert record.metrics["unsafe_recall_rate"] == 0.0
        assert record.metrics["clarification_rate"] == 1.0
        assert record.metrics["lexical_shared_identifier"] is False
        assert record.metrics["rerank_success_rate"] == 0.0
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        by_id = {r["id"]: r for r in rows}
        assert by_id["unsafe_no_policy"]["route"] == "recall"
        assert by_id["safe_margin_fallback"]["route"] == "core"
        assert by_id["safe_margin_fallback"]["lookup_source"] == "ambiguous"
        assert by_id["safe_margin_fallback"]["clarified"] is True


def test_semantic_rerank_smoke_recovers_unshared_identifier_ambiguity():
    class FakeEncoder:
        name = "fake-semantic-encoder"

        def encode(self, text):
            if "appointment" in text or "clinic scheduling" in text:
                return torch.tensor([1.0, 0.0])
            if "parcel" in text or "warehouse returns" in text:
                return torch.tensor([0.0, 1.0])
            if "seedlings" in text or "garden irrigation" in text:
                return torch.tensor([1.0, 1.0])
            if "invoice" in text or "finance billing" in text:
                return torch.tensor([1.0, -1.0])
            if "account" in text or "identity credentials" in text:
                return torch.tensor([2.0, 0.0])
            if "roof" in text or "property repairs" in text:
                return torch.tensor([0.0, 2.0])
            if "allergy" in text or "nutrition dietary" in text:
                return torch.tensor([2.0, 2.0])
            if "security patch" in text or "software vulnerability" in text:
                return torch.tensor([2.0, -2.0])
            if "hire offer" in text or "human resources" in text:
                return torch.tensor([-2.0, 0.0])
            if "blood sample" in text or "laboratory specimen" in text:
                return torch.tensor([0.0, -2.0])
            if "tenant deposit" in text or "leasing security" in text:
                return torch.tensor([-2.0, -2.0])
            if "service outage" in text or "operations incident" in text:
                return torch.tensor([-1.0, 2.0])
            return torch.tensor([0.1, 0.9])

        def nbytes(self):
            return 12

    with tempfile.TemporaryDirectory() as d:
        record = run_semantic_rerank_smoke(d, encoder=FakeEncoder())
        assert record.task_name == "semantic-rerank-smoke"
        assert record.backend == "factstore+semantic-rerank+deterministic-core"
        assert record.score == 1.0
        assert record.metrics["suite_path"] == semantic_ambiguity_suite_path()
        assert record.metrics["route_score"] == 1.0
        assert record.metrics["semantic_recovery_rate"] == 1.0
        assert record.metrics["lexical_fallback_rate"] == 1.0
        assert record.metrics["unsafe_wrong_recall_rate"] == 1.0
        assert record.metrics["semantic_rerank_rate"] == 1.0
        assert record.metrics["scenario_count"] == 12
        assert record.metrics["semantic_right_score"] > record.metrics["semantic_wrong_score"]
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        assert len(rows) == 36
        assert sum(int(r["case"] == "unsafe_no_policy" and r["unsafe_wrong_recall"]) for r in rows) == 12
        assert sum(int(r["case"] == "lexical_guard" and r["route"] == "core") for r in rows) == 12
        semantic = [r for r in rows if r["case"] == "semantic_rerank"]
        assert len(semantic) == 12
        assert all(r["route"] == "recall" for r in semantic)
        assert all(r["lookup_source"] == "rerank" for r in semantic)
        assert all(r["prediction"] == r["gold"] for r in semantic)


def test_semantic_rerank_smoke_accepts_jsonl_suite_path():
    class FakeEncoder:
        name = "fake-semantic-encoder"

        def encode(self, text):
            if "refund" in text or "billing refund" in text:
                return torch.tensor([1.0, 0.0])
            return torch.tensor([0.0, 1.0])

        def nbytes(self):
            return 8

    with tempfile.TemporaryDirectory() as d:
        suite = os.path.join(d, "semantic_suite.jsonl")
        with open(suite, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "refund",
                "query_text": "How should a refund request be handled?",
                "right_text": "billing refund exception procedure",
                "wrong_text": "garden irrigation maintenance guide",
                "right_value": "billing refund",
                "wrong_value": "garden watering",
                "clarify": "clarify: need finance or facilities domain",
            }, sort_keys=True) + "\n")
        scenarios = load_semantic_ambiguity_scenarios(suite)
        assert len(scenarios) == 1
        record = run_semantic_rerank_smoke(d, encoder=FakeEncoder(), suite_path=suite)
        assert record.metrics["suite_path"] == suite
        assert record.metrics["scenario_count"] == 1
        assert record.prompt_count == 3
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        assert rows[-1]["prediction"] == "billing refund"


def test_generated_ambiguity_suite_matches_semantic_contract():
    rows = make_ambiguity_suite(16)
    assert len(rows) == 16
    required = {"id", "query_text", "right_text", "right_value", "wrong_text", "wrong_value", "clarify"}
    assert all(required.issubset(row) for row in rows)
    assert all(row["right_value"] != row["wrong_value"] for row in rows)
    assert len({row["id"] for row in rows}) == 16
    assert all(row["clarify"].startswith("clarify: need ") for row in rows)


def test_semantic_rerank_smoke_accepts_generated_ambiguity_suite():
    class FakeEncoder:
        name = "fake-generated-ambiguity-encoder"

        def encode(self, text):
            if "RIGHT_UNIQUE" in text:
                return torch.tensor([1.0, 0.0])
            if "WRONG_UNIQUE" in text:
                return torch.tensor([0.0, 1.0])
            return torch.tensor([1.0, 0.0])

        def nbytes(self):
            return 8

    with tempfile.TemporaryDirectory() as d:
        suite = os.path.join(d, "ambiguity.jsonl")
        rows = make_ambiguity_suite(8)
        for row in rows:
            row["right_text"] = f"RIGHT_UNIQUE {row['right_text']}"
            row["wrong_text"] = f"WRONG_UNIQUE {row['wrong_text']}"
        write_suite(suite, rows)
        scenarios = load_semantic_ambiguity_scenarios(suite)
        assert len(scenarios) == 8
        record = run_semantic_rerank_smoke(d, encoder=FakeEncoder(), suite_path=suite)
        assert record.metrics["scenario_count"] == 8
        assert record.metrics["suite_path"] == suite
        assert record.metrics["semantic_recovery_rate"] == 1.0
        assert record.prompt_count == 24


def test_suite_large_scale_uses_guarded_index():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-large-scale", d, suite_sizes=[64])
        assert len(records) == 1
        record = records[0]
        assert record.task_name == "suite-large-scale"
        assert record.score == 1.0
        assert record.metrics["lookup_mode"] == "indexed-guarded"
        assert record.metrics["bulk_load"] is True
        assert record.metrics["facts"] == 64
        assert record.metrics["conflicts"] == 1
        assert record.metrics["add_wall_s"] >= 0
        assert record.metrics["index_build_s"] > 0
        assert record.metrics["lookup_comparisons_mean"] < record.metrics["store_records_final"]


def test_suite_100k_smoke_samples_locality():
    with tempfile.TemporaryDirectory() as d:
        records = run_benchmark("suite-100k-smoke", d, suite_sizes=[128])
        assert len(records) == 1
        record = records[0]
        assert record.task_name == "suite-100k-smoke"
        assert record.score == 1.0
        assert record.metrics["facts"] == 128
        assert record.metrics["edited"] == 32
        assert record.metrics["lookup_mode"] == "indexed-guarded"
        assert record.metrics["locality_sampled"] == record.metrics["locality_population"]
        assert record.metrics["efficacy"] == 1.0
        assert record.metrics["generalization"] == 1.0
        assert record.metrics["locality"] == 1.0
        assert record.metrics["delete_fallback"] == 1.0
        assert record.metrics["lookup_comparisons_mean"] < record.metrics["store_records_final"]
        assert os.path.exists(record.raw_predictions_path)
        assert sum(1 for _ in open(record.raw_predictions_path, encoding="utf-8")) == record.metrics["trace_events"]


def test_scale_report_mentions_largest_run_and_limits():
    records = [
        {
            "task_name": "suite-100k-smoke",
            "created_at": "2026-06-09T10:17:03Z",
            "score": 1.0,
            "prompt_count": 54095,
            "peak_rss_mb": 471.0,
            "wall_time_s": 60.2,
            "metrics": {
                "scale_n": 100000,
                "lookup_mode": "indexed-guarded",
                "efficacy": 1.0,
                "generalization": 1.0,
                "locality": 1.0,
                "edited": 25000,
                "total": 54095,
                "locality_sampled": 4096,
                "locality_population": 75000,
                "lookup_comparisons_mean": 1608.86,
                "lookup_fallback_rate": 0.000259,
                "index_buckets": 949,
                "encode_wall_s": 12.5,
                "encoder_batch_size": 32,
                "store_bytes_final": {"total": 6599934},
            },
        }
    ]
    text = scale_report_markdown(records)
    assert "Latest largest run" in text
    assert "n=100000" in text
    assert "4096/75000" in text
    assert "12.500" in text
    assert "32" in text
    assert "949" in text
    assert "Current Limits" in text
    assert "encoder-scale rows use the real RetrievalEncoder" in text
    assert "runtime FactStore tombstones" in text
    assert "encoder-space ambiguity" in text
    assert "answer precision and abstain rate" in text


def test_scale_report_includes_encoder_jsonl_task():
    records = [
        {
            "task_name": "suite-encoder-jsonl-scale",
            "created_at": "2026-06-09T11:10:00Z",
            "score": 1.0,
            "prompt_count": 11,
            "peak_rss_mb": 340.0,
            "wall_time_s": 0.01,
            "metrics": {
                "scale_n": 8,
                "lookup_mode": "indexed-guarded",
                "efficacy": 1.0,
                "generalization": 1.0,
                "locality": 1.0,
                "edited": 4,
                "total": 11,
                "lookup_comparisons_mean": 7.0,
                "lookup_fallback_rate": 0.0,
                "encode_wall_s": 0.2,
                "encoder_batch_size": 32,
                "store_bytes_final": {"total": 2702},
            },
        }
    ]
    text = scale_report_markdown(records)
    assert "suite-encoder-jsonl-scale" in text


def test_scale_report_includes_answer_precision_and_abstain_columns():
    records = [
        {
            "task_name": "suite-encoder-jsonl-scale",
            "created_at": "2026-06-09T13:39:08Z",
            "score": 0.836,
            "prompt_count": 1279,
            "peak_rss_mb": 318.0,
            "wall_time_s": 0.75,
            "metrics": {
                "scale_n": 1024,
                "suite_path": "kef_results/suites/jsonl_encoder_1024.jsonl",
                "lookup_mode": "indexed-guarded",
                "efficacy": 0.82,
                "generalization": 0.85,
                "locality": 0.512,
                "edited": 256,
                "total": 1279,
                "lookup_comparisons_mean": 1174.279,
                "lookup_fallback_rate": 0.000781,
                "lookup_answer_precision": 1.0,
                "lookup_abstain_rate": 0.358874,
                "lookup_rerank_rate": 0.0,
                "store_bytes_final": {"total": 394878},
            },
        }
    ]
    text = scale_report_markdown(records)
    assert "ans | abs | rr" in text
    assert "1.000 | 0.359 | 0.000" in text


def test_scale_report_includes_native_runtime_section():
    records = [
        {
            "task_name": "native-smoke",
            "backend": "llama.cpp-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:07:48Z",
            "score": 1.0,
            "tokens_per_second": 5.87,
            "peak_rss_mb": 14.7,
            "wall_time_s": 2.38,
            "metrics": {
                "model_bytes": 668788096,
                "eval_tokens_per_second_mean": 80.35,
            },
        },
        {
            "task_name": "native-resident-smoke",
            "backend": "llama.cpp-server-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:17:48Z",
            "score": 1.0,
            "tokens_per_second": 75.31,
            "peak_rss_mb": 20.1,
            "wall_time_s": 1.04,
            "metrics": {
                "model_bytes": 668788096,
                "predicted_tokens_per_second_mean": 127.0,
                "prompt_tokens_per_second_mean": 176.35,
                "server_startup_s": 0.83,
                "generation_wall_s": 0.21,
            },
        },
        {
            "task_name": "native-kef-smoke",
            "backend": "factstore+llama.cpp-server-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:26:41Z",
            "score": 1.0,
            "tokens_per_second": 21.72,
            "peak_rss_mb": 178.75,
            "wall_time_s": 1.74,
            "metrics": {
                "model_bytes": 668788096,
                "recall_rows": 2,
                "core_rows": 2,
                "core_called_rate": 0.5,
                "predicted_tokens_per_second_mean": 88.43,
                "prompt_tokens_per_second_mean": 28.28,
                "server_startup_s": 0.91,
                "generation_wall_s": 0.83,
            },
        },
        {
            "task_name": "native-prompt-cache-smoke",
            "backend": "llama.cpp-server-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:30:41Z",
            "score": 1.0,
            "tokens_per_second": 80.0,
            "peak_rss_mb": 21.0,
            "wall_time_s": 1.00,
            "metrics": {
                "model_bytes": 668788096,
                "predicted_tokens_per_second_mean": 88.0,
                "prompt_tokens_per_second_mean": 200.0,
                "cold_prompt_tokens": 32,
                "warm_prompt_tokens": 32,
                "cold_prompt_eval_tokens": 32,
                "warm_prompt_eval_tokens": 1,
                "warm_prompt_cache_tokens": 31,
                "prompt_eval_reduction": 0.96875,
                "server_startup_s": 0.80,
                "generation_wall_s": 0.20,
            },
        },
        {
            "task_name": "native-ambiguity-core-smoke",
            "backend": "factstore+llama.cpp-server-gguf+ambiguity-core",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:34:41Z",
            "score": 0.667,
            "tokens_per_second": 42.0,
            "peak_rss_mb": 22.0,
            "wall_time_s": 1.20,
            "metrics": {
                "model_bytes": 668788096,
                "core_rows": 12,
                "core_called_rate": 1.0,
                "clarification_quality_rate": 0.0,
                "clarification_strict_quality_rate": 0.0,
                "core_raw_strict_quality_rate": 0.0,
                "core_raw_strict_failure_rate": 1.0,
                "completion_rate": 1.0,
                "core_prompt_strategy": "raw",
                "core_output_policy": "raw",
                "core_output_repair_rate": 0.0,
                "predicted_tokens_per_second_mean": 90.0,
                "prompt_tokens_per_second_mean": 210.0,
                "server_startup_s": 0.80,
                "generation_wall_s": 0.40,
            },
        },
        {
            "task_name": "native-ambiguity-core-smoke",
            "backend": "factstore+llama.cpp-server-gguf+ambiguity-core",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:34:42Z",
            "score": 1.0,
            "tokens_per_second": 42.0,
            "peak_rss_mb": 22.0,
            "wall_time_s": 1.20,
            "metrics": {
                "model_bytes": 668788096,
                "core_rows": 12,
                "core_called_rate": 1.0,
                "clarification_quality_rate": 1.0,
                "clarification_strict_quality_rate": 1.0,
                "core_raw_strict_quality_rate": 0.0,
                "core_raw_strict_failure_rate": 1.0,
                "completion_rate": 1.0,
                "core_prompt_strategy": "raw",
                "core_output_policy": "strict-domain-repair",
                "core_output_repair_rate": 1.0,
                "predicted_tokens_per_second_mean": 90.0,
                "prompt_tokens_per_second_mean": 210.0,
                "server_startup_s": 0.80,
                "generation_wall_s": 0.40,
            },
        },
        {
            "task_name": "native-ambiguity-core-smoke",
            "backend": "factstore+llama.cpp-server-gguf+ambiguity-core",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:35:41Z",
            "score": 1.0,
            "tokens_per_second": 42.0,
            "peak_rss_mb": 22.0,
            "wall_time_s": 1.20,
            "metrics": {
                "model_bytes": 668788096,
                "core_rows": 12,
                "core_called_rate": 1.0,
                "clarification_quality_rate": 1.0,
                "clarification_strict_quality_rate": 1.0,
                "core_raw_strict_quality_rate": 1.0,
                "core_raw_strict_failure_rate": 0.0,
                "completion_rate": 1.0,
                "core_prompt_strategy": "fewshot-domain-question",
                "core_output_policy": "raw",
                "core_output_repair_rate": 0.0,
                "predicted_tokens_per_second_mean": 90.0,
                "prompt_tokens_per_second_mean": 210.0,
                "server_startup_s": 0.80,
                "generation_wall_s": 0.40,
            },
        },
        {
            "task_name": "native-ambiguity-core-smoke",
            "backend": "factstore+llama.cpp-server-gguf+ambiguity-core",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:35:42Z",
            "score": 0.972,
            "tokens_per_second": 41.0,
            "peak_rss_mb": 22.0,
            "wall_time_s": 1.20,
            "metrics": {
                "model_bytes": 668788096,
                "core_rows": 12,
                "core_called_rate": 1.0,
                "clarification_quality_rate": 1.0,
                "clarification_strict_quality_rate": 0.917,
                "core_raw_strict_quality_rate": 0.917,
                "core_raw_strict_failure_rate": 0.083,
                "completion_rate": 1.0,
                "core_prompt_strategy": "fewshot-domain-question",
                "core_output_policy": "raw",
                "core_output_repair_rate": 0.0,
                "predicted_tokens_per_second_mean": 91.0,
                "prompt_tokens_per_second_mean": 211.0,
                "server_startup_s": 0.81,
                "generation_wall_s": 0.41,
            },
        },
        {
            "task_name": "native-ambiguity-core-smoke",
            "backend": "factstore+llama.cpp-server-gguf+ambiguity-core",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T14:35:43Z",
            "score": 1.0,
            "tokens_per_second": 42.0,
            "peak_rss_mb": 22.0,
            "wall_time_s": 1.20,
            "metrics": {
                "model_bytes": 668788096,
                "core_rows": 12,
                "core_called_rate": 1.0,
                "clarification_quality_rate": 1.0,
                "clarification_strict_quality_rate": 1.0,
                "core_raw_strict_quality_rate": 1.0,
                "core_raw_strict_failure_rate": 0.0,
                "completion_rate": 1.0,
                "core_prompt_strategy": "fewshot-domain-question",
                "core_output_policy": "raw",
                "core_output_repair_rate": 0.0,
                "predicted_tokens_per_second_mean": 92.0,
                "prompt_tokens_per_second_mean": 212.0,
                "server_startup_s": 0.82,
                "generation_wall_s": 0.42,
            },
        },
    ]
    text = scale_report_markdown(records)
    assert "Native Runtime" in text
    assert "llama.cpp-gguf" in text
    assert "llama.cpp-server-gguf" in text
    assert "tiny.gguf" in text
    assert "80.350" in text
    assert "127.000" in text
    assert "0.830" in text
    assert "factstore+llama.cpp-server-gguf" in text
    assert "2 | 2 | 0.500 |  | " in text
    assert "factstore+llama.cpp-server-gguf+ambiguity-core" in text
    assert "fewshot-domain-question | raw | 1.000 |  | 12 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000" in text
    assert "strategy | policy | score" in text
    assert "raw | raw | 0.667" in text
    assert "raw | strict-domain-repair | 1.000" in text
    assert "raw | strict-domain-repair | 1.000 |  | 12 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000" in text
    assert "fewshot-domain-question | raw | 1.000" in text
    assert "fewshot-domain-question | raw | 0.972" not in text
    assert "clarify q | raw strict | raw fail | final strict | repair | complete" in text
    assert "cold peval | warm peval | warm cache | cache red" in text
    assert "32 | 1 | 31 | 0.969" in text


def test_scale_report_includes_quantization_damage_section():
    records = [
        {
            "task_name": "native-quant-damage-smoke",
            "backend": "llama.cpp-quantize+perplexity",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T15:00:00Z",
            "score": 1.0,
            "tokens_per_second": 70.0,
            "wall_time_s": 3.0,
            "quantization_recipe": "Q5_K_M",
            "metrics": {
                "quantization_recipe": "Q5_K_M",
                "source_model_bytes": 100 * 1024 * 1024,
                "quant_model_bytes": 50 * 1024 * 1024,
                "byte_ratio": 0.5,
                "source_bpw_reported": 4.85,
                "quant_bpw_reported": 5.68,
                "source_ppl": 1.0,
                "quant_ppl": 1.02,
                "ppl_delta": 0.02,
                "requantized_from_quantized": True,
            },
        }
    ]
    text = scale_report_markdown(records)
    assert "Quantization Damage" in text
    assert "Q5_K_M" in text
    assert "100.000 | 50.000 | 0.500" in text
    assert "1.000 | 1.020 | 0.020" in text
    assert "requantized" in text


def test_scale_report_includes_kv_cache_policy_section():
    records = [
        {
            "task_name": "native-kv-cache-smoke",
            "backend": "llama.cpp-server-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T15:10:00Z",
            "score": 1.0,
            "tokens_per_second": 100.0,
            "wall_time_s": 1.0,
            "metrics": {
                "kv_policy": "kv-f16",
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "ctx_size": 512,
                "server_rss_mb": 100.0,
                "rss_ratio_vs_baseline": 1.0,
                "predicted_tokens_per_second_mean": 100.0,
                "tps_ratio_vs_baseline": 1.0,
                "generation_wall_s": 0.1,
                "server_startup_s": 0.9,
            },
        },
        {
            "task_name": "native-kv-cache-smoke",
            "backend": "llama.cpp-server-gguf",
            "model_id": "/tmp/tiny.gguf",
            "created_at": "2026-06-09T15:10:01Z",
            "score": 1.0,
            "tokens_per_second": 80.0,
            "wall_time_s": 1.1,
            "metrics": {
                "kv_policy": "kv-q8_0",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "ctx_size": 512,
                "server_rss_mb": 80.0,
                "rss_ratio_vs_baseline": 0.8,
                "predicted_tokens_per_second_mean": 80.0,
                "tps_ratio_vs_baseline": 0.8,
                "generation_wall_s": 0.2,
                "server_startup_s": 0.9,
            },
        },
    ]
    text = scale_report_markdown(records)
    assert "KV Cache Policy" in text
    assert "kv-f16" in text
    assert "kv-q8_0" in text
    assert "80.000 | 0.800" in text
    assert "q8_0 | q8_0" in text


def test_scale_report_includes_ambiguity_fallback_section():
    records = [
        {
            "task_name": "ambiguity-fallback-smoke",
            "backend": "factstore+margin-policy+deterministic-core",
            "model_id": "deterministic-core",
            "created_at": "2026-06-09T15:20:00Z",
            "score": 1.0,
            "metrics": {
                "route_score": 1.0,
                "fallback_quality": 1.0,
                "unsafe_recall_rate": 0.0,
                "clarification_rate": 1.0,
                "lookup_abstain_rate": 0.5,
                "rerank_success_rate": 0.0,
                "lexical_shared_identifier": False,
            },
        },
        {
            "task_name": "semantic-rerank-smoke",
            "backend": "factstore+semantic-rerank+deterministic-core",
            "model_id": "retrieval-encoder",
            "created_at": "2026-06-09T15:30:00Z",
            "score": 1.0,
            "metrics": {
                "route_score": 1.0,
                "scenario_count": 12,
                "lexical_fallback_rate": 1.0,
                "semantic_recovery_rate": 1.0,
                "unsafe_wrong_recall_rate": 1.0,
                "clarification_rate": 0.3333333333333333,
                "lookup_abstain_rate": 0.3333333333333333,
                "rerank_success_rate": 0.5,
                "semantic_rerank_rate": 1.0,
                "semantic_margin": 0.3,
                "lexical_shared_identifier": False,
            },
        },
        {
            "task_name": "semantic-rerank-smoke",
            "backend": "factstore+semantic-rerank+deterministic-core",
            "model_id": "retrieval-encoder",
            "created_at": "2026-06-09T15:40:00Z",
            "score": 0.745,
            "metrics": {
                "route_score": 0.944,
                "scenario_count": 48,
                "suite_path": "kef_results/suites/ambiguity_48.jsonl",
                "lexical_fallback_rate": 0.833,
                "semantic_recovery_rate": 0.458,
                "unsafe_wrong_recall_rate": 1.0,
                "clarification_rate": 0.278,
                "lookup_abstain_rate": 0.278,
                "rerank_success_rate": 0.583,
                "semantic_rerank_rate": 1.0,
                "semantic_margin": -0.008,
                "lexical_shared_identifier": False,
            },
        },
    ]
    text = scale_report_markdown(records)
    assert "Ambiguity Fallback" in text
    assert "suite | scenarios" in text
    assert "factstore+margin-policy+deterministic-core" in text
    assert "factstore+semantic-rerank+deterministic-core" in text
    assert " | 12 | 1.000 | 1.000 | 1.000 | 1.000 |  | 1.000 | 0.333 | 0.333 | 0.500 | 1.000 | 0.300 | False" in text
    assert "ambiguity_48.jsonl | 48 | 0.745 | 0.944 | 0.833 | 0.458" in text


def test_scale_report_keeps_jsonl_suite_and_probe_variants():
    records = [
        {
            "task_name": "suite-encoder-jsonl-scale",
            "created_at": "2026-06-09T11:24:13Z",
            "score": 0.9992156862745099,
            "prompt_count": 1279,
            "peak_rss_mb": 427.625,
            "wall_time_s": 0.235,
            "metrics": {
                "scale_n": 1024,
                "suite_path": "kef_results/suites/jsonl_encoder_keyed_1024.jsonl",
                "lookup_mode": "indexed-guarded",
                "efficacy": 1.0,
                "generalization": 0.996078431372549,
                "locality": 1.0,
                "edited": 256,
                "total": 1279,
                "lookup_comparisons_mean": 329.3776387802971,
                "lookup_fallback_rate": 0.0007818608287724785,
                "index_buckets": 96,
                "index_probe_source": "auto",
                "encode_wall_s": 1.697,
                "encoder_batch_size": 64,
                "store_bytes_final": {"total": 394878},
            },
        },
        {
            "task_name": "suite-encoder-jsonl-scale",
            "created_at": "2026-06-09T11:47:29Z",
            "score": 1.0,
            "prompt_count": 1279,
            "peak_rss_mb": 424.859,
            "wall_time_s": 0.687,
            "metrics": {
                "scale_n": 1024,
                "suite_path": "kef_results/suites/jsonl_encoder_keyed_1024.jsonl",
                "lookup_mode": "indexed-guarded",
                "index_probe": 16,
                "efficacy": 1.0,
                "generalization": 1.0,
                "locality": 1.0,
                "edited": 256,
                "total": 1279,
                "lookup_comparisons_mean": 775.4941360437842,
                "lookup_fallback_rate": 0.0007818608287724785,
                "index_buckets": 32,
                "index_probe_source": "cli",
                "encode_wall_s": 1.17,
                "encoder_batch_size": 64,
                "store_bytes_final": {"total": 394878},
            },
        },
    ]
    text = scale_report_markdown(records)
    assert "329.378" in text
    assert "775.494" in text
    assert "| suite-encoder-jsonl-scale | 1024 | jsonl_encoder_keyed_1024.jsonl | indexed-guarded | 16 | cli | 32 | 1.000" in text
    assert "1k keyed JSONL exact row reaches 1.0" in text
    assert "finer default bucketization" in text


def test_scale_report_includes_4096_keyed_auto_probe_gate():
    records = [
        {
            "task_name": "suite-encoder-jsonl-exact",
            "created_at": "2026-06-09T12:20:15Z",
            "score": 1.0,
            "prompt_count": 5119,
            "peak_rss_mb": 459.453,
            "wall_time_s": 3.885,
            "metrics": {
                "scale_n": 4096,
                "suite_path": "kef_results/suites/jsonl_encoder_keyed_4096.jsonl",
                "lookup_mode": "flat",
                "efficacy": 1.0,
                "generalization": 1.0,
                "locality": 1.0,
                "edited": 1024,
                "total": 5119,
                "lookup_comparisons_mean": 4095.0,
                "lookup_fallback_rate": 0.0,
                "encode_wall_s": 5.579,
                "encoder_batch_size": 64,
                "store_bytes_final": {"total": 1580670},
            },
        },
        {
            "task_name": "suite-encoder-jsonl-scale",
            "created_at": "2026-06-09T12:27:33Z",
            "score": 1.0,
            "prompt_count": 5119,
            "peak_rss_mb": 415.484,
            "wall_time_s": 1.329,
            "metrics": {
                "scale_n": 4096,
                "suite_path": "kef_results/suites/jsonl_encoder_keyed_4096.jsonl",
                "lookup_mode": "indexed-guarded",
                "index_probe": 8,
                "index_probe_source": "auto",
                "index_buckets": 192,
                "efficacy": 1.0,
                "generalization": 1.0,
                "locality": 1.0,
                "edited": 1024,
                "total": 5119,
                "lookup_comparisons_mean": 638.8235983590545,
                "lookup_fallback_rate": 0.0001953506544246923,
                "encode_wall_s": 6.591,
                "encoder_batch_size": 64,
                "store_bytes_final": {"total": 1580670},
            },
        },
    ]
    text = scale_report_markdown(records)
    assert "| suite-encoder-jsonl-scale | 4096 | jsonl_encoder_keyed_4096.jsonl | indexed-guarded | 8 | auto | 192 | 1.000" in text
    assert "638.824" in text
    assert "4095.000" in text
    assert "manual probe override" in text


def test_kef_edit_multitoken_scores_multi_token_facts():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("kef-edit-multitoken", d, suite_path=None, n_facts=24, n_paraphrases=3, n_distractors=3)
        assert record.task_name == "kef-edit-multitoken"
        assert record.backend == "factstore-multitoken"
        assert record.metrics["facts"] == 24
        assert record.metrics["multi_token_facts"] > 0
        assert record.metrics["efficacy"] == 1.0
        assert record.metrics["generalization"] == 1.0
        assert record.metrics["locality"] == 1.0
        assert record.metrics["locality_distractor"] == 1.0
        assert record.metrics["exact_match_rate"] == 1.0
        assert record.metrics["semantic_match_rate"] == 1.0
        assert record.metrics["conflicts"] >= 0
        assert record.metrics["store_records_final"] > 0
        assert record.metrics["total_eval_rows"] > 0
        assert os.path.exists(record.raw_predictions_path)
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        # Should have add events, edit events, and eval rows
        events = [r.get("event") for r in rows]
        assert "add" in events
        assert "edit" in events
        eval_rows = [r for r in rows if r.get("event") == "eval"]
        assert any(r.get("multi_token") for r in eval_rows if r.get("kind") == "efficacy")


def test_adapter_gate_smoke_detects_damage_controls():
    with tempfile.TemporaryDirectory() as d:
        record = run_benchmark("adapter-gate-smoke", d)
        assert record.task_name == "adapter-gate-smoke"
        assert record.backend == "deterministic-adapter-gate"
        assert record.score == 1.0
        assert record.metrics["good_adapter_accepted"] is True
        assert record.metrics["damaged_adapter_rejected"] is True
        assert record.metrics["verbose_adapter_rejected"] is True
        assert record.metrics["refusing_adapter_rejected"] is True
        assert record.metrics["good_target_delta"] > 0
        assert record.metrics["damaged_fact_damage"] > 0
        assert record.metrics["verbose_ratio"] > 2.0
        assert record.metrics["refusing_rate"] > 0.2
        rows = [json.loads(x) for x in open(record.raw_predictions_path, encoding="utf-8")]
        assert len(rows) == 4
        by_adapter = {r["adapter"]: r for r in rows}
        assert by_adapter["good_adapter"]["accepted"] is True
        assert by_adapter["damaged_adapter"]["accepted"] is False
        assert by_adapter["verbose_adapter"]["accepted"] is False
        assert by_adapter["refusing_adapter"]["accepted"] is False
        assert "fact_damage" in by_adapter["damaged_adapter"]["reasons"][0] or any("fact_damage" in r for r in by_adapter["damaged_adapter"]["reasons"])


def test_heldout_ambiguity_suite_generates_partitions():
    from bitx.suite import make_heldout_ambiguity_suite
    rows = make_heldout_ambiguity_suite(48)
    assert len(rows) == 48
    required = {"id", "query_text", "right_text", "right_value", "wrong_text", "wrong_value", "clarify", "partition"}
    assert all(required.issubset(row) for row in rows)
    partitions = {row["partition"] for row in rows}
    assert partitions == {"train", "heldout"}
    heldout = [r for r in rows if r["partition"] == "heldout"]
    train = [r for r in rows if r["partition"] == "train"]
    assert len(heldout) == 16
    assert len(train) == 32
    assert all("query_variants" in row for row in rows)
    assert all(len(row["query_variants"]) >= 2 for row in rows)


def test_cli_accepts_new_phase_tasks():
    from bitx.cli import build_parser
    args = build_parser().parse_args(["bench", "--task", "native-quant-damage-suite", "--model-id", "/tmp/model.gguf"])
    assert args.task == "native-quant-damage-suite"
    args = build_parser().parse_args(["bench", "--task", "kef-edit-multitoken"])
    assert args.task == "kef-edit-multitoken"
    args = build_parser().parse_args(["bench", "--task", "heldout-ambiguity-core", "--model-id", "/tmp/model.gguf"])
    assert args.task == "heldout-ambiguity-core"
    args = build_parser().parse_args(["bench", "--task", "adapter-gate-smoke"])
    assert args.task == "adapter-gate-smoke"
    args = build_parser().parse_args(["suite", "make", "--size", "48", "--kind", "heldout-ambiguity", "--out", "/tmp/ha.jsonl"])
    assert args.kind == "heldout-ambiguity"
    args = build_parser().parse_args(["suite", "make", "--size", "24", "--kind", "multitoken", "--out", "/tmp/mt.jsonl"])
    assert args.kind == "multitoken"


if __name__ == "__main__":
    test_smoke_benchmark_writes_contract()
    test_kef_edit_smoke_reports_edit_metrics()
    test_edit_comparison_smoke_returns_four_system_rows()
    test_summary_includes_edit_metrics()
    test_summary_includes_trace_metrics()
    test_summary_includes_vector_source()
    test_edit_mini_uses_factstore_backends()
    test_deterministic_backend_contract()
    test_llamacpp_backend_parses_perf_output()
    test_llamacpp_tools_parse_perplexity_and_quantize_output()
    test_llamacpp_server_backend_base_url()
    test_routed_rows_uses_recall_before_core()
    test_native_smoke_writes_contract_with_llamacpp_backend()
    test_native_resident_smoke_writes_contract_with_server_backend()
    test_native_prompt_cache_smoke_records_cold_and_warm_prompt_eval()
    test_native_kv_cache_smoke_compares_cache_policies()
    test_native_quant_damage_smoke_writes_damage_contract()
    test_native_kef_smoke_routes_recall_before_resident_core()
    test_native_kef_suite_smoke_loads_jsonl_and_limits_core_calls()
    test_native_ambiguity_core_smoke_routes_lexical_misses_to_resident_core()
    test_bitx_cli_accepts_native_smoke_task()
    test_edit_trace_mini_records_audit_events()
    test_edit_suite_mini_scores_batch_edits()
    test_edit_suite_mini_records_vector_source()
    test_edit_suite_data_mini_loads_jsonl_facts()
    test_edit_suite_data_mini_accepts_custom_suite_path()
    test_generated_suite_runs_data_benchmark()
    test_keyed_suite_has_structured_prompts()
    test_suite_scale_returns_one_record_per_size()
    test_summary_includes_scale_fields()
    test_suite_index_scale_compares_flat_and_indexed()
    test_summary_includes_index_fields()
    test_lookup_min_margin_records_ambiguity_policy()
    test_ambiguity_fallback_smoke_routes_unshared_identifier_ambiguity_to_core()
    test_semantic_rerank_smoke_recovers_unshared_identifier_ambiguity()
    test_semantic_rerank_smoke_accepts_jsonl_suite_path()
    test_suite_large_scale_uses_guarded_index()
    test_suite_100k_smoke_samples_locality()
    test_scale_report_mentions_largest_run_and_limits()
    test_scale_report_includes_encoder_jsonl_task()
    test_scale_report_includes_answer_precision_and_abstain_columns()
    test_scale_report_includes_native_runtime_section()
    test_scale_report_includes_quantization_damage_section()
    test_scale_report_includes_kv_cache_policy_section()
    test_scale_report_includes_ambiguity_fallback_section()
    test_scale_report_keeps_jsonl_suite_and_probe_variants()
    test_scale_report_includes_4096_keyed_auto_probe_gate()
    test_kef_edit_multitoken_scores_multi_token_facts()
    test_adapter_gate_smoke_detects_damage_controls()
    test_heldout_ambiguity_suite_generates_partitions()
    test_cli_accepts_new_phase_tasks()
    print("ALL BITX BENCH TESTS PASS")
