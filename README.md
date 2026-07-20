# KEF — Knowledge-Externalization Framework

> A research prototype exploring a simple idea: **don't compress the giant thing
> — separate the small thing that *thinks* from the large thing it *knows*, and
> store each in its proper currency.**

## BitX north star

KEF is the research core of BitX. The larger target is not a demo and not a
single-domain trick: BitX aims to become an open local intelligence stack that
combines editable external knowledge, adapter-based behavior updates, calibrated
mixed-precision runtime, and reproducible quality gates.

Read [NORTH_STAR.md](NORTH_STAR.md) for the end state and [ROADMAP.md](ROADMAP.md)
for the execution plan.

**This is NOT a "lossless compression" framework.** Information theory forbids
losslessly shrinking a model that genuinely uses its capacity (Shannon + the
~2 bit/param knowledge-capacity law — which we reproduce in `kef/diagnose.py`).
What KEF does instead is **relocate** high-entropy *facts* out of the weights
into an editable, retrievable external store, keeping only low-entropy reusable
*skills/reasoning* in the weights. Smaller accelerator footprint is a
by-product; the real value is **knowledge maintainability + unified
recall/derive**.

Internal/technical name: **Parametric Footprint Minimization**.

## The architecture (three decoupled modules)

```
query → RetrievalEncoder (MiniLM, dedicated) → FactStore.gated_lookup
      → hit?  yes → editable external fact (value)
              no  → ReasoningCore (frozen LM: thinks / derives)
```

- **RetrievalEncoder** — a *dedicated* small encoder produces the lookup key.
  The generation model **cannot** be its own retriever (its hidden states are
  template-dominated: cos(France, Japan) ≈ 0.998). We verified this; a separate
  encoder (MiniLM) gives a clean margin (paraphrase 0.94 vs neighbor 0.51).
- **FactStore** — `add/edit/delete` (O(1), no weights touched), semantic search
  with a similarity gate, a sublinear hierarchical index (~2√N), value
  quantization, edit-conflict detection, tombstone-aware delete fallback, and
  metadata-indexed subject confirmation for auditable structured keys.
- **ReasoningCore** — a frozen LM. Its byte size is **independent of the number
  of facts N** — the whole point.

## Headline results (real model: gpt2-medium, CPU)

| method | efficacy | generalization | locality |
|--------|----------|----------------|----------|
| finetune-edit (2 steps) | 1.00 | 0.50 | **0.50** |
| **externalized edit**   | 1.00 | 1.00 | **1.00** |

Editing one fact by finetuning damaged 50% of *other* facts; the externalized
edit changed exactly one fact, generalized to paraphrases, and left everything
else intact — **no weight update**. Batch editing stays robust: 24 sequential
edits on the same template → efficacy 1.00, zero cross-fire, locality 1.00.

vs plain RAG: identical on fact lookup; KEF additionally **derives** answers to
unseen compute queries (1.00 vs 0.00) — the "compute-instead-of-store" leg.

Byte accounting: the reasoning core is constant while a monolithic baseline
grows with N (21.6× at N=4096; the gap widens without bound).

## Reproduce

```bash
pip install torch transformers numpy            # CPU is fine
python3 -m bitx bench --task smoke              # BitX benchmark contract smoke
python3 -m bitx bench --task kef-edit-smoke     # edit/generalize/locality smoke
python3 -m bitx bench --task edit-comparison-smoke
python3 -m bitx bench --task edit-mini          # FactStore-backed mini comparison
python3 -m bitx bench --task edit-core-mini     # FactStore + real core fallback
python3 -m bitx bench --task edit-trace-mini    # edit/delete/conflict audit trace
python3 -m bitx bench --task edit-suite-mini    # batch edit suite smoke
python3 -m bitx bench --task edit-suite-data-mini  # suite loaded from JSONL
python3 -m bitx bench --task edit-suite-encoder-mini  # same suite with RetrievalEncoder
python3 -m bitx bench --task edit-suite-data-mini --suite bitx/data/edit_suite_capitals.jsonl
python3 -m bitx suite make --size 100 --out kef_results/suites/generated_100.jsonl
python3 -m bitx suite make --size 1024 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_1024.jsonl
python3 -m bitx bench --task edit-suite-data-mini --suite kef_results/suites/generated_100.jsonl
python3 -m bitx bench --task suite-scale --sizes 32,128,512
python3 -m bitx bench --task suite-index-scale --sizes 128,512,2048
python3 -m bitx bench --task suite-large-scale --sizes 10000
python3 -m bitx bench --task suite-100k-smoke --sizes 100000
python3 -m bitx bench --task suite-encoder-scale --sizes 64,128 --encoder-batch-size 32
python3 -m bitx bench --task suite-encoder-keyed-scale --sizes 64,128 --encoder-batch-size 32
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite bitx/data/edit_suite_capitals.jsonl --encoder-batch-size 32
python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64 --lookup-min-margin 0.01
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64 --lookup-min-margin 0.01 --lookup-rerank lexical
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --encoder-batch-size 64
python3 -m bitx suite make --size 4096 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_4096.jsonl
python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64
python3 -m bitx suite make --size 16384 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_16384.jsonl
python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-keyed --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task core-smoke         # real tiny HF generation smoke
python3 -m bitx bench --task native-smoke --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task native-resident-smoke --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task native-prompt-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task native-kv-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 32
python3 -m bitx bench --task native-quant-damage-smoke --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task native-kef-smoke --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task native-kef-suite-smoke --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx bench --task ambiguity-fallback-smoke
python3 -m bitx bench --task semantic-rerank-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl
python3 -m bitx bench --task native-ambiguity-core-smoke --suite bitx/data/semantic_ambiguity_suite.jsonl --model-id /path/to/model.gguf --max-new-tokens 8
python3 -m bitx summarize
python3 -m bitx report --out kef_results/bitx_bench/SCALE_REPORT.md
python3 run_eval.py --tiny     # fast smoke test (distilgpt2)
python3 run_eval.py            # full numbers (gpt2-medium, ~40s on CPU)
python3 -m kef.diagnose        # the facts-vs-rules capacity wall
python3 tests/test_factstore.py
python3 tests/test_kef_ingest.py
python3 tests/test_kef_cli.py
python3 tests/test_synthetic_core.py
```

## Use it (CLI)

The framework ships a CLI. The fact store is persisted to disk, so knowledge
survives between commands; each command loads only the model(s) it needs.

```bash
# the editability story, end to end:
python3 -m kef ask    "The capital of France is"     # -> Paris   [source: core]
python3 -m kef edit   "The capital of France is" " Lyon"   # no weights changed
python3 -m kef ask    "The capital of France is"     # -> Lyon    [source: recall]
python3 -m kef ask    "France's capital city is"     # -> Lyon    [generalizes]
python3 -m kef ask    "The capital of Japan is"      # -> Tokyo   [locality: untouched]

# other commands:
python3 -m kef teach  "The CEO of Acme is" "Alice"   # add a new fact
python3 -m kef teach  "Registry record bx-alpha" "Alice" --subject bx-alpha
python3 -m kef ask    "Who owns bx-alpha?" --subject bx-alpha
python3 -m kef ask    "Who owns this ambiguous record?" --min-margin 0.01
python3 -m kef ask    "Who owns record bx-alpha?" --min-margin 0.01 --rerank lexical
python3 -m kef forget "The capital of France is"     # delete a fact
python3 -m kef list                                  # show stored facts
python3 -m kef bytes                                 # store byte accounting
python3 -m kef eval [--tiny]                          # run the eval suite
python3 -m kef diagnose [--tiny]                      # capacity probe
```

Add `--tiny` to `ask` to use distilgpt2 (faster, but it doesn't actually know
capitals). Use `--store PATH` to point at a different knowledge base. Use
`--subject` when the prompt carries an explicit registry key; that routes through
metadata confirmation before semantic fallback. Use `--min-margin` on unkeyed
queries when you prefer fallback over a low-margin external-memory answer. Use
`--rerank lexical` when the query and stored key text share auditable identifiers
that can recover a low-margin vector tie.


All experiments are CPU-friendly, fixed-seed, and run **one model per process**
(a hard memory rule, learned the hard way). Artifacts land in `kef_results/`.

## What's validated vs future work

See [LIMITATIONS.md](LIMITATIONS.md). Short version: the *mechanisms* are
validated on synthetic tasks and a small real model. This is **not** a
large-scale benchmark, and several pieces (multi-token answers, end-to-end joint
training, >1B models, million-scale stores) are explicitly future work.

## Layout

- `kef/` — the framework (config, encoder, factstore, core, framework, diagnose)
- `bitx/` — benchmark contract and future runtime tooling
- `kef/eval/` — staged, memory-minimal evaluation
- `kef/synthetic_core.py` — research variants (unified recall/derive routing)
- `tests/` — unit + regression tests locking in the core claims
- `RESULTS.md` — full experiment log with honest course-corrections
- `exp*.py` — the original minimal experiments (kept for provenance)

## Contributing

Delivery loop: **Issue → PR into `main` → CI green → merge** (Issue closes on merge via `Fixes #N` / `Closes #N`).

- Humans: [CONTRIBUTING.md](CONTRIBUTING.md)
- Coding agents: [AGENTS.md](AGENTS.md)
- Required check: **CI / test** (`.github/workflows/ci.yml`)

