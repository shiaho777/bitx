# Next Engineering Moves

These are the next concrete moves toward the BitX north star.

## 1. Create The Benchmark Contract

Build `bitx bench` or an equivalent script that records JSONL results with:

- run id
- git commit
- model id
- backend
- quantization recipe
- adapter id
- task name
- prompt count
- raw predictions path
- score
- tokens/s
- first-token latency
- peak RSS
- notes about OS cache or external memory

First supported comparisons:

- base model
- plain RAG
- KEF edit
- fine-tuned edit

## 2. Move Runtime To A Native Backend

Keep Python streaming as a fallback. Start the main path with one native backend:

- GGUF through llama.cpp, or
- MLX on Apple Silicon

Required first proof:

- one local GGUF model loads through `llama-completion`
- one prompt generates
- eval tokens/s and wall-clock tokens/s are measured separately
- resident footprint is measured
- command is reproducible

Current foothold:

`native-smoke` records the first llama.cpp GGUF startup row. `native-resident-smoke`
keeps `llama-server` alive for multiple prompts and separates startup cost from
generation wall time. `native-prompt-cache-smoke` now measures prompt-cache
reuse with cold/warm prompt eval tokens and cache tokens. `native-kv-cache-smoke`
now compares KV cache dtype policies with resident server RSS and speed; the
first short local run shows `q8_0/q8_0` is not automatically better than
`f16/f16`. `native-kef-smoke` combines resident native generation with KEF recall
so external-memory hits and core fallback are measured in one path.
`native-kef-suite-smoke` now loads a JSONL suite into KEF, applies edits and
deletes, checks a bounded batch of recall rows, and sends only bounded misses to
resident llama.cpp. `native-ambiguity-core-smoke` now routes ambiguity-suite
lexical misses to resident llama.cpp and records generation completion plus a
simple clarification-quality rate using a few-shot domain-question prompt
strategy. The next step is longer-context KV measurements, larger-model rows,
and stronger clarification quality scoring.

## 2.5 Add An Ambiguity Recovery Layer

Structured keys now have a runtime confirmation path. Unkeyed ambiguity has a
first safety guard through `--lookup-min-margin`; lexical rerank recovers
identifier-grounded ambiguity, and `ambiguity-fallback-smoke` now proves that
low-margin queries without shared lexical identifiers can be routed to core
clarification instead of unsafe recall. `semantic-rerank-smoke` now adds a
small JSONL-backed RetrievalEncoder-scored recovery slice for the no-shared-
token case. `native-ambiguity-core-smoke` adds resident llama.cpp core routing
and a simple clarification-quality rate for the same lexical misses. These are
still controlled smokes, not general semantic recovery; the current core prompt
strategy is few-shot and template-shaped.

Required first proof:

- low-margin unkeyed queries are detected
- answer precision and abstain rate are reported
- lexical rerank recovers useful coverage when shared identifiers exist
- a semantic reranker or real core clarification set recovers useful coverage
  across a held-out no-shared-identifier ambiguity set
- the tradeoff is shown beside the keyed-confirmed row

## 3. Build Quantization Damage Reports

For each quantization recipe, report:

- perplexity delta
- small reasoning score delta
- KEF edit score delta
- generation speed
- resident memory

The output should make it obvious when Q4 is too damaged and when a mixed Q5/Q6
recipe is worth the extra bytes.

Current foothold:

`native-quant-damage-smoke` now writes the contract row: source and quantized
bytes, BPW, fixed-slice PPL, PPL delta, generation speed, and raw artifacts. The
first real local row is intentionally labeled as requantized because the source
GGUF is already Q4_K_M. The next step is adding an original F16/F32 GGUF baseline
and running Q4/Q5/Q6/Q8 rows against it.

## 4. Upgrade KEF Edit Evaluation

The next KEF win needs multi-token answers and stronger locality.

Minimum eval set:

- 100 facts
- 3 paraphrases per fact
- 3 neighboring distractors per fact
- add, edit, delete operations
- exact-match and semantic-match scoring

Compare against:

- prompt-only RAG
- fine-tuned edit
- unedited base

## 5. Preserve Adapter Discipline

The persona and reasoning LoRA experiments already contain a useful instinct:
never accept an adapter without damage controls.

Turn this into a reusable gate:

- target score improves
- known-answer controls stay correct
- math controls do not regress
- verbosity and refusal behavior stay inside bounds
- health curve is saved

Current foothold:

`kef/adapter_gate.py` now provides `AdapterGate` with `GateControls` for
known answers, math problems, verbosity ratio, and over-refusal rate. The
`adapter-gate-smoke` benchmark task proves all five gate criteria work with
deterministic generate functions. The next step is connecting the gate to a
real LoRA training loop and running it under AdapterGate acceptance criteria.

## 6. Phase 1 Expansion: Held-Out Ambiguity Core

The 12-scenario ambiguity core smoke needs to become a broader held-out set
with partition-aware scoring.

Current foothold:

`heldout-ambiguity-core` benchmark task generates a 48-scenario suite (1/3
held out), routes each through margin-guarded lookup to the resident core, and
reports train vs held-out clarification quality separately. Query variant
consistency is checked across phrasings. `bitx suite make --kind
heldout-ambiguity` generates deterministic suites with partition markers. The
next step is running this against a real GGUF model and comparing
fewshot-domain-question vs raw strategies at scale.

## 7. Phase 2 Expansion: Multi-Recipe Damage Suite

The single-recipe quantization damage smoke needs to become a multi-recipe
comparison with baseline awareness.

Current foothold:

`native-quant-damage-suite` benchmark task runs Q4_K_M, Q5_K_M, Q6_K, and Q8_0
against the same source model in one benchmark group. The `is_baseline` metric
distinguishes F16/F32 sources (BPW >= 14) from requantized sources. The report
labels baseline vs requantized rows separately. The next step is running this
against a real F16 GGUF model to get the first true baseline damage comparison.

## 8. Phase 3: Multi-Token KEF Edit Benchmark

The single-token edit suites need multi-token answer control to be
category-defining.

Current foothold:

`kef-edit-multitoken` benchmark task stores multi-token string values and
checks exact-match + semantic-match across efficacy, generalization
(paraphrases), locality (distractors), delete fallback, and conflict detection.
The `make_multitoken_suite` function generates deterministic suites with
configurable fact count, paraphrase count, and distractor count. The next step
is scaling to 1k and 10k facts and adding plain-RAG and fine-tuned-edit
baselines for the same multi-token suite.

## Immediate First Patch

The first patch is the benchmark contract:

```bash
python3 -m bitx bench --task smoke
python3 -m bitx bench --task kef-edit-smoke
python3 -m bitx bench --task edit-comparison-smoke
python3 -m bitx bench --task edit-mini
python3 -m bitx bench --task edit-core-mini
python3 -m bitx bench --task edit-trace-mini
python3 -m bitx bench --task edit-suite-mini
python3 -m bitx bench --task edit-suite-data-mini
python3 -m bitx bench --task edit-suite-encoder-mini
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
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --encoder-batch-size 64
python3 -m bitx suite make --size 4096 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_4096.jsonl
python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64
python3 -m bitx suite make --size 16384 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_16384.jsonl
python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task suite-encoder-jsonl-keyed --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64
python3 -m bitx bench --task core-smoke
python3 -m bitx summarize
python3 -m bitx report --out kef_results/bitx_bench/SCALE_REPORT.md
python3 tests/test_bitx_bench.py
```

It is the spine that lets every later miracle be measured instead of argued.
