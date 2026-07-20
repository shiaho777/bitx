# BitX Roadmap

This roadmap converts the north star into engineering checkpoints.

## Phase 0: Stop Measuring The Wrong Thing

Goal: replace fragile one-off claims with a benchmark contract.

Deliverables:

1. `bitx bench` command that writes one JSONL row per run.
2. Metrics for resident RSS, wall memory note, tokens/s, first-token latency,
   perplexity delta, task score, edit efficacy, edit generalization, edit
   locality, and retrieval hit rate.
3. Fixed eval slices for fast smoke, nightly, and public report.
4. Artifact directory containing config, commit hash, model id, adapter id,
   quantization recipe, and raw predictions.

Exit gate:

The same command can compare base, RAG, fine-tune edit, and KEF edit without
manual spreadsheet work.

Current foothold:

`python3 -m bitx bench --task smoke` verifies the record contract.
`python3 -m bitx bench --task kef-edit-smoke` verifies deterministic edit
efficacy, paraphrase generalization, and locality metrics.
`python3 -m bitx bench --task edit-comparison-smoke` verifies that base,
plain-RAG, fine-tuned-edit, and KEF-edit rows can be emitted into the same
comparison group.
`python3 -m bitx bench --task edit-mini` runs the same four-row comparison with
plain-RAG and KEF-edit backed by the real `FactStore` path.
`python3 -m bitx bench --task edit-core-mini` adds real HF core generation as the
base row and as the fallback path behind external memory.
`python3 -m bitx bench --task edit-trace-mini` writes an audit trace for
add/edit/delete/conflict operations and records whether delete fallback and
post-delete locality hold.
`python3 -m bitx bench --task edit-suite-mini` scales the same checks to a
small batch: multiple facts, multiple edits, paraphrases, locality controls,
conflict detection, and delete fallback.
`python3 -m bitx bench --task edit-suite-data-mini` loads that suite from
`bitx/data/edit_suite_capitals.jsonl`, making the benchmark data-driven.
`--suite PATH` lets data-driven suite tasks run any compatible JSONL suite.
`python3 -m bitx suite make --size 100 --out kef_results/suites/generated_100.jsonl`
generates deterministic synthetic suites for scale smoke runs.
`python3 -m bitx bench --task edit-suite-encoder-mini` runs that batch through
the real `RetrievalEncoder` key path instead of one-hot vectors. First run may
download the encoder model.
`python3 -m bitx bench --task suite-scale --sizes 32,128,512` emits a scale
curve over generated suites, sharing one run group and recording facts, store
bytes, edit metrics, trace events, wall time, tokens/s, and RSS for each size.
`python3 -m bitx bench --task suite-index-scale --sizes 128,512,2048` emits
paired flat/indexed/indexed-guarded retrieval rows for the same generated suite
and records average lookup comparisons, fallback rate, and index build time.
`python3 -m bitx bench --task suite-large-scale --sizes 10000` uses the guarded
indexed path with fast generated-suite loading to prove the same edit/delete/
conflict/locality checks at the first five-digit store size.
`python3 -m bitx bench --task suite-100k-smoke --sizes 100000` uses the same
guarded indexed path with streaming raw output, full edited/generalization
checks, delete/conflict checks, and bounded locality sampling for the first
six-digit store smoke.
`python3 -m bitx bench --task suite-encoder-scale --sizes 64,128 --encoder-batch-size 32`
runs the scale contract with real `RetrievalEncoder` keys and records encoder
wall time, encoder byte size, and encoder batch size.
`python3 -m bitx bench --task suite-encoder-keyed-scale --sizes 64,128 --encoder-batch-size 32`
repeats the encoder run with a structured key suite to separate encoder weakness
from the weak numeric generated-entity benchmark.
`python3 -m bitx bench --task suite-encoder-jsonl-scale --suite bitx/data/edit_suite_capitals.jsonl --encoder-batch-size 32`
runs the real encoder path against a JSONL suite, which is the interface needed
for public benchmark data.
`python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_1024.jsonl --encoder-batch-size 64`
runs the same JSONL encoder suite with flat exact retrieval to expose whether
misses come from approximate indexing or from encoder-space ambiguity.
`python3 -m bitx suite make --size 1024 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_1024.jsonl`
generates a structured-key JSONL suite to separate weak generated-entity naming
from retrieval/index behavior.
`python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --encoder-batch-size 64`
runs the keyed JSONL suite with the default indexed path and records the
bucket/probe accuracy/comparison tradeoff.
`python3 -m bitx suite make --size 4096 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_4096.jsonl`
generates the next keyed JSONL scale gate.
`python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64`
verifies that any 4096 indexed miss is not caused by the keyed schema or encoder
space itself.
`python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_4096.jsonl --encoder-batch-size 64`
runs that 4096-fact keyed gate through the real encoder and default indexed path.
`python3 -m bitx suite make --size 16384 --kind keyed --out kef_results/suites/jsonl_encoder_keyed_16384.jsonl`
generates the current 16k keyed boundary suite.
`python3 -m bitx bench --task suite-encoder-jsonl-exact --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64`
proves whether 16k misses are caused by exact encoder-space ambiguity.
`python3 -m bitx bench --task suite-encoder-jsonl-scale --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64`
records the indexed behavior at that boundary.
`python3 -m bitx bench --task suite-encoder-jsonl-keyed --suite kef_results/suites/jsonl_encoder_keyed_16384.jsonl --encoder-batch-size 64`
uses explicit registry-key confirmation to prove structured keys can eliminate
the 16k exact ambiguity when the query carries the key.
`python3 -m bitx bench --task core-smoke` records real HF causal-LM generation
speed and memory through the backend interface.
`python3 -m bitx summarize` prints a compact table from the JSONL records.
`python3 -m bitx report --out kef_results/bitx_bench/SCALE_REPORT.md` generates
a Markdown evidence report from the benchmark JSONL, including headline scale
numbers, scale tables, what is proven, current limits, and the next gate.

`edit-mini` is a component benchmark, not a model-quality claim. It proves that
the comparison harness can drive real external-memory operations and record
edit metrics. The next benchmark must replace deterministic base/fine-tune rows
with real model calls.

`core-smoke` is also not a quality benchmark. Its score only means generation
completed. It exists to prove real model backends can write the same contract.

`edit-suite-mini` is the first batch edit suite smoke. It is still synthetic, but
it exercises the same edit categories the public suite must scale: multiple
facts, multiple edits, paraphrases, locality controls, conflict detection, and
delete fallback.

`edit-suite-data-mini` moves those cases out of code and into JSONL. This is the
shape needed for 100, 1k, and eventually million-fact benchmark splits.

`bitx suite make` is the first scale-data generator. It has been smoke-tested at
32 facts through the same data-driven benchmark contract.

`edit-suite-encoder-mini` is the next realism step: the facts are still synthetic
country-capital edits, but the keys now come from the same dedicated encoder used
by the KEF runtime path.

Current record fields:

`run_id`, `created_at`, `git_commit`, `model_id`, `backend`,
`quantization_recipe`, `adapter_id`, `task_name`, `prompt_count`,
`raw_predictions_path`, `score`, `tokens_per_second`,
`first_token_latency_s`, `peak_rss_mb`, `wall_time_s`, `notes`, `metrics`.

Current summary columns include edit metrics (`eff`, `gen`, `loc`), audit
metrics (`trace`, `conf`, `del`), scale/storage fields (`n`, `store_b`),
retrieval mode/comparison fields (`mode`, `cmp`, `fb`), and key source (`vec`) so
correctness, maintainability, storage growth, retrieval realism, and retrieval
work can be read from the same result table.

Generated scale tasks use fast initial loading so large runs measure edit,
delete, conflict, index build, and lookup behavior instead of O(N²) import-time
duplicate scans. The deliberate conflict row still exercises conflict detection.
The 100k smoke bounds locality rows so the benchmark can keep moving while still
reporting the sampled locality count and population.

Current encoder-scale finding:

The first `suite-encoder-scale` run at 32 facts scored 0.80 because delete
fallback failed: after deleting `entity_00007`, the real encoder query retrieved
neighbor `entity_00008`. Tombstone-aware delete semantics now live in the
runtime `FactStore` API and persist with the store. With runtime tombstones,
`suite-encoder-scale` and `suite-encoder-keyed-scale` pass at 32 and 64 facts
with the real `RetrievalEncoder`. The next encoder gate is a larger public JSONL
suite.

`RetrievalEncoder.encode_batch` now performs real tokenizer/model batching
instead of looping over `encode`, so larger encoder gates measure encoder cost
with the intended path.

Current JSONL encoder finding:

`suite-encoder-jsonl-scale` at 1024 facts with batch size 64 scored 0.9992. The
only miss was the paraphrase for `entity_00189`, which retrieved
`entity_00198`. `suite-encoder-jsonl-exact` reproduced the same miss under flat
exact retrieval, so the remaining error is encoder-space ambiguity rather than
approximate-index loss.

The keyed 1024 JSONL suite splits that error apart. Exact flat retrieval reaches
1.0, proving the structured key schema removes the generated-neighbor ambiguity.
The first default indexed run still missed one paraphrase at probe 4, so
approximate candidate recall remained a separate problem. Raising
`--index-probe` to 16 reached 1.0 with average comparisons around 775 against a
1023-row flat scan, which was a useful precision mode but not the final index
design. The current index uses finer default bucketization and auto-probe
scaling. It reaches 1.0 on the keyed 1024 JSONL suite at probe 4 with average
comparisons around 203, and reaches 1.0 on the keyed 4096 JSONL suite at auto
probe 8 with average comparisons around 639 against a 4095-row flat scan. Exact
flat retrieval also reaches 1.0 on the keyed 4096 suite, so the earlier 4096
probe-4 misses were index candidate misses rather than keyed-schema ambiguity.

The keyed 16k JSONL suite is the current boundary. Indexed default retrieval
scores 0.9998 at auto probe 32 with average comparisons around 3414. Exact flat
retrieval still misses two paraphrases (`bx-HJE-02503` and `bx-JJG-03656`), so
this is no longer only an index candidate problem. The next gate is stronger
keyed disambiguation or semantic match policy that recovers those 16k exact
misses without treating every near key as acceptable.

Structured key confirmation recovers the 16k keyed suite to 1.0 with average
lookup comparisons of 1.0 by confirming the explicit registry key in the query.
This is now a first-class `FactStore.lookup(..., subject=...)` runtime path
backed by a metadata index, exposed through `KEFramework`, `KEFModel`, CLI
`--subject`, JSONL/export metadata, and the benchmark contract. It is not a
general semantic fix; it applies to structured knowledge where the key is
present and auditable. The remaining open problem is unstructured ambiguity,
where there is no exact key to confirm. A first safety policy now exists:
`min_margin` can make unstructured lookup abstain when top candidates are too
close, and the benchmark reports answer precision separately from abstain rate.
A lexical rerank path can recover synthetic identifier ambiguity when query text
and stored key text share grounded tokens; rerank rate is reported so this
recovery path stays visible. A deterministic ambiguity-fallback smoke now proves
the route discipline for low-margin natural-language queries without shared
lexical identifiers: unsafe recall is visible, and the guarded path falls back
to clarification. `semantic-rerank-smoke` now reads a JSONL ambiguity suite and
adds the first RetrievalEncoder-scored recovery row set for no-shared-token
ambiguity that lexical rerank cannot resolve. `native-ambiguity-core-smoke`
routes the same lexical misses to a resident llama.cpp core and records
generation completion and a simple clarification-quality rate separately from
semantic recovery; the current native row uses a few-shot domain-question prompt
strategy. The next step is expanding this small suite into a larger held-out
semantic rerank and stronger core clarification quality set. The
`heldout-ambiguity-core` task expands this to a 48-scenario held-out suite
with partition-aware scoring and query variant consistency.

## Phase 1: Native Runtime First

Goal: stop treating Python disk streaming as the main inference engine.

Deliverables:

1. GGUF or MLX backend adapter behind one BitX interface.
2. Prompt cache and KV cache enabled for normal generation.
3. Quantization recipes for Q4, Q5, Q6, Q8, and mixed per-tensor modes.
4. Throughput report on the local machine for at least one 1B-class and one
   larger model when available.

Exit gate:

A normal prompt must run through the native backend with documented tokens/s,
resident footprint, and reproducible command output.

Current foothold:

`python3 -m bitx bench --task native-smoke --model-id /path/to/model.gguf --max-new-tokens 8`
runs a GGUF model through `llama-completion`, records model bytes, eval
tokens/s, wall time, and RSS in the same benchmark contract. This is a startup
smoke, not a resident-server throughput result.
`python3 -m bitx bench --task native-resident-smoke --model-id /path/to/model.gguf --max-new-tokens 8`
runs the same class of model through a long-lived `llama-server` process and
separates server startup from generation wall time.
`python3 -m bitx bench --task native-prompt-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 8`
repeats one prompt through the resident server with prompt caching enabled and
records cold/warm prompt eval tokens and cache tokens. The current local GGUF
row records cold prompt eval 41, warm prompt eval 1, warm cache 40, and prompt
eval reduction around 0.976.
`python3 -m bitx bench --task native-kv-cache-smoke --model-id /path/to/model.gguf --max-new-tokens 32`
compares KV cache dtype policies with resident server RSS and speed. On the
current short ctx=512 local GGUF smoke, `q8_0/q8_0` did not improve RSS versus
`f16/f16` and was slower, which is a useful negative result rather than a
failure to hide.
`python3 -m bitx bench --task native-kef-smoke --model-id /path/to/model.gguf --max-new-tokens 8`
is the first combined row: external-memory hits bypass the resident core, and
misses fall back to llama.cpp in the same measured path. It is still a small
routing smoke, not a full serving benchmark.
`python3 -m bitx bench --task native-kef-suite-smoke --suite kef_results/suites/jsonl_encoder_keyed_1024.jsonl --model-id /path/to/model.gguf --max-new-tokens 8`
loads a JSONL suite into KEF, applies edits and deletes, checks many recall rows,
and keeps resident llama.cpp fallback calls bounded. The current local GGUF row
records 1024 facts, 32 recall rows, 2 core rows, route score 1.0, recall value
score 1.0, server startup around 0.87s, and generation wall around 0.18s.
`python3 -m bitx bench --task heldout-ambiguity-core --model-id /path/to/model.gguf --max-new-tokens 8`
is the Phase 1 expansion: a larger held-out ambiguity suite (default 48
scenarios, 1/3 held out) routes through margin-guarded lookup to the resident
core. It reports train vs held-out clarification quality separately, plus
per-variant route consistency across query phrasings. This is the step from
the 12-scenario smoke to a broader held-out set with partition-aware scoring.
`python3 -m bitx suite make --size 48 --kind heldout-ambiguity --out kef_results/suites/heldout_ambiguity_48.jsonl`
generates a deterministic held-out ambiguity suite with explicit train/heldout
partition markers and query variants for consistency checking.

## Phase 2: Quantization With A Damage Budget

Goal: make near-lossless a measured engineering target.

Deliverables:

1. Calibration dataset builder from reasoning, general language, code, and KEF
   edit prompts.
2. Per-tensor sensitivity report.
3. Mixed-precision quant recipe generator.
4. Regression gate that rejects a recipe if quality loss exceeds the configured
   budget.

Exit gate:

BitX can say which tensors deserve more bits and show the quality reason.

Current foothold:

`python3 -m bitx bench --task native-quant-damage-smoke --model-id /path/to/model.gguf --max-new-tokens 8`
creates a quantization damage row with source bytes, quantized bytes, BPW,
fixed-slice PPL, PPL delta, generation speed, and raw artifacts. The current
local GGUF row is a contract smoke from an already-quantized Q4_K_M source to
Q5_K_M: byte ratio about 1.17, PPL delta about 0.026, and the report labels it
as requantized. The next gate needs an original F16/F32 GGUF baseline before any
near-lossless claim is public-quality.
`python3 -m bitx bench --task native-quant-damage-suite --model-id /path/to/model.gguf --max-new-tokens 8`
is the Phase 2 multi-recipe damage suite: runs Q4_K_M, Q5_K_M, Q6_K, and Q8_0
against the same source model in one benchmark group. Each row records source
and quantized bytes, BPW, PPL delta, generation speed, and whether the source
is a true F16/F32 baseline (BPW >= 14) or requantized from an already-quantized
source. This replaces the single-recipe smoke with a damage-budget comparison
so Q4 vs Q5 vs Q6 vs Q8 decisions are evidence-based. The `is_baseline` and
`source_is_f16_f32` metrics make the distinction visible in the report.

## Phase 3: KEF Edit Benchmark

Goal: make externalized knowledge the category-defining result.

Deliverables:

1. Multi-token answer control.
2. Paraphrase eval for every edited fact.
3. Locality eval against neighboring facts.
4. Conflict detection report.
5. Plain RAG and fine-tuned-edit baselines.
6. Store scaling runs at 1k, 10k, 100k, and 1M facts.

Exit gate:

A public report shows that editing one fact changes that fact, generalizes to
paraphrases, and does not damage neighbors, with latency and byte accounting.

Current foothold:

`python3 -m bitx bench --task kef-edit-multitoken`
is the Phase 3 multi-token KEF edit benchmark. Unlike the single-token edit
suites, this benchmark stores multi-token string values (e.g. "ancient
kingdom") and checks exact-match + semantic-match scoring across efficacy,
generalization (paraphrases), locality (distractor queries), delete fallback,
and conflict detection. The suite is data-driven from JSONL or generated if no
path is given, with configurable fact count (default 100), paraphrase count
(default 3), and distractor count (default 3). The `make_multitoken_suite`
function generates deterministic multi-token facts with paraphrase and
distractor variants, and `multitoken_semantic_match` provides a looser
token-overlap match alongside exact string equality.
`python3 -m bitx suite make --size 100 --kind multitoken --out kef_results/suites/multitoken_100.jsonl`
generates a deterministic multi-token fact suite that can be loaded by the
`kef-edit-multitoken` task via `--suite`.
`python3 -m bitx bench --task kef-edit-multitoken --suite kef_results/suites/multitoken_100.jsonl`
runs the multi-token benchmark against the generated suite.

## Phase 4: Adapter Discipline

Goal: make small behavioral and reasoning updates useful without damaging the
base model.

Deliverables:

1. Adapter training harness with validation-guided early stopping.
2. Health curve persisted for every run.
3. Negative controls for over-refusal, verbosity drift, fact damage, and math
   degradation.
4. Adapter merge and adapter stack evaluation.

Exit gate:

An adapter is accepted only when it improves its target trait and passes damage
controls.

Current foothold:

`kef/adapter_gate.py` is the Phase 4 reusable adapter acceptance gate. It
is a standalone module with five gate criteria: target trait improvement, fact
damage detection, math damage detection, verbosity drift bounds, and over-refusal
rate bounds. The `AdapterGate.evaluate` method compares base vs adapted generate
functions and returns a `GateResult` with per-criterion pass/fail, reasons for
rejection, and a persisted health curve. The `evaluate_with_history` method
evaluates across epoch checkpoints and picks the healthiest one.
`python3 -m bitx bench --task adapter-gate-smoke`
is the Phase 4 contract smoke: it uses deterministic generate functions to
prove the gate can detect target improvement, fact damage, math damage,
verbosity drift, and over-refusal. A good adapter is accepted, and damaged,
verbose, and refusing adapters are all rejected. The health curve is persisted
as JSONL for every evaluation.

## Phase 5: Public Release Shape

Goal: make the project reproducible enough that outsiders can attack it and
still reproduce the core result.

Deliverables:

1. One-command smoke test.
2. One-command public benchmark subset.
3. Model/backend compatibility matrix.
4. Reproducible report generator.
5. Clear claim ladder: proven, promising, unproven, rejected.

Exit gate:

A new user can clone the repository, run the smoke test, run a small benchmark,
edit a fact, and see the same class of result without reading the research log.

## Current Strategic Decision

The old full-precision disk-streaming path stays as a research fallback. It is
valuable because it proves memory can be decoupled from model depth, but it is
too slow to be the product path.

The main route is:

```
native quantized runtime
  + calibrated mixed precision
  + KEF editable memory
  + adapter training with health gates
  + benchmark evidence
```

This is the path that can become public, useful, and hard to dismiss.
