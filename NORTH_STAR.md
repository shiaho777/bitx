# BitX North Star

BitX is not trying to make a smaller demo model.

The goal is to build the first open local intelligence stack where a model can
reason, remember, adapt, and serve efficiently without stuffing every fact,
style, update, and runtime constraint into the same weight file.

The miracle is this:

> Separate intelligence into the right currencies, then make the whole system
> measurably better than a monolithic model at the things users actually need:
> speed, editability, memory, freshness, reproducibility, and retained reasoning.

## Ultimate Target

BitX should become an open system that can run 1B-7B class models on ordinary
personal hardware while preserving near-base-model quality, supporting instant
knowledge edits, retaining reasoning ability, and producing benchmark evidence
that survives outside scrutiny.

The public claim we should earn is:

> Same model family, same hardware class, lower resident footprint, faster useful
> iteration, editable knowledge, measurable reasoning retention, and no hidden
> benchmark theater.

Not a toy. Not a single-domain patch. Not a private trick. A reproducible stack.

## Core Thesis

Different kinds of intelligence deserve different storage and execution paths.

| Component | Belongs in | Why |
|---|---|---|
| Stable reasoning skills | Base model weights | Reused everywhere, low edit frequency |
| Mutable facts | External fact store | High entropy, stale quickly, must be editable |
| Persona and task habits | Full-weight variants | Separate checkpoints or merged weights; swappable without baking facts into base |
| Hot inference path | Native runtime | Python disk streaming is a fallback, not the product |
| Quality proof | Benchmarks and traces | Claims must be reproducible |

The current KEF work proves the most important direction: facts can live outside
weights and edits can avoid damaging unrelated knowledge. BitX turns that idea
into a full inference and adaptation system.

## What Number One Means

Number one does not mean claiming to beat every lab on every leaderboard.

Number one means owning a new category:

1. Best open stack for editable local intelligence.
2. Best evidence that knowledge should be externalized instead of repeatedly
   baked into weights.
3. Best practical combination of quantized serving, full-weight behavior
   variants (route or merge), and verifiable knowledge edits.
4. Best reproducible benchmark suite for the tradeoff between model bytes,
   speed, reasoning, and edit locality.

If BitX wins, it wins by changing the comparison itself: monolithic models are
not the only unit of intelligence.

## Non-Negotiables

1. Quality is measured, not assumed.
2. Every compression or externalization step gets a damage budget.
3. Knowledge edits must report efficacy, paraphrase generalization, and locality.
4. Runtime speed must be measured in tokens per second and latency, not just
   memory screenshots.
5. Benchmarks must be reproducible from clean commands.
6. Streaming full-precision weights remains an extreme fallback, not the main
   route.
7. No claim of losslessness unless the exact invariant is proven.
8. No single tiny eval set may justify a broad claim.

## Architecture To Chase

```
user query
  -> router
      -> editable fact memory
      -> reasoning core
      -> weight variant (checkpoint route or merged full weights)
  -> native quantized runtime
  -> answer with provenance and measured confidence
```

The target implementation has four planes:

1. Runtime plane: GGUF/MLX/vLLM-class serving, KV cache, prompt cache, and native
   kernels.
2. Memory plane: KEF fact store, conflict detection, provenance, ANN backend, and
   million-scale edit tests.
3. Adaptation plane: full-weight fine-tunes or offline merges with health-curve early
   stopping and regression gates.
4. Evaluation plane: benchmark harness that reports quality, speed, memory,
   edit locality, retrieval accuracy, and reproducibility artifacts.

## First Public Proof

The first public proof should not be a grand claim. It should be a clean table:

| System | Resident bytes | Tokens/s | Reasoning score | Edit efficacy | Edit locality | Freshness cost |
|---|---:|---:|---:|---:|---:|---:|
| Base model | baseline | baseline | baseline | none | none | retrain or prompt |
| Fine-tuned edit | higher | baseline | measured | measured | measured | training |
| Plain RAG | external | measured | measured | measured | measured | index write |
| BitX/KEF | lower | measured | retained | high | high | instant edit |

The first win is not a slogan. It is a benchmark row that makes the old
comparison look incomplete.

## Technical Anchors

- Full fine-tunes and checkpoint merges show behavior can move without
  stuffing mutable facts into the same file as stable reasoning.
- Quantized serving and careful weight updates can preserve most behavior
  quality while lowering resident memory; they complement external facts rather
  than replacing the need for editable memory.
- llama.cpp/GGUF gives a practical native path for quantized local serving and
  tensor-specific quantization control.
- vLLM/PagedAttention establishes the importance of KV memory management for
  throughput and serving efficiency.

BitX should use these as foundations where they are already strong and innovate
where they do not solve the actual problem: editable, measurable, local
intelligence.
