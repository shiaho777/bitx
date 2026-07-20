# Limitations & honest scope

This project is a **research prototype**. Its purpose is to test, with minimal
experiments, whether the "relocate facts, keep skills" idea holds up — not to
ship a production system or claim a benchmark win. We separate what is actually
validated from what is not, so the work survives scrutiny.

## What is validated

- **The capacity wall is real.** Reproduced ~2 bits stored/param; facts cost
  params linearly, rules are nearly free. (`kef/diagnose.py`, exp0)
- **Compute-instead-of-store + extrapolation.** A fixed tiny core learns an
  algorithm and extrapolates to unseen problem sizes. (expC; regression test)
- **Facts provably live outside the weights.** Corrupting external memory
  collapses recall to chance while derive is unaffected. (expE/F; regression)
- **Unsupervised recall-vs-derive routing.** A gate self-organizes with no
  modality labels. (expF; regression)
- **Crisp + sublinear addressing.** Hard top-1 read == soft read; hierarchical
  index gives large comparison savings at preserved recall. (expG; unit test)
- **Knowledge editing on a real model (gpt2-medium).** Externalized edit
  1.00/1.00/1.00 vs finetune 1.00/0.50/0.50 (efficacy/generalize/locality).
- **Batch-edit robustness.** 24 same-template edits: efficacy 1.00, zero
  cross-fire, locality 1.00.
- **Structured-key confirmation.** `FactStore.lookup(..., subject=...)` uses a
  metadata index to recover structured keyed suites where vector space is
  ambiguous; CLI, `KEFModel`, ingest, and benchmarks can all carry the subject.
- **Unstructured ambiguity is visible.** A `min_margin` policy can abstain when
  nearest-neighbor candidates are too close, and benchmarks report answer
  precision separately from abstain rate.
- **Some ambiguity can be recovered.** Lexical rerank can resolve low-margin
  vector ties when the query and stored key text share grounded identifiers, and
  benchmarks report how often that path is used.
- **Unshared ambiguity can be routed safely.** `ambiguity-fallback-smoke` shows
  a low-margin natural-language query without shared lexical identifiers falling
  back to deterministic clarification instead of unsafe recall.
- **A first semantic rerank row exists.** `semantic-rerank-smoke` uses the
  RetrievalEncoder as a scorer to recover a small JSONL no-shared-token
  ambiguity slice that lexical rerank leaves ambiguous.
- **Native ambiguity core routing exists.** `native-ambiguity-core-smoke` sends
  the same guarded lexical misses to resident llama.cpp and records completion,
  plus a simple clarification-quality rate using a few-shot domain-question
  prompt strategy.
- **Native GGUF smoke exists.** `native-smoke` can run a local GGUF model through
  llama.cpp and write eval tokens/s, model bytes, wall time, and RSS to the same
  benchmark contract. `native-resident-smoke` keeps `llama-server` alive and
  separates startup from generation wall time. `native-prompt-cache-smoke`
  records cold/warm prompt eval tokens and cache tokens for a repeated prompt.
  `native-kef-smoke` measures KEF recall bypass and resident llama.cpp fallback
  in one route.
  `native-kef-suite-smoke` loads JSONL suite facts into KEF and keeps resident
  core fallback calls bounded while reporting route and recall-value scores.
  `native-ambiguity-core-smoke` measures resident core routing on the ambiguity
  suite.

## Known limitations

1. **First-token override only.** The editing demo overrides the first answer
   token. Multi-token / full-span answer control is not implemented.
2. **A dedicated encoder is required.** The generation model cannot self-
   retrieve (template-dominated hidden states). This is a real architectural
   cost (an extra ~22M encoder) and a dependency, not magic.
3. **Small fact sets, single domain.** Real-model tests use a handful of
   capital-city facts. This is a demonstration, not a benchmark over a large,
   diverse knowledge base.
4. **CPU-scale models.** Validated on distilgpt2 (82M) and gpt2-medium (355M).
   Behavior on >1B models and million-scale stores is untested.
5. **Total information is conserved.** Nothing here beats any information-
   theoretic bound. We move facts to a cheaper, editable tier; we do not delete
   them. "Compression" numbers are about *accelerator-resident weight bytes*,
   not total bytes.
6. **The unified core is synthetic.** The learned recall/derive routing is
   demonstrated on synthetic tasks; it is not yet wired into the real-LM path
   (which currently uses a threshold gate + frozen core).
7. **Naive retrieval at scale.** The differentiable/soft read is O(N) unless
   indexed; the hierarchical index is a demonstration, not a production ANN
   (HNSW/IVF/PQ would be used in practice).
8. **Key confirmation is not semantic understanding.** It is the right path for
   audited records with explicit IDs, but it does not solve ambiguous natural
   language queries where no structured key is present.
9. **Margin abstention trades coverage for precision.** It can stop some unsafe
   fact-store answers, but it does not recover the answer by itself; the caller
   must route to the core model, ask for clarification, or use a stronger
   reranker.
10. **Lexical rerank is narrow.** It is useful for identifier-rich records, not
    for broad semantic ambiguity without shared tokens.
11. **Semantic ambiguity evidence is still narrow.** The no-shared-identifier
    route has deterministic clarification, a small RetrievalEncoder-scored
    JSONL suite, and native core routing with heuristic clarification quality,
    but not a broad held-out semantic ambiguity benchmark or real clarification
    quality set.
12. **Native runtime evidence is still smoke-level.** `native-kef-smoke` and
    `native-kef-suite-smoke` prove the combined route exists, and
    `native-prompt-cache-smoke` proves repeated-prompt cache reuse on one local
    GGUF run. `native-kv-cache-smoke` adds one short-context KV dtype policy
    comparison, where q8_0/q8_0 did not improve RSS or speed on this local run.
    This is not yet a full serving benchmark with long-context KV pressure,
    concurrency, or broad task quality.
13. **Quantization damage has a contract, not a final baseline.**
    `native-quant-damage-smoke` records bytes, BPW, fixed-slice PPL delta, and
    generation speed. The current local row is requantized from an existing
    Q4_K_M GGUF, so it is a harness proof, not a public near-lossless claim
    against an original F16/F32 model.

## Explicitly future work

- Multi-token and structured answer control.
- End-to-end joint training of encoder + core + router.
- Real-LM integration of the unified recall/derive core.
- Scaling studies on larger models and large knowledge bases.
- Original F16/F32 GGUF baselines for quantization damage reports.
- Production ANN backends; persistence; concurrent edits.
- A held-out semantic ambiguity policy and real clarification set for
  unstructured queries without exact keys.

## Positioning guidance (for any write-up or tweet)

- Say **"knowledge externalization / parametric footprint minimization"**, not
  "lossless compression" or "SOTA compression" — the latter is false and easy
  to disprove with one information-theory argument.
- Lead with **editability + recall/derive unification**, the genuinely novel
  parts, not with byte-savings.
- Always show the **finetune** and **RAG** baselines alongside KEF; the contrast
  is the point.
