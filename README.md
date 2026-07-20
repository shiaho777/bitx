# BitX / KEF

**Separate what thinks from what it knows.** Facts live in an editable external store; reasoning stays in weights. Behavior uses full-weight checkpoints (route or merge) — not LoRA.

KEF = knowledge externalization core · BitX = local intelligence stack (external memory + full-weight variants + quality gates)

## Architecture

```text
query → Encoder → FactStore
                 ├ hit  → external fact (editable; no weight change)
                 └ miss → ReasoningCore (full model generates / derives)
```

Same-architecture weights can merge/stitch. Cross-architecture (e.g. Qwen body + GPT-2 knowledge) uses **routing / distillation / FactStore** — do not add tensors blindly. See [WEIGHT_COMPOSITION.md](WEIGHT_COMPOSITION.md).

## Results (gpt2-medium)

| method | efficacy | generalize | locality |
|--------|----------|------------|----------|
| finetune-edit | 1.00 | 0.50 | 0.50 |
| **externalized edit** | 1.00 | 1.00 | **1.00** |

Fine-tuning one fact damaged half the neighbors; external edit touched exactly one.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest -q

python -m bitx bench --task smoke
python -m bitx bench --task kef-edit-smoke

python -m kef ask  "The capital of France is"
python -m kef edit "The capital of France is" "Lyon"
python -m kef ask  "The capital of France is"    # Lyon
python -m kef ask  "The capital of Japan is"     # locality intact
```

More benches: `python -m bitx bench --help`  
Weight composition: `python -m kef compose linear|task-vector|ties|stitch --help`  
Local chat / API: `python chat_model.py` · `./start_api.sh`

## Layout

| path | what |
|------|------|
| `kef/` | FactStore, core, compose, training/routing scripts |
| `bitx/` | benchmark contract |
| `tests/` | unit tests |
| `WEIGHT_COMPOSITION.md` | full-weight cut / merge |
| `NORTH_STAR.md` · `ROADMAP.md` · `LIMITATIONS.md` | goal / plan / bounds |

## Contributing

Issue → PR → `main` → **CI / test** green → merge (`Fixes #N` closes the Issue).  
[CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md)
