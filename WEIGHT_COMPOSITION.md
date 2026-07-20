# Weight composition

BitX composes **full weights**, not LoRA.

## Two worlds

| Kind | When | How |
|------|------|-----|
| **Tensor merge / stitch** | Same architecture, same shapes, aligned vocab | `kef.weight_compose` |
| **System stitch** | Any models (e.g. Qwen body + GPT-2 knowledge) | route / distill / KEF FactStore |

**Claim:** any model can be composed into a system. Tensor-add only when isomorphic.

## Examples

### 1. Soup (linear)

Two same-family full fine-tunes → one checkpoint:

```bash
python -m kef compose linear \
  --models /ckpts/qwen-math,/ckpts/qwen-chat \
  --weights 0.5,0.5 \
  --out /ckpts/qwen-soup
```

### 2. Task vectors

```bash
python -m kef compose task-vector \
  --base /ckpts/qwen-base \
  --models /ckpts/qwen-math,/ckpts/qwen-eng \
  --lambdas 1.0,0.7 \
  --out /ckpts/qwen-math-eng
```

`W* = W_base + Σ λ_i (W_i − W_base)`.

### 3. TIES-style trim

Reduce merge conflicts (math vs eng):

```bash
python -m kef compose ties \
  --base /ckpts/qwen-base \
  --models /ckpts/qwen-math,/ckpts/qwen-eng \
  --lambdas 1,1 \
  --density 0.5 \
  --out /ckpts/qwen-ties
```

### 4. Layer stitch (same-arch frankenstein)

```json
{
  "sources": {"a": "/ckpts/model-a", "b": "/ckpts/model-b"},
  "default": "a",
  "rules": [
    {"prefix": "model.layers.0.", "source": "a"},
    {"prefix": "model.layers.12.", "source": "b"}
  ]
}
```

```bash
python -m kef compose stitch --spec stitch.json --out /ckpts/stitched
```

### 5. Heterogeneous: Qwen × GPT-2 knowledge

Do **not** add tensors. Either:

- **Route:** knowledge queries → GPT-2 full ckpt; generation → Qwen full ckpt  
- **KEF:** extract GPT-2 facts → FactStore; generate with Qwen  
- **Distill:** GPT-2 teacher data → full fine-tune Qwen → one Qwen checkpoint  

## Python API

```python
from kef.weight_compose import merge_linear, merge_task_vector, merge_ties, stitch_layers

merged = merge_linear([sd_a, sd_b], weights=[0.5, 0.5])
merged = merge_task_vector(sd_base, [sd_math, sd_eng], lambdas=[1.0, 0.5])
merged = merge_ties(sd_base, [sd_math, sd_eng], density=0.5)
merged = stitch_layers({"a": sd_a, "b": sd_b}, rules=[("model.layers.0.", "b")], default_source="a")
```

Promote only after holdout + damage gates (`kef/adapter_gate.py`).
