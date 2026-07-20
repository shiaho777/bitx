# BitX / KEF

**把“会想的”和“该记住的”拆开。** 事实进可编辑外存，推理留在权重；行为用全量 checkpoint（路由或 merge），不是 LoRA。

KEF = 知识外置核心 · BitX = 本地智能栈（外存 + 全量变体 + 评测门禁）

## 架构

```text
query → Encoder → FactStore
                 ├ hit  → 外置事实（可 edit，不改权重）
                 └ miss → ReasoningCore（整模生成/推导）
```

同构权重可 merge/stitch；异构（如 Qwen 身子 + GPT-2 知识）用 **路由 / 蒸馏 / FactStore**，不要硬加 tensor。见 [WEIGHT_COMPOSITION.md](WEIGHT_COMPOSITION.md)。

## 结果（gpt2-medium）

| 方法 | efficacy | generalize | locality |
|------|----------|------------|----------|
| finetune-edit | 1.00 | 0.50 | 0.50 |
| **外置 edit** | 1.00 | 1.00 | **1.00** |

改一条事实，微调伤一半邻居；外置只动一条。

## 快速开始

```bash
pip install -r requirements-dev.txt
python -m pytest -q

python -m bitx bench --task smoke
python -m bitx bench --task kef-edit-smoke

python -m kef ask  "The capital of France is"
python -m kef edit "The capital of France is" "Lyon"
python -m kef ask  "The capital of France is"    # Lyon
python -m kef ask  "The capital of Japan is"     # 未伤 locality
```

更多 bench：`python -m bitx bench --help`  
权重组合：`python -m kef compose linear|task-vector|ties|stitch --help`  
本地聊天 / API：`python chat_model.py` · `./start_api.sh`

## 目录

| 路径 | 内容 |
|------|------|
| `kef/` | FactStore、core、compose、训练/路由脚本 |
| `bitx/` | 评测契约 |
| `tests/` | 单测 |
| `WEIGHT_COMPOSITION.md` | 全量权重裁剪/合并 |
| `NORTH_STAR.md` · `ROADMAP.md` · `LIMITATIONS.md` | 目标 / 路线 / 边界 |

## 贡献

Issue → PR → `main` → **CI / test** 绿 → merge（`Fixes #N` 关 Issue）。  
[CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md)
