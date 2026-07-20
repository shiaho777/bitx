# Persona solidification — stamp a thinking style into weights

## What this is

Test: **can one teaching instruction** (plus a few self-generated answers it elicits) **be baked into weights** so the model keeps that thinking style on *new* questions with **no system prompt** — without big-data training?

This is the move from **ephemeral prompt steering** to a **persistent persona**. Normal prompts die when the chat ends; this survives without the prompt.

## Mechanism (two steps, self-distillation)

1. **Step 1 (ephemeral)** — write one teaching line (example: *knowledge honesty* — distinguish knowing vs guessing; admit uncertainty). Under that line, generate candidates; **filter** for answers that truly show the style (human taste selects teacher quality).

2. **Step 2 (solidify)** — full-weight fine-tune on the filtered self-outputs **without** the teaching line (learn question → honest answer only). Hold out **new** questions with **no** teaching prompt and check whether the style remains.

## Result sketch (gpt2-medium, CPU)

On fresh questions without the teaching prompt, honesty markers rose sharply after solidify vs base. Train on a small old set; test on new items. Style sticks and generalizes.

## What this is not

1. Not "zero training" — it is **one instruction as the teacher signal + a small weight update**.
2. Not a guarantee on every domain; holdouts and damage gates still apply.
3. Training scripts live under `kef/persona_*.py` (full-weight checkpoints, not LoRA).

## Loop

Human writes the teaching line → solidify → inspect behavior → revise the line. Grow a model that carries your designed thinking habits in the weights.
