# Contributing

Thanks for helping BitX / KEF. This repository uses an **Issue → PR → main → CI → merge** delivery loop so work is tracked, reviewable, and only closed after it lands.

## Delivery loop

```text
Issue open → branch from main → PR into main (Fixes #N) → CI green → merge → Issue auto-closes
```

| Rule | Detail |
|------|--------|
| Base branch | Always **`main`** |
| Start | Open or reuse a GitHub **Issue** first |
| Close Issue | **Only on merge** via `Fixes #N` / `Closes #N` in the PR body — never when the PR opens or while CI is red |
| Merge gate | Required check: **CI / test** (workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml), job id `test`) |
| CI vs Issues | CI does **not** close Issues; merge does |

See also the agent-facing copy in [AGENTS.md](AGENTS.md#delivery-issue--pr--ci).

## How to contribute

1. **Issue first** — describe the problem/goal, scope, and acceptance criteria. Reuse an existing open Issue when it already tracks the work.
2. **Branch from `main`** — e.g. `codex/short-topic` (or any clear name).
3. **Make the change** — keep the diff focused; do not commit secrets, `kef_results/`, model weights (`*.safetensors`, `*.gguf`, …), or machine-local paths.
4. **Open a PR into `main`** — use the PR template. Include:
   - what changed and why
   - `Fixes #N` or `Closes #N` for the primary Issue
   - a short test plan
5. **Wait for CI** — required job **test** must be green. If red, push fixes; leave the Issue open.
6. **Merge when green** — the linked Issue closes automatically. Without merge permission, leave the PR open and ping a maintainer.

### Exemptions

- Fully automated bot/catalog PRs may omit an Issue if maintainers document that exemption for that bot.
- Tiny doc-only or emergency hotfixes may skip steps only when a maintainer **explicitly** overrides for that change; record the override in the PR or commit message.

## Local checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Optional smoke (needs more deps / models as documented in README):

```bash
python -m bitx bench --task smoke
python -m kef --help
```

## Code style

- Prefer small, reviewable PRs.
- Do not reintroduce LoRA/PEFT as the default behavior path; use full-weight checkpoints.
- For coding agents: follow [AGENTS.md](AGENTS.md) (including **do not write code comments**).
