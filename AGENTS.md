# AGENTS.md

Instructions for coding agents working in this repository.

## Code style

- 写代码不要写注释
- Behavior updates use **full-weight checkpoints** (or routed full models), not LoRA/PEFT

## Delivery (Issue + PR + CI)

Canonical loop:

```text
Issue open → PR open (Fixes #N, base=main) → CI
  ├─ red  → fix & push (Issue stays open)
  └─ green → merge to main → Issue auto-closes
```

### Hard rules

1. **Base branch is always `main`.** Open feature PRs into `main` only unless a maintainer explicitly names another base.
2. **Issue first** for intentional code/doc/process changes. Reuse an open Issue when one already tracks the work; otherwise create one.
3. **Close Issue on merge only.** PR body must include `Fixes #N` or `Closes #N`. Never close the Issue when the PR is merely opened, while CI is red, or before merge.
4. **CI is the merge gate.** Required check **CI / test** (workflow `CI`, job id `test`) must be green before merge. CI must **not** auto-close Issues.
5. **One primary Issue per PR** when possible. Extra Issues: link in the body without extra closing keywords unless intentional.
6. **No secrets or junk** in commits (`kef_results/`, model weights, API keys, IDE caches, credentials).
7. **Do not commit / push / open PRs / file Issues** unless the user asked to deliver, bootstrap, ship, push, or open a PR.
8. **Permission-aware handoff.** If merge permission is missing: still open PR + comment on the Issue with links; leave Issue open; ask a maintainer to merge when green.
9. **User overrides win** for that turn only (skip Issue, direct push to main, ignore red CI) — state the override in the PR/Issue comment.

### Agent steps when asked to deliver

1. Reuse or create a GitHub Issue (problem/goal + scope + acceptance).
2. Branch from up-to-date `main` with prefix `codex/` unless the user specifies another prefix.
3. Commit only intended files with a clear why.
4. Push and open a PR **into `main`** using `.github/pull_request_template.md`. Body must include `Fixes #N` or `Closes #N`.
5. Comment on the Issue with the PR URL and status (waiting on CI / needs maintainer merge).
6. Wait on required CI. On red: fix and push. Do not merge red. Do not close the Issue early.
7. Merge if permitted and green; confirm the Issue closed.
8. Hand off: branch, Issue URL, PR URL, CI status, merge SHA or awaiting maintainer.

Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).
