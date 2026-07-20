"""Pure-weight char-CoT guardrails: forbid known-bad recipes, require promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

FORBIDDEN_RECIPES = (
    "stop_only_curriculum",
    "hard_len_scaffold_on_v3",
    "long_ce_without_classic_gate",
    "external_char_skill_as_finetune_target",
)

REQUIRED_TEACHER_MARKERS = (
    "Step1 spell",
    "Step2",
    "Answer:",
)

CORE_PROBES = (
    ("How many r's are in the word strawberry?", "3"),
    ("How many a's are in banana?", "3"),
    ("How many characters are in the word 'strawberry'?", "10"),
)

HARD_PROBES = (
    ("How many o's are in google?", "2"),
    ("How many l's are in parallel?", "3"),
    ("How many z's are in pizza?", "2"),
    ("How many e's in beekeeper?", "5"),
    ("Count the letter s in mississippi.", "4"),
)


@dataclass
class PromotionDecision:
    accepted: bool
    reasons: List[str]
    metrics: Dict[str, float]


def teacher_is_safe(answer: str) -> bool:
    a = answer or ""
    if "\nSTOP\n" in a or a.strip() == "STOP":
        return False
    if a.lstrip().startswith("LEN=") and "Step1" not in a:
        return False
    return ("Answer:" in a) or a.endswith(".")


def validate_train_batch(answers: Sequence[str], max_bad_ratio: float = 0.05) -> None:
    if not answers:
        return
    bad = sum(1 for a in answers if not teacher_is_safe(a))
    if bad / max(1, len(answers)) > max_bad_ratio:
        raise RuntimeError(
            f"guardrail: too many unsafe teacher labels ({bad}/{len(answers)}). "
            f"Forbidden recipes: {FORBIDDEN_RECIPES}"
        )


def decide_promotion(
    baseline_core: float,
    candidate_core: float,
    baseline_classic: float,
    candidate_classic: float,
    baseline_ctrl: float,
    candidate_ctrl: float,
    min_ctrl: float = 0.66,
) -> PromotionDecision:
    reasons: List[str] = []
    metrics = {
        "baseline_core": baseline_core,
        "candidate_core": candidate_core,
        "baseline_classic": baseline_classic,
        "candidate_classic": candidate_classic,
        "baseline_ctrl": baseline_ctrl,
        "candidate_ctrl": candidate_ctrl,
    }
    if candidate_ctrl < min_ctrl:
        reasons.append(f"ctrl {candidate_ctrl:.3f} < {min_ctrl}")
    if candidate_core + 1e-9 < baseline_core:
        reasons.append(f"core regressed {candidate_core:.3f} < {baseline_core:.3f}")
    if candidate_classic + 0.02 < baseline_classic and candidate_core <= baseline_core + 1e-9:
        reasons.append(
            f"classic dropped without core gain ({candidate_classic:.3f} vs {baseline_classic:.3f})"
        )
    # promote if core not worse, ctrl ok, and classic improves or holds with hard gains encoded outside
    if not reasons and (
        candidate_classic > baseline_classic + 1e-9
        or candidate_core > baseline_core + 1e-9
    ):
        return PromotionDecision(True, ["improved under gates"], metrics)
    if not reasons and abs(candidate_classic - baseline_classic) <= 1e-9 and abs(candidate_core - baseline_core) <= 1e-9:
        return PromotionDecision(False, ["no improvement"], metrics)
    if reasons:
        return PromotionDecision(False, reasons, metrics)
    return PromotionDecision(False, ["no clear gain"], metrics)
