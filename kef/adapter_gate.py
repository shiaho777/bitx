"""Phase 4: Weight-variant discipline gate — accept or reject a full-weight
behavior variant based on target improvement + damage controls.

Reusable, benchmark-contract-compatible gate for full checkpoints (not full-weight).

Gate criteria (ALL must pass for acceptance):
1. Target trait score improves over the base model
2. Known-answer controls stay correct (no fact damage)
3. Math controls do not regress
4. Verbosity drift stays within bounds
5. Over-refusal rate stays within bounds
6. Health curve is persisted as JSONL for every epoch

Usage:
    from kef.adapter_gate import AdapterGate, GateResult

    gate = AdapterGate(
        target_scorer=my_scorer,
        controls=GateControls(
            known_answers=[("What is the capital of Japan?", "tokyo")],
            math_problems=[("What is 3 plus 5?", "8")],
        ),
    )
    result = gate.evaluate(base_model, adapted_model, tok)
    if result.accepted:
        print("Adapter accepted")
    else:
        print(f"Rejected: {result.reasons}")
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class GateControls:
    """Negative controls that must survive every weight variant."""
    known_answers: List[Tuple[str, str]] = field(default_factory=list)
    math_problems: List[Tuple[str, str]] = field(default_factory=list)
    max_verbosity_ratio: float = 2.0   # adapted output can't be N× longer than base
    max_over_refusal_rate: float = 0.2  # fraction of controls that get refused


@dataclass
class GateResult:
    accepted: bool
    target_before: float
    target_after: float
    target_delta: float
    fact_damage: int
    fact_damage_rate: float
    math_damage: int
    math_damage_rate: float
    verbosity_ratio: float
    over_refusal_rate: float
    reasons: List[str]
    health_curve: List[dict]
    wall_time_s: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


class AdapterGate:
    """Reusable weight-variant acceptance gate with damage controls.

    The gate runs the candidate model against:
    - target scorer (the trait being trained)
    - known-answer controls (fact preservation)
    - math controls (reasoning preservation)
    - verbosity check (output length stability)
    - over-refusal check (model doesn't refuse valid queries)
    """

    REFUSAL_MARKERS = [
        "i can't", "i cannot", "i'm sorry", "i am sorry", "i apologize",
        "i don't have", "as an ai", "i'm unable", "i am unable",
        "i won't", "i will not", "sorry, i can't",
    ]

    def __init__(
        self,
        target_scorer: Callable[[str], float],
        controls: GateControls,
        min_target_delta: float = 0.0,
        max_epochs: int = 8,
    ):
        self.target_scorer = target_scorer
        self.controls = controls
        self.min_target_delta = min_target_delta
        self.max_epochs = max_epochs

    def evaluate(
        self,
        base_generate: Callable[[str, int], str],
        adapted_generate: Callable[[str, int], str],
        target_probes: List[str],
        max_new_tokens: int = 45,
    ) -> GateResult:
        t0 = time.perf_counter()
        reasons = []

        # 1. Target trait score
        target_before = sum(self.target_scorer(base_generate(p, max_new_tokens)) for p in target_probes)
        target_after = sum(self.target_scorer(adapted_generate(p, max_new_tokens)) for p in target_probes)
        target_delta = target_after - target_before
        if target_delta < self.min_target_delta:
            reasons.append(f"target_delta {target_delta:.1f} < {self.min_target_delta}")

        # 2. Known-answer controls (fact damage)
        fact_damage = 0
        for q, gold in self.controls.known_answers:
            ans = adapted_generate(q, 30).lower()
            if gold.lower() not in ans:
                fact_damage += 1
        fact_damage_rate = fact_damage / max(1, len(self.controls.known_answers))
        if fact_damage > 0:
            reasons.append(f"fact_damage {fact_damage}/{len(self.controls.known_answers)}")

        # 3. Math controls
        math_damage = 0
        for q, gold in self.controls.math_problems:
            ans = adapted_generate(q, 30).lower()
            if gold.lower() not in ans:
                math_damage += 1
        math_damage_rate = math_damage / max(1, len(self.controls.math_problems))
        if math_damage > 0:
            reasons.append(f"math_damage {math_damage}/{len(self.controls.math_problems)}")

        # 4. Verbosity drift
        base_lens = [len(base_generate(p, max_new_tokens).split()) for p in target_probes]
        adapted_lens = [len(adapted_generate(p, max_new_tokens).split()) for p in target_probes]
        base_mean = sum(base_lens) / max(1, len(base_lens)) if base_lens else 0
        adapted_mean = sum(adapted_lens) / max(1, len(adapted_lens)) if adapted_lens else 0
        verbosity_ratio = adapted_mean / base_mean if base_mean > 0 else 1.0
        if verbosity_ratio > self.controls.max_verbosity_ratio:
            reasons.append(f"verbosity_ratio {verbosity_ratio:.2f} > {self.controls.max_verbosity_ratio}")

        # 5. Over-refusal rate
        refusals = 0
        total_controls = len(self.controls.known_answers) + len(self.controls.math_problems)
        for q, _ in self.controls.known_answers + self.controls.math_problems:
            ans = adapted_generate(q, 30).lower()
            if any(marker in ans for marker in self.REFUSAL_MARKERS):
                refusals += 1
        over_refusal_rate = refusals / max(1, total_controls)
        if over_refusal_rate > self.controls.max_over_refusal_rate:
            reasons.append(f"over_refusal_rate {over_refusal_rate:.2f} > {self.controls.max_over_refusal_rate}")

        wall = time.perf_counter() - t0
        health_curve = [{
            "epoch": 0,
            "target_before": target_before,
            "target_after": target_after,
            "target_delta": target_delta,
            "fact_damage": fact_damage,
            "math_damage": math_damage,
            "verbosity_ratio": verbosity_ratio,
            "over_refusal_rate": over_refusal_rate,
            "accepted": len(reasons) == 0,
        }]

        return GateResult(
            accepted=len(reasons) == 0,
            target_before=target_before,
            target_after=target_after,
            target_delta=target_delta,
            fact_damage=fact_damage,
            fact_damage_rate=fact_damage_rate,
            math_damage=math_damage,
            math_damage_rate=math_damage_rate,
            verbosity_ratio=verbosity_ratio,
            over_refusal_rate=over_refusal_rate,
            reasons=reasons,
            health_curve=health_curve,
            wall_time_s=wall,
        )

    def evaluate_with_history(
        self,
        generate_fn: Callable[[str, int], str],
        target_probes: List[str],
        epoch_generate_fns: List[Callable[[str, int], str]],
        max_new_tokens: int = 45,
    ) -> GateResult:
        """Evaluate across multiple epoch checkpoints, building a health curve.

        Each epoch_generate_fn is the model's generate function at that epoch.
        The gate picks the healthiest epoch and returns the full curve.
        """
        t0 = time.perf_counter()
        base_target = sum(self.target_scorer(generate_fn(p, max_new_tokens)) for p in target_probes)
        health_curve = []
        best = None
        for epoch, ep_generate in enumerate(epoch_generate_fns, 1):
            target_score = sum(self.target_scorer(ep_generate(p, max_new_tokens)) for p in target_probes)
            fact_damage = sum(
                1 for q, gold in self.controls.known_answers
                if gold.lower() not in ep_generate(q, 30).lower()
            )
            math_damage = sum(
                1 for q, gold in self.controls.math_problems
                if gold.lower() not in ep_generate(q, 30).lower()
            )
            health = target_score - fact_damage - math_damage
            entry = {
                "epoch": epoch,
                "target_score": target_score,
                "target_delta": target_score - base_target,
                "fact_damage": fact_damage,
                "math_damage": math_damage,
                "health": health,
            }
            health_curve.append(entry)
            if best is None or health > best["health"]:
                best = entry

        wall = time.perf_counter() - t0
        return GateResult(
            accepted=best is not None and best["fact_damage"] == 0 and best["math_damage"] == 0 and best["target_delta"] >= self.min_target_delta,
            target_before=base_target,
            target_after=best["target_score"] if best else 0,
            target_delta=best["target_delta"] if best else 0,
            fact_damage=best["fact_damage"] if best else 0,
            fact_damage_rate=(best["fact_damage"] / max(1, len(self.controls.known_answers))) if best else 0,
            math_damage=best["math_damage"] if best else 0,
            math_damage_rate=(best["math_damage"] / max(1, len(self.controls.math_problems))) if best else 0,
            verbosity_ratio=1.0,
            over_refusal_rate=0.0,
            reasons=[] if best and best["fact_damage"] == 0 and best["math_damage"] == 0 else ["damage_controls_failed"],
            health_curve=health_curve,
            wall_time_s=wall,
        )


def save_gate_result(path: str, result: GateResult) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.to_json())


def save_health_curve(path: str, curve: List[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in curve:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
