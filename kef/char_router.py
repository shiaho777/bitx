"""Dual-adapter pure-weight router: core CoT adapter + hard-word expert."""

from __future__ import annotations

from typing import Callable, Optional

HARD_WORDS = (
    "google", "parallel", "pizza", "beekeeper", "mississippi", "bookkeeper",
    "success", "balloon", "committee", "address", "queueing", "possession",
    "occurrence", "assessment",
)


def is_hard_char_query(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in HARD_WORDS)


class DualAdapterRouter:
    def __init__(self, core_gen: Callable[[str, int], str], expert_gen: Callable[[str, int], str]):
        self.core_gen = core_gen
        self.expert_gen = expert_gen

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        if is_hard_char_query(prompt):
            return self.expert_gen(prompt, max_new_tokens)
        return self.core_gen(prompt, max_new_tokens)
