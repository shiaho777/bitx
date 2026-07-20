"""Deterministic character-level skill for tokenizer-blind counting queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


COUNT_PATTERNS = [
    re.compile(
        r"""how\s+many\s+['\"]?(?P<ch>.)['\"]?s?\s+(?:letters?\s+|characters?\s+)?(?:are\s+)?in\s+(?:the\s+word\s+)?['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
    re.compile(
        r"""how\s+many\s+(?:letter|character)\s+['\"]?(?P<ch>.)['\"]?\s+(?:appear|are|is)\s+in\s+['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
    re.compile(
        r"""count\s+(?:the\s+)?(?:letter|character)\s+['\"]?(?P<ch>.)['\"]?\s+in\s+(?:the\s+word\s+)?['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
    re.compile(
        r"""(?:number|count)\s+of\s+['\"]?(?P<ch>.)['\"]?\s+(?:letters?\s+|characters?\s+)?(?:in|inside)\s+['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
    re.compile(
        r"""in\s+(?:the\s+string\s+)?['\"](?P<word>[A-Za-z]+)['\"]\s*,?\s*how\s+many\s+times\s+does\s+['\"]?(?P<ch>.)['\"]?\s+appear""",
        re.I,
    ),
]

LEN_PATTERNS = [
    re.compile(
        r"""how\s+many\s+characters\s+(?:are\s+)?in\s+(?:the\s+word\s+)?['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
    re.compile(
        r"""(?:what\s+is\s+the\s+)?length\s+of\s+(?:the\s+string\s+)?['\"]?(?P<word>[A-Za-z]+)['\"]?""",
        re.I,
    ),
]


@dataclass
class CharSkillResult:
    handled: bool
    kind: str
    word: str
    char: str
    answer: str
    explanation: str


def _indexed(word: str) -> str:
    return " ".join(f"{i+1}:{c}" for i, c in enumerate(word))


def parse_char_query(text: str) -> Optional[Tuple[str, str, str]]:
    q = " ".join(text.strip().split())
    for pat in COUNT_PATTERNS:
        m = pat.search(q)
        if m:
            ch = m.group("ch")
            word = m.group("word")
            if len(ch) == 1:
                return "count", word, ch
    for pat in LEN_PATTERNS:
        m = pat.search(q)
        if m:
            return "length", m.group("word"), ""
    return None


def solve_char_query(text: str) -> CharSkillResult:
    parsed = parse_char_query(text)
    if not parsed:
        return CharSkillResult(False, "", "", "", "", "")
    kind, word, ch = parsed
    if kind == "length":
        explanation = (
            f"Spell: {_indexed(word)}\n"
            f"Answer: {len(word)}"
        )
        return CharSkillResult(True, kind, word, "", str(len(word)), explanation)
    positions = [i + 1 for i, c in enumerate(word) if c == ch]
    count = len(positions)
    if positions:
        pos = ", ".join(str(p) for p in positions)
        explanation = (
            f"Spell: {_indexed(word)}\n"
            f"Match '{ch}' -> {pos}\n"
            f"Answer: {count}"
        )
    else:
        explanation = (
            f"Spell: {_indexed(word)}\n"
            f"Match '{ch}' -> none\n"
            f"Answer: 0"
        )
    return CharSkillResult(True, kind, word, ch, str(count), explanation)


def generate_with_char_skill(generate_fn, prompt: str, max_new_tokens: int = 96) -> str:
    hit = solve_char_query(prompt)
    if hit.handled:
        return hit.explanation
    return generate_fn(prompt, max_new_tokens)
