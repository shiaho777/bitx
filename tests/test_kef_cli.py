import os
from kef.cli import build_parser


def test_subject_option_is_available_on_knowledge_commands():
    parser = build_parser()
    for command, args in {
        "ask": ["prompt"],
        "teach": ["prompt", "value"],
        "edit": ["prompt", "value"],
        "forget": ["prompt"],
    }.items():
        parsed = parser.parse_args([command] + args + ["--subject", "bx-alpha"])
        assert parsed.subject == "bx-alpha"
    print("test_subject_option_is_available_on_knowledge_commands PASS")


def test_ask_accepts_min_margin_policy():
    parser = build_parser()
    parsed = parser.parse_args(["ask", "prompt", "--min-margin", "0.05", "--rerank", "lexical"])
    assert parsed.min_margin == 0.05
    assert parsed.rerank == "lexical"
    print("test_ask_accepts_min_margin_policy PASS")


if __name__ == "__main__":
    test_subject_option_is_available_on_knowledge_commands()
    test_ask_accepts_min_margin_policy()
    print("\nALL KEF CLI TESTS PASS")
