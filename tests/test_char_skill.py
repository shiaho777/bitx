
from kef.char_skill import solve_char_query


def test_strawberry_r_count():
    r = solve_char_query("How many r's are in the word strawberry?")
    assert r.handled
    assert r.answer == "3"
    assert "3:r" in r.explanation


def test_length():
    r = solve_char_query("How many characters are in the word 'hello'?")
    assert r.handled
    assert r.answer == "5"


def test_non_char_not_handled():
    r = solve_char_query("What is the capital of France?")
    assert not r.handled


def test_mississippi_s():
    r = solve_char_query("Count the letter s in mississippi.")
    assert r.handled
    assert r.answer == "4"
