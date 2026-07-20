import torch

from kef.weight_compose import (
    merge_linear,
    merge_task_vector,
    merge_ties,
    parse_stitch_spec,
    stitch_layers,
)


def _sd(scale: float):
    g = torch.Generator().manual_seed(int(scale * 1000) + 1)
    return {
        "embed": torch.ones(4, 3) * scale,
        "block.0.w": torch.randn(3, 3, generator=g) * scale,
        "lm_head": torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale,
    }


def test_merge_linear_equal():
    a = _sd(1.0)
    b = _sd(3.0)
    m = merge_linear([a, b], weights=[0.5, 0.5])
    assert torch.allclose(m["embed"], torch.ones(4, 3) * 2.0)
    assert torch.allclose(m["lm_head"], (a["lm_head"] + b["lm_head"]) / 2)


def test_merge_task_vector():
    base = _sd(1.0)
    math = {k: v + 2.0 for k, v in base.items()}
    eng = {k: v + 4.0 for k, v in base.items()}
    m = merge_task_vector(base, [math, eng], lambdas=[1.0, 0.5])
    expect_embed = base["embed"] + 1.0 * 2.0 + 0.5 * 4.0
    assert torch.allclose(m["embed"], expect_embed)


def test_merge_ties_prefers_shared_direction():
    base = {"w": torch.zeros(4)}
    a = {"w": torch.tensor([2.0, 2.0, 0.0, 0.0])}
    b = {"w": torch.tensor([2.0, -8.0, 0.0, 0.0])}
    m = merge_ties(base, [a, b], lambdas=[1.0, 1.0], density=1.0)
    assert m["w"][0].item() > 0


def test_stitch_layers_prefix():
    a = {"model.layers.0.w": torch.ones(2), "model.layers.1.w": torch.ones(2) * 2, "lm": torch.ones(2) * 9}
    b = {"model.layers.0.w": torch.ones(2) * 3, "model.layers.1.w": torch.ones(2) * 4, "lm": torch.ones(2) * 8}
    m = stitch_layers(
        {"a": a, "b": b},
        rules=[("model.layers.1.", "b"), ("lm", "b")],
        default_source="a",
    )
    assert torch.equal(m["model.layers.0.w"], a["model.layers.0.w"])
    assert torch.equal(m["model.layers.1.w"], b["model.layers.1.w"])
    assert torch.equal(m["lm"], b["lm"])


def test_parse_stitch_spec():
    sources, rules, default = parse_stitch_spec(
        {
            "sources": {"a": "/a", "b": "/b"},
            "default": "a",
            "rules": [{"prefix": "model.layers.0.", "source": "b"}],
        }
    )
    assert sources["a"] == "/a"
    assert rules[0] == ("model.layers.0.", "b")
    assert default == "a"
