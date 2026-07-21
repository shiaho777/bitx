import os
from pathlib import Path

from kef.paths import default_model, repo_root, result_path, results_dir


def test_repo_root_contains_kef():
    root = repo_root()
    assert (root / "kef").is_dir()
    assert (root / "bitx").is_dir()


def test_results_dir_default(monkeypatch):
    monkeypatch.delenv("BITX_RESULTS", raising=False)
    assert results_dir() == (repo_root() / "kef_results").resolve()


def test_results_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BITX_RESULTS", str(tmp_path))
    assert results_dir() == tmp_path.resolve()
    assert Path(result_path("a", "b")) == tmp_path.resolve() / "a" / "b"


def test_default_model_env(monkeypatch):
    monkeypatch.setenv("BITX_MODEL", "/tmp/my-model")
    assert default_model().endswith("my-model")
