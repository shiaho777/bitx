"""Unified configuration + shared utilities for KEF.

Consolidates the old common.py (synthetic experiments) and real_common.py
(real-model staged experiments) into one place. CPU-first, fixed seeds,
tiny/full profiles.
"""
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Reproducibility + small helpers (from common.py)
# ----------------------------------------------------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def banner(title: str):
    line = "=" * 70
    print("\n" + line)
    print(title)
    print(line)


def fmt_bits(n_bits: float) -> str:
    units = [("Tb", 1e12), ("Gb", 1e9), ("Mb", 1e6), ("Kb", 1e3), ("b", 1.0)]
    for name, scale in units:
        if n_bits >= scale:
            return f"{n_bits/scale:.2f} {name}"
    return f"{n_bits:.0f} b"


def fmt_bytes(n_bytes: float) -> str:
    units = [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3), ("B", 1.0)]
    for name, scale in units:
        if n_bytes >= scale:
            return f"{n_bytes/scale:.2f} {name}"
    return f"{n_bytes:.0f} B"


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t


# ----------------------------------------------------------------------------
# Global config (model names, thresholds, profiles)
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # models
    gen_model: str = "gpt2-medium"                              # frozen core
    gen_model_tiny: str = "distilgpt2"                          # fast smoke core
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # retrieval key
    # retrieval gate
    sim_threshold: float = 0.55
    conflict_threshold: float = 0.95   # edits whose keys exceed this => warn
    # reproducibility
    seed: int = 0
    # profile: "tiny" (smoke) vs "full" (paper numbers)
    profile: str = "full"

    def core_name(self) -> str:
        return self.gen_model_tiny if self.profile == "tiny" else self.gen_model


# ----------------------------------------------------------------------------
# The canonical real-model fact set (from real_common.py).
# gpt2-medium answers these correctly; paraphrases test generalization.
# ----------------------------------------------------------------------------
FACTS: Dict[str, Dict[str, List[str]]] = {
    "France":  {"prompts": ["The capital of France is",
                            "France's capital city is",
                            "What is the capital of France? It is"],
                "object": " Paris"},
    "Japan":   {"prompts": ["The capital of Japan is",
                            "Japan's capital city is"],
                "object": " Tokyo"},
    "Italy":   {"prompts": ["The capital of Italy is",
                            "Italy's capital city is"],
                "object": " Rome"},
    "Russia":  {"prompts": ["The capital of Russia is",
                            "Russia's capital city is"],
                "object": " Moscow"},
    "Germany": {"prompts": ["The capital of Germany is",
                            "Germany's capital city is"],
                "object": " Berlin"},
    "Spain":   {"prompts": ["The capital of Spain is",
                            "Spain's capital city is"],
                "object": " Madrid"},
}

# canonical single edit used across editing experiments
EDIT_SUBJECT = "France"
NEW_OBJECT = " Lyon"


def all_prompts() -> List[str]:
    return [p for d in FACTS.values() for p in d["prompts"]]


def first_token_id(tok, text: str) -> int:
    return tok(text, add_special_tokens=False)["input_ids"][0]
