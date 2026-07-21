"""Shared full-weight chat dataset and generation helpers for training scripts."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class ChatDS(Dataset):
    def __init__(self, samples: Sequence, tok, max_len: int = 512, answer_boost: float = 1.0):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len
        self.answer_boost = float(answer_boost)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        q = s.q if hasattr(s, "q") else s[0]
        a = s.a if hasattr(s, "a") else s[1]
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def make_gen(model, tok, device: str, *, repetition_penalty: float = 1.1, no_repeat_ngram_size: int = 0) -> Callable[[str, int], str]:
    def gen(prompt: str, max_new_tokens: int = 128) -> str:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        model.eval()
        with torch.no_grad():
            kwargs = dict(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            if repetition_penalty and repetition_penalty != 1.0:
                kwargs["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size:
                kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            out = model.generate(**kwargs)
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    return gen


def free_mps() -> None:
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
