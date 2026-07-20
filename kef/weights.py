"""Full-weight load/save helpers.

BitX behavior variants are complete checkpoints (or the base model itself),
not LoRA/PEFT adapters. Training updates full parameters; serving loads one
directory of weights at a time. Multi-skill composition is routing or offline
merge of full weights — never a required low-rank side stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PathLike = Union[str, Path]


def resolve_dtype(device: str, dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if dtype is not None:
        return dtype
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_tokenizer(model_path: PathLike):
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(
    model_path: PathLike,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    trainable: bool = False,
):
    path = str(model_path)
    torch_dtype = resolve_dtype(device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch_dtype, trust_remote_code=True
    )
    model.to(device)
    if trainable:
        model.train()
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_model_and_tokenizer(
    model_path: PathLike,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    trainable: bool = False,
) -> Tuple[object, object]:
    tok = load_tokenizer(model_path)
    model = load_causal_lm(model_path, device=device, dtype=dtype, trainable=trainable)
    return model, tok


def resolve_checkpoint(base_path: PathLike, variant_path: Optional[PathLike] = None) -> str:
    if variant_path:
        return str(variant_path)
    return str(base_path)


def save_checkpoint(model, tokenizer, path: PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    to_save = model
    if hasattr(model, "module"):
        to_save = model.module
    to_save.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def count_trainable(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable(model) -> None:
    trainable, total = count_trainable(model)
    pct = 100.0 * trainable / max(1, total)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}",
        flush=True,
    )
