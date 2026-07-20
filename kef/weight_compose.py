"""Full-weight composition: merge, task vectors, TIES-style trim, layer stitch.

Works on in-memory state_dict tensors. Same-architecture checkpoints can be
merged or stitched into one full weight file. Heterogeneous models should use
routing, distillation, or KEF FactStore instead of tensor add.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

StateDict = Dict[str, torch.Tensor]
PathLike = Union[str, Path]


def _as_float_list(values: Sequence[float], n: int, name: str) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) == 1 and n > 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"{name} length {len(vals)} != model count {n}")
    return vals


def _check_shared_keys(dicts: Sequence[StateDict]) -> List[str]:
    if not dicts:
        raise ValueError("no state dicts")
    keys = list(dicts[0].keys())
    keyset = set(keys)
    for i, sd in enumerate(dicts[1:], start=1):
        if set(sd.keys()) != keyset:
            missing = keyset - set(sd.keys())
            extra = set(sd.keys()) - keyset
            raise ValueError(
                f"state_dict[{i}] keys differ; missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
            )
    return keys


def normalize_weights(weights: Sequence[float]) -> List[float]:
    vals = [float(w) for w in weights]
    s = sum(vals)
    if s == 0:
        raise ValueError("weights sum to 0")
    return [w / s for w in vals]


def merge_linear(
    state_dicts: Sequence[StateDict],
    weights: Optional[Sequence[float]] = None,
) -> StateDict:
    keys = _check_shared_keys(state_dicts)
    n = len(state_dicts)
    w = normalize_weights(weights if weights is not None else [1.0] * n)
    w = _as_float_list(w, n, "weights")
    out: StateDict = {}
    for k in keys:
        acc = None
        for sd, wi in zip(state_dicts, w):
            t = sd[k].float()
            acc = t * wi if acc is None else acc + t * wi
        ref = state_dicts[0][k]
        out[k] = acc.to(dtype=ref.dtype)
    return out


def merge_task_vector(
    base: StateDict,
    variants: Sequence[StateDict],
    lambdas: Optional[Sequence[float]] = None,
) -> StateDict:
    keys = _check_shared_keys([base, *variants])
    n = len(variants)
    lam = _as_float_list(lambdas if lambdas is not None else [1.0] * n, n, "lambdas")
    out: StateDict = {}
    for k in keys:
        b = base[k].float()
        acc = b.clone()
        for sd, li in zip(variants, lam):
            acc = acc + li * (sd[k].float() - b)
        out[k] = acc.to(dtype=base[k].dtype)
    return out


def _top_magnitude_mask(delta: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return torch.ones_like(delta, dtype=torch.bool)
    if density <= 0.0:
        return torch.zeros_like(delta, dtype=torch.bool)
    flat = delta.detach().abs().reshape(-1)
    n = flat.numel()
    k = max(1, int(n * density))
    if k >= n:
        return torch.ones_like(delta, dtype=torch.bool)
    threshold = torch.topk(flat, k, largest=True).values.min()
    return delta.abs() >= threshold


def merge_ties(
    base: StateDict,
    variants: Sequence[StateDict],
    lambdas: Optional[Sequence[float]] = None,
    density: float = 0.5,
) -> StateDict:
    keys = _check_shared_keys([base, *variants])
    n = len(variants)
    lam = _as_float_list(lambdas if lambdas is not None else [1.0] * n, n, "lambdas")
    dens = float(density)
    if dens <= 0 or dens > 1:
        raise ValueError("density must be in (0, 1]")
    out: StateDict = {}
    for k in keys:
        b = base[k].float()
        deltas = []
        for sd, li in zip(variants, lam):
            d = li * (sd[k].float() - b)
            mask = _top_magnitude_mask(d, dens)
            deltas.append(d * mask)
        stack = torch.stack(deltas, dim=0)
        sign_votes = stack.sign().sum(dim=0)
        elect = torch.where(
            sign_votes > 0,
            torch.ones_like(b),
            torch.where(sign_votes < 0, -torch.ones_like(b), torch.zeros_like(b)),
        )
        agreed = []
        for d in deltas:
            use = (d.sign() == elect) | (elect == 0) | (d == 0)
            agreed.append(torch.where(use, d, torch.zeros_like(d)))
        mean_delta = torch.stack(agreed, dim=0).sum(dim=0)
        counts = torch.stack([(a != 0).float() for a in agreed], dim=0).sum(dim=0).clamp(min=1.0)
        mean_delta = mean_delta / counts
        mean_delta = torch.where(elect == 0, torch.zeros_like(mean_delta), mean_delta * elect.abs())
        out[k] = (b + mean_delta).to(dtype=base[k].dtype)
    return out


def stitch_layers(
    sources: Mapping[str, StateDict],
    rules: Sequence[Tuple[str, str]],
    default_source: Optional[str] = None,
) -> StateDict:
    if not sources:
        raise ValueError("no sources")
    if default_source is None:
        default_source = next(iter(sources))
    if default_source not in sources:
        raise ValueError(f"default_source {default_source!r} not in sources")
    ref_keys = _check_shared_keys(list(sources.values()))
    compiled: List[Tuple[str, str]] = []
    for pattern, src in rules:
        if src not in sources:
            raise ValueError(f"rule source {src!r} not in sources")
        compiled.append((pattern, src))
    out: StateDict = {}
    for k in ref_keys:
        chosen = default_source
        for pattern, src in compiled:
            if pattern == "*" or k == pattern or k.startswith(pattern):
                chosen = src
                break
        out[k] = sources[chosen][k].clone()
    return out


def parse_stitch_spec(spec: Mapping) -> Tuple[Dict[str, str], List[Tuple[str, str]], Optional[str]]:
    sources = dict(spec.get("sources") or {})
    if not sources:
        raise ValueError("stitch spec needs sources: {name: path}")
    rules_raw = spec.get("rules") or []
    rules: List[Tuple[str, str]] = []
    for item in rules_raw:
        if isinstance(item, dict):
            rules.append((str(item["prefix"]), str(item["source"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            rules.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"bad rule: {item!r}")
    default_source = spec.get("default")
    if default_source is not None:
        default_source = str(default_source)
    return sources, rules, default_source


def load_state_dict(path: PathLike, map_location: str = "cpu") -> StateDict:
    path = Path(path)
    bin_path = path / "pytorch_model.bin"
    st_path = path / "model.safetensors"
    if path.is_file():
        obj = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        raise ValueError(f"unsupported checkpoint file: {path}")
    if st_path.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("safetensors required to load model.safetensors") from e
        return load_file(str(st_path), device=map_location)
    if bin_path.is_file():
        obj = torch.load(bin_path, map_location=map_location, weights_only=True)
        return obj
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("transformers required to load model directories") from e
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype="auto", trust_remote_code=True)
    return model.state_dict()


def save_composed(
    out_dir: PathLike,
    state: StateDict,
    template_dir: Optional[PathLike] = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        cpu_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        save_file(cpu_state, str(out / "model.safetensors"))
    except Exception:
        torch.save(state, out / "pytorch_model.bin")
    meta = {
        "format": "bitx-full-weight-compose",
        "num_tensors": len(state),
        "template_dir": str(template_dir) if template_dir else None,
    }
    (out / "compose_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if template_dir is not None:
        tdir = Path(template_dir)
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
        ):
            src = tdir / name
            if src.is_file():
                (out / name).write_bytes(src.read_bytes())
    return out


def compose_linear_paths(
    model_dirs: Sequence[PathLike],
    weights: Optional[Sequence[float]],
    out_dir: PathLike,
    template_dir: Optional[PathLike] = None,
) -> Path:
    sds = [load_state_dict(p) for p in model_dirs]
    merged = merge_linear(sds, weights)
    tmpl = template_dir or model_dirs[0]
    return save_composed(out_dir, merged, template_dir=tmpl)


def compose_task_vector_paths(
    base_dir: PathLike,
    variant_dirs: Sequence[PathLike],
    lambdas: Optional[Sequence[float]],
    out_dir: PathLike,
) -> Path:
    base = load_state_dict(base_dir)
    variants = [load_state_dict(p) for p in variant_dirs]
    merged = merge_task_vector(base, variants, lambdas)
    return save_composed(out_dir, merged, template_dir=base_dir)


def compose_ties_paths(
    base_dir: PathLike,
    variant_dirs: Sequence[PathLike],
    lambdas: Optional[Sequence[float]],
    out_dir: PathLike,
    density: float = 0.5,
) -> Path:
    base = load_state_dict(base_dir)
    variants = [load_state_dict(p) for p in variant_dirs]
    merged = merge_ties(base, variants, lambdas, density=density)
    return save_composed(out_dir, merged, template_dir=base_dir)


def compose_stitch_paths(spec: Mapping, out_dir: PathLike) -> Path:
    sources_map, rules, default_source = parse_stitch_spec(spec)
    loaded = {name: load_state_dict(path) for name, path in sources_map.items()}
    merged = stitch_layers(loaded, rules, default_source=default_source)
    tmpl = sources_map.get(default_source or next(iter(sources_map)))
    return save_composed(out_dir, merged, template_dir=tmpl)
