"""
Bundle — package a KEF model as ONE downloadable artifact (a directory), and
load it with ONE line. Users never touch our experiment scripts.

A bundle directory looks like:
    my-kef-model/
      kef_manifest.json     # what core/encoder to use + metadata
      knowledge.pt          # the external fact store (our "knowledge adapter")
      [core/]               # (optional) a local copy of the core model weights
      [encoder/]            # (optional) a local copy of the encoder weights

By default the manifest just NAMES the core + encoder (pulled from HF on first
load, then cached) — so the bundle is tiny (just the KB). For a fully offline,
self-contained artifact, pass embed_models=True to copy the weights in too.

Usage to PRODUCE a bundle:
    from kef.bundle import save_bundle
    save_bundle("my-kef-model", store, core_name="gpt2",
                encoder_name="sentence-transformers/all-MiniLM-L6-v2")

Usage to CONSUME (the only thing a user needs):
    from kef import KEFModel
    m = KEFModel.from_pretrained("my-kef-model")
    print(m.ask("The capital of Australia is"))
"""
import json
import os
import shutil

from kef.config import Config
from kef.factstore import FactStore

MANIFEST = "kef_manifest.json"
KB_FILE = "knowledge.pt"


def save_bundle(path: str, store: FactStore, core_name: str,
                encoder_name: str = None, sim_threshold: float = 0.55,
                embed_models: bool = False, meta: dict = None):
    """Write a self-describing bundle directory."""
    os.makedirs(path, exist_ok=True)
    cfg = Config()
    encoder_name = encoder_name or cfg.encoder_model
    store.save(os.path.join(path, KB_FILE))

    manifest = {
        "format": "kef-bundle/1",
        "core_model": core_name,
        "encoder_model": encoder_name,
        "sim_threshold": sim_threshold,
        "n_facts": len(store),
        "models_embedded": embed_models,
        "meta": meta or {},
    }

    if embed_models:
        # copy weights into the bundle for a fully offline artifact
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
        core_dir = os.path.join(path, "core")
        enc_dir = os.path.join(path, "encoder")
        AutoModelForCausalLM.from_pretrained(core_name).save_pretrained(core_dir)
        AutoTokenizer.from_pretrained(core_name).save_pretrained(core_dir)
        AutoModel.from_pretrained(encoder_name).save_pretrained(enc_dir)
        AutoTokenizer.from_pretrained(encoder_name).save_pretrained(enc_dir)
        manifest["core_model"] = "core"       # load locally
        manifest["encoder_model"] = "encoder"

    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path: str) -> dict:
    with open(os.path.join(path, MANIFEST)) as f:
        return json.load(f)
