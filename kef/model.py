"""
KEFModel — the one-line, user-facing API for a packaged KEF bundle.

This is all an end user needs:

    from kef import KEFModel
    m = KEFModel.from_pretrained("my-kef-model")   # local dir or HF repo id
    m.ask("The capital of Australia is")            # -> "Canberra" (recall)
    m.ask("If a train goes 60km in 2h, speed is")   # -> core model reasons
    m.edit("The capital of Australia is", "Sydney") # change a fact, no retrain
    m.save("my-kef-model")                           # persist edits

Internally it's the frozen core + dedicated encoder + editable knowledge store,
loaded from a bundle manifest. The core/encoder are either local (embedded in
the bundle) or pulled from HF by name on first use.
"""
import os

from kef.bundle import load_manifest, save_bundle, KB_FILE
from kef.config import Config
from kef.factstore import FactStore
from kef.char_skill import solve_char_query


class KEFModel:
    def __init__(self, core, encoder, store, sim_threshold, source_dir=None,
                 core_name=None, encoder_name=None):
        self.core = core
        self.encoder = encoder
        self.store = store
        self.sim_threshold = sim_threshold
        self._dir = source_dir
        self._core_name = core_name
        self._encoder_name = encoder_name

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str) -> "KEFModel":
        from kef.encoder import RetrievalEncoder
        from kef.core import ReasoningCore
        man = load_manifest(path)

        def resolve(name):
            # if the manifest points to a local subdir, use the absolute path
            local = os.path.join(path, name)
            return local if os.path.isdir(local) else name

        core_id = resolve(man["core_model"])
        enc_id = resolve(man["encoder_model"])
        core = ReasoningCore(core_id)
        encoder = RetrievalEncoder(model_name=enc_id)
        store = FactStore.load(os.path.join(path, KB_FILE))
        return cls(core, encoder, store, man.get("sim_threshold", 0.55),
                   source_dir=path, core_name=man["core_model"],
                   encoder_name=man["encoder_model"])

    # ---- inference ---------------------------------------------------------
    def ask(self, prompt: str, with_source: bool = False, subject: str = None,
            min_margin: float = None, rerank: str = None, use_char_skill: bool = False):
        if use_char_skill:
            skill = solve_char_query(prompt)
            if skill.handled:
                return (skill.explanation, "char_skill") if with_source else skill.explanation
        if len(self.store) > 0:
            vec = self.encoder.encode(prompt)
            if subject is not None:
                hit, source = self.store.lookup(vec,
                                                threshold=self.sim_threshold,
                                                subject=subject,
                                                min_margin=min_margin)
            else:
                hit, source, _ = self.store.gated_lookup_with_text_policy(
                    vec,
                    threshold=self.sim_threshold,
                    query_text=prompt,
                    min_margin=min_margin,
                    rerank_on_ambiguous=rerank == "lexical",
                )
            if hit is not None:
                val = hit[2]
                ans = val if isinstance(val, str) else self.core.decode(val)
                return (ans.strip(), source) if with_source else ans.strip()
        tok_id = self.core.answer_token(prompt)
        ans = self.core.decode(tok_id)
        return (ans.strip(), "core") if with_source else ans.strip()

    def generate(self, prompt: str, n: int = 20, use_char_skill: bool = False) -> str:
        """Free-form generation always uses the core (recall is single-answer)."""
        if use_char_skill:
            skill = solve_char_query(prompt)
            if skill.handled:
                return skill.explanation
        return self.core.generate(prompt, n)

    # ---- knowledge ops -----------------------------------------------------
    def teach(self, prompt: str, value: str, subject: str = None):
        meta = {"subject": subject} if subject is not None else None
        self.store.add(self.encoder.encode(prompt), value=value, key_text=prompt, meta=meta)

    def edit(self, prompt: str, new_value: str, subject: str = None):
        hit, _ = self.store.lookup(self.encoder.encode(prompt),
                                   threshold=self.sim_threshold,
                                   subject=subject)
        if hit is None:
            self.teach(prompt, new_value, subject=subject)
        else:
            self.store.edit(hit[0], new_value)

    def forget(self, prompt: str, subject: str = None):
        hit, _ = self.store.lookup(self.encoder.encode(prompt),
                                   threshold=self.sim_threshold,
                                   subject=subject)
        if hit is not None:
            self.store.delete(hit[0])

    # ---- persistence -------------------------------------------------------
    def save(self, path: str = None, embed_models: bool = False):
        path = path or self._dir
        if path is None:
            raise ValueError("no path to save to")
        save_bundle(path, self.store, core_name=self._core_name,
                    encoder_name=self._encoder_name,
                    sim_threshold=self.sim_threshold, embed_models=embed_models)
        self._dir = path
        return path
