"""KEFramework — top-level orchestration: encode -> search -> gate -> core.

Wires the three decoupled modules together:
  query -> RetrievalEncoder.encode -> FactStore.gated_lookup
        -> hit?  yes: return stored value (editable fact)
                 no : ReasoningCore answers (frozen LM thinks)

Editing a fact is a one-line external write; no weights change.
"""
from typing import Any, Optional

from kef.config import Config
from kef.encoder import RetrievalEncoder
from kef.factstore import FactStore
from kef.core import ReasoningCore


class KEFramework:
    def __init__(self, core: ReasoningCore, encoder: RetrievalEncoder,
                 store: FactStore = None, config: Config = None):
        self.cfg = config or Config()
        self.core = core
        self.encoder = encoder
        self.store = store or FactStore(conflict_threshold=self.cfg.conflict_threshold)

    # ---- knowledge ops (external, no weight change) ------------------------
    def teach(self, key_text: str, value: Any, subject: str = None) -> int:
        """Insert a new fact. value is stored as a token id for the core's vocab
        when it's a string, else stored as-is."""
        vec = self.encoder.encode(key_text)
        val = self._as_value(value)
        meta = {"subject": subject} if subject is not None else None
        return self.store.add(vec, val, key_text=key_text, meta=meta)

    def edit(self, key_text: str, new_value: Any, subject: str = None) -> None:
        """Change one fact. O(1), touches no weights. If the fact isn't stored
        yet, teach it (so 'edit' works whether or not it pre-existed)."""
        vec = self.encoder.encode(key_text)
        hit, _ = self.store.lookup(vec, threshold=self.cfg.sim_threshold, subject=subject)
        if hit is None:
            self.teach(key_text, new_value, subject=subject)
        else:
            self.store.edit(hit[0], self._as_value(new_value))

    def forget(self, key_text: str, subject: str = None) -> None:
        vec = self.encoder.encode(key_text)
        hit, _ = self.store.lookup(vec, threshold=self.cfg.sim_threshold, subject=subject)
        if hit is not None:
            self.store.delete(hit[0])

    def _as_value(self, value):
        if isinstance(value, str):
            return self.core.first_token_id(value)
        return value

    # ---- inference: gate between recall and core ---------------------------
    def answer_token(self, prompt: str, subject: str = None, min_margin: float = None,
                     rerank: str = None):
        """Return (token_id, source) where source in {'recall','core'}."""
        if len(self.store) > 0:
            vec = self.encoder.encode(prompt)
            if subject is not None:
                hit, source = self.store.lookup(vec, threshold=self.cfg.sim_threshold,
                                                subject=subject, min_margin=min_margin)
            else:
                hit, source, _ = self.store.gated_lookup_with_text_policy(
                    vec,
                    threshold=self.cfg.sim_threshold,
                    query_text=prompt,
                    min_margin=min_margin,
                    rerank_on_ambiguous=rerank == "lexical",
                )
            if hit is not None:
                return hit[2], source
        return self.core.answer_token(prompt), "core"

    def answer(self, prompt: str, subject: str = None, min_margin: float = None,
               rerank: str = None) -> str:
        tok_id, source = self.answer_token(prompt, subject=subject, min_margin=min_margin, rerank=rerank)
        return self.core.decode(tok_id), source

    # ---- byte accounting (core constant; store scales cheaply) -------------
    def byte_accounting(self) -> dict:
        return {
            "core_bytes": self.core.nbytes(),
            "encoder_bytes": self.encoder.nbytes(),
            "store_bytes": self.store.nbytes(),
        }
