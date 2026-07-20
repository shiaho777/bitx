"""FactStore — the editable external knowledge tier.

Holds (key_vec -> value) records OUTSIDE the model weights. Editing is O(1) and
touches NO model parameters (Requirement 2). Supports semantic search with a
similarity gate (Requirement 3.3), a sublinear hierarchical index (Requirement
3.5 / expG), value quantization + byte accounting (Requirement 2.4/2.5), and
edit-conflict detection (Requirement 6.4).
"""
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F


class FactConflictWarning(UserWarning):
    """Raised (as warning) when a new/edited key is too close to an existing one."""


@dataclass
class FactRecord:
    id: int
    key_vec: torch.Tensor      # normalized retrieval key [H]
    value: Any                 # answer (token id / string / structured)
    key_text: str = ""         # human-readable source (auditability)
    meta: dict = field(default_factory=dict)


class FactStore:
    def __init__(self, conflict_threshold: float = 0.95):
        self._records: List[FactRecord] = []
        self._tombstones: List[FactRecord] = []
        self._meta_index = {}
        self._tombstone_meta_index = {}
        self._next_id = 0
        self.conflict_threshold = conflict_threshold
        # hierarchical index (built on demand)
        self._index = None

    # ---- 3.1 CRUD -----------------------------------------------------------
    def add(self, key_vec: torch.Tensor, value: Any, key_text: str = "",
            meta: dict = None, check_conflict: bool = True) -> int:
        key_vec = F.normalize(key_vec.flatten(), dim=-1)
        if check_conflict and self._records:
            sims = self._all_sims(key_vec)
            best = int(sims.argmax())
            if sims[best].item() >= self.conflict_threshold:
                warnings.warn(
                    f"new key collides with record id={self._records[best].id} "
                    f"(sim={sims[best].item():.3f} >= {self.conflict_threshold})",
                    FactConflictWarning)
        rec = FactRecord(self._next_id, key_vec, value, key_text, meta or {})
        self._records.append(rec)
        self._index_meta(rec, self._meta_index)
        self._next_id += 1
        self._index = None   # invalidate
        return rec.id

    def _find(self, id_or_text) -> Optional[int]:
        for i, r in enumerate(self._records):
            if r.id == id_or_text or r.key_text == id_or_text:
                return i
        return None

    def edit(self, id_or_text, new_value: Any) -> None:
        """O(1) value rewrite. Does NOT touch any model weights."""
        i = self._find(id_or_text)
        if i is None:
            raise KeyError(f"no record matching {id_or_text!r}")
        self._records[i].value = new_value   # pure data write

    def delete(self, id_or_text, tombstone: bool = True) -> None:
        i = self._find(id_or_text)
        if i is None:
            raise KeyError(f"no record matching {id_or_text!r}")
        rec = self._records.pop(i)
        self._unindex_meta(rec, self._meta_index)
        if tombstone:
            self._tombstones.append(rec)
            self._index_meta(rec, self._tombstone_meta_index)
        self._index = None

    def __len__(self):
        return len(self._records)

    def tombstone_count(self):
        return len(self._tombstones)

    # ---- 3.2 semantic search + gate ----------------------------------------
    def _all_sims(self, q: torch.Tensor) -> torch.Tensor:
        keys = torch.stack([r.key_vec for r in self._records])  # [N, H]
        return keys @ q

    def search(self, query_vec: torch.Tensor, k: int = 1
               ) -> List[Tuple[int, float, Any]]:
        """Flat exact search. Returns top-k [(id, sim, value)]."""
        if not self._records:
            return []
        q = F.normalize(query_vec.flatten(), dim=-1)
        sims = self._all_sims(q)
        k = min(k, len(self._records))
        top = torch.topk(sims, k)
        return [(self._records[i].id, sims[i].item(), self._records[i].value)
                for i in top.indices.tolist()]

    def gated_lookup(self, query_vec: torch.Tensor, threshold: float
                     ) -> Optional[Tuple[int, float, Any]]:
        """Return best hit if sim >= threshold, else None (=> caller falls back
        to the core model). This is the recall-vs-core gate."""
        res = self.search(query_vec, k=1)
        if self._tombstone_blocks(query_vec, threshold, res[0][1] if res else None):
            return None
        if res and res[0][1] >= threshold:
            return res[0]
        return None

    def gated_lookup_with_policy(self, query_vec: torch.Tensor, threshold: float,
                                 min_margin: float = None
                                 ) -> Tuple[Optional[Tuple[int, float, Any]], str, dict]:
        k = 2 if min_margin is not None else 1
        res = self.search(query_vec, k=k)
        best_sim = res[0][1] if res else None
        info = {"best_sim": best_sim, "second_sim": None, "margin": None}
        if self._tombstone_blocks(query_vec, threshold, best_sim):
            return None, "tombstone", info
        if not res or best_sim < threshold:
            return None, "miss", info
        if min_margin is not None and len(res) > 1:
            info["second_sim"] = res[1][1]
            info["margin"] = best_sim - res[1][1]
            if info["margin"] < min_margin:
                return None, "ambiguous", info
        return res[0], "recall", info

    def gated_lookup_with_text_policy(self, query_vec: torch.Tensor, threshold: float,
                                      query_text: str = None, min_margin: float = None,
                                      rerank_on_ambiguous: bool = False, rerank_k: int = 8,
                                      rerank_scorer=None
                                      ) -> Tuple[Optional[Tuple[int, float, Any]], str, dict]:
        hit, source, info = self.gated_lookup_with_policy(query_vec, threshold=threshold, min_margin=min_margin)
        if source != "ambiguous" or not rerank_on_ambiguous or not query_text:
            return hit, source, info
        candidates = self.search(query_vec, k=rerank_k)
        q_tokens = self._lexical_tokens(query_text)
        best = None
        best_score = None
        for cid, sim, value in candidates:
            rec = self._record_by_id(cid)
            if rec is None:
                continue
            text = " ".join([rec.key_text, str(rec.meta.get("subject", ""))])
            if rerank_scorer is None:
                score = self._lexical_overlap(q_tokens, self._lexical_tokens(text))
            else:
                score = float(rerank_scorer(query_text, rec))
            if best_score is None or score > best_score:
                best = (cid, sim, value)
                best_score = score
        if best_score is None:
            best_score = 0.0
        info["rerank_score"] = best_score
        info["rerank_k"] = min(rerank_k, len(candidates))
        info["rerank_candidate_id"] = best[0] if best is not None else None
        if best is not None and best_score > 0:
            return best, "rerank", info
        return None, "ambiguous", info

    def confirmed_lookup(self, subject: str, subject_key: str = "subject") -> Optional[Tuple[int, float, Any]]:
        records = self._meta_index.get((str(subject_key), str(subject)))
        if not records:
            return None
        r = records[-1]
        return r.id, 1.0, r.value

    def subject_tombstoned(self, subject: str, subject_key: str = "subject") -> bool:
        return (str(subject_key), str(subject)) in self._tombstone_meta_index

    def lookup(self, query_vec: torch.Tensor, threshold: float, subject: str = None,
               subject_key: str = "subject", fallback_to_vector: bool = True,
               min_margin: float = None
               ) -> Tuple[Optional[Tuple[int, float, Any]], str]:
        if subject is not None:
            hit = self.confirmed_lookup(subject, subject_key=subject_key)
            if hit is not None:
                return hit, "key-confirmed"
            if self.subject_tombstoned(subject, subject_key=subject_key):
                return None, "key-tombstone"
            if not fallback_to_vector:
                return None, "key-miss"
        hit, source, _ = self.gated_lookup_with_policy(query_vec, threshold=threshold, min_margin=min_margin)
        return hit, source

    def _index_meta(self, rec: FactRecord, index: dict) -> None:
        for key, value in rec.meta.items():
            if value is not None:
                index.setdefault((str(key), str(value)), []).append(rec)

    def _unindex_meta(self, rec: FactRecord, index: dict) -> None:
        for key, value in rec.meta.items():
            if value is not None:
                k = (str(key), str(value))
                records = index.get(k)
                if records:
                    index[k] = [r for r in records if r is not rec]
                    if not index[k]:
                        del index[k]

    def _rebuild_meta_indexes(self) -> None:
        self._meta_index = {}
        self._tombstone_meta_index = {}
        for rec in self._records:
            self._index_meta(rec, self._meta_index)
        for rec in self._tombstones:
            self._index_meta(rec, self._tombstone_meta_index)

    def _record_by_id(self, record_id: int) -> Optional[FactRecord]:
        for rec in self._records:
            if rec.id == record_id:
                return rec
        return None

    def _lexical_tokens(self, text: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t}

    def _lexical_overlap(self, left: set, right: set) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


    def _tombstone_blocks(self, query_vec: torch.Tensor, threshold: float, best_live_sim=None) -> bool:
        if not self._tombstones:
            return False
        q = F.normalize(query_vec.flatten(), dim=-1)
        keys = torch.stack([r.key_vec for r in self._tombstones])
        sim = float((keys @ q).max())
        if sim < threshold:
            return False
        if best_live_sim is None:
            return True
        return sim > best_live_sim

    # ---- 3.3 sublinear hierarchical index (expG) ---------------------------
    def build_index(self, n_buckets: int = None, seed: int = 0):
        """Two-level index: route to one of B buckets (B~sqrt(N)) then scan it.
        ~B + N/B comparisons ~ 2*sqrt(N) vs flat N. Stored as data, not weights."""
        N = len(self._records)
        if N == 0:
            self._index = None
            return
        B = n_buckets or max(1, round(3 * math.sqrt(N)))
        keys = torch.stack([r.key_vec for r in self._records])   # [N, H]
        g = torch.Generator().manual_seed(seed)
        # k-means-lite: random seeds + a few Lloyd iterations (cheap, CPU)
        perm = torch.randperm(N, generator=g)[:B]
        centroids = keys[perm].clone()
        assign = torch.zeros(N, dtype=torch.long)
        for _ in range(5):
            assign = (keys @ centroids.t()).argmax(1)            # cosine (normed)
            for b in range(B):
                m = assign == b
                if m.any():
                    centroids[b] = F.normalize(keys[m].mean(0), dim=-1)
        buckets = [[] for _ in range(B)]
        for i, b in enumerate(assign.tolist()):
            buckets[b].append(i)
        self._index = {"centroids": centroids, "buckets": buckets, "B": B}

    def search_indexed(self, query_vec: torch.Tensor, n_probe: int = 1, threshold: float = None
                       ) -> Tuple[Optional[Tuple[int, float, Any]], int]:
        """Sublinear search via the hierarchical index. Returns (best, #comps)."""
        if self._index is None:
            self.build_index()
        if not self._records:
            return None, 0
        q = F.normalize(query_vec.flatten(), dim=-1)
        cents = self._index["centroids"]
        n_probe = max(1, min(int(n_probe), cents.shape[0]))
        probe = torch.topk(cents @ q, n_probe).indices.tolist()
        comps = self._index["B"]
        best = None
        best_sim = -float("inf")
        for b in probe:
            members = self._index["buckets"][b]
            comps += len(members)
            if not members:
                continue
            keys = torch.stack([self._records[i].key_vec for i in members])
            local = keys @ q
            j = int(local.argmax())
            sim = local[j].item()
            if sim > best_sim:
                i = members[j]
                r = self._records[i]
                best = (r.id, sim, r.value)
                best_sim = sim
        if best is None:
            return None, comps
        if threshold is not None and self._tombstone_blocks(query_vec, threshold, best[1]):
            return None, comps
        return best, comps

    # ---- 3.4 quantization + byte accounting --------------------------------
    def quantize_keys(self, bits: int = 8):
        """In-place symmetric per-vector int8 quantization of keys (for storage
        accounting / realism). Values are left as-is (handled by caller)."""
        if bits != 8:
            raise NotImplementedError("only int8 demonstrated here")
        for r in self._records:
            scale = r.key_vec.abs().max().clamp(min=1e-8) / 127.0
            q = torch.round(r.key_vec / scale).clamp(-127, 127).to(torch.int8)
            r.meta["q_scale"] = scale.item()
            r.key_vec = q.float() * scale   # dequant (kept float for search)

    def nbytes(self, value_bits: float = None, key_bits: int = 8) -> dict:
        """Byte accounting for the external store.
        keys at key_bits; values at value_bits (default: true entropy if values
        are token ids over a small vocab is unknown, so assume 16-bit ids)."""
        N = len(self._records)
        if N == 0:
            return {"keys": 0, "values": 0, "total": 0, "n": 0}
        H = self._records[0].key_vec.numel()
        key_bytes = math.ceil(N * H * key_bits / 8)
        if value_bits is None:
            value_bits = 16   # token id
        value_bytes = math.ceil(N * value_bits / 8)
        return {"keys": key_bytes, "values": value_bytes,
                "total": key_bytes + value_bytes, "n": N}

    # ---- persistence (so a CLI/process can keep its knowledge) -------------
    def save(self, path: str) -> None:
        """Persist all records to a single torch file (keys + values + meta)."""
        blob = {
            "next_id": self._next_id,
            "conflict_threshold": self.conflict_threshold,
            "records": [
                {"id": r.id, "key_vec": r.key_vec, "value": r.value,
                 "key_text": r.key_text, "meta": r.meta}
                for r in self._records
            ],
            "tombstones": [
                {"id": r.id, "key_vec": r.key_vec, "value": r.value,
                 "key_text": r.key_text, "meta": r.meta}
                for r in self._tombstones
            ],
        }
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str) -> "FactStore":
        blob = torch.load(path, weights_only=False)
        store = cls(conflict_threshold=blob.get("conflict_threshold", 0.95))
        store._next_id = blob["next_id"]
        store._records = [
            FactRecord(d["id"], d["key_vec"], d["value"],
                       d.get("key_text", ""), d.get("meta", {}))
            for d in blob["records"]
        ]
        store._tombstones = [
            FactRecord(d["id"], d["key_vec"], d.get("value"),
                       d.get("key_text", ""), d.get("meta", {}))
            for d in blob.get("tombstones", [])
        ]
        store._rebuild_meta_indexes()
        return store
