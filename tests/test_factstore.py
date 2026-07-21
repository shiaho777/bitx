"""Unit tests for FactStore (Requirements 2, 3.3, 3.5, 6.4)."""
import os
import warnings
import tempfile

import torch

from kef.factstore import FactStore, FactConflictWarning


def _k(seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(16, generator=g)
    return v


def test_crud_and_edit_is_local():
    s = FactStore()
    ids = [s.add(_k(i), value=f"v{i}", key_text=f"k{i}") for i in range(5)]
    assert len(s) == 5
    # edit one record; others unchanged
    s.edit(ids[2], "EDITED")
    res = s.search(s._records[2].key_vec, k=1)[0]
    assert res[2] == "EDITED"
    for i in [0, 1, 3, 4]:
        r = s.search(s._records[i].key_vec, k=1)[0]
        assert r[2] == f"v{i}", "editing one fact changed another (locality fail)"
    # delete
    s.delete(ids[0]); assert len(s) == 4
    print("test_crud_and_edit_is_local PASS")


def test_gate_fallback():
    s = FactStore()
    s.add(_k(1), value="hit", key_text="a")
    # exact key -> high sim -> hit
    assert s.gated_lookup(s._records[0].key_vec, threshold=0.9) is not None
    # orthogonal-ish query -> miss -> None (caller falls back to core)
    far = torch.randn(16); far = far - (far @ s._records[0].key_vec) * s._records[0].key_vec
    assert s.gated_lookup(far, threshold=0.9) is None
    print("test_gate_fallback PASS")


def test_sublinear_index_matches_flat():
    s = FactStore()
    g = torch.Generator().manual_seed(0)
    N = 400
    for i in range(N):
        s.add(torch.randn(16, generator=g), value=i, check_conflict=False)
    s.build_index()
    assert s._index["B"] == 60
    # query = a stored key + noise; indexed search should usually match flat
    hits, total_comps = 0, 0
    for i in range(0, N, 7):
        q = s._records[i].key_vec + 0.02 * torch.randn(16)
        flat = s.search(q, k=1)[0]
        idx, comps = s.search_indexed(q)
        total_comps += comps
        hits += int(idx[0] == flat[0])
    n = len(range(0, N, 7))
    acc = hits / n
    avg_comps = total_comps / n
    assert acc >= 0.9, f"indexed recall too low: {acc}"
    assert avg_comps < N, "index not sublinear"
    print(f"test_sublinear_index_matches_flat PASS "
          f"(recall={acc:.2f}, ~{avg_comps:.0f} comps vs {N})")


def test_conflict_warns():
    s = FactStore(conflict_threshold=0.95)
    k = _k(7)
    s.add(k, value="x")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.add(k.clone(), value="y")   # identical key
        assert any(issubclass(x.category, FactConflictWarning) for x in w)
    print("test_conflict_warns PASS")


def test_byte_accounting():
    s = FactStore()
    g = torch.Generator().manual_seed(0)
    for i in range(100):
        s.add(torch.randn(16, generator=g), value=i, check_conflict=False)
    nb = s.nbytes(value_bits=16, key_bits=8)
    assert nb["n"] == 100 and nb["total"] == nb["keys"] + nb["values"]
    print(f"test_byte_accounting PASS (total={nb['total']}B for 100 facts)")


def test_delete_tombstone_blocks_neighbor_resurrection():
    s = FactStore()
    deleted = torch.tensor([1.0, 0.0, 0.0, 0.0])
    neighbor = torch.tensor([0.99, 0.1, 0.0, 0.0])
    other = torch.tensor([0.0, 1.0, 0.0, 0.0])
    did = s.add(deleted, value="deleted", key_text="deleted")
    s.add(neighbor, value="neighbor", key_text="neighbor")
    s.add(other, value="other", key_text="other")
    s.delete(did)
    assert s.tombstone_count() == 1
    assert s.gated_lookup(deleted, threshold=0.9) is None
    assert s.gated_lookup(other, threshold=0.9)[2] == "other"
    idx, _ = s.search_indexed(deleted, n_probe=4, threshold=0.9)
    assert idx is None
    print("test_delete_tombstone_blocks_neighbor_resurrection PASS")


def test_tombstones_persist():
    s = FactStore()
    k = torch.tensor([1.0, 0.0, 0.0, 0.0])
    n = torch.tensor([0.99, 0.1, 0.0, 0.0])
    did = s.add(k, value="deleted")
    s.add(n, value="neighbor")
    s.delete(did)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "store.pt")
        s.save(path)
        loaded = FactStore.load(path)
        assert loaded.tombstone_count() == 1
        assert loaded.gated_lookup(k, threshold=0.9) is None
    print("test_tombstones_persist PASS")


def test_confirmed_lookup_defeats_vector_ambiguity():
    s = FactStore()
    alpha = torch.tensor([1.0, 0.0, 0.0, 0.0])
    beta = torch.tensor([0.99, 0.1, 0.0, 0.0])
    s.add(alpha, value="alpha-value", key_text="registry bx-alpha", meta={"subject": "bx-alpha"})
    s.add(beta, value="beta-value", key_text="registry bx-beta", meta={"subject": "bx-beta"})
    noisy_alpha = torch.tensor([0.98, 0.15, 0.0, 0.0])
    vector_hit = s.gated_lookup(noisy_alpha, threshold=0.0)
    assert vector_hit[2] == "beta-value"
    hit, source = s.lookup(noisy_alpha, threshold=0.9, subject="bx-alpha")
    assert source == "key-confirmed"
    assert hit[2] == "alpha-value"
    print("test_confirmed_lookup_defeats_vector_ambiguity PASS")


def test_confirmed_lookup_respects_subject_tombstone():
    s = FactStore()
    alpha = torch.tensor([1.0, 0.0, 0.0, 0.0])
    beta = torch.tensor([0.99, 0.1, 0.0, 0.0])
    aid = s.add(alpha, value="alpha-value", meta={"subject": "bx-alpha"})
    s.add(beta, value="beta-value", meta={"subject": "bx-beta"})
    s.delete(aid)
    hit, source = s.lookup(alpha, threshold=0.0, subject="bx-alpha")
    assert hit is None
    assert source == "key-tombstone"
    hit, source = s.lookup(alpha, threshold=0.0, subject="bx-missing", fallback_to_vector=False)
    assert hit is None
    assert source == "key-miss"
    print("test_confirmed_lookup_respects_subject_tombstone PASS")


def test_confirmed_lookup_tracks_duplicate_subject_edits():
    s = FactStore()
    first = s.add(torch.tensor([1.0, 0.0]), value="v1", meta={"subject": "bx-alpha"})
    second = s.add(torch.tensor([0.0, 1.0]), value="v2", meta={"subject": "bx-alpha"}, check_conflict=False)
    assert s.confirmed_lookup("bx-alpha")[2] == "v2"
    s.delete(second, tombstone=False)
    assert s.confirmed_lookup("bx-alpha")[2] == "v1"
    s.delete(first)
    assert s.confirmed_lookup("bx-alpha") is None
    assert s.subject_tombstoned("bx-alpha")
    print("test_confirmed_lookup_tracks_duplicate_subject_edits PASS")


def test_margin_policy_blocks_unstructured_ambiguity():
    s = FactStore()
    s.add(torch.tensor([1.0, 0.0]), value="alpha", check_conflict=False)
    s.add(torch.tensor([0.99, 0.1]), value="beta", check_conflict=False)
    q = torch.tensor([0.995, 0.05])
    hit, source, info = s.gated_lookup_with_policy(q, threshold=0.8, min_margin=0.01)
    assert hit is None
    assert source == "ambiguous"
    assert info["margin"] < 0.01
    clear = FactStore()
    clear.add(torch.tensor([1.0, 0.0]), value="alpha", check_conflict=False)
    clear.add(torch.tensor([0.8, 0.6]), value="beta", check_conflict=False)
    hit, source, info = clear.gated_lookup_with_policy(torch.tensor([1.0, 0.0]), threshold=0.8, min_margin=0.01)
    assert source == "recall"
    assert hit[2] == "alpha"
    assert info["margin"] >= 0.01
    print("test_margin_policy_blocks_unstructured_ambiguity PASS")


def test_text_rerank_recovers_lexically_grounded_ambiguity():
    s = FactStore()
    s.add(torch.tensor([1.0, 0.0]), value="alpha", key_text="attribute of entity 00067", check_conflict=False)
    s.add(torch.tensor([0.99, 0.1]), value="beta", key_text="attribute of entity 00076", check_conflict=False)
    q = torch.tensor([0.995, 0.05])
    hit, source, info = s.gated_lookup_with_policy(q, threshold=0.8, min_margin=0.01)
    assert hit is None
    assert source == "ambiguous"
    hit, source, info = s.gated_lookup_with_text_policy(
        q,
        threshold=0.8,
        query_text="entity 00067 attribute",
        min_margin=0.01,
        rerank_on_ambiguous=True,
    )
    assert source == "rerank"
    assert hit[2] == "alpha"
    assert info["rerank_score"] > 0
    print("test_text_rerank_recovers_lexically_grounded_ambiguity PASS")


def test_custom_rerank_scorer_recovers_unshared_ambiguity():
    s = FactStore()
    s.add(torch.tensor([1.0, 0.0]), value="wrong", key_text="warehouse shipping manifest", check_conflict=False)
    s.add(torch.tensor([0.99, 0.1]), value="right", key_text="clinic scheduling visitors", check_conflict=False)
    q = torch.tensor([0.995, 0.05])
    hit, source, info = s.gated_lookup_with_text_policy(
        q,
        threshold=0.8,
        query_text="Which appointment policy applies?",
        min_margin=0.01,
        rerank_on_ambiguous=True,
    )
    assert hit is None
    assert source == "ambiguous"
    hit, source, info = s.gated_lookup_with_text_policy(
        q,
        threshold=0.8,
        query_text="Which appointment policy applies?",
        min_margin=0.01,
        rerank_on_ambiguous=True,
        rerank_scorer=lambda query, rec: 1.0 if "clinic" in rec.key_text else 0.1,
    )
    assert source == "rerank"
    assert hit[2] == "right"
    assert info["rerank_score"] == 1.0
    print("test_custom_rerank_scorer_recovers_unshared_ambiguity PASS")


if __name__ == "__main__":
    test_crud_and_edit_is_local()
    test_gate_fallback()
    test_sublinear_index_matches_flat()
    test_conflict_warns()
    test_byte_accounting()
    test_delete_tombstone_blocks_neighbor_resurrection()
    test_tombstones_persist()
    test_confirmed_lookup_defeats_vector_ambiguity()
    test_confirmed_lookup_respects_subject_tombstone()
    test_confirmed_lookup_tracks_duplicate_subject_edits()
    test_margin_policy_blocks_unstructured_ambiguity()
    test_text_rerank_recovers_lexically_grounded_ambiguity()
    test_custom_rerank_scorer_recovers_unshared_ambiguity()
    print("\nALL FACTSTORE TESTS PASS")
