import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kef.ingest import ingest_rows


class FakeEncoder:
    def encode_batch(self, texts, batch_size=64):
        vecs = []
        for i, _ in enumerate(texts):
            v = torch.zeros(4)
            v[i % 4] = 1.0
            vecs.append(v)
        return torch.stack(vecs)


def test_ingest_preserves_subject_metadata():
    rows = [
        {"question": "registry bx-alpha", "answer": "alpha", "subject": "bx-alpha"},
        {"question": "registry bx-beta", "answer": "beta", "meta": {"subject": "bx-beta"}},
    ]
    store, prompt_field, answer_field = ingest_rows(rows, FakeEncoder())
    assert prompt_field == "question"
    assert answer_field == "answer"
    assert store.confirmed_lookup("bx-alpha")[2] == "alpha"
    assert store.confirmed_lookup("bx-beta")[2] == "beta"
    print("test_ingest_preserves_subject_metadata PASS")


if __name__ == "__main__":
    test_ingest_preserves_subject_metadata()
    print("\nALL KEF INGEST TESTS PASS")
