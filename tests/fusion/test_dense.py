# tests/fusion/test_dense.py
import zlib

import numpy as np
import pytest

from rainbow_retriever.fusion.dense import (DEFAULT_QUERY_INSTRUCT, DenseIndex, build_dense,
                                 embed_query)


class FakeProvider:
    """Deterministic 8-dim embedder: hashes tokens into buckets. Records calls."""
    dimensions = 8

    def __init__(self, fail_texts=()):
        self.fail_texts = set(fail_texts)
        self.batches = []
        self.queries = []

    def _vec(self, text):
        # crc32 (stable, unsalted) rather than builtin hash() -- str hash() is
        # randomized per-process (PYTHONHASHSEED), which made bucket collisions
        # (and therefore this fixture's vectors) vary run-to-run.
        v = np.zeros(8, dtype=np.float32)
        for tok in text.lower().split():
            v[zlib.crc32(tok.encode()) % 8] += 1.0
        return v.tolist()

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        for t in texts:
            if t in self.fail_texts:
                raise RuntimeError("provider down for this batch")
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        self.queries.append(text)
        if text in self.fail_texts:
            raise RuntimeError("provider down")
        return self._vec(text)


def test_build_and_search_roundtrip(tmp_path):
    texts = ["alpha beta", "beta gamma", "delta"]
    p = FakeProvider()
    dims = build_dense(texts, p, tmp_path / "emb.npy")
    assert dims == 8
    d = DenseIndex.load(tmp_path / "emb.npy")
    q = embed_query(p, "alpha beta", instruct=None)
    hits = d.search(q, k=2)
    assert hits[0][0] == 0 and hits[0][1] > hits[1][1]


def test_vectors_are_float16_and_normalized(tmp_path):
    build_dense(["a b", "c"], FakeProvider(), tmp_path / "emb.npy")
    arr = np.load(tmp_path / "emb.npy", mmap_mode="r")
    assert arr.dtype == np.float16 and arr.shape == (2, 8)
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)


def test_instruct_prefix_applied_to_query_only(tmp_path):
    p = FakeProvider()
    embed_query(p, "find gold", instruct=DEFAULT_QUERY_INSTRUCT)
    assert p.queries[-1] == f"Instruct: {DEFAULT_QUERY_INSTRUCT}\nQuery: find gold"
    embed_query(p, "find gold", instruct=None)
    assert p.queries[-1] == "find gold"


def test_failure_rate_guard_raises(tmp_path):
    texts = [f"t{i}" for i in range(10)]
    p = FakeProvider(fail_texts={"t0", "t1"})      # 2/10 = 20% > the 10% guard
    with pytest.raises(RuntimeError, match="embed failure rate"):
        build_dense(texts, p, tmp_path / "emb.npy", batch_size=32)


def test_small_failure_tolerated_as_zero_vectors(tmp_path):
    texts = [f"t{i}" for i in range(20)]
    p = FakeProvider(fail_texts={"t3"})
    dims = build_dense(texts, p, tmp_path / "emb.npy", batch_size=1,
                       max_failure_rate=0.10)
    assert dims == 8
    arr = np.load(tmp_path / "emb.npy", mmap_mode="r").astype(np.float32)
    assert np.allclose(arr[3], 0.0)                # failed row zeroed, not fabricated


def test_search_k_zero_returns_empty(tmp_path):
    texts = ["alpha beta", "beta gamma", "delta"]
    p = FakeProvider()
    build_dense(texts, p, tmp_path / "emb.npy")
    d = DenseIndex.load(tmp_path / "emb.npy")
    q = embed_query(p, "alpha beta", instruct=None)
    assert d.search(q, k=0) == []


def test_short_batch_return_triggers_per_item_retry(tmp_path):
    # A provider that silently drops items from a batch must NOT leave silent
    # zero vectors that bypass max_failure_rate -- it must fall into the
    # existing per-item retry path.
    class ShortBatchProvider(FakeProvider):
        def embed_batch(self, texts):
            super().embed_batch(texts)                  # record the call
            return [self._vec(t) for t in texts[:-1]]   # drops the last item

    texts = [f"tok{i}" for i in range(6)]
    p = ShortBatchProvider()
    build_dense(texts, p, tmp_path / "emb.npy", batch_size=3)
    assert len(p.queries) == 6                  # every item re-embedded singly
    arr = np.load(tmp_path / "emb.npy", mmap_mode="r").astype(np.float32)
    assert np.all(np.any(arr, axis=1))          # no silent zero rows


def test_all_zero_vectors_count_as_failures(tmp_path):
    class ZeroProvider(FakeProvider):
        def embed_batch(self, texts):
            return [[0.0] * 8 for _ in texts]   # right length, all-zero garbage
        def embed_one(self, text):
            return [0.0] * 8

    with pytest.raises(RuntimeError, match="embed failure rate"):
        build_dense([f"t{i}" for i in range(10)], ZeroProvider(),
                    tmp_path / "emb.npy", batch_size=4)
