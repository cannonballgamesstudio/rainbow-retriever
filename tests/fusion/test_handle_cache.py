# tests/fusion/test_handle_cache.py
import pytest

from rainbow_retriever.fusion.fusion_backend import FusionBackend
from tests.fusion.test_dense import FakeProvider


@pytest.fixture(scope="module")
def indexed(srclight_index, tmp_path_factory):
    wt, _ = srclight_index
    idx = tmp_path_factory.mktemp("cache_idx")
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    return str(idx)


def test_handles_loaded_once_across_queries(indexed, monkeypatch):
    from rainbow_retriever.fusion import lex as lex_mod
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    loads = []
    real_load = lex_mod.LexIndex.load.__func__
    monkeypatch.setattr(lex_mod.LexIndex, "load",
                        classmethod(lambda cls, d: loads.append(1) or real_load(cls, d)))
    for _ in range(3):
        be.query(indexed, "r", "m", "AddGoldCoins", mode="keyword", limit=5)
    assert len(loads) == 1


def test_cache_invalidated_on_reindex(indexed, srclight_index, tmp_path):
    wt, _ = srclight_index
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    a = be.query(indexed, "r", "m", "AddGoldCoins", mode="keyword", limit=5)
    be.index(str(wt), indexed)                     # rebuild bumps mtimes
    b = be.query(indexed, "r", "m", "AddGoldCoins", mode="keyword", limit=5)
    assert [h.symbol for h in a] == [h.symbol for h in b]   # fresh handles, same result


def test_dense_matrix_float32_cached_once(tmp_path):
    import numpy as np
    from rainbow_retriever.fusion.dense import DenseIndex, build_dense, embed_query
    p = FakeProvider()
    build_dense(["alpha beta", "gamma"], p, tmp_path / "emb.npy")
    d = DenseIndex.load(tmp_path / "emb.npy")
    m1 = d._arr32
    d.search(embed_query(p, "alpha", None), k=1)
    assert d._arr32 is m1                          # no per-search re-materialization
    assert m1.dtype == np.float32


def test_keyword_only_backend_skips_dense_load(indexed, monkeypatch):
    """A keyword-only backend querying an index that HAS an emb sidecar must not
    load/materialize the dense matrix (up to ~800MB it can never use)."""
    from rainbow_retriever.fusion import dense as dense_mod
    be = FusionBackend()                        # no embed_model -> keyword-only
    loads = []
    real = dense_mod.DenseIndex.load.__func__
    monkeypatch.setattr(dense_mod.DenseIndex, "load",
                        classmethod(lambda cls, p: loads.append(1) or real(cls, p)))
    hits = be.query(indexed, "r", "m", "AddGoldCoins", mode="keyword", limit=5)
    assert hits and not loads


def test_dense_load_holds_no_file_handle(tmp_path):
    """Windows: os.replace over a file with an open mapped section raises
    OSError -- DenseIndex kept a dead mmap (_arr) alive though only _arr32 is
    used, blocking in-place reindex of fusion_emb.npy."""
    import os
    from rainbow_retriever.fusion.dense import DenseIndex, build_dense, embed_query
    p = FakeProvider()
    build_dense(["alpha beta", "gamma"], p, tmp_path / "emb.npy")
    d = DenseIndex.load(tmp_path / "emb.npy")
    build_dense(["delta", "epsilon zeta"], p, tmp_path / "emb2.npy")
    os.replace(tmp_path / "emb2.npy", tmp_path / "emb.npy")  # raised pre-fix
    assert d.search(embed_query(p, "alpha", None), k=1)      # old copy serves on
    assert not hasattr(d, "_arr")                            # dead handle gone
