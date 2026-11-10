# tests/fusion/test_fusion_backend_index.py
from pathlib import Path

from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.fusion.identity import FUSION_FORMAT_VERSION
from tests.fusion.test_dense import FakeProvider


def test_capabilities_keyword_only_without_embed_model():
    cap = FusionBackend().capabilities()
    assert cap.backend_id == "fusion"
    assert cap.modes == ("keyword",)
    assert cap.supports_callgraph is True
    assert cap.supports_ops == ("deps", "file", "tests", "map", "hierarchy", "reachable")
    assert cap.backend_version.startswith(f"fusion@{FUSION_FORMAT_VERSION}+")


def test_capabilities_with_embed_model():
    cap = FusionBackend(embed_model="qwen3-embedding", dims=8).capabilities()
    assert cap.modes == ("keyword", "semantic", "hybrid")
    assert cap.supports_dims == (8,)


def test_index_keyword_only_builds_srclight_fusion_and_bm25(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "idx"
    res = FusionBackend().index(str(wt), str(idx))
    assert (idx / "index.db").exists()
    assert (idx / "fusion.db").exists()
    assert (idx / "fusion_bm25").is_dir()
    assert not (idx / "fusion_emb.npy").exists()      # no embed model -> no dense
    assert res.symbols > 0 and res.files_indexed > 0
    assert res.backend_version.startswith("fusion@")
    assert res.index_dir == str(idx)


def test_index_with_provider_builds_dense_sidecar(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "idx2"
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    import numpy as np
    arr = np.load(idx / "fusion_emb.npy", mmap_mode="r")
    from rainbow_retriever.fusion import store
    conn = store.open_db(idx / "fusion.db")
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert arr.shape == (n, 8)
