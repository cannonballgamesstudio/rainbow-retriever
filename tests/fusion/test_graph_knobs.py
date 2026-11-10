# tests/fusion/test_graph_knobs.py
import sqlite3

import pytest

from rainbow_retriever.fusion import graph as graph_mod
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from tests.fusion.test_dense import FakeProvider


def _conn(edges):
    """In-memory adjacency: edges = [(src, dst, w, edge_type)]."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(graph_mod._SCHEMA)
    conn.executemany("INSERT INTO adjacency(src, dst, w, edge_type) VALUES(?,?,?,?)",
                     edges)
    return conn


def test_expand_defaults_unchanged():
    # stored w for a confidence-1.0 "calls" edge = 0.9 (EDGE_WEIGHTS)
    conn = _conn([(1, 2, 0.9, "calls"), (2, 3, 0.9, "calls")])
    out = dict(graph_mod.expand(conn, [(1, 1.0)], hops=2))
    assert out[2] == pytest.approx(0.9)              # hop 1, factor 1.0
    assert out[3] == pytest.approx(0.9 * 0.9 * 0.5)  # hop 2, damped x0.5


def test_expand_damp_parameter():
    conn = _conn([(1, 2, 0.9, "calls"), (2, 3, 0.9, "calls")])
    out = dict(graph_mod.expand(conn, [(1, 1.0)], hops=2, damp=0.3))
    assert out[3] == pytest.approx(0.9 * 0.9 * 0.3)


def test_expand_flat_variant_divides_out_spec_weight():
    conn = _conn([(1, 2, 0.9, "calls")])             # conf 1.0: 0.9 = 0.9 * 1.0
    out = dict(graph_mod.expand(conn, [(1, 1.0)], edge_variant="flat"))
    assert out[2] == pytest.approx(1.0)              # 0.9 / 0.9 * 1.0


def test_expand_calls_led_variant():
    conn = _conn([(1, 2, 1.0, "contains"), (1, 3, 0.9, "calls")])
    out = dict(graph_mod.expand(conn, [(1, 1.0)], edge_variant="calls_led"))
    assert out[2] == pytest.approx(0.6)              # 1.0/1.0 * 0.6
    assert out[3] == pytest.approx(1.0)              # 0.9/0.9 * 1.0


def test_backend_rejects_bad_knobs():
    with pytest.raises(ValueError, match="graph_edge_variant"):
        FusionBackend(graph_edge_variant="bogus")
    with pytest.raises(ValueError, match="graph_seed_arm"):
        FusionBackend(graph_seed_arm="lexical")


@pytest.fixture()
def graph_backend_idx(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "gk_idx"
    be = FusionBackend(embed_model="fake", dims=8, w_graph=1.0,
                       provider_factory=lambda m: FakeProvider())
    be.index(str(wt), str(idx))
    return str(idx)


def _captured_seeds(be, idx, text, monkeypatch):
    captured = {}
    real = graph_mod.expand

    def spy(conn, seeds, **kw):
        captured["seeds"] = list(seeds)
        return real(conn, seeds, **kw)

    monkeypatch.setattr(graph_mod, "expand", spy)
    be.query(idx, "r", "m", text, mode="hybrid", limit=5)
    return captured["seeds"]


def test_seed_arm_dense_changes_seed_pool(graph_backend_idx, monkeypatch):
    kw = dict(embed_model="fake", dims=8, w_graph=1.0, lex_mode="full",
              provider_factory=lambda m: FakeProvider())
    fused_seeds = _captured_seeds(FusionBackend(**kw), graph_backend_idx,
                                  "AddGoldCoins inventory", monkeypatch)
    dense_seeds = _captured_seeds(FusionBackend(graph_seed_arm="dense", **kw),
                                  graph_backend_idx, "AddGoldCoins inventory",
                                  monkeypatch)
    assert fused_seeds and dense_seeds
    assert fused_seeds != dense_seeds        # lex arm influences fused, not dense


def test_seed_exclude_docs_drops_doc_rows(graph_backend_idx, monkeypatch):
    from rainbow_retriever.fusion import store
    from rainbow_retriever.fusion.fuse import DOC_KINDS, DOC_EXTS
    be = FusionBackend(embed_model="fake", dims=8, w_graph=1.0, lex_mode="full",
                       graph_seed_exclude_docs=True,
                       provider_factory=lambda m: FakeProvider())
    seeds = _captured_seeds(be, graph_backend_idx, "inventory system",
                            monkeypatch)
    conn = store.open_db(f"{graph_backend_idx}/fusion.db")
    try:
        doc_rows = {r["emb_row"] for r in
                    store.chunks_by_rows(conn, [s for s, _ in seeds])
                    if (r["kind"] or "").lower() in DOC_KINDS
                    or (r["file"] or "").lower().endswith(DOC_EXTS)}
    finally:
        conn.close()
    assert not doc_rows                      # README section never seeds expansion
