# tests/fusion/test_graph_build.py
import pytest

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.graph import EDGE_WEIGHTS, build_adjacency


def _adj(index_dir):
    ingest.ingest_srclight(index_dir)
    n = build_adjacency(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    rows = conn.execute(
        "SELECT src, dst, w, edge_type FROM adjacency").fetchall()
    name_of = {r["emb_row"]: r["name"] for r in conn.execute("SELECT * FROM chunks")}
    conn.close()
    return n, rows, name_of


def test_containment_edges_bidirectional(srclight_index):
    _, index_dir = srclight_index
    n, rows, name_of = _adj(index_dir)
    pairs = {(name_of[r["src"]], name_of[r["dst"]], r["edge_type"]) for r in rows}
    assert ("AddGoldCoins", "PlayerInventory", "contains") in pairs
    assert ("PlayerInventory", "AddGoldCoins", "contains") in pairs


def test_call_edges_present_with_weights(srclight_index):
    _, index_dir = srclight_index
    n, rows, name_of = _adj(index_dir)
    call_ws = {r["w"] for r in rows if r["edge_type"] == "calls"}
    # w = EDGE_WEIGHTS["calls"] * srclight confidence: full-confidence edges hit the
    # type weight exactly; lower-confidence (e.g. cross-file) edges land below it.
    assert call_ws and all(0.0 < w <= EDGE_WEIGHTS["calls"] + 1e-9 for w in call_ws)
    assert any(abs(w - EDGE_WEIGHTS["calls"]) < 1e-9 for w in call_ws)
    # fixture: top_level_fn calls Greeter/greet_user (srclight emits 'calls' edges)
    names = {(name_of[r["src"]], name_of[r["dst"]]) for r in rows if r["edge_type"] == "calls"}
    assert ("top_level_fn", "Greeter") in names or ("top_level_fn", "greet_user") in names


def test_rebuild_idempotent(srclight_index):
    _, index_dir = srclight_index
    n1, _, _ = _adj(index_dir)
    n2 = build_adjacency(index_dir)
    assert n1 == n2 > 0


def test_edges_reference_valid_emb_rows(srclight_index):
    _, index_dir = srclight_index
    n, rows, name_of = _adj(index_dir)
    assert all(r["src"] in name_of and r["dst"] in name_of for r in rows)


def test_symmetric_same_type_edges_keep_max_weight(tmp_path):
    """Two independent same-type edges A->B and B->A with different confidences
    must aggregate to max() in BOTH directions (explicit policy), not to
    whichever insert happens last."""
    from srclight.db import Database, EdgeRecord
    from srclight.indexer import Indexer, IndexConfig

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "a.py").write_text("def f():\n    return g()\n\ndef g():\n    return 1\n")
    idx = tmp_path / "idx"
    idx.mkdir()
    db = Database(idx / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=wt, embed_model=None)).index(wt)
    f_id = db.get_symbols_by_name("f")[0].id
    g_id = db.get_symbols_by_name("g")[0].id
    # NOTE: the indexer does not resolve this trivial free-function call into a
    # "calls" edge (db.get_callees returns [] for both f and g here), so both
    # edges are inserted manually -- the point of this test is max()
    # aggregation of two independent same-type edges, not indexer behavior.
    db.insert_edge(EdgeRecord(id=None, source_id=f_id, target_id=g_id,
                              edge_type="calls", confidence=1.0, metadata=None))
    db.insert_edge(EdgeRecord(id=None, source_id=g_id, target_id=f_id,
                              edge_type="calls", confidence=0.5, metadata=None))
    db.commit()
    db.close()

    ingest.ingest_srclight(idx)
    build_adjacency(idx)
    conn = store.open_db(idx / "fusion.db")
    name_of = {r["emb_row"]: r["name"] for r in conn.execute("SELECT * FROM chunks")}
    ws = {(name_of[r["src"]], name_of[r["dst"]]): r["w"] for r in
          conn.execute("SELECT src, dst, w FROM adjacency WHERE edge_type='calls'")}
    conn.close()
    assert ws[("f", "g")] == pytest.approx(0.9)   # max(0.9*1.0, 0.9*0.5), both directions
    assert ws[("g", "f")] == pytest.approx(0.9)
