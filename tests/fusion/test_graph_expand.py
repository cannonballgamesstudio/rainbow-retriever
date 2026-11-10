# tests/fusion/test_graph_expand.py
import sqlite3

import pytest

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.fuse import weighted_rrf, weighted_rrf_scored
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.fusion.graph import build_adjacency, expand
from tests.fusion.test_dense import FakeProvider


def test_weighted_rrf_scored_matches_unscored():
    arms = {"a": [1, 2, 3], "b": [2, 3, 4]}
    scored = weighted_rrf_scored(arms)
    assert [r for r, _ in scored] == weighted_rrf(arms)
    assert all(s > 0 for _, s in scored)
    assert scored[0][1] >= scored[-1][1]


def test_expand_math_exact(tmp_path):
    conn = sqlite3.connect(tmp_path / "g.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE adjacency(src INT, dst INT, w REAL, edge_type TEXT);")
    conn.executemany("INSERT INTO adjacency VALUES(?,?,?,?)",
                     [(0, 5, 0.9, "calls"), (1, 5, 0.8, "inherits"), (1, 6, 1.0, "contains")])
    out = dict(expand(conn, [(0, 1.0), (1, 0.5)]))
    assert out[5] == pytest.approx(1.0 * 0.9 + 0.5 * 0.8)
    assert out[6] == pytest.approx(0.5 * 1.0)


def test_expand_two_hops_damped(tmp_path):
    conn = sqlite3.connect(tmp_path / "g.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE adjacency(src INT, dst INT, w REAL, edge_type TEXT);")
    conn.executemany("INSERT INTO adjacency VALUES(?,?,?,?)",
                     [(0, 1, 1.0, "calls"), (1, 2, 1.0, "calls")])
    one = dict(expand(conn, [(0, 1.0)], hops=1))
    two = dict(expand(conn, [(0, 1.0)], hops=2))
    assert 2 not in one
    assert two[2] == pytest.approx(1.0 * 1.0 * 1.0 * 0.5)   # hop1 score * w * damp


def test_expand_deterministic_and_capped(tmp_path):
    conn = sqlite3.connect(tmp_path / "g.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE adjacency(src INT, dst INT, w REAL, edge_type TEXT);")
    conn.executemany("INSERT INTO adjacency VALUES(?,?,?,?)",
                     [(0, d, 0.5, "calls") for d in range(1, 100)])
    out = expand(conn, [(0, 1.0)], cap=10)
    assert len(out) == 10
    assert out == expand(conn, [(0, 1.0)], cap=10)          # deterministic
    assert [r for r, _ in out] == sorted(r for r, _ in out)  # equal scores -> row asc


def test_missing_adjacency_fails_soft(tmp_path, srclight_index):
    wt, _ = srclight_index
    idx = tmp_path / "noadj"
    be = FusionBackend(embed_model="fake", dims=8, w_graph=1.0,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    conn = store.open_db(idx / "fusion.db")
    conn.execute("DROP TABLE adjacency")
    conn.commit()
    conn.close()
    hits = be.query(str(idx), "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
    assert hits                                              # graph dropped, query fine


def test_graph_arm_surfaces_cross_file_neighbor(srclight_index, tmp_path):
    """Rank-comparison design (not set-equality): caller.py's boot_sequence
    calls top_level_fn (sample.py), reachable ONLY via the "calls" edge. A
    no-op graph arm must fail this test.

    Prior version asserted sample.py was IN the result set with dedup_file=True
    and fetch=25 (limit=5 * _DEDUP_OVERFETCH). On this 5-file fixture that
    over-fetch already pulls every file into the pool regardless of w_graph, so
    the SAME file set came back whether w_graph was 0 or 1 -- a broken/no-op
    graph arm would still pass. Instead: index ONCE (index() doesn't depend on
    w_graph -- only query() does), construct two backends differing ONLY in
    w_graph (0.0 control vs 1.0 graph-on), both with dedup_file=False and
    lex_mode="ids" so fetch is capped at limit=10 (no dedup over-fetch masking
    the arm), and compare the RANK of the top_level_fn symbol between them.

    Query choice, verified empirically (3x, FakeProvider is deterministic so
    it's stable once passing): "boot_sequence" alone ties both arms at rank 6
    (no discrimination); "boot_sequence init" makes the graph arm rank
    top_level_fn WORSE (1 -> 3), the wrong direction. "boot_sequence startup"
    discriminates correctly both ways: control ranks top_level_fn at 3 (not 0,
    so strict improvement remains possible) and graph-on ranks it at 2.
    """
    wt, _ = srclight_index
    idx = tmp_path / "gidx"
    control = FusionBackend(embed_model="fake", dims=8, w_graph=0.0,
                            dedup_file=False, lex_mode="ids",
                            provider_factory=lambda model: FakeProvider())
    graph_on = FusionBackend(embed_model="fake", dims=8, w_graph=1.0,
                             dedup_file=False, lex_mode="ids",
                             provider_factory=lambda model: FakeProvider())
    control.index(str(wt), str(idx))    # build once; w_graph only affects query()

    query = "boot_sequence startup"
    hits_control = control.query(str(idx), "r", "m", query, mode="hybrid", limit=10)
    hits_graph = graph_on.query(str(idx), "r", "m", query, mode="hybrid", limit=10)

    syms_control = [h.symbol for h in hits_control]
    syms_graph = [h.symbol for h in hits_graph]

    def rank_of(syms: list[str]) -> int:
        return syms.index("top_level_fn") if "top_level_fn" in syms else len(syms)

    assert syms_graph != syms_control                    # graph-on reorders results
    assert rank_of(syms_graph) < rank_of(syms_control)    # ...and strictly promotes the callee


def test_w_graph_zero_is_single_phase(srclight_index, tmp_path, monkeypatch):
    wt, _ = srclight_index
    idx = tmp_path / "g0"
    be = FusionBackend(embed_model="fake", dims=8, w_graph=0.0,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    import rainbow_retriever.fusion.graph as graph_mod
    called = []
    monkeypatch.setattr(graph_mod, "expand",
                        lambda *a, **k: called.append(1) or [])
    be.query(str(idx), "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
    assert not called                                        # expansion never invoked
