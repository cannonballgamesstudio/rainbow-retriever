# tests/ragd/test_reachable.py
"""Tests for the `reachable` op (transitive call-graph reachability).

Backend unit tests build a tiny real srclight index (keyword-only -> fully
deterministic, no model/network) over a fixture call graph:

    alpha -> beta -> gamma -> alpha   (cycle back to alpha)
    alpha -> delta

(names must be real identifiers with an arg -- srclight's call-edge
extraction does not register single-letter/no-arg names as call sites.)

Route tests mirror tests/ragd/test_comprehension.py's FakeBackend pattern.
"""
import pytest
from fastapi.testclient import TestClient

from rainbow_retriever.backend import AmbiguousSymbol, ReachNode, ReachResult
from rainbow_retriever.srclight_backend import SrclightBackend
from rainbow_retriever.fake_backend import FakeBackend
from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig

AUTH = {"X-User-Login": "tester@example.com"}


def _make_cycle_worktree(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "graph.py").write_text(
        "def alpha(n):\n    return beta(n) + delta(n)\n\n"
        "def beta(n):\n    return gamma(n)\n\n"
        "def gamma(n):\n    return alpha(n)\n\n"
        "def delta(n):\n    return 1\n"
    )
    return wt


def _indexed_backend(tmp_path):
    wt = _make_cycle_worktree(tmp_path)
    idx_dir = tmp_path / ".srclight"
    be = SrclightBackend()
    be.index(str(wt), str(idx_dir))
    return be, str(idx_dir)


# ---- backend unit tests -----------------------------------------------

def test_reachable_callees_bfs_respects_max_depth(tmp_path):
    be, idx = _indexed_backend(tmp_path)
    res = be.reachable(idx, "r", "main", ["alpha"], "callees", 2)
    assert isinstance(res, ReachResult)
    assert res.entry == ["alpha"]
    by_symbol = {n.symbol: n.depth for n in res.reachable}
    assert by_symbol == {"beta": 1, "delta": 1, "gamma": 2}
    assert "alpha" not in by_symbol            # entry itself never reappears
    assert res.truncated is False


def test_reachable_orders_by_depth_then_symbol(tmp_path):
    be, idx = _indexed_backend(tmp_path)
    res = be.reachable(idx, "r", "main", ["alpha"], "callees", 2)
    ordered = [(n.depth, n.symbol) for n in res.reachable]
    assert ordered == sorted(ordered)


def test_reachable_cycle_terminates(tmp_path):
    # Unbounded depth (max_depth=0) over a graph with a cycle back to the
    # entry must still terminate (visited-set dedup), not loop forever.
    be, idx = _indexed_backend(tmp_path)
    res = be.reachable(idx, "r", "main", ["alpha"], "callees", 0)
    symbols = {n.symbol for n in res.reachable}
    assert symbols == {"beta", "gamma", "delta"}
    assert "alpha" not in symbols


def test_reachable_callers_direction(tmp_path):
    be, idx = _indexed_backend(tmp_path)
    res = be.reachable(idx, "r", "main", ["gamma"], "callers", 5)
    by_symbol = {n.symbol: n.depth for n in res.reachable}
    # gamma is called by beta; beta is called by alpha -- caller-direction BFS.
    assert by_symbol == {"beta": 1, "alpha": 2}


def test_reachable_ambiguous_entry_raises(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "one.py").write_text("def foo():\n    return 1\n")
    (wt / "two.py").write_text("def foo():\n    return 2\n")
    idx_dir = tmp_path / ".srclight"
    be = SrclightBackend()
    be.index(str(wt), str(idx_dir))
    with pytest.raises(AmbiguousSymbol):
        be.reachable(str(idx_dir), "r", "main", ["foo"], "callees", 5)


def test_reachable_limit_sets_truncated(tmp_path):
    be, idx = _indexed_backend(tmp_path)
    res = be.reachable(idx, "r", "main", ["alpha"], "callees", 5, None, 1)
    assert res.truncated is True
    assert len(res.reachable) == 1


def test_reachable_in_supports_ops_srclight():
    assert "reachable" in SrclightBackend().capabilities().supports_ops


def test_reachable_in_supports_ops_fusion():
    from rainbow_retriever.fusion.fusion_backend import FusionBackend
    assert "reachable" in FusionBackend().capabilities().supports_ops


def test_fusion_reachable_delegates(tmp_path, monkeypatch):
    from rainbow_retriever.fusion.fusion_backend import FusionBackend
    fb = FusionBackend()
    sentinel = ReachResult(served_ref="main", entry=["A"], reachable=[], truncated=False)
    calls = {}

    def fake_reachable(index_dir, repo, ref, symbols, direction="callees",
                       max_depth=5, file=None, limit=2000):
        calls["args"] = (index_dir, repo, ref, symbols, direction, max_depth, file, limit)
        return sentinel

    monkeypatch.setattr(fb._delegate, "reachable", fake_reachable)
    out = fb.reachable("idx", "r", "main", ["A"], "callees", 3, None, 500)
    assert out is sentinel
    assert calls["args"] == ("idx", "r", "main", ["A"], "callees", 3, None, 500)


# ---- fake_backend stub --------------------------------------------------

def test_fake_backend_reachable_stub_and_ambiguity():
    node = ReachNode(symbol="B", file="m.py", line=4, kind="function", depth=1)
    fixture = ReachResult(served_ref="main", entry=["A"], reachable=[node], truncated=False)
    fb = FakeBackend(reachable_result=fixture)
    out = fb.reachable("idx", "r", "main", ["A"])
    assert out is fixture

    fb2 = FakeBackend(ambiguous=[{"symbol": "foo", "file": "a.py", "line": 1},
                                 {"symbol": "foo", "file": "b.py", "line": 9}])
    with pytest.raises(AmbiguousSymbol):
        fb2.reachable("idx", "r", "main", ["foo"])


def test_fake_backend_supports_reachable_by_default():
    assert "reachable" in FakeBackend().capabilities().supports_ops


# ---- route tests ----------------------------------------------------------

def _app(**kw):
    from rainbow_retriever.registry import Registry, RegistryRO
    import tempfile, os
    d = tempfile.mkdtemp()
    reg = Registry.init_db(os.path.join(d, "r.db"))
    reg.upsert_ref("example/repo", "main", "sha1", os.path.join(d, "wt"),
                   "ready", 100, "srclight@x+none+0", "cfg1")
    return create_app(FakeBackend(**kw), RegistryRO(os.path.join(d, "r.db")), RagdConfig())


def _get(app, url, **params):
    return TestClient(app).get(url, params={"repo": "example/repo", "branch": "main", **params}, headers=AUTH)


def test_route_reachable_returns_dto():
    node = ReachNode(symbol="B", file="graph.py", line=4, kind="function", depth=1)
    fixture = ReachResult(served_ref="main", entry=["A"], reachable=[node], truncated=False)
    app = _app(reachable_result=fixture)
    r = _get(app, "/v1/reachable", symbol="A", direction="callees", depth=2)
    assert r.status_code == 200
    body = r.json()
    assert body["served_ref"] == "main"
    assert body["entry"] == ["A"]
    assert body["reachable"] == [{"symbol": "B", "file": "graph.py", "line": 4,
                                  "kind": "function", "depth": 1}]
    assert body["truncated"] is False


def test_route_reachable_requires_identity():
    node = ReachNode(symbol="B", file="graph.py", line=4, kind="function", depth=1)
    fixture = ReachResult(served_ref="main", entry=["A"], reachable=[node], truncated=False)
    r = TestClient(_app(reachable_result=fixture)).get(
        "/v1/reachable", params={"repo": "example/repo", "branch": "main", "symbol": "A"})
    assert r.status_code == 401


def test_route_reachable_ambiguous_is_409():
    app = _app(ambiguous=[{"symbol": "foo", "file": "a.py", "line": 1},
                          {"symbol": "foo", "file": "b.py", "line": 9}])
    r = _get(app, "/v1/reachable", symbol="foo")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "ambiguous_symbol"


def test_route_reachable_unsupported_op():
    # Mirrors test_unsupported_op_is_400 for the other ops: `_require_op` maps
    # an unsupported op to UnsupportedOp (http_status=400), not 503 -- there is
    # no separate "backend lacks this op at call time" path in the existing
    # wiring, so a backend without "reachable" in supports_ops is gated at the
    # same 400 boundary as deps/file/tests/map/hierarchy.
    r = _get(_app(supports_ops=()), "/v1/reachable", symbol="A")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_op"
