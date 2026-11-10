# tests/ragd/test_comprehension.py  (Create)
from fastapi.testclient import TestClient
from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig
from rainbow_retriever.backend import GraphEdge, FileSymbol, DirEntry, HotspotFile, Structure, Hierarchy
from rainbow_retriever.fake_backend import FakeBackend

AUTH = {"X-User-Login": "tester@example.com"}
EDGE = GraphEdge(symbol="greet", file="m.py", line=1, kind="function", edge_type="calls", confidence=1.0)


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


def test_deps_returns_edges():
    r = _get(_app(deps_edges=[EDGE]), "/v1/deps", symbol="hello")
    assert r.status_code == 200
    assert r.json()["edges"][0]["symbol"] == "greet"


def test_file_returns_symbols():
    fs = FileSymbol(symbol="foo", kind="function", line=2, signature="def foo()")
    r = _get(_app(file_symbols=[fs]), "/v1/file", path="m.py")
    assert r.status_code == 200 and r.json()["symbols"][0]["line"] == 2


def test_tests_returns_edges():
    r = _get(_app(test_edges=[EDGE]), "/v1/tests", symbol="greet")
    assert r.status_code == 200
    assert r.json()["edges"][0]["edge_type"] == "calls"


def test_map_returns_structure():
    st = Structure(dirs=[DirEntry("pkg", 1, 3, ["python"])], hotspots=[HotspotFile("pkg/a.py", "python", 9, 3)])
    r = _get(_app(structure=st), "/v1/map")
    assert r.status_code == 200
    body = r.json()
    assert body["dirs"][0]["path"] == "pkg" and body["hotspots"][0]["symbols"] == 3


def test_hierarchy_returns_groups():
    r = _get(_app(hierarchy=Hierarchy(base=[EDGE], subclasses=[], implementors=[])), "/v1/hierarchy", symbol="C")
    assert r.status_code == 200
    assert r.json()["base"][0]["symbol"] == "greet"


def test_ambiguous_symbol_is_409():
    app = _app(ambiguous=[{"symbol": "foo", "file": "a.py", "line": 1},
                          {"symbol": "foo", "file": "b.py", "line": 9}])
    r = _get(app, "/v1/deps", symbol="foo")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "ambiguous_symbol"
    assert len(r.json()["error"]["details"]["candidates"]) == 2


def test_comprehension_requires_identity():
    r = TestClient(_app(deps_edges=[EDGE])).get("/v1/deps", params={"repo": "example/repo", "branch": "main", "symbol": "x"})
    assert r.status_code == 401


def test_capabilities_lists_supports_ops():
    r = TestClient(_app()).get("/v1/capabilities", headers=AUTH)
    assert r.json()["supports_ops"] == ["deps", "file", "tests", "map", "hierarchy", "reachable"]


def test_unsupported_op_is_400():
    r = _get(_app(supports_ops=()), "/v1/deps", symbol="x")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_op"
