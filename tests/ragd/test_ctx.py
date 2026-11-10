from tests.ragd.conftest import AUTH

BODY = {"repo": "example/repo", "branch": "main", "local_head_sha": None, "symbol": "hello"}


def test_ctx_found(client):
    b = client.post("/v1/ctx", json=BODY, headers=AUTH).json()
    assert b["found"] is True
    assert b["graph"]["symbol"] == "hello" and b["graph"]["callers"][0]["symbol"] == "greet"


def test_ctx_not_found_is_200_false(client):
    r = client.post("/v1/ctx", json={**BODY, "symbol": "missing"}, headers=AUTH)
    assert r.status_code == 200
    b = r.json()
    assert b["found"] is False and b["graph"] is None


def test_ctx_ambiguous_is_409(seeded_ro):
    """ctx must 409 on an ambiguous symbol (like deps/hierarchy), not silently pick one."""
    from fastapi.testclient import TestClient
    from rainbow_retriever.fake_backend import FakeBackend
    from rainbow_retriever.ragd.app import create_app

    cands = [{"symbol": "GardenScene", "file": "docs/x.md", "line": 1},
             {"symbol": "GardenScene", "file": "src/GardenScene.ts", "line": 9}]
    app = create_app(FakeBackend(ambiguous=cands), seeded_ro)
    r = TestClient(app).post("/v1/ctx", json={**BODY, "symbol": "GardenScene"}, headers=AUTH)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "ambiguous_symbol"
    assert len(r.json()["error"]["details"]["candidates"]) == 2


def test_ctx_file_disambiguates(seeded_ro):
    """Passing `file` resolves the ambiguity instead of 409."""
    from fastapi.testclient import TestClient
    from rainbow_retriever.fake_backend import FakeBackend
    from rainbow_retriever.backend import SymbolGraph
    from rainbow_retriever.ragd.app import create_app

    g = SymbolGraph(symbol="GardenScene", defined_file="src/GardenScene.ts",
                    defined_line=9, signature="class GardenScene", callers=[], callees=[])
    app = create_app(FakeBackend(graph=g, ambiguous=[{"symbol": "x", "file": "a", "line": 1},
                                                     {"symbol": "x", "file": "b", "line": 2}]), seeded_ro)
    r = TestClient(app).post("/v1/ctx",
                             json={**BODY, "symbol": "GardenScene", "file": "src/GardenScene.ts"},
                             headers=AUTH)
    assert r.status_code == 200
    assert r.json()["found"] is True
    assert r.json()["graph"]["defined_file"] == "src/GardenScene.ts"


def test_ctx_unsupported_404(seeded_ro):
    from fastapi.testclient import TestClient
    from rainbow_retriever.fake_backend import FakeBackend
    from rainbow_retriever.backend import Capabilities
    from rainbow_retriever.ragd.app import create_app

    class NoGraph(FakeBackend):
        def capabilities(self):
            return Capabilities("fake", "fake@1+none+0", ("keyword",), False, ())

    r = TestClient(create_app(NoGraph(), seeded_ro)).post("/v1/ctx", json=BODY, headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "ctx_unsupported"
