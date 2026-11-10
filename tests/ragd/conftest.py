import pytest
from fastapi.testclient import TestClient
from rainbow_retriever.registry import Registry, RegistryRO
from rainbow_retriever.fake_backend import FakeBackend
from rainbow_retriever.backend import Hit, SymbolGraph, GraphEdge
from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig

AUTH = {"X-User-Login": "tester@example.com"}


def _hit(symbol, score, snippet="x" * 8):
    return Hit(repo="?", ref="?", file=f"{symbol}.py", line=1,
               symbol=symbol, score=score, snippet=snippet, kind="function")


@pytest.fixture
def seeded_ro(tmp_path):
    reg = Registry.init_db(str(tmp_path / "registry.db"))
    reg.upsert_ref("example/repo", "main", "indexed_sha_1", str(tmp_path / "wt_main"),
                   "ready", 100, "srclight@0.15.1+none+0", "cfg1")
    reg.upsert_ref("example/repo", "wip", "wip_sha", str(tmp_path / "wt_wip"),
                   "indexing", 50, "srclight@0.15.1+none+0", "cfg1")
    return RegistryRO(str(tmp_path / "registry.db"))


@pytest.fixture
def fake_backend():
    graph = SymbolGraph("hello", "a.py", 4, "def hello()",
                        callers=[GraphEdge("greet", "a.py", 1, "function", "calls", 1.0)],
                        callees=[])
    return FakeBackend(hits=[_hit("greet", -1.0), _hit("hello", -2.0)], graph=graph)


@pytest.fixture
def app(fake_backend, seeded_ro):
    return create_app(fake_backend, seeded_ro, RagdConfig())


@pytest.fixture
def client(app):
    return TestClient(app)
