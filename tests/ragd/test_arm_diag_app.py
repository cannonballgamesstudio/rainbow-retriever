# tests/ragd/test_arm_diag_app.py
from fastapi.testclient import TestClient

from rainbow_retriever.backend import Hit
from rainbow_retriever.fake_backend import FakeBackend
from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig
from tests.ragd.conftest import AUTH

BODY = {"repo": "example/repo", "branch": "main", "local_head_sha": None,
        "text": "find total", "mode": "keyword", "limit": 20}


class DiagBackend(FakeBackend):
    """FakeBackend whose query exposes the fusion diagnostics contract."""

    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20,
              dims=None, code_only=False, rerank=False, dedup=None,
              max_per_file=None, *, arm_diag=None, expected_commit=None):
        if arm_diag is not None:
            arm_diag.update({
                "active": ["lex", "dense"],
                "dropped": {"graph": "OperationalError: no such table: adjacency"},
                "index_commit": "a" * 40,
                "expected_commit": expected_commit,
                "commit_mismatch": False,
            })
        return super().query(index_dir, repo, ref, text, mode, limit, dims,
                             code_only, rerank, dedup)


def test_query_response_gains_arms_block(seeded_ro):
    be = DiagBackend(hits=[Hit("?", "?", "a.py", 1, "s", 1.0, "x" * 8, "function")])
    client = TestClient(create_app(be, seeded_ro, RagdConfig()))
    b = client.post("/v1/query", json=BODY, headers=AUTH).json()
    assert b["arms"]["active"] == ["lex", "dense"]
    assert "graph" in b["arms"]["dropped"]
    assert b["arms"]["index_commit"] == "a" * 40
    # ragd threads the registry's recorded commit (conftest seeds head_sha)
    assert b["arms"]["expected_commit"] == "indexed_sha_1"
    assert b["hits"]                       # hits payload unchanged (additive)


def test_diagless_backend_response_unchanged(client):
    b = client.post("/v1/query", json=BODY, headers=AUTH).json()
    assert "arms" not in b                 # additive: absent without support
