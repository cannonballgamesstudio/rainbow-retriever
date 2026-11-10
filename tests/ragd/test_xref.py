import pytest
from fastapi.testclient import TestClient

from rainbow_retriever.backend import Hit
from rainbow_retriever.fake_backend import FakeBackend
from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig
from rainbow_retriever.registry import Registry, RegistryRO
from tests.ragd.conftest import AUTH

BODY = {"repo": "example/repo", "branch": "main", "local_head_sha": None,
        "text": "X", "refs": ["main", "wip", "ghost"], "limit": 20}


def test_xref_merges_resolved_skips_unresolved(client):
    b = client.post("/v1/xref", json=BODY, headers=AUTH).json()
    assert b["served_refs"] == ["main"]                 # wip=indexing, ghost=missing -> unresolved
    assert set(b["unresolved_refs"]) == {"wip", "ghost"}
    assert len(b["hits"]) == 2


def test_xref_all_unresolved_404(client):
    r = client.post("/v1/xref", json={**BODY, "refs": ["wip", "ghost"]}, headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "ref_not_found"


def test_xref_hits_sorted_by_score_desc(client):
    b = client.post("/v1/xref", json={**BODY, "refs": ["main"]}, headers=AUTH).json()
    scores = [h["score"] for h in b["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_xref_unindexed_working_branch_is_not_404(client):
    # working branch 'ghost' is unindexed and no fallbacks, but ref 'main' resolves
    body = {"repo": "example/repo", "branch": "ghost", "local_head_sha": None,
            "text": "X", "refs": ["main"], "fallback_refs": [], "limit": 20}
    r = client.post("/v1/xref", json=body, headers=AUTH)
    assert r.status_code == 200
    b = r.json()
    assert b["served_refs"] == ["main"] and len(b["hits"]) == 2
    assert b["staleness"]["stale"] is True and b["staleness"]["indexed_sha"] is None


class SizeBiasedBackend(FakeBackend):
    """Emits the OLD fusion score shape (n - i): the 'main' ref's index returns 5
    rows, 'alt' returns 2 -- pre-fix, every main hit outranks alt's rank-1 hit."""

    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20, dims=None,
              code_only=False, rerank=False, dedup=None, max_per_file=None):
        n = 5 if ref == "main" else 2
        return [Hit(repo=repo, ref=ref, file=f"{ref}_{i}.py", line=1,
                    symbol=f"s{i}", score=float(n - i), snippet="x" * 8,
                    kind="function")
                for i in range(n)][:limit]


@pytest.fixture
def two_ref_client(tmp_path):
    reg = Registry.init_db(str(tmp_path / "reg2.db"))
    reg.upsert_ref("example/repo", "main", "a" * 40, str(tmp_path / "wt_main"),
                   "ready", 100, "fake@1+none+0", "cfg1")
    reg.upsert_ref("example/repo", "alt", "b" * 40, str(tmp_path / "wt_alt"),
                   "ready", 100, "fake@1+none+0", "cfg1")
    ro = RegistryRO(str(tmp_path / "reg2.db"))
    return TestClient(create_app(SizeBiasedBackend(), ro, RagdConfig()))


def test_xref_interleaves_by_rank_not_index_size(two_ref_client):
    body = {"repo": "example/repo", "branch": "main", "local_head_sha": None,
            "text": "X", "refs": ["main", "alt"], "limit": 10}
    b = two_ref_client.post("/v1/xref", json=body, headers=AUTH).json()
    assert set(b["served_refs"]) == {"main", "alt"}
    files = [h["file"] for h in b["hits"]]
    # rank-1 of BOTH refs must precede any rank-2 hit
    assert set(files[:2]) == {"main_0.py", "alt_0.py"}
    # and scores in the response are per-ref reciprocal rank, sorted desc
    scores = [h["score"] for h in b["hits"]]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
