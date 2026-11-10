from tests.ragd.conftest import AUTH

BODY = {"repo": "example/repo", "branch": "main", "local_head_sha": None,
        "text": "find total", "mode": "keyword", "limit": 20}


def test_query_happy_path(client):
    r = client.post("/v1/query", json=BODY, headers=AUTH)
    assert r.status_code == 200
    b = r.json()
    assert b["served_ref"] == "main" and len(b["hits"]) == 2
    assert b["hits"][0]["repo"] == "example/repo" and b["hits"][0]["ref"] == "main"
    assert b["staleness"]["stale"] is False and b["truncated"] is False


def test_query_stale_when_sha_mismatch(client):
    body = {**BODY, "local_head_sha": "f" * 40}
    b = client.post("/v1/query", json=body, headers=AUTH).json()
    assert b["staleness"]["stale"] is True and b["staleness"]["push_hint"]


def test_query_fallback_served(client):
    body = {**BODY, "branch": "feature/x", "fallback_refs": ["wip", "main"]}
    b = client.post("/v1/query", json=body, headers=AUTH).json()
    assert b["served_ref"] == "main" and b["staleness"]["stale"] is True   # wip not ready -> main (fallback)


def test_query_unsupported_mode_422(client):
    # FakeBackend(modes default keyword,semantic,hybrid) -> use a keyword-only via a fresh app? Here 'bogus':
    r = client.post("/v1/query", json={**BODY, "mode": "bogus"}, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsupported_mode"


def test_query_ref_not_found_404(client):
    r = client.post("/v1/query", json={**BODY, "branch": "ghost", "fallback_refs": []}, headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "ref_not_found"


def test_query_budget_truncates(client):
    b = client.post("/v1/query", json={**BODY, "budget": 1}, headers=AUTH).json()
    assert b["truncated"] is True and len(b["hits"]) == 1


def test_query_invalid_body_400(client):
    r = client.post("/v1/query", json={"repo": "r", "branch": "m"}, headers=AUTH)  # missing text
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_semantic_busy_503(seeded_ro, fake_backend):
    from fastapi.testclient import TestClient
    from rainbow_retriever.ragd.app import create_app
    from rainbow_retriever.ragd.config import RagdConfig, EmbedQueue
    app = create_app(fake_backend, seeded_ro, RagdConfig(), embed_queue=EmbedQueue(0))
    r = TestClient(app).post("/v1/query", json={**BODY, "mode": "semantic"}, headers=AUTH)
    assert r.status_code == 503 and r.json()["error"]["code"] == "backend_busy"


def test_query_reports_anchored(client):
    b = client.post("/v1/query", json=BODY, headers=AUTH).json()
    assert b["anchored"] == "none"                 # "find total": no id-shaped tokens
    body = {**BODY, "text": "why does compute_total_v2 drop rows"}
    b = client.post("/v1/query", json=body, headers=AUTH).json()
    assert b["anchored"] == "strong"


def test_query_forwards_rerank_flag(client, fake_backend):
    resp = client.post("/v1/query", json={
        "repo": "example/repo", "branch": "main", "fallback_refs": [],
        "text": "how does routing work", "mode": "keyword", "limit": 5,
        "rerank": True,
    }, headers=AUTH)
    assert resp.status_code == 200
    assert fake_backend.query_calls[-1]["rerank"] is True


def test_query_forwards_max_chunks_per_file(client, fake_backend):
    r = client.post("/v1/query", json={**BODY, "max_chunks_per_file": 2}, headers=AUTH)
    assert r.status_code == 200
    assert fake_backend.query_calls[-1]["max_per_file"] == 2


def test_query_max_chunks_per_file_defaults_from_config(seeded_ro, fake_backend):
    from fastapi.testclient import TestClient
    from rainbow_retriever.ragd.app import create_app
    from rainbow_retriever.ragd.config import RagdConfig
    app = create_app(fake_backend, seeded_ro, RagdConfig(query_max_per_file=3))
    r = TestClient(app).post("/v1/query", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert fake_backend.query_calls[-1]["max_per_file"] == 3     # request unset -> config default


def test_query_max_chunks_per_file_unset_is_none(client, fake_backend):
    r = client.post("/v1/query", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert fake_backend.query_calls[-1]["max_per_file"] is None  # default config -> no cap


def test_query_max_chunks_per_file_zero_rejected_400(client):
    r = client.post("/v1/query", json={**BODY, "max_chunks_per_file": 0}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
