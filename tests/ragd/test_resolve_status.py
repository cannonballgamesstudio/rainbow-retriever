from tests.ragd.conftest import AUTH


def test_resolve_any_status(client):
    b = client.get("/v1/resolve", params={"repo": "example/repo", "branch": "wip"}, headers=AUTH).json()
    assert b["ref"] == "wip" and b["status"] == "indexing"        # non-ready visible (A2)


def test_resolve_miss_404(client):
    r = client.get("/v1/resolve", params={"repo": "example/repo", "branch": "ghost"}, headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "ref_not_found"


def test_status_lists_all_rows(client):
    b = client.get("/v1/status", params={"repo": "example/repo"}, headers=AUTH).json()
    assert {x["ref"] for x in b["refs"]} == {"main", "wip"}


def test_resolve_requires_identity(client):
    assert client.get("/v1/resolve", params={"repo": "r", "branch": "m"}).status_code == 401


def test_status_requires_identity(client):
    assert client.get("/v1/status", params={"repo": "example/repo"}).status_code == 401
