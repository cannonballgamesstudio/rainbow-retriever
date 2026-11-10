from tests.ragd.conftest import AUTH


def test_healthz_no_identity(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "X-Request-Id" in r.headers


def test_v1_requires_identity(client):
    assert client.get("/v1/capabilities").status_code == 401
    err = client.get("/v1/capabilities").json()["error"]
    assert err["code"] == "identity_required" and "request_id" in err


def test_capabilities_with_identity(client):
    r = client.get("/v1/capabilities", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["backend_id"] == "fake" and "keyword" in body["modes"]
    assert r.headers["X-Backend-Version"] == "fake@1+none+0"


def test_honors_incoming_request_id(client):
    r = client.get("/healthz", headers={"X-Request-Id": "abc-123"})
    assert r.headers["X-Request-Id"] == "abc-123"
