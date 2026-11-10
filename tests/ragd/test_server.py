from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rainbow_retriever.registry import Registry
from rainbow_retriever.ragd.server import build_app

AUTH = {"X-User-Login": "tester@example.com"}


def test_build_app_serves_healthz(tmp_path):
    db = tmp_path / "registry.db"
    Registry.init_db(str(db))
    app = build_app({"RR_BACKEND": "fake", "RR_REGISTRY_DB": str(db)})
    r = TestClient(app).get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_build_app_wires_registry_from_env(tmp_path):
    db = tmp_path / "registry.db"
    reg = Registry.init_db(str(db))
    reg.upsert_ref("example/repo", "main", "sha1", str(tmp_path / "wt"),
                   "ready", 100, "srclight@x+none+0", "cfg")
    app = build_app({"RR_BACKEND": "fake", "RR_REGISTRY_DB": str(db)})
    r = TestClient(app).get("/v1/status", params={"repo": "example/repo"}, headers=AUTH)
    assert r.status_code == 200
    assert [x["ref"] for x in r.json()["refs"]] == ["main"]


def test_build_app_requires_registry_db():
    with pytest.raises(RuntimeError, match="RR_REGISTRY_DB"):
        build_app({"RR_BACKEND": "fake"})
