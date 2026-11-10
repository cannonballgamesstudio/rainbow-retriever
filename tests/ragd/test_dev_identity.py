from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig
from rainbow_retriever.ragd.errors import IdentityRequired
from rainbow_retriever.ragd.identity import extract_identity

_BODY = {"repo": "example/repo", "branch": "main", "text": "hello", "mode": "keyword", "limit": 5}


def test_extract_identity_prefers_header_over_default():
    ident = extract_identity({"X-User-Login": "real@example.com"}, "fallback@example.com")
    assert ident.login == "real@example.com"


def test_extract_identity_uses_default_when_header_absent():
    assert extract_identity({}, "fallback@example.com").login == "fallback@example.com"


def test_extract_identity_raises_when_no_header_and_no_default():
    with pytest.raises(IdentityRequired):
        extract_identity({}, None)


def test_from_env_reads_default_identity():
    cfg = RagdConfig.from_env({"RR_DEFAULT_IDENTITY": "local@example.com"})
    assert cfg.default_identity == "local@example.com"
    assert RagdConfig.from_env({}).default_identity is None


def test_query_without_header_allowed_when_default_set(fake_backend, seeded_ro):
    app = create_app(fake_backend, seeded_ro, RagdConfig(default_identity="local@example.com"))
    r = TestClient(app).post("/v1/query", json=_BODY)  # no X-User-Login header
    assert r.status_code == 200


def test_query_without_header_401_when_no_default(fake_backend, seeded_ro):
    app = create_app(fake_backend, seeded_ro, RagdConfig())
    r = TestClient(app).post("/v1/query", json=_BODY)  # no header, no default
    assert r.status_code == 401
