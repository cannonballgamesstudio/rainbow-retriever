import pytest
from pydantic import ValidationError
from rainbow_retriever.ragd.wire import QueryRequest, CtxRequest, XrefRequest

GOOD_SHA = "9f3c1a2b4d5e6f7081920aabbccddeeff0011223"


def test_query_defaults_and_extra_ignored():
    q = QueryRequest(repo="r", branch="main", text="hi", junk="x")
    assert (q.mode, q.limit, q.fallback_refs, q.local_head_sha) == ("hybrid", 20, [], None)
    assert not hasattr(q, "junk")


def test_sha_validation():
    QueryRequest(repo="r", branch="m", text="t", local_head_sha=GOOD_SHA)   # ok
    QueryRequest(repo="r", branch="m", text="t", local_head_sha=None)        # ok
    with pytest.raises(ValidationError):
        QueryRequest(repo="r", branch="m", text="t", local_head_sha="ABC")   # uppercase/short


def test_limit_bounds_and_nonempty():
    with pytest.raises(ValidationError):
        QueryRequest(repo="r", branch="m", text="t", limit=0)
    with pytest.raises(ValidationError):
        QueryRequest(repo="r", branch="m", text="t", limit=201)
    with pytest.raises(ValidationError):
        QueryRequest(repo="", branch="m", text="t")


def test_ctx_and_xref():
    assert CtxRequest(repo="r", branch="m", symbol="Foo.bar").symbol == "Foo.bar"
    x = XrefRequest(repo="r", branch="m", text="t", refs=["main", "pr-1"])
    assert x.refs == ["main", "pr-1"]
    with pytest.raises(ValidationError):
        XrefRequest(repo="r", branch="m", text="t", refs=[])                  # non-empty
