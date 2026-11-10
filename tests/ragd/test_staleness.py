from rainbow_retriever.ragd.staleness import compute_staleness


def test_fresh_when_match_and_no_fallback():
    s = compute_staleness("aaaa", "aaaa", False, "main", "main")
    assert s["stale"] is False and s["push_hint"] is None


def test_sha_mismatch_is_stale():
    s = compute_staleness("aaaabbbb", "ccccdddd", False, "main", "main")
    assert s["stale"] is True and "push" in s["push_hint"].lower()


def test_fallback_is_stale_even_if_sha_none():
    s = compute_staleness("aaaa", None, True, "feature/x", "main")
    assert s["stale"] is True
    assert "feature/x" in s["push_hint"] and "main" in s["push_hint"]


def test_your_sha_none_no_fallback_is_fresh():
    s = compute_staleness("aaaa", None, False, "main", "main")
    assert s["stale"] is False and s["push_hint"] is None
