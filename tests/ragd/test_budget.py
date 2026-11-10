from rainbow_retriever.ragd.budget import apply_budget


def _h(n):  # snippet length 40 -> cost 10 tokens each
    return {"snippet": "x" * 40, "symbol": n}


def test_none_budget_keeps_all():
    hits = [_h(0), _h(1), _h(2)]
    assert apply_budget(hits, None) == (hits, False)


def test_truncates_by_token_cost():
    hits = [_h(0), _h(1), _h(2)]            # 10 tokens each
    kept, truncated = apply_budget(hits, 15)  # fits 1, second would exceed
    assert [h["symbol"] for h in kept] == [0] and truncated is True


def test_exact_fit_not_truncated():
    hits = [_h(0), _h(1)]
    kept, truncated = apply_budget(hits, 20)
    assert len(kept) == 2 and truncated is False
