from collections import Counter

import pytest

from rainbow_retriever.backend import Hit
from rainbow_retriever.fusion.fuse import dedup_by_file, demote_docs, is_doc, weighted_rrf


def _hit(file, kind="function", symbol="s", score=1.0):
    return Hit(repo="r", ref="m", file=file, line=1, symbol=symbol,
               score=score, snippet="", kind=kind)


def test_rrf_prefers_items_ranked_by_both_arms():
    fused = weighted_rrf({"lex": [1, 2, 3], "dense": [2, 3, 4]})
    assert fused[0] == 2                         # top consensus first
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_weights_shift_the_balance():
    arms = {"lex": [1, 2], "dense": [2, 1]}
    assert weighted_rrf(arms, {"lex": 2.0, "dense": 0.5})[0] == 1
    assert weighted_rrf(arms, {"lex": 0.5, "dense": 2.0})[0] == 2


def test_rrf_single_arm_passthrough():
    assert weighted_rrf({"lex": [7, 8, 9]}) == [7, 8, 9]


def test_is_doc_by_kind_and_extension():
    assert is_doc(_hit("README.md"))
    assert is_doc(_hit("a.py", kind="section"))
    assert not is_doc(_hit("a.py"))


def test_demote_docs_stable():
    hits = [_hit("README.md"), _hit("a.py"), _hit("b.md"), _hit("c.py")]
    out = demote_docs(hits)
    assert [h.file for h in out] == ["a.py", "c.py", "README.md", "b.md"]


def test_dedup_by_file_keeps_first_and_backfills():
    hits = [_hit("a.py", symbol="x"), _hit("a.py", symbol="y"),
            _hit("B.PY", symbol="z"), _hit("c.py", symbol="w")]
    out = dedup_by_file(hits)
    assert [h.symbol for h in out] == ["x", "z", "w"]   # first per case-normalized file


def test_dedup_cap_default_is_current_behavior():
    hits = [_hit("a.py", symbol="x"), _hit("a.py", symbol="y"),
            _hit("B.PY", symbol="z"), _hit("c.py", symbol="w")]
    assert [h.symbol for h in dedup_by_file(hits)] == \
           [h.symbol for h in dedup_by_file(hits, max_per_file=1)] == ["x", "z", "w"]


def test_dedup_cap2_keeps_first_two_per_file_order_preserving():
    hits = [_hit("a.py", symbol="a1"), _hit("a.py", symbol="a2"), _hit("a.py", symbol="a3"),
            _hit("b.py", symbol="b1"), _hit("A.PY", symbol="a4"),   # case-normalized: same file as a.py
            _hit("b.py", symbol="b2"), _hit("c.py", symbol="c1")]
    out = dedup_by_file(hits, max_per_file=2)
    assert [h.symbol for h in out] == ["a1", "a2", "b1", "b2", "c1"]


def test_dedup_cap_binds_at_limit20_cap2_on_5x_synthetic_pool():
    # The existing 5x _DEDUP_OVERFETCH pool sized for cap=1 must be
    # sufficient for cap>=2. Synthetic pool = limit(20) * 5 = 100 hits,
    # 10 files x 10 chunks, round-robin (rank-interleaved like a fused list).
    limit = 20
    pool = [_hit(f"f{i}.py", symbol=f"f{i}s{j}") for j in range(10) for i in range(10)]
    assert len(pool) == limit * 5
    out = dedup_by_file(pool, max_per_file=2)[:limit]
    assert len(out) == limit                                   # pool fills the limit
    counts = Counter(h.file for h in out)
    assert set(counts.values()) == {2}                         # the cap binds: exactly 2/file
    assert [h.symbol for h in out[:10]] == [f"f{i}s0" for i in range(10)]  # order preserved


def test_dedup_cap_invalid_raises():
    with pytest.raises(ValueError, match="max_per_file"):
        dedup_by_file([], max_per_file=0)
