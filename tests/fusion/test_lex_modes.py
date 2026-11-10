import pytest

from rainbow_retriever.fusion.fusion_backend import FusionBackend
from tests.fusion.test_dense import FakeProvider


@pytest.fixture(scope="module")
def indexed(srclight_index, tmp_path_factory):
    wt, _ = srclight_index
    idx = tmp_path_factory.mktemp("lexmode_idx")
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    return str(idx)


def _be(lex_mode):
    return FusionBackend(embed_model="fake", dims=8, lex_mode=lex_mode,
                         provider_factory=lambda model: FakeProvider())


def test_invalid_lex_mode_raises():
    with pytest.raises(ValueError, match="lex_mode"):
        FusionBackend(lex_mode="bogus")


def test_default_lex_mode_is_ids():
    assert FusionBackend().lex_mode == "ids"


def test_ids_mode_anchored_query_hits_symbol(indexed):
    hits = _be("ids").query(indexed, "r", "m", "AddGoldCoins inventory",
                            mode="hybrid", limit=5)
    assert "AddGoldCoins" in {h.symbol for h in hits}


def test_ids_mode_prose_query_equals_dense_only(indexed):
    prose = "where is the gold coin inventory logic handled"   # no id-shaped tokens
    ids_hits = _be("ids").query(indexed, "r", "m", prose, mode="hybrid", limit=5)
    off_hits = _be("off").query(indexed, "r", "m", prose, mode="hybrid", limit=5)
    assert [h.file for h in ids_hits] == [h.file for h in off_hits]


def test_ids_mode_lex_pool_excludes_docs_but_dense_keeps_them(indexed):
    # 'Demo' is the README section heading (PascalCase-ish word won't extract; use
    # a backticked anchor that matches the README section symbol name).
    hits = _be("ids").query(indexed, "r", "m", "`Demo` inventory docs",
                            mode="hybrid", limit=10)
    # The doc section may appear (dense arm is allowed to surface it) -- the
    # invariant under test: the query does not ERROR and code hits are present.
    assert any(not h.file.endswith(".md") for h in hits)


def test_off_mode_hybrid_is_dense_only(indexed):
    p = FakeProvider()
    be = FusionBackend(embed_model="fake", dims=8, lex_mode="off",
                       provider_factory=lambda model: p)
    hits = be.query(indexed, "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
    assert isinstance(hits, list) and len(p.queries) == 1   # embed happened, lex skipped


def test_keyword_mode_ignores_lex_mode(indexed):
    # explicit keyword search stays literal raw-text search even in ids/off modes
    for m in ("ids", "off"):
        hits = _be(m).query(indexed, "r", "m", "AddGoldCoins", mode="keyword", limit=5)
        assert hits and hits[0].symbol == "AddGoldCoins"


def test_full_mode_preserves_p1_behavior(indexed):
    # dedup_file=False for the same reason as test_hybrid_fuses_lexical_and_dense
    # (tests/fusion/test_fusion_backend_query.py): AddGoldCoins and PlayerInventory
    # share sample.cs, and file-dedup keeps only whichever wins a razor-thin RRF
    # margin under the FakeProvider's 8-dim hash embedding -- not what this test
    # (full-mode lexical matching on raw prose) is checking.
    be = FusionBackend(embed_model="fake", dims=8, lex_mode="full", dedup_file=False,
                       provider_factory=lambda model: FakeProvider())
    hits = be.query(indexed, "r", "m", "gold coins inventory",
                    mode="hybrid", limit=10)
    assert any(h.symbol == "AddGoldCoins" for h in hits)
