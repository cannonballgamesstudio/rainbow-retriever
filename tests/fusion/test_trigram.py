# tests/fusion/test_trigram.py
import pytest

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.fusion.trigram import build_trigrams, search
from tests.fusion.test_dense import FakeProvider


@pytest.fixture()
def tri_index(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    build_trigrams(index_dir)
    return index_dir


def test_build_is_idempotent_and_nonempty(tri_index):
    n1 = build_trigrams(tri_index)
    n2 = build_trigrams(tri_index)
    assert n1 == n2 > 0


def test_partial_identifier_match(tri_index):
    conn = store.open_db(tri_index / "fusion.db")
    # 'AddGoldCoin' (typo/prefix: no trailing s) -- BM25 whole-token misses this;
    # trigram containment still hits AddGoldCoins.
    hits = search(conn, "AddGoldCoin")
    name_of = {r["emb_row"]: r["name"] for r in conn.execute("SELECT * FROM chunks")}
    conn.close()
    assert hits and name_of[hits[0][0]] == "AddGoldCoins"
    assert 0.5 <= hits[0][1] <= 1.0


def test_exact_identifier_scores_full(tri_index):
    conn = store.open_db(tri_index / "fusion.db")
    hits = search(conn, "parseQuestData")
    name_of = {r["emb_row"]: r["name"] for r in conn.execute("SELECT * FROM chunks")}
    conn.close()
    assert hits and name_of[hits[0][0]] == "parseQuestData"
    assert hits[0][1] == pytest.approx(1.0)


def test_short_or_prose_tokens_inactive(tri_index):
    conn = store.open_db(tri_index / "fusion.db")
    assert search(conn, "") == []
    assert search(conn, "gold") == []          # < 6 chars: not an anchor
    conn.close()


def test_arm_gated_and_fail_soft(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "tri_idx"
    be = FusionBackend(embed_model="fake", dims=8, w_trigram=1.0,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    hits = be.query(str(idx), "r", "m", "AddGoldCoin inventory",
                    mode="hybrid", limit=5)
    assert any(h.symbol == "AddGoldCoins" for h in hits)   # trigram arm reached it
    conn = store.open_db(idx / "fusion.db")
    conn.execute("DROP TABLE trigrams")
    conn.commit()
    conn.close()
    be2 = FusionBackend(embed_model="fake", dims=8, w_trigram=1.0,
                        provider_factory=lambda model: FakeProvider())
    assert be2.query(str(idx), "r", "m", "AddGoldCoin", mode="hybrid", limit=5)  # fail-soft


def test_w_trigram_zero_never_touches_table(srclight_index, tmp_path, monkeypatch):
    wt, _ = srclight_index
    idx = tmp_path / "tri0"
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    import rainbow_retriever.fusion.trigram as tri_mod
    called = []
    monkeypatch.setattr(tri_mod, "search", lambda *a, **k: called.append(1) or [])
    be.query(str(idx), "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
    assert not called
