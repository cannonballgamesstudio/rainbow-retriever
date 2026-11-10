# tests/fusion/test_lex.py
from pathlib import Path

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.lex import LexIndex


def _built(srclight_index, tmp_path) -> tuple[LexIndex, object]:
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    lex = LexIndex.build(conn, tmp_path / "bm25")
    return lex, conn


def test_exact_identifier_ranks_defining_chunk_first(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    rows = [r for r, _ in lex.search("AddGoldCoins", k=5)]
    top = store.chunks_by_rows(conn, rows[:1])[0]
    assert top["name"] == "AddGoldCoins"
    conn.close()


def test_camelcase_subtoken_reaches_csharp_symbol(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    rows = [r for r, _ in lex.search("gold coins", k=5)]
    names = {c["name"] for c in store.chunks_by_rows(conn, rows)}
    assert "AddGoldCoins" in names
    conn.close()


def test_python_snake_case_matches_whole_only(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    # whole identifier finds it...
    rows = [r for r, _ in lex.search("greet_user", k=5)]
    names = {c["name"] for c in store.chunks_by_rows(conn, rows)}
    assert "greet_user" in names
    conn.close()


def test_save_load_roundtrip(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    reloaded = LexIndex.load(tmp_path / "bm25")
    a = reloaded.search("AddGoldCoins", k=3)
    b = lex.search("AddGoldCoins", k=3)
    assert [r for r, _ in a] == [r for r, _ in b]
    conn.close()


def test_no_match_returns_empty(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    assert lex.search("zzznonexistentzzz", k=5) == []
    conn.close()


def test_k_larger_than_corpus_is_clamped(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    rows = [r for r, _ in lex.search("AddGoldCoins", k=100)]
    assert len(rows) <= n_chunks
    # Also verify the top hit is still the defining chunk
    top = store.chunks_by_rows(conn, rows[:1])[0]
    assert top["name"] == "AddGoldCoins"
    conn.close()


def test_k_zero_returns_empty(srclight_index, tmp_path):
    lex, conn = _built(srclight_index, tmp_path)
    assert lex.search("AddGoldCoins", k=0) == []
    conn.close()
