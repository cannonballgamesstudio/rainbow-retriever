# tests/fusion/test_ingest.py
from rainbow_retriever.fusion import ingest, store


def test_ingest_creates_chunks_for_all_indexed_languages(srclight_index):
    _, index_dir = srclight_index
    n = ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    langs = {r[0] for r in conn.execute("SELECT DISTINCT language FROM chunks")}
    assert {"python", "csharp", "javascript", "markdown"} <= langs
    assert n == conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
    # emb_row is a dense 0..n-1 ordering (canonical row order for npy + bm25s)
    rows = [r[0] for r in conn.execute("SELECT emb_row FROM chunks ORDER BY emb_row")]
    assert rows == list(range(n))
    conn.close()


def test_context_header_in_embed_text(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    row = conn.execute(
        "SELECT embed_text, parent, signature FROM chunks WHERE name='AddGoldCoins'").fetchone()
    assert row is not None
    embed_text, parent, _sig = row
    assert "sample.cs" in embed_text            # relpath
    assert "[csharp]" in embed_text             # language tag
    assert parent == "PlayerInventory"          # enclosing class resolved
    assert "AddGoldCoins" in embed_text
    conn.close()


def test_snippet_capped_and_lex_fields_populated(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    for snippet, lex_name, lex_body in conn.execute(
            "SELECT snippet, lex_name, lex_body FROM chunks"):
        assert len(snippet) <= 600 and len(snippet.splitlines()) <= 15
        assert lex_name.strip()
    doc = conn.execute("SELECT lex_body FROM chunks WHERE name='greet_user'").fetchone()[0]
    assert "greeting" in doc                    # doc_comment folded into body field
    conn.close()


def test_gdscript_gap_tripwire(srclight_index):
    """srclight 0.15.1 does not index GDScript. When this test FAILS, upstream added
    support -- remove the tripwire and extend the fixture assertions to cover .gd."""
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    n = conn.execute("SELECT COUNT(*) FROM chunks WHERE file LIKE '%.gd'").fetchone()[0]
    assert n == 0
    conn.close()


def test_meta_records_counts_and_format(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    assert store.meta_get(conn, "fusion_fmt") == "1"
    assert int(store.meta_get(conn, "chunks")) > 0
    conn.close()
