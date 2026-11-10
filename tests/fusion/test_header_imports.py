# tests/fusion/test_header_imports.py
from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.fusion.ingest import _uses_line
from tests.fusion.test_dense import FakeProvider


def test_default_off_is_unchanged(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    for (embed_text,) in conn.execute("SELECT embed_text FROM chunks"):
        assert not any(line.startswith("uses: ") for line in embed_text.splitlines())
    assert store.meta_get(conn, "header_imports") == "0"
    conn.close()


def test_flag_adds_cross_file_uses_line(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir, header_imports=True)
    conn = store.open_db(index_dir / "fusion.db")

    # boot_sequence (caller.py) calls top_level_fn (sample.py) -- cross-file.
    boot_text = conn.execute(
        "SELECT embed_text FROM chunks WHERE name='boot_sequence'").fetchone()[0]
    uses_lines = [l for l in boot_text.splitlines() if l.startswith("uses: ")]
    assert len(uses_lines) == 1
    assert "top_level_fn" in uses_lines[0]
    # the uses line sits immediately after the header (relpath/language/parent) line
    lines = boot_text.splitlines()
    assert lines[1] == uses_lines[0]

    # top_level_fn (sample.py) calls Greeter and greet_user, both in sample.py --
    # same-file, so no uses: line should appear (verified via get_callees below).
    from srclight.db import Database
    sdb = Database(index_dir / "index.db")
    sdb.open()
    try:
        row = next(r for path in sdb.all_file_paths()
                   for r in sdb.symbols_in_file(path) if r.name == "top_level_fn")
        callee_files = {e["symbol"].file_path for e in sdb.get_callees(row.id)}
        assert callee_files == {row.file_path}, (
            "fixture assumption changed: top_level_fn now has a cross-file "
            "callee -- pick a different same-file-only symbol for this assertion")
    finally:
        sdb.close()

    top_level_text = conn.execute(
        "SELECT embed_text FROM chunks WHERE name='top_level_fn'").fetchone()[0]
    assert not any(l.startswith("uses: ") for l in top_level_text.splitlines())

    assert store.meta_get(conn, "header_imports") == "1"
    conn.close()


def test_cap_at_five():
    names = [f"sym_{i}" for i in range(8)]
    line = _uses_line(names)
    assert line == "uses: sym_0, sym_1, sym_2, sym_3, sym_4"


def test_uses_line_dedups_preserving_order():
    assert _uses_line(["a", "b", "a", "c"]) == "uses: a, b, c"


def test_uses_line_none_when_empty():
    assert _uses_line([]) is None


def test_backend_threads_flag(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "idx_hdr"
    be = FusionBackend(header_imports=True, embed_model="fake",
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    conn = store.open_db(idx / "fusion.db")
    assert store.meta_get(conn, "header_imports") == "1"
    conn.close()
