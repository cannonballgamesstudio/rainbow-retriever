# tests/fusion/test_store_ro.py
import sqlite3

import pytest

from rainbow_retriever.fusion import ingest, store


def test_open_db_ro_missing_raises_and_creates_no_file(tmp_path):
    # sqlite3.connect(path) CREATES an empty db on a missing path -- a read
    # path doing that masks the real error AND plants a zero-schema fusion.db
    # for the next reader.
    target = tmp_path / "fusion.db"
    with pytest.raises(FileNotFoundError, match="fusion db missing"):
        store.open_db_ro(target)
    assert not target.exists()


def test_open_db_ro_uninitialized_raises(tmp_path):
    target = tmp_path / "fusion.db"
    sqlite3.connect(target).close()                 # empty file, no schema
    with pytest.raises(RuntimeError, match="uninitialized"):
        store.open_db_ro(target)


def test_open_db_ro_rejects_writes(srclight_index):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db_ro(index_dir / "fusion.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO meta(key, value) VALUES('x','y')")
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
    finally:
        conn.close()


def test_query_against_empty_dir_raises_and_creates_nothing(tmp_path):
    from rainbow_retriever.fusion.fusion_backend import FusionBackend
    idx = tmp_path / "empty_idx"
    idx.mkdir()
    be = FusionBackend()
    with pytest.raises(FileNotFoundError):
        be.query(str(idx), "r", "m", "x", mode="keyword", limit=5)
    assert list(idx.iterdir()) == []                # no fusion.db planted
