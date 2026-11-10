import numpy as np
import pytest

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.dense import build_dense
from rainbow_retriever.fusion.lex import LexIndex
from tests.fusion.test_dense import FakeProvider


def test_dense_write_is_temp_then_replace(tmp_path, monkeypatch):
    """A crash mid-embed must never leave a truncated fusion_emb.npy in place."""
    out = tmp_path / "fusion_emb.npy"
    out.write_bytes(b"OLD-INDEX-CONTENT")           # pre-existing good artifact

    class ExplodingProvider(FakeProvider):
        def embed_batch(self, texts):
            raise RuntimeError("provider died mid-build")
        def embed_one(self, text):
            raise RuntimeError("provider died mid-build")

    with pytest.raises(RuntimeError):
        build_dense([f"t{i}" for i in range(20)], ExplodingProvider(),
                    out, batch_size=1)
    assert out.read_bytes() == b"OLD-INDEX-CONTENT"  # old artifact untouched


def test_dense_success_replaces_atomically(tmp_path):
    out = tmp_path / "fusion_emb.npy"
    build_dense(["a b", "c"], FakeProvider(), out)
    arr = np.load(out, mmap_mode="r")
    assert arr.shape == (2, 8)
    assert not out.with_suffix(".npy.tmp").exists()


def test_fusion_db_finalized_after_ingest(srclight_index):
    _, index_dir = srclight_index
    n = ingest.ingest_srclight(index_dir)
    assert (index_dir / "fusion.db").exists()
    assert not (index_dir / "fusion.db.tmp").exists()
    conn = store.open_db(index_dir / "fusion.db")
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n
    conn.close()


def test_lex_build_swaps_directory(srclight_index, tmp_path):
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)
    conn = store.open_db(index_dir / "fusion.db")
    out = tmp_path / "fusion_bm25"
    LexIndex.build(conn, out)
    conn.close()
    assert out.is_dir()
    assert not (tmp_path / "fusion_bm25.tmp").exists()
    assert LexIndex.load(out).search("AddGoldCoins", k=3)


def test_ingest_crash_preserves_previous_db(srclight_index, monkeypatch):
    """init_db must build into fusion.db.tmp so a crash mid-ingest keeps the old db."""
    _, index_dir = srclight_index
    ingest.ingest_srclight(index_dir)                 # good build
    before = (index_dir / "fusion.db").read_bytes()

    def boom(conn, rows):
        raise RuntimeError("ingest died mid-insert")
    monkeypatch.setattr(store, "insert_chunks", boom)
    with pytest.raises(RuntimeError):
        ingest.ingest_srclight(index_dir)
    assert (index_dir / "fusion.db").read_bytes() == before
