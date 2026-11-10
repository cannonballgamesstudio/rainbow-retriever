# src/rainbow_retriever/fusion/store.py
from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  id        INTEGER PRIMARY KEY,          -- srclight symbol id
  emb_row   INTEGER UNIQUE NOT NULL,      -- canonical row: npy row == bm25s doc position
  file      TEXT NOT NULL,
  line      INTEGER NOT NULL,
  name      TEXT NOT NULL,
  kind      TEXT NOT NULL,
  signature TEXT,
  language  TEXT,
  parent    TEXT,
  snippet   TEXT NOT NULL,
  embed_text TEXT NOT NULL,
  lex_name  TEXT NOT NULL,
  lex_sig   TEXT NOT NULL,
  lex_body  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _tmp_of(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def init_db(path: Path) -> sqlite3.Connection:
    """Create the schema in <path>.tmp; finalize_db(conn, path) swaps it in.

    Building into a temp file means a crash mid-ingest leaves any previous
    fusion.db untouched (the registry's ready-pointer covers cross-artifact
    atomicity; this covers single-artifact truncation)."""
    path = Path(path)
    tmp = _tmp_of(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    conn.executescript(_SCHEMA)
    return conn


def finalize_db(conn: sqlite3.Connection, path: Path) -> Path:
    """Commit + close conn, then atomically swap <path>.tmp into path.

    sqlite3.Connection is a C type with no __dict__, so it cannot carry the
    target path as an attribute -- callers pass path explicitly here (and to
    init_db), matching it up themselves."""
    import os
    path = Path(path)
    conn.commit()
    conn.close()
    os.replace(_tmp_of(path), path)
    return path


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def open_db_ro(path: Path) -> sqlite3.Connection:
    """Read-only open for every query path. sqlite3.connect(path) silently
    CREATES an empty file on a missing path -- masking the real error and
    planting a zero-schema fusion.db for the next reader. URI mode=ro cannot
    create; we additionally fail fast (path in the message) on a missing or
    uninitialized db. Build/augment paths keep the writable open_db."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fusion db missing: {path}")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ok = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    if ok is None:
        conn.close()
        raise RuntimeError(f"fusion db uninitialized (no chunks table): {path}")
    return conn


def insert_chunks(conn: sqlite3.Connection, rows: list[dict]) -> None:
    cols = ("id", "emb_row", "file", "line", "name", "kind", "signature",
            "language", "parent", "snippet", "embed_text",
            "lex_name", "lex_sig", "lex_body")
    conn.executemany(
        f"INSERT INTO chunks({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows])
    conn.commit()


def chunks_by_rows(conn: sqlite3.Connection, rows: list[int]) -> list[sqlite3.Row]:
    if not rows:
        return []
    q = (f"SELECT * FROM chunks WHERE emb_row IN ({','.join('?' * len(rows))})")
    by_row = {r["emb_row"]: r for r in conn.execute(q, rows)}
    return [by_row[i] for i in rows if i in by_row]


def iter_lex_corpus(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    return [(r["lex_name"], r["lex_sig"], r["lex_body"], r["language"] or "")
            for r in conn.execute("SELECT * FROM chunks ORDER BY emb_row")]


def embed_texts(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in
            conn.execute("SELECT embed_text FROM chunks ORDER BY emb_row")]


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
    conn.commit()


def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None
