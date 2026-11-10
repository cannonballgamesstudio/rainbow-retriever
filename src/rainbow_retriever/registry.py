import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    repo: str
    ref: str
    head_sha: str
    worktree: str
    status: str
    indexed_at: int
    backend_id: str
    backend_version: str
    config_hash: str
    clone_dataset: str | None = None
    origin_snapshot: str | None = None


_COLS = ("repo, ref, head_sha, worktree, status, indexed_at, "
         "backend_id, backend_version, config_hash, clone_dataset, origin_snapshot")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS refs (
  repo            TEXT NOT NULL,
  ref             TEXT NOT NULL,
  head_sha        TEXT NOT NULL,
  worktree        TEXT NOT NULL,
  status          TEXT NOT NULL,
  indexed_at      INTEGER NOT NULL,
  backend_id      TEXT NOT NULL DEFAULT 'srclight',
  backend_version TEXT NOT NULL,
  config_hash     TEXT NOT NULL,
  clone_dataset   TEXT,
  origin_snapshot TEXT,
  PRIMARY KEY (repo, ref)
);
"""

VALID_STATUS = {"indexing", "ready", "failed", "stale"}


class Registry:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @classmethod
    def init_db(cls, db_path: str) -> "Registry":
        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            existing = {row[1] for row in con.execute("PRAGMA table_info(refs)")}
            for col in ("clone_dataset", "origin_snapshot"):
                if col not in existing:
                    con.execute(f"ALTER TABLE refs ADD COLUMN {col} TEXT")
            con.commit()
        finally:
            con.close()
        return cls(db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def upsert_ref(self, repo, ref, head_sha, worktree, status, indexed_at,
                   backend_version, config_hash, backend_id="srclight") -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status: {status!r}")
        con = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO refs "
                "(repo, ref, head_sha, worktree, status, indexed_at, "
                " backend_id, backend_version, config_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (repo, ref, head_sha, worktree, status, indexed_at,
                 backend_id, backend_version, config_hash),
            )
            con.commit()
        finally:
            con.close()

    def set_status(self, repo, ref, status) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status: {status!r}")
        con = self._connect()
        try:
            cur = con.execute(
                "UPDATE refs SET status=? WHERE repo=? AND ref=?",
                (status, repo, ref),
            )
            if cur.rowcount == 0:
                raise KeyError(f"ref not found: {repo!r}/{ref!r}")
            con.commit()
        finally:
            con.close()

    def resolve(self, repo, ref) -> "Ref | None":
        con = self._connect()
        try:
            row = con.execute(
                f"SELECT {_COLS} FROM refs WHERE repo=? AND ref=?",
                (repo, ref),
            ).fetchone()
        finally:
            con.close()
        return Ref(*row) if row else None

    def stale_refs(self, backend_id, backend_version, config_hash) -> "list[Ref]":
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT {_COLS} FROM refs "
                "WHERE backend_id=? AND (backend_version<>? OR config_hash<>?)",
                (backend_id, backend_version, config_hash),
            ).fetchall()
        finally:
            con.close()
        return [Ref(*r) for r in rows]

    def atomic_swap(self, repo, ref, worktree, head_sha, indexed_at,
                    backend_version, config_hash, clone_dataset=None,
                    origin_snapshot=None) -> None:
        con = self._connect()
        try:
            with con:  # single transaction; commits on success, rolls back on error
                con.execute(
                    "INSERT INTO refs "
                    "(repo, ref, head_sha, worktree, status, indexed_at, "
                    " backend_id, backend_version, config_hash, "
                    " clone_dataset, origin_snapshot) "
                    "VALUES (?, ?, ?, ?, 'ready', ?, 'srclight', ?, ?, ?, ?) "
                    "ON CONFLICT(repo, ref) DO UPDATE SET "
                    "  head_sha=excluded.head_sha, "
                    "  worktree=excluded.worktree, "
                    "  status='ready', "
                    "  indexed_at=excluded.indexed_at, "
                    "  backend_version=excluded.backend_version, "
                    "  config_hash=excluded.config_hash, "
                    "  clone_dataset=excluded.clone_dataset, "
                    "  origin_snapshot=excluded.origin_snapshot",
                    (repo, ref, head_sha, worktree, indexed_at,
                     backend_version, config_hash, clone_dataset, origin_snapshot),
                )
        finally:
            con.close()

    def delete_ref(self, repo, ref) -> None:
        con = self._connect()
        try:
            with con:
                con.execute("DELETE FROM refs WHERE repo=? AND ref=?", (repo, ref))
        finally:
            con.close()

    def list_refs(self, repo=None) -> "list[Ref]":
        con = self._connect()
        try:
            if repo is None:
                rows = con.execute(f"SELECT {_COLS} FROM refs ORDER BY repo, ref").fetchall()
            else:
                rows = con.execute(
                    f"SELECT {_COLS} FROM refs WHERE repo=? ORDER BY ref", (repo,)
                ).fetchall()
        finally:
            con.close()
        return [Ref(*r) for r in rows]


class RegistryRO:
    """Read-only registry accessor for readers (ragd). Opens sqlite file:?mode=ro."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def resolve(self, repo, ref) -> "Ref | None":
        con = self._connect()
        try:
            row = con.execute(
                f"SELECT {_COLS} FROM refs WHERE repo=? AND ref=? AND status='ready'",
                (repo, ref),
            ).fetchone()
        finally:
            con.close()
        return Ref(*row) if row else None

    def get_ref(self, repo, ref) -> "Ref | None":
        con = self._connect()
        try:
            row = con.execute(
                f"SELECT {_COLS} FROM refs WHERE repo=? AND ref=?", (repo, ref)
            ).fetchone()
        finally:
            con.close()
        return Ref(*row) if row else None

    def list_refs(self, repo=None) -> "list[Ref]":
        con = self._connect()
        try:
            if repo is None:
                rows = con.execute(f"SELECT {_COLS} FROM refs ORDER BY repo, ref").fetchall()
            else:
                rows = con.execute(
                    f"SELECT {_COLS} FROM refs WHERE repo=? ORDER BY ref", (repo,)
                ).fetchall()
        finally:
            con.close()
        return [Ref(*r) for r in rows]
