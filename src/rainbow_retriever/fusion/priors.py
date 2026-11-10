# src/rainbow_retriever/fusion/priors.py
from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

_SCHEMA = """
DROP TABLE IF EXISTS priors;
CREATE TABLE priors(file TEXT PRIMARY KEY, score REAL NOT NULL) WITHOUT ROWID;
"""
_HALF_LIFE_DAYS = 180.0
_FREQ_SATURATION = 50.0
_LOG_LIMIT = 2000


def _norm(p: str) -> str:
    return p.replace("\\", "/").lower()


def build_priors(index_dir: Path, worktree: str) -> int:
    """Per-file git-history prior: 0.7*recency(exp decay, 180d) + 0.3*freq(log sat).

    Uses wall clock
    at BUILD time only. Fail-soft: git absent/failed -> empty table, returns 0."""
    from rainbow_retriever.fusion import store

    last: dict[str, int] = {}
    count: dict[str, int] = {}
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "-C", worktree, "log", "--name-only",
             "--pretty=format:%ct", "-n", str(_LOG_LIMIT)],
            capture_output=True, text=True, timeout=30, check=True,
            encoding="utf-8", errors="replace").stdout
        ts = 0
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                ts = int(line)
            else:
                f = _norm(line)
                last[f] = max(last.get(f, 0), ts)
                count[f] = count.get(f, 0) + 1
    except Exception:
        last, count = {}, {}

    now = time.time()
    conn = store.open_db(Path(index_dir) / "fusion.db")
    try:
        conn.executescript(_SCHEMA)
        rows = []
        for f, t in last.items():
            recency = math.exp(-max(0.0, (now - t) / 86400.0) / _HALF_LIFE_DAYS)
            freq = min(1.0, math.log1p(count[f]) / math.log1p(_FREQ_SATURATION))
            rows.append((f, 0.7 * recency + 0.3 * freq))
        conn.executemany("INSERT INTO priors(file, score) VALUES(?,?)", sorted(rows))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def load(conn) -> dict[str, float]:
    return {r["file"]: r["score"] for r in conn.execute("SELECT * FROM priors")}


def join_rate(conn, pri: dict[str, float]) -> float:
    """Fraction of distinct chunk file keys with a prior entry. A ~0% rate on a
    non-empty priors table exposes a path-normalization mismatch between the
    build-time keys (git log paths) and query-time lookup keys (chunk files)."""
    files = {_norm(r[0] or "") for r in
             conn.execute("SELECT DISTINCT file FROM chunks")}
    if not files:
        return 0.0
    return sum(1 for f in files if f in pri) / len(files)
