# src/rainbow_retriever/fusion/trigram.py
from __future__ import annotations

from pathlib import Path

_MIN_IDENT = 4      # identifiers shorter than this are not indexed
_MIN_QUERY = 6      # query tokens shorter than this are not anchors
_SCHEMA = """
DROP TABLE IF EXISTS trigrams;
CREATE TABLE trigrams(
  tri TEXT NOT NULL,
  emb_row INTEGER NOT NULL,
  PRIMARY KEY(tri, emb_row)
) WITHOUT ROWID;
"""


def _grams(tok: str) -> set[str]:
    t = tok.lower()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def build_trigrams(index_dir: Path) -> int:
    """Identifier-trigram postings (Zoekt/Blackbird pattern, identifier-scoped).

    Additive + idempotent + augmentable, same contract as adjacency."""
    from rainbow_retriever.fusion import store
    from rainbow_retriever.fusion.lex import tokenize

    conn = store.open_db(Path(index_dir) / "fusion.db")
    try:
        rows: set[tuple[str, int]] = set()
        for r in conn.execute("SELECT emb_row, lex_name, language FROM chunks"):
            for tok in tokenize(r["lex_name"] or "", r["language"]):
                if len(tok) >= _MIN_IDENT:
                    rows.update((g, r["emb_row"]) for g in _grams(tok))
        conn.executescript(_SCHEMA)
        conn.executemany("INSERT INTO trigrams(tri, emb_row) VALUES(?,?)",
                         sorted(rows))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def search(conn, query_ids: str, *, cap: int = 50,
           min_frac: float = 0.5) -> list[tuple[int, float]]:
    """Containment score per chunk: |matched query-token trigrams| / |token trigrams|,
    max over the query's identifier tokens. Catches prefix/partial identifier
    matches that whole-token BM25 misses."""
    best: dict[int, float] = {}
    for tok in (query_ids or "").split():
        if len(tok) < _MIN_QUERY:
            continue
        grams = sorted(_grams(tok))
        if not grams:
            continue
        q = (f"SELECT emb_row, COUNT(*) c FROM trigrams "
             f"WHERE tri IN ({','.join('?' * len(grams))}) GROUP BY emb_row")
        for r in conn.execute(q, grams):
            frac = r["c"] / len(grams)
            if frac >= min_frac:
                best[r["emb_row"]] = max(best.get(r["emb_row"], 0.0), frac)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:cap]
