# src/rainbow_retriever/fusion/ingest.py
from __future__ import annotations

from pathlib import Path

from rainbow_retriever.fusion import store
from rainbow_retriever.fusion.identity import FUSION_FORMAT_VERSION, srclight_format

_SNIPPET_LINES = 15
_SNIPPET_CHARS = 600
_EMBED_CONTENT_CHARS = 1200
_USES_CAP = 5


def _snippet(sym) -> str:
    text = (sym.content or sym.signature or sym.doc_comment or sym.name or "").strip()
    return "\n".join(text.splitlines()[:_SNIPPET_LINES])[:_SNIPPET_CHARS]


def _uses_line(names: list[str]) -> str | None:
    """Build the `uses: a, b, c` header line: dedup order-preserving, cap at 5.

    Pure helper (header_imports option) -- None when there are no
    names, so callers can omit the line entirely rather than emit `uses: `.
    """
    deduped: list[str] = []
    for n in names:
        if n and n not in deduped:
            deduped.append(n)
    deduped = deduped[:_USES_CAP]
    if not deduped:
        return None
    return "uses: " + ", ".join(deduped)


def _norm_path(p: str | None) -> str:
    return (p or "").replace("/", "\\")


def _cross_file_callee_names(sdb, sym) -> list[str]:
    """Names of symbols `sym` calls that live in a DIFFERENT file (srclight
    public API: get_callees returns [{"symbol": SymbolRecord, ...}, ...]).
    One query per symbol -- acceptable at index time (build_adjacency already
    pays this cost for the graph arm)."""
    own_file = _norm_path(sym.file_path)
    names = []
    for e in sdb.get_callees(sym.id):
        callee = e["symbol"]
        if _norm_path(callee.file_path) != own_file:
            names.append(callee.name)
    return names


def _embed_text(relpath: str, language: str, parent: str | None, sym,
                uses_line: str | None = None) -> str:
    # Context header: relpath, language tag, enclosing class, [uses:
    # cross-file callees -- header_imports flag], signature, doc
    # comment, then capped code.
    header = f"{relpath} [{language}]" + (f" {parent}" if parent else "")
    parts = [header]
    if uses_line:
        parts.append(uses_line)
    parts.append(sym.signature or sym.name or "")
    if sym.doc_comment:
        parts.append(sym.doc_comment.strip())
    if sym.content:
        parts.append(sym.content[:_EMBED_CONTENT_CHARS])
    return "\n".join(parts)


def ingest_srclight(index_dir: Path, *, header_imports: bool = False) -> int:
    """Read srclight's index.db (public API only) into fusion.db. Returns chunk count.

    header_imports (default False, leaves embed_text unchanged):
    when True, each chunk's embed_text gains a `uses: name1, name2, ...` line
    naming the cross-file symbols it calls (capped at 5, omitted when empty).
    Experimental flag; recorded in meta["header_imports"].
    """
    from srclight.db import Database

    index_dir = Path(index_dir)
    sdb = Database(index_dir / "index.db")
    sdb.open()
    try:
        conn = store.init_db(index_dir / "fusion.db")
        rows: list[dict] = []
        for path in sorted(sdb.all_file_paths()):
            frec = sdb.get_file(path)
            language = (getattr(frec, "language", None) or "").lower()
            for sym in sdb.symbols_in_file(path):
                parent = None
                if sym.parent_symbol_id is not None:
                    p = sdb.get_symbol_by_id(sym.parent_symbol_id)
                    if p is not None and p.kind in ("class", "struct", "interface",
                                                    "namespace", "module"):
                        parent = p.name
                relpath = (sym.file_path or path).replace("\\", "/")
                uses_line = None
                if header_imports:
                    uses_line = _uses_line(_cross_file_callee_names(sdb, sym))
                rows.append({
                    "id": sym.id,
                    "emb_row": len(rows),
                    "file": relpath,
                    "line": sym.start_line or 0,
                    "name": sym.name or "",
                    "kind": sym.kind or "",
                    "signature": sym.signature,
                    "language": language,
                    "parent": parent,
                    "snippet": _snippet(sym),
                    "embed_text": _embed_text(relpath, language, parent, sym, uses_line),
                    "lex_name": " ".join(filter(None, [sym.name, sym.qualified_name])),
                    "lex_sig": sym.signature or "",
                    "lex_body": " ".join(filter(None, [sym.doc_comment, sym.content])),
                })
        store.insert_chunks(conn, rows)
        store.meta_set(conn, "fusion_fmt", str(FUSION_FORMAT_VERSION))
        store.meta_set(conn, "srclight_fmt", srclight_format())
        store.meta_set(conn, "chunks", str(len(rows)))
        store.meta_set(conn, "header_imports", "1" if header_imports else "0")
        store.finalize_db(conn, index_dir / "fusion.db")
        return len(rows)
    except Exception:
        conn.close()
        raise
    finally:
        sdb.close()
