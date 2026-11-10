from __future__ import annotations

import re

# Finding from prior work (arXiv 2605.18561): subtoken decomposition
# HURTS snake_case-conventional languages; whole identifiers carry the signal there.
NOSPLIT_LANGUAGES = frozenset({"python", "gdscript"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str, language: str | None = None, *, query: bool = False) -> list[str]:
    lang = (language or "").lower()
    split = query or lang not in NOSPLIT_LANGUAGES
    out: list[str] = []
    for ident in _IDENT.findall(text or ""):
        out.append(ident.lower())
        if not split:
            continue
        parts: list[str] = []
        for piece in ident.split("_"):
            parts.extend(_CAMEL.split(piece))
        if len(parts) > 1:
            out.extend(p.lower() for p in parts if len(p) >= 2)
    return out


class LexIndex:
    """bm25s index over fusion chunks. Doc position == chunks.emb_row.

    Field weighting is done by token duplication (name ×w_name, signature ×w_sig,
    body ×w_body) -- BM25's k1 saturation keeps this bounded, and it works with any
    plain-BM25 engine (BLUiR-style structured retrieval).
    """

    def __init__(self, retriever):
        self._retriever = retriever

    @classmethod
    def build(cls, conn, out_dir, w_name: int = 4, w_sig: int = 2, w_body: int = 1):
        """Build the bm25s index and save it into out_dir.

        Saves into a sibling `<out_dir>.tmp` directory first, then swaps it into
        out_dir (rmtree old + rename tmp). The swap itself is NOT atomic -- there
        is a small window between rmtree(out_dir) and the rename where out_dir
        doesn't exist. Acceptable here: cached LexIndex handles (FIFO-2, see
        FusionBackend._handles) are keyed on fusion.db mtime, and cross-process
        atomicity is the registry's job (fresh directory + repoint), not this
        directory's. This only narrows the crash window to that rename.
        """
        import bm25s
        import shutil
        from pathlib import Path

        from rainbow_retriever.fusion import store

        corpus: list[list[str]] = []
        for name, sig, body, lang in store.iter_lex_corpus(conn):
            toks: list[str] = []
            toks += tokenize(name, lang) * w_name
            toks += tokenize(sig, lang) * w_sig
            toks += tokenize(body, lang) * w_body
            corpus.append(toks or ["__empty__"])
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        out_dir = Path(out_dir)
        tmp_dir = Path(str(out_dir) + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(tmp_dir))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
        return cls(retriever)

    @classmethod
    def load(cls, out_dir):
        import bm25s
        return cls(bm25s.BM25.load(str(out_dir), mmap=True))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query, query=True)
        if not toks:
            return []
        if k <= 0:
            return []
        n_docs = self._retriever.scores["num_docs"]
        k = min(k, n_docs)
        if k <= 0:                                 # empty corpus (n_docs == 0)
            return []
        idx, scores = self._retriever.retrieve([toks], k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if s > 0.0]


# Identifier-shaped token extraction for the "ids" lexical mode: on long natural-
# language queries, OR-scored BM25 over the full text floods results with prose-
# matching docs (measured: docs/benchmark.md). Feeding the lexical
# arm only the query's identifier-shaped tokens makes it surgical on anchored
# queries and silent on anchorless ones (rule-based Reformulate-Retrieve-Localize).
_ID_SHAPES = re.compile(
    r"`([^`\n]{2,80})`"                                            # backticked spans
    r"|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b"  # dotted.path
    r"|\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"                            # snake_case
    r"|\b([a-z][a-z0-9]+[A-Z][A-Za-z0-9]+)\b"                      # camelCase
    r"|\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+)\b"                      # PascalCase
    r"|\b([A-Z][A-Z0-9_]{3,})\b")                                  # SCREAMING_CONST


def id_tokens(text: str, cap: int = 64) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for m in _ID_SHAPES.finditer(text or ""):
        tok = next(g for g in m.groups() if g)
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            out.append(tok)
            if len(out) >= cap:
                break
    return " ".join(out)


# Anchor strength rule (frozen): strong = >=1 id token of len>=6 containing
# '.', '_', or a lowercase->uppercase transition; weak = id tokens present but
# none strong; none = no id tokens. Reported in the server's /v1/query
# `anchored` field.
_STRONG_SHAPE = re.compile(r"\.|_|[a-z][A-Z]")


def anchor_strength(text: str) -> str:
    toks = id_tokens(text).split()
    if not toks:
        return "none"
    if any(len(t) >= 6 and _STRONG_SHAPE.search(t) for t in toks):
        return "strong"
    return "weak"
