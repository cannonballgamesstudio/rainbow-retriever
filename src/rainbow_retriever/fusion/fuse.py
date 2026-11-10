from __future__ import annotations

from rainbow_retriever.backend import Hit

# Same doc-detection sets as the srclight adapter (duplicated on purpose so the
# adapters do not couple).
DOC_KINDS = frozenset({"section", "document", "paragraph", "heading", "page", "doc"})
DOC_EXTS = (".md", ".rst", ".txt", ".markdown")


def weighted_rrf_scored(arms: dict[str, list[int]],
                        weights: dict[str, float] | None = None,
                        k: int = 60) -> list[tuple[int, float]]:
    weights = weights or {}
    scores: dict[int, float] = {}
    for arm, rows in arms.items():
        w = weights.get(arm, 1.0)
        for rank, row in enumerate(rows):
            scores[row] = scores.get(row, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_rrf(arms: dict[str, list[int]], weights: dict[str, float] | None = None,
                 k: int = 60) -> list[int]:
    return [row for row, _ in weighted_rrf_scored(arms, weights, k)]


def is_doc(h: Hit) -> bool:
    return ((h.kind or "").lower() in DOC_KINDS
            or (h.file or "").lower().endswith(DOC_EXTS))


def demote_docs(hits: list[Hit]) -> list[Hit]:
    code = [h for h in hits if not is_doc(h)]
    docs = [h for h in hits if is_doc(h)]
    return code + docs


def dedup_by_file(hits: list[Hit], max_per_file: int = 1) -> list[Hit]:
    """Keep the first `max_per_file` hits per case-normalized file, order-preserving.

    max_per_file=1 is the historical binary file-dedup; higher caps trade
    breadth for depth under a fixed pack budget. Hits with an
    empty file key are always kept (unchanged behavior).
    """
    if max_per_file < 1:
        raise ValueError(f"max_per_file must be >= 1, got {max_per_file}")
    counts: dict[str, int] = {}
    out: list[Hit] = []
    for h in hits:
        key = (h.file or "").replace("\\", "/").lower()
        if key:
            c = counts.get(key, 0)
            if c >= max_per_file:
                continue
            counts[key] = c + 1
        out.append(h)
    return out
