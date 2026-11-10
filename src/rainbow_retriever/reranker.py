from __future__ import annotations

import dataclasses
import json
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class FakeReranker:
    """Deterministic reranker for tests. `scores` returns a fixed list (truncated
    to the document count); `by_query` maps a query string to its score list."""

    def __init__(self, scores: list[float] | None = None,
                 by_query: dict[str, list[float]] | None = None):
        self._scores = scores
        self._by_query = by_query or {}

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if query in self._by_query:
            return list(self._by_query[query])
        if self._scores is not None:
            return list(self._scores)[: len(documents)]
        # default: descending scores that preserve input order
        return [float(len(documents) - i) for i in range(len(documents))]


class HttpReranker:
    """Client for an Infinity-style `/rerank` endpoint. Fail-open: any error or a
    malformed response yields an empty score list, so the caller keeps the original
    order (search never breaks because the reranker is down)."""

    def __init__(self, endpoint: str, model: str, *, timeout: float = 10.0,
                 post_json=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post_json = post_json or self._urllib_post

    def _urllib_post(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {"model": self.model, "query": query,
                   "documents": documents, "return_documents": False}
        try:
            body = self._post_json(f"{self.endpoint}/rerank", payload)
            scores = [0.0] * len(documents)
            for r in body.get("results", []):
                i = r.get("index")
                if isinstance(i, int) and 0 <= i < len(documents):
                    scores[i] = float(r.get("relevance_score", 0.0))
            return scores
        except Exception:            # noqa: BLE001 - fail-open on any transport/parse error
            return []


def _doc_text(hit) -> str:
    # Give the cross-encoder the locator + code, not signature-only.
    return f"{hit.file}::{hit.symbol}\n{hit.snippet}"


def rerank_hits(reranker, query: str, hits: list):
    """Reorder `hits` by reranker score, descending, and rewrite each hit's
    score to fresh reciprocal rank (1, 1/2, 1/3, ...) so score order always
    agrees with the returned order -- stale pre-rerank scores contradicted the
    reranked order, and /v1/xref's score sort silently undid the rerank.
    Fail-open: returns `hits` unchanged (scores included) when there is no
    reranker, no hits, or the score list length does not match the hit count."""
    if reranker is None or not hits:
        return hits
    scores = reranker.rerank(query, [_doc_text(h) for h in hits])
    if len(scores) != len(hits):
        return hits
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [dataclasses.replace(hits[i], score=1.0 / (rank + 1))
            for rank, i in enumerate(order)]
