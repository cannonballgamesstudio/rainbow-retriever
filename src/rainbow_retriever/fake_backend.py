from __future__ import annotations

from dataclasses import replace

from rainbow_retriever.backend import (SymbolGraph, Capabilities, IndexResult, MODES,
                            GraphEdge, FileSymbol, Structure, Hierarchy, AmbiguousSymbol,
                            ReachResult)


class FakeBackend:
    """Deterministic in-memory RetrievalBackend for tests and ragd wiring."""

    def __init__(self, hits=None, graph=None, modes=MODES, *, deps_edges=None,
                 file_symbols=None, test_edges=None, structure=None, hierarchy=None,
                 ambiguous=None, reachable_result=None,
                 supports_ops=("deps", "file", "tests", "map", "hierarchy", "reachable")):
        self._hits = list(hits or [])
        self._graph = graph
        self._modes = tuple(modes)
        self.index_calls = []
        self._deps = list(deps_edges or [])
        self._file_symbols = list(file_symbols or [])
        self._tests = list(test_edges or [])
        self._structure = structure or Structure(dirs=[], hotspots=[])
        self._hierarchy = hierarchy or Hierarchy(base=[], subclasses=[], implementors=[])
        self._ambiguous = list(ambiguous or [])
        self._reachable = reachable_result or ReachResult(
            served_ref="", entry=[], reachable=[], truncated=False)
        self._supports_ops = tuple(supports_ops)
        self.query_calls = []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            backend_id="fake",
            backend_version="fake@1+none+0",
            modes=self._modes,
            supports_callgraph=True,
            supports_dims=(),
            supports_ops=self._supports_ops,
        )

    def index(self, worktree_path: str, index_dir: str) -> IndexResult:
        self.index_calls.append((worktree_path, index_dir))
        return IndexResult(
            backend_version="fake@1+none+0",
            files_indexed=1,
            symbols=len(self._hits),
            edges=0,
            index_dir=index_dir,
        )

    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20, dims=None,
              code_only=False, rerank=False, dedup=None, max_per_file=None):
        if mode not in self._modes:
            raise ValueError(f"unsupported mode: {mode!r}")
        self.query_calls.append({"text": text, "mode": mode, "limit": limit,
                                 "code_only": code_only, "rerank": rerank, "dedup": dedup,
                                 "max_per_file": max_per_file})
        return [replace(h, repo=repo, ref=ref) for h in self._hits[:limit]]

    def ctx(self, index_dir, repo, ref, symbol, *, file=None) -> SymbolGraph | None:
        # Mirror the real backend: ambiguous name -> 409 unless `file` disambiguates.
        if self._ambiguous and not file:
            raise AmbiguousSymbol(self._ambiguous)
        if self._graph is not None and self._graph.symbol == symbol:
            return self._graph
        return None

    def deps(self, index_dir, repo, ref, symbol, *, file=None, transitive=False, max_depth=5):
        if self._ambiguous:
            raise AmbiguousSymbol(self._ambiguous)
        return self._deps

    def file_overview(self, index_dir, repo, ref, path):
        return self._file_symbols

    def tests(self, index_dir, repo, ref, symbol):
        return self._tests

    def structure(self, index_dir, repo, ref, *, path=None, max_depth=3):
        return self._structure

    def hierarchy(self, index_dir, repo, ref, symbol, *, file=None):
        if self._ambiguous:
            raise AmbiguousSymbol(self._ambiguous)
        return self._hierarchy

    def reachable(self, index_dir, repo, ref, symbols, direction="callees",
                  max_depth=5, file=None, limit=2000):
        if self._ambiguous:
            raise AmbiguousSymbol(self._ambiguous)
        return self._reachable
