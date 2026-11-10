from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

MODES = ("keyword", "semantic", "hybrid")


@dataclass(frozen=True)
class Hit:
    repo: str
    ref: str
    file: str
    line: int
    symbol: str
    score: float
    snippet: str
    kind: str


@dataclass(frozen=True)
class GraphEdge:
    symbol: str
    file: str
    line: int
    kind: str
    edge_type: str
    confidence: float


@dataclass(frozen=True)
class SymbolGraph:
    symbol: str
    defined_file: str
    defined_line: int
    signature: Optional[str]
    callers: list[GraphEdge]
    callees: list[GraphEdge]


@dataclass(frozen=True)
class FileSymbol:
    symbol: str
    kind: str
    line: int
    signature: Optional[str]


@dataclass(frozen=True)
class DirEntry:
    path: str
    files: int
    symbols: int
    languages: list[str]


@dataclass(frozen=True)
class HotspotFile:
    path: str
    language: Optional[str]
    lines: int
    symbols: int


@dataclass(frozen=True)
class Structure:
    dirs: list[DirEntry]
    hotspots: list[HotspotFile]


@dataclass(frozen=True)
class Hierarchy:
    base: list[GraphEdge]
    subclasses: list[GraphEdge]
    implementors: list[GraphEdge]


@dataclass(frozen=True)
class ReachNode:
    symbol: str
    file: str
    line: int
    kind: str
    depth: int


@dataclass(frozen=True)
class ReachResult:
    served_ref: str
    entry: list[str]
    reachable: list[ReachNode]
    truncated: bool


class AmbiguousSymbol(Exception):
    """A symbol name matched more than one definition; carries the candidates
    (each {symbol, file, line}) so the caller can re-address with a file."""

    def __init__(self, candidates: list[dict]):
        super().__init__(f"ambiguous symbol: {len(candidates)} matches")
        self.candidates = candidates


@dataclass(frozen=True)
class Capabilities:
    backend_id: str
    backend_version: str
    modes: tuple[str, ...]
    supports_callgraph: bool
    supports_dims: tuple[int, ...]
    backend_running_version: str = ""
    supports_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexResult:
    backend_version: str
    files_indexed: int
    symbols: int
    edges: int
    index_dir: str


@runtime_checkable
class RetrievalBackend(Protocol):
    def capabilities(self) -> Capabilities: ...
    def index(self, worktree_path: str, index_dir: str) -> IndexResult: ...
    def query(self, index_dir: str, repo: str, ref: str, text: str,
              mode: str = "hybrid", limit: int = 20,
              dims: int | None = None, code_only: bool = False,
              rerank: bool = False) -> list[Hit]: ...
    def ctx(self, index_dir: str, repo: str, ref: str,
            symbol: str) -> SymbolGraph | None: ...
    def deps(self, index_dir: str, repo: str, ref: str, symbol: str, *,
             file: str | None = None, transitive: bool = False,
             max_depth: int = 5) -> list[GraphEdge]: ...
    def file_overview(self, index_dir: str, repo: str, ref: str,
                      path: str) -> list[FileSymbol]: ...
    def tests(self, index_dir: str, repo: str, ref: str,
              symbol: str) -> list[GraphEdge]: ...
    def structure(self, index_dir: str, repo: str, ref: str, *,
                  path: str | None = None, max_depth: int = 3) -> Structure: ...
    def hierarchy(self, index_dir: str, repo: str, ref: str, symbol: str, *,
                  file: str | None = None) -> Hierarchy: ...
    def reachable(self, index_dir: str, repo: str, ref: str, symbols: list[str],
                  direction: str = "callees", max_depth: int = 5,
                  file: str | None = None, limit: int = 2000) -> ReachResult: ...
