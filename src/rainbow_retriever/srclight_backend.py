from __future__ import annotations

from rainbow_retriever.backend import (Capabilities, Hit, GraphEdge, SymbolGraph, IndexResult,
                            FileSymbol, DirEntry, HotspotFile, Structure, Hierarchy,
                            AmbiguousSymbol, ReachNode, ReachResult)

# Instruction-tuned embedding models (e.g. Qwen3-Embedding) are trained for
# ASYMMETRIC retrieval: the query carries a task instruction, documents stay
# plain. srclight indexes docs plain (correct) but embeds the query raw.
# Prefixing the query with this code-search instruction improved recall in our
# tests (the exact wording beat 7 alternatives). Config-gated via
# RR_QUERY_INSTRUCT.
DEFAULT_QUERY_INSTRUCT = "Given a code-search intent, retrieve the most relevant source code."


class SrclightBackend:
    """RetrievalBackend adapter that calls Srclight's query primitives in-process.

    No Srclight CLI, no MCP server, no subprocess.
    """

    def __init__(self, embed_model: str | None = None, dims: int | None = None,
                 identity_mode: str = "minor", reranker=None,
                 query_instruct: str | None = None, dedup_file: bool = False):
        self.embed_model = embed_model
        self.dims = dims
        self.identity_mode = identity_mode
        self.reranker = reranker
        # query_instruct: instruction text prefixed to the EMBEDDED query only
        # (FTS/keyword search keeps the raw text). dedup_file: collapse the hit
        # list to one hit per file (best-scoring), over-fetching to backfill.
        self.query_instruct = query_instruct
        self.dedup_file = dedup_file

    def _index_identity(self) -> str:
        import srclight
        if self.identity_mode == "full":
            fmt = srclight.__version__
        else:
            fmt = getattr(srclight, "INDEX_FORMAT_VERSION", None)
            if fmt is None:
                # Conservative fallback: patches within a minor are assumed format-compatible
                # until upstream ships a stable INDEX_FORMAT_VERSION marker.
                fmt = "fmt-" + ".".join(srclight.__version__.split(".")[:2])
        return f"srclight@{fmt}+{self.embed_model or 'none'}+{self.dims or 0}"

    def _running_version(self) -> str:
        import srclight
        return srclight.__version__          # diagnostics/logging only -- never stored

    def _version(self) -> str:               # the value stored + compared (now the identity)
        return self._index_identity()

    def capabilities(self) -> Capabilities:
        modes = ["keyword"]
        if self.embed_model:
            modes += ["semantic", "hybrid"]
        return Capabilities(
            backend_id="srclight",
            backend_version=self._version(),
            backend_running_version=self._running_version(),
            modes=tuple(modes),
            supports_callgraph=True,
            supports_dims=(self.dims,) if self.dims else (),
            supports_ops=("deps", "file", "tests", "map", "hierarchy", "reachable"),
        )

    def index(self, worktree_path: str, index_dir: str) -> IndexResult:
        from pathlib import Path
        from srclight.db import Database
        from srclight.indexer import Indexer, IndexConfig

        from rainbow_retriever import srclight_patches
        srclight_patches.apply()  # capture pointer-returning C/C++ functions srclight drops

        idx = Path(index_dir)
        idx.mkdir(parents=True, exist_ok=True)
        db = Database(idx / "index.db")
        try:
            db.open()
            db.initialize()                 # CREATE TABLE IF NOT EXISTS -- safe to re-run
            root = Path(worktree_path)
            Indexer(db, IndexConfig(root=root, embed_model=self.embed_model)).index(root)
            # Cumulative DB totals (db.stats keys: files, symbols, edges). Srclight's
            # incremental indexer returns per-run DELTAS (0 for unchanged files on a
            # re-run); reporting the index's STATE keeps all three fields consistent.
            db_stats = db.stats()
            # Guard the silent-degradation path: srclight's embedder fails SOFT (returns 0
            # on a provider error), so a dead embedding server yields a keyword-only index
            # that still answers mode="hybrid" -- a crippled retriever masquerading as hybrid.
            # If an embed model was requested, refuse an index that isn't actually embedded.
            if self.embed_model:
                try:
                    cov = (db.embedding_stats() or {}).get("coverage_pct")
                except AttributeError:
                    cov = None           # older db / test double without embedding_stats
                if cov is not None and cov < 90.0:
                    raise RuntimeError(
                        f"embedding coverage {cov:.1f}% (<90%) with embed_model={self.embed_model!r}: "
                        "the embedding provider likely failed -- refusing to ship a keyword-only "
                        "'hybrid' index. Check the embedding server, then re-index.")
        finally:
            db.close()
        return IndexResult(
            backend_version=self._version(),
            files_indexed=db_stats["files"],
            symbols=db_stats["symbols"],
            edges=db_stats["edges"],
            index_dir=str(idx),
        )

    # srclight indexes docs (markdown sections, etc.) as symbols. It's tempting to filter them
    # from code search, BUT docs helped recall on repos with design docs that name
    # the involved code files (they act as a map for cross-cutting features). So docs are KEPT
    # by default; `code_only=True` is an opt-in that demotes docs below code (over-fetch backfills)
    # for repos where docs are genuinely noise.
    _DOC_KINDS = frozenset({"section", "document", "paragraph", "heading", "page", "doc"})
    _DOC_EXTS = (".md", ".rst", ".txt", ".markdown")
    _DOC_OVERFETCH = 3
    _RERANK_OVERFETCH = 5   # over-fetch candidates before reranking, then truncate to limit
    _DEDUP_OVERFETCH = 5    # over-fetch before file-dedup so distinct files backfill to limit

    def _is_doc(self, h) -> bool:
        return ((h.kind or "").lower() in self._DOC_KINDS
                or (h.file or "").lower().endswith(self._DOC_EXTS))

    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20, dims=None,
              code_only=False, rerank=False, dedup=None, max_per_file=None):
        # dedup override: None -> self.dedup_file default; False -> keep multiple
        # chunks per file (depth pack). See fusion_backend.query for the rationale.
        from pathlib import Path
        from srclight.db import Database
        from rainbow_retriever.reranker import rerank_hits

        supported = self.capabilities().modes
        if mode not in supported:
            raise ValueError(f"unsupported mode {mode!r}; this backend supports {supported}")
        active_rerank = rerank and self.reranker is not None and mode != "keyword"
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            fetch = limit
            if code_only:
                fetch = max(fetch, limit * self._DOC_OVERFETCH)
            if active_rerank:
                fetch = max(fetch, limit * self._RERANK_OVERFETCH)
            if max_per_file is not None and max_per_file < 1:
                raise ValueError(f"max_per_file must be >= 1, got {max_per_file}")
            if (self.dedup_file if dedup is None else dedup) or max_per_file is not None:
                fetch = max(fetch, limit * self._DEDUP_OVERFETCH)
            if mode == "keyword":
                rows = db.search_symbols(text, limit=fetch)
            else:
                rows = self._semantic_or_hybrid(db, text, mode, fetch, dims)
            hits = [self._row_to_hit(db, repo, ref, r) for r in rows]
            if active_rerank:
                hits = rerank_hits(self.reranker, text, hits)   # precision pass over the pool
            if code_only:
                # opt-in: demote docs below code (stable), code backfills.
                code = [h for h in hits if not self._is_doc(h)]
                docs = [h for h in hits if self._is_doc(h)]
                hits = code + docs
            do_dedup = self.dedup_file if dedup is None else dedup
            if max_per_file is not None or do_dedup:
                # one..cap hits per file (first == best in the current order), so a
                # top-K of distinct FILES backfills from the over-fetched pool.
                # max_per_file generalizes the binary dedup: when set
                # it overrides the dedup decision; cap=1 == historical behavior.
                cap = max_per_file if max_per_file is not None else 1
                counts, deduped = {}, []
                for h in hits:
                    key = (h.file or "").replace("\\", "/").lower()
                    if key:
                        c = counts.get(key, 0)
                        if c >= cap:
                            continue
                        counts[key] = c + 1
                    deduped.append(h)
                hits = deduped
            return hits[:limit]
        finally:
            db.close()

    def _semantic_or_hybrid(self, db, text, mode, limit, dims):
        from srclight.embeddings import get_provider, vector_to_bytes, rrf_merge

        if not self.embed_model:
            raise RuntimeError(
                "semantic/hybrid requires an embed_model; this backend was built without one"
            )
        provider = get_provider(self.embed_model)
        # Asymmetric retrieval: instruction goes on the EMBEDDED query only; the
        # FTS/keyword search below keeps the raw text (an instruction would just be
        # noise tokens to bm25).
        embed_text = text
        if self.query_instruct:
            embed_text = f"Instruct: {self.query_instruct}\nQuery: {text}"
        query_vec = provider.embed_one(embed_text)
        # Prefer the index's ACTUAL embedding dims (authoritative, like upstream's
        # hybrid_search) so a config/provider mismatch can't silently blank the vector arm:
        # db.vector_search filters WHERE e.dimensions = ?, so a wrong dims returns 0 rows.
        try:
            index_dims = (db.embedding_stats() or {}).get("dimensions")
        except AttributeError:
            index_dims = None            # older db / test double without embedding_stats
        dimensions = dims or index_dims or self.dims or provider.dimensions
        emb_rows = db.vector_search(vector_to_bytes(query_vec), dimensions, limit=limit)
        if mode == "semantic":
            return emb_rows
        fts_rows = db.search_symbols(text, limit=limit)
        return rrf_merge(fts_rows, emb_rows)

    def ctx(self, index_dir, repo, ref, symbol, *, file=None) -> SymbolGraph | None:
        from pathlib import Path
        from srclight.db import Database

        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            # Resolve like deps/hierarchy: detect ambiguity (raise AmbiguousSymbol -> 409) and
            # honor `file`, instead of get_symbol_by_name's silent first-match -- which could
            # return a docs/markdown symbol over the code one with no signal.
            sid = self._resolve_symbol_id(db, symbol, file)
            if sid is None:
                return None
            sym = db.get_symbol_by_id(sid)
            if sym is None:
                return None
            callers = [self._edge(r) for r in db.get_callers(sym.id)]
            callees = [self._edge(r) for r in db.get_callees(sym.id)]
            return SymbolGraph(
                symbol=sym.name,
                defined_file=sym.file_path,
                defined_line=sym.start_line,
                signature=sym.signature,
                callers=callers,
                callees=callees,
            )
        finally:
            db.close()

    def _edge(self, r) -> GraphEdge:
        s = r["symbol"]   # SymbolRecord
        return GraphEdge(
            symbol=s.name,
            file=s.file_path,
            line=s.start_line,
            kind=s.kind,
            edge_type=r.get("edge_type", ""),
            confidence=float(r.get("confidence", 1.0)),
        )

    @staticmethod
    def _snippet_from_symbol(sym, max_lines: int = 15, max_chars: int = 600) -> str:
        """A real code preview for a hit: the symbol's source, capped.

        The whole point of rag is to hand the caller the code so it need not open
        the file. Keyword rows carry only an FTS highlight (`>>>name<<<`) and
        vector rows carry no snippet at all -- so prefer the symbol's stored
        `content`, falling back to signature/doc/name.
        """
        text = (sym.content or sym.signature or sym.doc_comment or sym.name or "").strip()
        if not text:
            return ""
        return "\n".join(text.splitlines()[:max_lines])[:max_chars]

    def _row_to_hit(self, db, repo, ref, r) -> Hit:
        # Score is mode-relative (list order is authoritative): keyword rows carry
        # `rank` (FTS bm25), vector rows `similarity` (cosine), hybrid `rrf_score`.
        score = r.get("rrf_score")
        if score is None:
            score = r.get("similarity")
        if score is None:
            score = r.get("rank")
        file = r.get("file", "")
        name = r.get("name", "")
        kind = r.get("kind", "")
        # keyword rows carry no line; vector rows have start_line; both resolve via
        # the symbol record, which also gives us the real source for the snippet.
        line = r.get("start_line") or 0
        snippet = ""
        sid = r.get("symbol_id")
        if sid is not None:
            sym = db.get_symbol_by_id(sid)
            if sym is not None:
                line = sym.start_line or line
                file = file or (sym.file_path or "")
                name = name or (sym.name or "")
                kind = kind or (sym.kind or "")
                snippet = self._snippet_from_symbol(sym)
        if not snippet:
            snippet = r.get("snippet", "")
        return Hit(
            repo=repo, ref=ref,
            file=file,
            line=line,
            symbol=name,
            score=float(score) if score is not None else 0.0,
            snippet=snippet,
            kind=kind,
        )

    def _resolve_symbol_id(self, db, name, file):
        # limit=10000 avoids truncating before the file filter: without an explicit
        # generous limit, srclight's default (20) can silently drop the intended
        # match when a name has >20 definitions across the index.
        syms = db.get_symbols_by_name(name, limit=10000)
        if file:
            syms = [s for s in syms if (s.file_path or "").replace("\\", "/").endswith(file.replace("\\", "/"))]
        if not syms:
            return None
        if len(syms) > 1:
            raise AmbiguousSymbol([
                {"symbol": s.name, "file": s.file_path, "line": s.start_line} for s in syms])
        return syms[0].id

    def deps(self, index_dir, repo, ref, symbol, *,
             file=None, transitive=False, max_depth=5) -> list[GraphEdge]:
        from pathlib import Path
        from srclight.db import Database
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            sid = self._resolve_symbol_id(db, symbol, file)
            if sid is None:
                return []
            return [self._edge(r) for r in db.get_dependents(sid, transitive=transitive, max_depth=max_depth)]
        finally:
            db.close()

    def reachable(self, index_dir, repo, ref, symbols, direction="callees",
                  max_depth=5, file=None, limit=2000) -> ReachResult:
        """BFS over symbol_edges in the callee/caller direction from one or more
        entry symbols. Structurally identical to db.get_dependents's transitive
        caller-walk (srclight db.py:838-863), but reusable for callees/both and
        exposed as a first-class op instead of buried inside deps(transitive=True).

        `max_depth=0` means unbounded, guarded only by `limit` (node cap) and the
        `visited` set (which also terminates cycles). Results are ordered by
        (depth, symbol) for determinism.
        """
        from collections import deque
        from pathlib import Path
        from srclight.db import Database

        if direction not in ("callees", "callers", "both"):
            raise ValueError(f"direction must be callees|callers|both, got {direction!r}")
        if isinstance(symbols, str):
            symbols = [symbols]

        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            entry_ids: list[int] = []
            entry_names: list[str] = []
            for name in symbols:
                sid = self._resolve_symbol_id(db, name, file)
                if sid is None:
                    continue
                sym = db.get_symbol_by_id(sid)
                entry_names.append(sym.name if sym is not None else name)
                entry_ids.append(sid)

            visited: set[int] = set(entry_ids)
            queue: deque[tuple[int, int]] = deque((sid, 0) for sid in entry_ids)
            result: list[ReachNode] = []
            truncated = False
            unbounded = max_depth <= 0

            while queue:
                sid, depth = queue.popleft()
                if not unbounded and depth >= max_depth:
                    continue          # at the depth cap: don't expand further
                rows = []
                if direction in ("callees", "both"):
                    rows += db.get_callees(sid)
                if direction in ("callers", "both"):
                    rows += db.get_callers(sid)
                for r in rows:
                    nsym = r["symbol"]
                    nid = nsym.id
                    if nid in visited:
                        continue       # visited set both dedups and breaks cycles
                    visited.add(nid)
                    if len(result) >= limit:
                        truncated = True
                        break
                    new_depth = depth + 1
                    result.append(ReachNode(symbol=nsym.name, file=nsym.file_path,
                                            line=nsym.start_line, kind=nsym.kind,
                                            depth=new_depth))
                    queue.append((nid, new_depth))
                if truncated:
                    break

            result.sort(key=lambda n: (n.depth, n.symbol))
            return ReachResult(served_ref=ref, entry=entry_names,
                               reachable=result, truncated=truncated)
        finally:
            db.close()

    def file_overview(self, index_dir, repo, ref, path) -> list[FileSymbol]:
        from pathlib import Path
        from srclight.db import Database
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            # The index may store '/' or '\\' separators (OS-dependent), and
            # symbols_in_file is an exact match -- try the path as given, then the
            # other separator, so the caller need not know the stored form.
            syms = db.symbols_in_file(path)
            if not syms:
                alt = path.replace("/", "\\") if "/" in path else path.replace("\\", "/")
                if alt != path:
                    syms = db.symbols_in_file(alt)
            return [FileSymbol(symbol=s.name, kind=s.kind, line=s.start_line, signature=s.signature)
                    for s in syms]
        finally:
            db.close()

    def tests(self, index_dir, repo, ref, symbol) -> list[GraphEdge]:
        from pathlib import Path
        from srclight.db import Database
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            return [self._edge(r) for r in db.get_tests_for(symbol)]
        finally:
            db.close()

    def structure(self, index_dir, repo, ref, *,
                  path=None, max_depth=3) -> Structure:
        from pathlib import Path
        from srclight.db import Database
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            # Normalize stored separators to '/' on the way out so the caller sees
            # consistent forward-slash paths regardless of the indexing OS.
            dirs = [DirEntry(path=d["path"].replace("\\", "/"), files=d["files"], symbols=d["symbols"],
                             languages=d["languages"]) for d in db.directory_summary(max_depth=max_depth)]
            hots = [HotspotFile(path=h["path"].replace("\\", "/"), language=h["language"],
                                lines=h["lines"], symbols=h["symbols"]) for h in db.hotspot_files()]
            if path:
                p = path.replace("\\", "/").rstrip("/")
                dirs = [d for d in dirs if d.path == p or d.path.startswith(p + "/")]
                hots = [h for h in hots if h.path.startswith(p + "/")]
            return Structure(dirs=dirs, hotspots=hots)
        finally:
            db.close()

    def hierarchy(self, index_dir, repo, ref, symbol, *,
                  file=None) -> Hierarchy:
        from pathlib import Path
        from srclight.db import Database
        db = Database(Path(index_dir) / "index.db")
        try:
            db.open()
            sid = self._resolve_symbol_id(db, symbol, file)
            if sid is None:
                return Hierarchy(base=[], subclasses=[], implementors=[])
            return Hierarchy(
                base=[self._edge(r) for r in db.get_base_classes(sid)],
                subclasses=[self._edge(r) for r in db.get_subclasses(sid)],
                implementors=[self._edge(r) for r in db.get_implementors(sid)],
            )
        finally:
            db.close()
