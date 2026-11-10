# src/rainbow_retriever/fusion/fusion_backend.py
from __future__ import annotations

from pathlib import Path

from rainbow_retriever.backend import Capabilities, IndexResult
from rainbow_retriever.fusion.identity import fusion_identity


def _git_head(worktree: str) -> str | None:
    """HEAD sha of the worktree being indexed, or None (fail-soft: no git,
    no repo, timeout). Index-time only."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             check=True)
        sha = out.stdout.strip()
        return sha if len(sha) == 40 else None
    except Exception:
        return None


class FusionBackend:
    """Multi-signal RetrievalBackend.

    Composes srclight as an unmodified library: srclight indexes (keyword/graph
    only), fusion ingests and owns the whole query pipeline. Graph ops delegate
    to an internal SrclightBackend.
    """

    def __init__(self, embed_model: str | None = None, dims: int | None = None,
                 reranker=None, query_instruct: str | None = None,
                 dedup_file: bool = True, w_lex: float = 1.0, w_dense: float = 1.0,
                 lex_mode: str = "ids",
                 # library default stays 0.0 (conservative); the SERVED default is 1.0
                 # via RagdConfig (see the note in ragd/config.py).
                 w_graph: float = 0.0, seeds: int = 20, graph_hops: int = 1,
                 graph_damp: float = 0.5, graph_edge_variant: str = "spec",
                 graph_seed_arm: str = "fused", graph_seed_exclude_docs: bool = False,
                 w_trigram: float = 0.0, w_prior: float = 0.0,
                 header_imports: bool = False,
                 strict_arms: bool = False,
                 provider_factory=None):
        self.embed_model = embed_model
        self.dims = dims
        self.reranker = reranker
        self.query_instruct = query_instruct
        self.dedup_file = dedup_file
        self.w_lex = w_lex
        self.w_dense = w_dense
        if lex_mode not in ("ids", "full", "off"):
            raise ValueError(f"lex_mode must be ids|full|off, got {lex_mode!r}")
        self.lex_mode = lex_mode
        self.w_graph = w_graph
        self.seeds = seeds
        self.graph_hops = graph_hops
        self.graph_damp = graph_damp
        if graph_edge_variant not in ("spec", "flat", "calls_led"):
            raise ValueError(f"graph_edge_variant must be spec|flat|calls_led, "
                             f"got {graph_edge_variant!r}")
        self.graph_edge_variant = graph_edge_variant
        if graph_seed_arm not in ("fused", "dense"):
            raise ValueError(f"graph_seed_arm must be fused|dense, got {graph_seed_arm!r}")
        self.graph_seed_arm = graph_seed_arm
        self.graph_seed_exclude_docs = graph_seed_exclude_docs
        self.w_trigram = w_trigram
        self.w_prior = w_prior
        # Experimental flag (adds a `uses:` line of cross-file callees to the
        # embed text). Not exposed through the server config.
        self.header_imports = header_imports
        self.strict_arms = strict_arms
        self._warned_arms: set[tuple[str, str]] = set()   # (index_dir, arm) warn-once
        self._prior_join_logged: set[str] = set()          # index_dirs already logged
        if provider_factory is None:
            def provider_factory(model):
                from srclight.embeddings import get_provider
                return get_provider(model)
        self._provider_factory = provider_factory
        self._provider = None

    def _get_provider(self):
        if self._provider is None:
            self._provider = self._provider_factory(self.embed_model)
        return self._provider

    def _version(self) -> str:
        return fusion_identity(self.embed_model, self.dims)

    def capabilities(self) -> Capabilities:
        import srclight
        modes = ["keyword"]
        if self.embed_model:
            modes += ["semantic", "hybrid"]
        return Capabilities(
            backend_id="fusion",
            backend_version=self._version(),
            backend_running_version=srclight.__version__,
            modes=tuple(modes),
            supports_callgraph=True,
            supports_dims=(self.dims,) if self.dims else (),
            supports_ops=("deps", "file", "tests", "map", "hierarchy", "reachable"),
        )

    def index(self, worktree_path: str, index_dir: str) -> IndexResult:
        from srclight.db import Database
        from srclight.indexer import Indexer, IndexConfig

        from rainbow_retriever.fusion import store
        from rainbow_retriever.fusion.dense import build_dense
        from rainbow_retriever.fusion.ingest import ingest_srclight
        from rainbow_retriever.fusion.lex import LexIndex

        from rainbow_retriever import srclight_patches
        srclight_patches.apply()  # capture pointer-returning C/C++ functions srclight drops

        idx = Path(index_dir)
        idx.mkdir(parents=True, exist_ok=True)

        # Drop any cached handles for this index_dir BEFORE rebuilding: a cached
        # LexIndex/DenseIndex mmaps fusion_bm25/fusion_emb.npy, and on Windows a
        # rewrite of a file with an open mapped section raises OSError (EINVAL) --
        # so a reindex() on the same backend instance that just queried this idx
        # would otherwise fail. Evicting drops the last reference, closing the
        # mmap immediately (CPython refcounting).
        cache = getattr(self, "_handle_cache", None)
        if cache:
            for key in [k for k in cache if k[0] == str(idx)]:
                del cache[key]

        # 1) srclight indexes WITHOUT embeddings -- fusion owns dense vectors,
        #    so the GPU isn't paid twice for the same corpus.
        db = Database(idx / "index.db")
        try:
            db.open()
            db.initialize()
            root = Path(worktree_path)
            Indexer(db, IndexConfig(root=root, embed_model=None)).index(root)
            db_stats = db.stats()
        finally:
            db.close()

        # 2) ingest -> fusion.db (chunks + headers)
        ingest_srclight(idx, header_imports=self.header_imports)

        # 3) lexical arm
        conn = store.open_db(idx / "fusion.db")
        try:
            LexIndex.build(conn, idx / "fusion_bm25")
            texts = store.embed_texts(conn)
        finally:
            conn.close()

        # 3b) graph arm: adjacency table (contains/calls/inherits/...) for the
        # graph-expand arm. Additive + idempotent -- safe to re-run as
        # a standalone augment on an already-built fusion.db.
        from rainbow_retriever.fusion.graph import build_adjacency
        build_adjacency(idx)

        # 3c) trigram arm: identifier-trigram postings for the anchor arm
        # (containment-scored). Additive/idempotent, same as adjacency.
        from rainbow_retriever.fusion.trigram import build_trigrams
        build_trigrams(idx)

        # 3d) git-history priors: per-file recency+frequency score for the
        # final-ranking boost (query-independent, see query()).
        # Index-time only: wall-clock use here is allowed (build-time artifact).
        from rainbow_retriever.fusion.priors import build_priors
        build_priors(idx, worktree_path)

        # 3e) index-commit stamp: query() surfaces this SHA (and compares it to
        # the registry's expectation) so callers can verify which worktree state
        # an index was built from (detects index/commit drift).
        # Fail-soft: no git -> no stamp.
        sha = _git_head(worktree_path)
        if sha:
            conn_m = store.open_db(idx / "fusion.db")
            try:
                store.meta_set(conn_m, "index_commit", sha)
            finally:
                conn_m.close()

        # 4) dense arm (only when an embed model is configured)
        if self.embed_model:
            build_dense(texts, self._get_provider(), idx / "fusion_emb.npy")

        return IndexResult(
            backend_version=self._version(),
            files_indexed=db_stats["files"],
            symbols=db_stats["symbols"],
            edges=db_stats["edges"],
            index_dir=str(idx),
        )

    _DOC_OVERFETCH = 3
    _RERANK_OVERFETCH = 5
    _DEDUP_OVERFETCH = 5

    def _handles(self, idx: Path):
        """Cache (LexIndex, DenseIndex|None) per index_dir, keyed by mtimes.

        FIFO-2 eviction: avoids re-loading bm25s/npy on every query() call (measured
        600-950ms per query for the dense float32 copy alone, on top of the
        lexical index load) while staying correct across reindex -- the mtime
        key busts the cache the moment fusion.db or fusion_emb.npy changes.
        Acceptable for a 1-2 index working set; degrades to an
        extra reload if >2 indices are queried sequentially on the same backend.

        CAVEAT (cross-process): the dense sidecar (fusion_emb.npy) is now read
        eagerly at load -- DenseIndex holds NO open handle, so it no longer
        participates in Windows file locking. The bm25s LexIndex mmap remains a
        holder: an in-place reindex from ANOTHER process can still hit Windows
        file locking (OSError EINVAL) during the fusion_bm25 swap. Same-process
        eviction via the index() method covers only this instance;
        build_dense/LexIndex.build write into a temp path and swap into place
        (store.finalize_db does the same for fusion.db), which removes
        truncate-in-place as a failure mode -- but the durable cross-process fix
        is to build each new index in a fresh directory and repoint the
        registry, not this cache.
        """
        from rainbow_retriever.fusion.dense import DenseIndex
        from rainbow_retriever.fusion.lex import LexIndex

        emb = idx / "fusion_emb.npy"
        key = (str(idx), (idx / "fusion.db").stat().st_mtime_ns,
               emb.stat().st_mtime_ns if emb.exists() else 0)
        cache = getattr(self, "_handle_cache", None)
        if cache is None:
            cache = self._handle_cache = {}
        if key not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            dense = DenseIndex.load(emb) if (emb.exists() and self.embed_model) else None
            cache[key] = (LexIndex.load(idx / "fusion_bm25"), dense)
        return cache[key]

    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20, dims=None,
              code_only=False, rerank=False, dedup=None, max_per_file=None, *,
              arm_diag=None, expected_commit=None):
        # dedup=None -> use the backend default (self.dedup_file). A per-request
        # override exists because file-dedup maximizes distinct-file recall, while
        # consumers that pack the top hits into a fixed-size context block may
        # prefer several chunks per file (dedup=False).
        from rainbow_retriever.backend import Hit
        from rainbow_retriever.fusion import store
        from rainbow_retriever.fusion.dense import embed_query
        from rainbow_retriever.fusion.fuse import dedup_by_file, demote_docs
        from rainbow_retriever.reranker import rerank_hits

        supported = self.capabilities().modes
        if mode not in supported:
            raise ValueError(f"unsupported mode {mode!r}; this backend supports {supported}")
        idx = Path(index_dir)
        lex_h, dense_h = self._handles(idx)

        fetch = limit
        if code_only:
            fetch = max(fetch, limit * self._DOC_OVERFETCH)
        active_rerank = rerank and self.reranker is not None and mode != "keyword"
        if active_rerank:
            fetch = max(fetch, limit * self._RERANK_OVERFETCH)
        # max_per_file: an integer cap on chunks per file. When set
        # (>=1) it OVERRIDES the binary dedup decision (1 == dedup-ON, higher ==
        # middle arms). The 5x _DEDUP_OVERFETCH pool sized for cap=1 is sufficient
        # for cap>=2 (unit-tested: cap binds at limit=20, cap=2).
        if max_per_file is not None and max_per_file < 1:
            raise ValueError(f"max_per_file must be >= 1, got {max_per_file}")
        if (self.dedup_file if dedup is None else dedup) or max_per_file is not None:
            fetch = max(fetch, limit * self._DEDUP_OVERFETCH)

        import logging
        log = logging.getLogger("rainbow_retriever.fusion")
        active: list[str] = []
        dropped: dict[str, str] = {}

        arms: dict[str, list[int]] = {}
        if mode == "keyword":
            arms["lex"] = [r for r, _ in lex_h.search(text, k=fetch)]
            active.append("lex")
        elif mode == "hybrid" and self.lex_mode != "off":
            from rainbow_retriever.fusion.lex import id_tokens
            lex_text = id_tokens(text) if self.lex_mode == "ids" else text
            if lex_text:
                if self.lex_mode == "ids":
                    # code-only pool: identifier-rich docs (changelogs) otherwise
                    # flood the arm (measured; see docs/benchmark.md)
                    rows_l = [r for r, _ in lex_h.search(lex_text, k=fetch * 2)]
                    arms["lex"] = self._code_rows(idx, rows_l)[:fetch]
                else:
                    arms["lex"] = [r for r, _ in lex_h.search(lex_text, k=fetch)]
                active.append("lex")
            else:
                # not a failure -- diagnostic: ids-mode goes silent by design
                # on anchorless queries
                dropped["lex"] = "no identifier tokens in query (lex_mode=ids)"
        if mode in ("semantic", "hybrid"):
            if dense_h is None:
                raise RuntimeError(
                    "semantic/hybrid requested but this index has no dense sidecar "
                    "(fusion_emb.npy missing) -- likely built by a keyword-only backend; re-index.")
            qvec = embed_query(self._get_provider(), text, self.query_instruct)
            arms["dense"] = [r for r, _ in dense_h.search(qvec, k=fetch)]
            active.append("dense")
            # Extension seam for subclasses that add arms reusing the query
            # vector (e.g. a second dense sidecar). Base returns {}.
            for name, rows_x in self._extra_arms(idx, qvec, fetch).items():
                arms[name] = rows_x
                active.append(name)
        if mode == "hybrid" and self.w_trigram > 0:
            from rainbow_retriever.fusion.lex import id_tokens
            from rainbow_retriever.fusion import trigram as tri_mod
            ids_q = id_tokens(text)
            if ids_q:
                conn_t = store.open_db_ro(idx / "fusion.db")
                try:
                    arms["trigram"] = [r for r, _ in tri_mod.search(conn_t, ids_q,
                                                                    cap=fetch)]
                    active.append("trigram")
                except Exception as e:
                    self._drop_arm(idx, "trigram", e, dropped)
                finally:
                    conn_t.close()

        from rainbow_retriever.fusion.fuse import weighted_rrf_scored
        first = weighted_rrf_scored(
            arms, {"lex": self.w_lex, "dense": self.w_dense, "trigram": self.w_trigram,
                   **self._extra_weights()})
        if self.w_graph > 0 and mode != "keyword":
            from rainbow_retriever.fusion import graph as graph_mod
            conn_g = store.open_db_ro(idx / "fusion.db")
            try:
                seed_pool = first
                if self.graph_seed_arm == "dense" and "dense" in arms:
                    # structural variant: seed expansion from the dense arm only
                    seed_pool = weighted_rrf_scored({"dense": arms["dense"]},
                                                    {"dense": self.w_dense})
                if self.graph_seed_exclude_docs and seed_pool:
                    # structural variant: no doc-kind chunks as seeds (3x overfetch
                    # so the filter doesn't shrink the pool below `seeds`)
                    keep = set(self._code_rows(
                        idx, [r for r, _ in seed_pool[: max(0, self.seeds) * 3]]))
                    seed_pool = [(r, s) for r, s in seed_pool if r in keep]
                g = graph_mod.expand(conn_g, seed_pool[: max(0, self.seeds)],
                                     hops=self.graph_hops, cap=fetch,
                                     damp=self.graph_damp,
                                     edge_variant=self.graph_edge_variant)
                arms["graph"] = [r for r, _ in g]
                active.append("graph")
            except Exception as e:
                self._drop_arm(idx, "graph", e, dropped)
            finally:
                conn_g.close()
            first = weighted_rrf_scored(
                arms, {"lex": self.w_lex, "dense": self.w_dense,
                       "graph": self.w_graph, "trigram": self.w_trigram,
                       **self._extra_weights()})
        fused = [r for r, _ in first][:fetch]

        index_commit = None
        conn = store.open_db_ro(idx / "fusion.db")
        try:
            index_commit = store.meta_get(conn, "index_commit")
            rows = store.chunks_by_rows(conn, fused)
            # Git-history prior boost: applied on the FINAL fused scores, not as
            # an RRF arm -- a query-independent ranking injected into RRF would
            # surface the same hot files for every query. Mode-agnostic by
            # design (keyword included): a file's history is not query-specific.
            if self.w_prior > 0:
                try:
                    from rainbow_retriever.fusion import priors as priors_mod
                    pri = priors_mod.load(conn)
                    if str(idx) not in self._prior_join_logged:
                        self._prior_join_logged.add(str(idx))
                        rate = priors_mod.join_rate(conn, pri)
                        log.info("priors join rate %.1f%% (index=%s)",
                                 100.0 * rate, idx)
                        if pri and rate < 0.01:
                            log.warning(
                                "priors join rate ~0%% -- path-normalization "
                                "mismatch between priors keys and chunk files "
                                "(index=%s)", idx)
                    smap = dict(first)
                    rows = sorted(
                        rows,
                        key=lambda r: -(smap.get(r["emb_row"], 0.0)
                                        * (1.0 + self.w_prior
                                           * pri.get((r["file"] or "").replace("\\", "/").lower(), 0.0))))
                    active.append("prior")
                except Exception as e:
                    self._drop_arm(idx, "prior", e, dropped)
        finally:
            conn.close()
        hits = [Hit(repo=repo, ref=ref, file=r["file"], line=r["line"],
                    symbol=r["name"],
                    # rank-derived; order is authoritative. Reciprocal rank (not
                    # n - i): comparable across refs at the /v1/xref merge -- a
                    # bigger index no longer outranks a smaller one at equal rank.
                    score=1.0 / (i + 1),
                    snippet=r["snippet"], kind=r["kind"])
                for i, r in enumerate(rows)]

        mismatch = bool(expected_commit and index_commit
                        and expected_commit != index_commit)
        if mismatch:
            key = (str(idx), "index_commit")
            if key not in self._warned_arms:
                self._warned_arms.add(key)
                log.warning("index commit drift: index built at %s but caller "
                            "expected %s (index=%s) -- warning only, never a "
                            "hard failure", index_commit, expected_commit, idx)
        if arm_diag is not None:
            arm_diag.update({"active": active, "dropped": dropped,
                             "index_commit": index_commit,
                             "expected_commit": expected_commit,
                             "commit_mismatch": mismatch})

        if active_rerank:
            hits = rerank_hits(self.reranker, text, hits)
        if code_only:
            hits = demote_docs(hits)
        do_dedup = self.dedup_file if dedup is None else dedup
        if max_per_file is not None:
            hits = dedup_by_file(hits, max_per_file=max_per_file)
        elif do_dedup:
            hits = dedup_by_file(hits)
        return hits[:limit]

    def _drop_arm(self, idx: Path, arm: str, exc: Exception,
                  dropped: dict[str, str]) -> None:
        """Record + (once per index+arm) warn a dropped arm; re-raise under
        strict mode. The failure class this closes: a missing adjacency/
        priors table silently serving lex-only fusion while config claims
        w_graph=1.0."""
        import logging
        reason = f"{type(exc).__name__}: {exc}"
        dropped[arm] = reason
        if self.strict_arms:
            raise exc
        key = (str(idx), arm)
        if key not in self._warned_arms:
            self._warned_arms.add(key)
            logging.getLogger("rainbow_retriever.fusion").warning(
                "%s arm dropped (index=%s): %s", arm, idx, reason)

    def _extra_arms(self, idx: Path, qvec, fetch: int) -> dict[str, list[int]]:
        """Extension seam: extra RRF arms computed from the query vector.
        The base backend contributes none; subclasses may override."""
        return {}

    def _extra_weights(self) -> dict[str, float]:
        """RRF weights for _extra_arms; keys must match. Served: none."""
        return {}

    def _code_rows(self, idx: Path, rows: list[int]) -> list[int]:
        from rainbow_retriever.fusion import store
        from rainbow_retriever.fusion.fuse import DOC_KINDS, DOC_EXTS
        conn = store.open_db_ro(idx / "fusion.db")
        try:
            doc = {r["emb_row"] for r in store.chunks_by_rows(conn, rows)
                   if (r["kind"] or "").lower() in DOC_KINDS
                   or (r["file"] or "").lower().endswith(DOC_EXTS)}
        finally:
            conn.close()
        return [r for r in rows if r not in doc]

    # ---- graph ops: pure delegation to the (untouched) srclight adapter, which
    # reads the index.db that fusion's index() already builds. Parity by reuse.

    @property
    def _delegate(self):
        if getattr(self, "_delegate_backend", None) is None:
            from rainbow_retriever.srclight_backend import SrclightBackend
            self._delegate_backend = SrclightBackend()   # graph ops need no embeddings
        return self._delegate_backend

    def ctx(self, index_dir, repo, ref, symbol, *, file=None):
        return self._delegate.ctx(index_dir, repo, ref, symbol, file=file)

    def deps(self, index_dir, repo, ref, symbol, *, file=None,
             transitive=False, max_depth=5):
        return self._delegate.deps(index_dir, repo, ref, symbol, file=file,
                                   transitive=transitive, max_depth=max_depth)

    def file_overview(self, index_dir, repo, ref, path):
        return self._delegate.file_overview(index_dir, repo, ref, path)

    def tests(self, index_dir, repo, ref, symbol):
        return self._delegate.tests(index_dir, repo, ref, symbol)

    def structure(self, index_dir, repo, ref, *, path=None, max_depth=3):
        return self._delegate.structure(index_dir, repo, ref,
                                        path=path, max_depth=max_depth)

    def hierarchy(self, index_dir, repo, ref, symbol, *, file=None):
        return self._delegate.hierarchy(index_dir, repo, ref, symbol, file=file)

    def reachable(self, *a, **k):
        return self._delegate.reachable(*a, **k)
