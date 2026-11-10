from __future__ import annotations

import dataclasses
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from rainbow_retriever.fusion.lex import anchor_strength
from rainbow_retriever.ragd.budget import apply_budget
from rainbow_retriever.ragd.config import RagdConfig, EmbedQueue
from rainbow_retriever.backend import AmbiguousSymbol as BackendAmbiguous
from rainbow_retriever.ragd.errors import (
    RagdError, IdentityRequired, Internal, InvalidRequest, make_error_response,
    UnsupportedMode, UnsupportedDims, BackendDegraded, CtxUnsupported, RefNotFound,
    AmbiguousSymbol, UnsupportedOp,
)
from rainbow_retriever.ragd.identity import extract_identity, audit_log
from rainbow_retriever.ragd.resolve import index_dir_for, resolve_ref
from rainbow_retriever.ragd.staleness import compute_staleness
from rainbow_retriever.ragd.wire import QueryRequest, CtxRequest, XrefRequest

_AUDIT = logging.getLogger("ragd.audit")


def _hit_to_dict(h) -> dict:
    return dataclasses.asdict(h)


def _graph_to_dict(g) -> dict | None:
    if g is None:
        return None
    return dataclasses.asdict(g)


def _ref_to_dict(ref) -> dict:
    return {"repo": ref.repo, "ref": ref.ref, "head_sha": ref.head_sha,
            "worktree": ref.worktree, "status": ref.status, "indexed_at": ref.indexed_at,
            "backend_id": ref.backend_id, "backend_version": ref.backend_version,
            "config_hash": ref.config_hash}


def _validate_modes_dims(cap, mode, query_dims):
    if mode not in cap.modes:
        raise UnsupportedMode(f"backend {cap.backend_id!r} does not support mode {mode!r}",
                              details={"requested": mode, "supported": list(cap.modes)})
    if query_dims is not None and query_dims not in cap.supports_dims:
        raise UnsupportedDims(f"dims {query_dims} unsupported",
                              details={"requested": query_dims, "supported": list(cap.supports_dims)})


def _run_backend_query(backend, *args, **kwargs):
    try:
        return backend.query(*args, **kwargs)
    except ValueError as e:
        raise UnsupportedMode(str(e))
    except Exception as e:                      # noqa: BLE001 - any backend fault -> 503
        raise BackendDegraded(str(e), details={"retryable": True})


def create_app(backend, registry_ro, config=None, embed_queue=None) -> FastAPI:
    config = config or RagdConfig()
    embed_queue = embed_queue or EmbedQueue(config.embed_queue_size)
    backend_version = backend.capabilities().backend_version
    import inspect
    # Diagnostics contract detection: fusion's query() accepts
    # arm_diag/expected_commit; fake/srclight do not -- the `arms` response
    # field is additive and appears only when the backend supports it.
    supports_arm_diag = "arm_diag" in inspect.signature(backend.query).parameters
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.state.backend = backend
    app.state.ro = registry_ro
    app.state.config = config
    app.state.embed_queue = embed_queue

    def _gated_query(index_dir, repo, ref_name, text, mode, limit, dims,
                     code_only=False, rerank=False, dedup=None, max_per_file=None,
                     **kwargs):
        if max_per_file is None:
            max_per_file = config.query_max_per_file     # env RR_QUERY_MAX_PER_FILE
        args = (index_dir, repo, ref_name, text, mode, limit, dims,
                code_only, rerank, dedup, max_per_file)
        if mode in ("semantic", "hybrid"):
            with embed_queue:
                return _run_backend_query(backend, *args, **kwargs)
        return _run_backend_query(backend, *args, **kwargs)

    @app.middleware("http")
    async def _stamp(request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        response.headers["X-Backend-Version"] = backend_version
        return response

    def _err(request: Request, err: RagdError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None) or str(uuid.uuid4())
        resp = JSONResponse(status_code=err.http_status,
                            content=make_error_response(err, rid))
        resp.headers["X-Request-Id"] = rid
        resp.headers["X-Backend-Version"] = backend_version
        return resp

    @app.exception_handler(RagdError)
    async def _ragd_error(request: Request, err: RagdError):
        return _err(request, err)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, err: RequestValidationError):
        return _err(request, InvalidRequest("invalid request body"))

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation(request: Request, err: PydanticValidationError):
        return _err(request, InvalidRequest("invalid request body"))

    @app.exception_handler(Exception)
    async def _uncaught(request: Request, err: Exception):
        return _err(request, Internal("internal error"))

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        extract_identity(request.headers, config.default_identity)
        return _capabilities_dict(backend)

    @app.post("/v1/query")
    async def query(req: QueryRequest, request: Request):
        ident = extract_identity(request.headers, config.default_identity)
        cap = backend.capabilities()
        _validate_modes_dims(cap, req.mode, req.query_dims)
        ref, is_fallback = resolve_ref(registry_ro, req.repo, req.branch, req.fallback_refs)
        diag: dict | None = {} if supports_arm_diag else None
        extra = {}
        if diag is not None:
            extra["arm_diag"] = diag
            if ref.head_sha:
                extra["expected_commit"] = ref.head_sha
        hits = _gated_query(index_dir_for(ref), req.repo, ref.ref, req.text, req.mode, req.limit,
                            req.query_dims, req.code_only, req.rerank, req.dedup,
                            req.max_chunks_per_file, **extra)
        hit_dicts = [_hit_to_dict(h) for h in hits]
        kept, truncated = apply_budget(hit_dicts, req.budget)
        staleness = compute_staleness(ref.head_sha, req.local_head_sha, is_fallback,
                                      req.branch, ref.ref)
        audit_log(_AUDIT, ident, request.state.request_id, req.repo, ref.ref, "query",
                  staleness["stale"], text=req.text, log_raw=config.log_raw_text)
        resp = {"served_ref": ref.ref, "hits": kept, "staleness": staleness,
                "capabilities": _capabilities_dict(backend), "truncated": truncated,
                "anchored": anchor_strength(req.text)}
        if diag is not None:
            resp["arms"] = diag
        return resp

    @app.post("/v1/ctx")
    async def ctx(req: CtxRequest, request: Request):
        ident = extract_identity(request.headers, config.default_identity)
        cap = backend.capabilities()
        if not cap.supports_callgraph:
            raise CtxUnsupported(f"backend {cap.backend_id!r} has no call-graph")
        ref, is_fallback = resolve_ref(registry_ro, req.repo, req.branch, req.fallback_refs)
        # _run_op maps AmbiguousSymbol -> 409 (with candidates) and other faults -> 503,
        # matching deps/hierarchy/tests. (ctx previously swallowed everything into 503.)
        graph = _run_op(lambda: backend.ctx(index_dir_for(ref), req.repo, ref.ref, req.symbol,
                                            file=req.file))
        staleness = compute_staleness(ref.head_sha, req.local_head_sha, is_fallback,
                                      req.branch, ref.ref)
        audit_log(_AUDIT, ident, request.state.request_id, req.repo, ref.ref, "ctx",
                  staleness["stale"], symbol=req.symbol, log_raw=config.log_raw_text)
        return {"served_ref": ref.ref, "found": graph is not None,
                "graph": _graph_to_dict(graph), "staleness": staleness,
                "capabilities": _capabilities_dict(backend)}

    @app.post("/v1/xref")
    async def xref(req: XrefRequest, request: Request):
        ident = extract_identity(request.headers, config.default_identity)
        cap = backend.capabilities()
        _validate_modes_dims(cap, req.mode, req.query_dims)
        served, unresolved, merged = [], [], []
        for r in req.refs:
            ref = registry_ro.resolve(req.repo, r)        # exact, ready-only, no fallback
            if ref is None:
                unresolved.append(r)
                continue
            served.append(r)
            hits = _gated_query(index_dir_for(ref), req.repo, ref.ref, req.text, req.mode, req.limit,
                            req.query_dims, req.code_only, req.rerank, req.dedup,
                            req.max_chunks_per_file)
            # Normalize per-ref scores to reciprocal rank BEFORE the cross-ref
            # score sort: backends emit mode-relative scores (srclight's FTS5
            # bm25 `rank` is negative/more-negative-is-better; old fusion scores
            # were index-size-dependent) that are not comparable across refs.
            # Order within a ref is authoritative; rank is the only
            # cross-backend invariant. Idempotent for backends already emitting
            # reciprocal rank.
            for pos, h in enumerate(hits):
                d = _hit_to_dict(h)
                d["score"] = 1.0 / (pos + 1)
                merged.append(d)
        if not served:
            raise RefNotFound("no requested ref is indexed",
                              details={"unresolved_refs": unresolved})
        merged.sort(key=lambda h: h["score"], reverse=True)
        merged = merged[: req.limit]
        kept, truncated = apply_budget(merged, req.budget)
        try:
            work_ref, is_fallback = resolve_ref(registry_ro, req.repo, req.branch, req.fallback_refs)
            staleness = compute_staleness(work_ref.head_sha, req.local_head_sha, is_fallback,
                                          req.branch, work_ref.ref)
        except RefNotFound:
            staleness = {"indexed_sha": None, "your_sha": req.local_head_sha, "stale": True,
                         "push_hint": f"working branch {req.branch!r} is not indexed -- push to index it"}
        audit_log(_AUDIT, ident, request.state.request_id, req.repo, ",".join(served), "xref",
                  staleness["stale"], text=req.text, log_raw=config.log_raw_text)
        return {"served_refs": served, "unresolved_refs": unresolved, "hits": kept,
                "staleness": staleness, "capabilities": _capabilities_dict(backend),
                "truncated": truncated}

    @app.get("/v1/resolve")
    async def resolve(request: Request, repo: str, branch: str):
        extract_identity(request.headers, config.default_identity)
        ref = registry_ro.get_ref(repo, branch)
        if ref is None:
            raise RefNotFound(f"no row for {repo!r} {branch!r}")
        return _ref_to_dict(ref)

    @app.get("/v1/status")
    async def status(request: Request, repo: str | None = None):
        extract_identity(request.headers, config.default_identity)
        return {"refs": [_ref_to_dict(r) for r in registry_ro.list_refs(repo)]}

    def _resolve(req_repo, branch):
        ref, _ = resolve_ref(registry_ro, req_repo, branch, [])
        return ref

    def _run_op(fn):
        try:
            return fn()
        except BackendAmbiguous as a:
            raise AmbiguousSymbol("ambiguous symbol", details={"candidates": a.candidates})
        except (BackendDegraded, RagdError):
            raise
        except Exception as e:                              # noqa: BLE001
            raise BackendDegraded(str(e), details={"retryable": True})

    def _require_op(op):
        if op not in backend.capabilities().supports_ops:
            raise UnsupportedOp(f"backend does not support {op!r}")

    @app.get("/v1/deps")
    async def deps(request: Request, repo: str, branch: str, symbol: str,
                   transitive: bool = False, depth: int = 5, file: str | None = None):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("deps")
        ref = _resolve(repo, branch)
        edges = _run_op(lambda: backend.deps(index_dir_for(ref), repo, ref.ref, symbol,
                                             file=file, transitive=transitive, max_depth=depth))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "deps", False, symbol=symbol)
        return {"served_ref": ref.ref, "edges": [dataclasses.asdict(e) for e in edges]}

    @app.get("/v1/file")
    async def file_overview(request: Request, repo: str, branch: str, path: str):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("file")
        ref = _resolve(repo, branch)
        syms = _run_op(lambda: backend.file_overview(index_dir_for(ref), repo, ref.ref, path))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "file", False, symbol=path)
        return {"served_ref": ref.ref, "symbols": [dataclasses.asdict(s) for s in syms]}

    @app.get("/v1/tests")
    async def tests(request: Request, repo: str, branch: str, symbol: str):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("tests")
        ref = _resolve(repo, branch)
        edges = _run_op(lambda: backend.tests(index_dir_for(ref), repo, ref.ref, symbol))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "tests", False, symbol=symbol)
        return {"served_ref": ref.ref, "edges": [dataclasses.asdict(e) for e in edges]}

    @app.get("/v1/map")
    async def structure(request: Request, repo: str, branch: str,
                        path: str | None = None, depth: int = 3):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("map")
        ref = _resolve(repo, branch)
        st = _run_op(lambda: backend.structure(index_dir_for(ref), repo, ref.ref, path=path, max_depth=depth))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "map", False, symbol=path or "/")
        return {"served_ref": ref.ref,
                "dirs": [dataclasses.asdict(d) for d in st.dirs],
                "hotspots": [dataclasses.asdict(h) for h in st.hotspots]}

    @app.get("/v1/hierarchy")
    async def hierarchy(request: Request, repo: str, branch: str, symbol: str, file: str | None = None):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("hierarchy")
        ref = _resolve(repo, branch)
        h = _run_op(lambda: backend.hierarchy(index_dir_for(ref), repo, ref.ref, symbol, file=file))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "hierarchy", False, symbol=symbol)
        return {"served_ref": ref.ref,
                "base": [dataclasses.asdict(e) for e in h.base],
                "subclasses": [dataclasses.asdict(e) for e in h.subclasses],
                "implementors": [dataclasses.asdict(e) for e in h.implementors]}

    @app.get("/v1/reachable")
    async def reachable(request: Request, repo: str, branch: str, symbol: str,
                        direction: str = "callees", depth: int = 5,
                        file: str | None = None, limit: int = 2000):
        ident = extract_identity(request.headers, config.default_identity)
        _require_op("reachable")
        ref = _resolve(repo, branch)
        symbols = [s.strip() for s in symbol.split(",") if s.strip()]
        res = _run_op(lambda: backend.reachable(index_dir_for(ref), repo, ref.ref, symbols,
                                                direction, depth, file, limit))
        audit_log(_AUDIT, ident, request.state.request_id, repo, ref.ref, "reachable", False, symbol=symbol)
        return {"served_ref": ref.ref, "entry": res.entry,
                "reachable": [dataclasses.asdict(n) for n in res.reachable],
                "truncated": res.truncated}

    return app


def _capabilities_dict(backend) -> dict:
    cap = backend.capabilities()
    return {"backend_id": cap.backend_id, "backend_version": cap.backend_version,
            "modes": list(cap.modes), "supports_callgraph": cap.supports_callgraph,
            "supports_dims": list(cap.supports_dims),
            "supports_ops": list(cap.supports_ops)}
