from __future__ import annotations

from dataclasses import dataclass

from rainbow_retriever.ragd.errors import BackendBusy


@dataclass
class RagdConfig:
    backend: str = "fake"
    embed_model: str | None = None
    dims: int | None = None
    embed_queue_size: int = 4
    registry_db: str | None = None
    log_raw_text: bool = False
    default_identity: str | None = None
    rerank_endpoint: str | None = None
    rerank_model: str | None = None
    query_instruct: str | None = None
    dedup_file: bool = True
    # Integer per-file chunk cap applied when a request doesn't set
    # max_chunks_per_file. Default unset (None) -- binary dedup_file governs.
    query_max_per_file: int | None = None
    fusion_w_lex: float = 1.0
    fusion_w_dense: float = 1.0
    fusion_lex_mode: str = "ids"
    # Served default 1.0: the graph arm improved MRR on a held-out set
    # (see docs/benchmark.md).
    fusion_w_graph: float = 1.0
    fusion_seeds: int = 20
    fusion_graph_hops: int = 1
    fusion_w_trigram: float = 0.0
    # Served default 0.3: the git-history prior gave a small paired MRR gain
    # (+0.015, CI [+0.001, +0.035]) on SWE-bench localization queries.
    fusion_w_prior: float = 0.3
    # Strict mode: re-raise arm exceptions instead of warn-and-drop, so a
    # missing adjacency/priors table cannot silently serve lex-only fusion
    # while config claims w_graph=1.0. Set RR_STRICT_ARMS=1 for benchmarking;
    # the served default stays fail-soft.
    strict_arms: bool = False

    @classmethod
    def from_env(cls, env) -> "RagdConfig":
        dims = env.get("RR_DIMS")
        # instruct-prefixing the embedded query + file-dedup are validated free wins
        # (instruct lifts recall/MRR; dedup gives distinct-file top-K for localization),
        # so both default ON. RR_QUERY_INSTRUCT: unset -> the validated default
        # instruction; "0"/"off"/"" -> disabled; "1"/"on" -> default; any other value ->
        # used verbatim. RR_QUERY_DEDUP_FILE: "0"/"off" opts out. (code_only stays an
        # explicit per-request opt-in: docs helped recall on repos with design docs.)
        #
        # Consumers that pack the top-K hits into a fixed-size context block may
        # prefer RR_QUERY_DEDUP_FILE=0 (several chunks per file). RR_QUERY_MAX_PER_FILE
        # generalizes this to an integer per-file cap (1 == dedup on); unset by default.
        from rainbow_retriever.srclight_backend import DEFAULT_QUERY_INSTRUCT
        qi = env.get("RR_QUERY_INSTRUCT")
        if qi is None or qi.lower() in ("1", "true", "on", "yes"):
            qi = DEFAULT_QUERY_INSTRUCT
        elif qi.lower() in ("0", "false", "off", "no", ""):
            qi = None
        dedup = (env.get("RR_QUERY_DEDUP_FILE") or "on").lower() not in ("0", "false", "off", "no")
        mpf = env.get("RR_QUERY_MAX_PER_FILE")
        mpf_i = int(mpf) if mpf and mpf.strip() else 0
        return cls(
            backend=env.get("RR_BACKEND", "fake"),
            embed_model=env.get("RR_EMBED_MODEL"),
            dims=int(dims) if dims else None,
            embed_queue_size=int(env.get("RR_EMBED_QUEUE", "4")),
            registry_db=env.get("RR_REGISTRY_DB"),
            log_raw_text=env.get("RR_LOG_RAW_TEXT") == "1",
            default_identity=env.get("RR_DEFAULT_IDENTITY"),
            rerank_endpoint=env.get("RR_RERANK_ENDPOINT"),
            rerank_model=env.get("RR_RERANK_MODEL"),
            query_instruct=qi or None,
            dedup_file=dedup,
            query_max_per_file=mpf_i if mpf_i >= 1 else None,
            fusion_w_lex=float(env.get("RR_FUSION_W_LEX", "1.0")),
            fusion_w_dense=float(env.get("RR_FUSION_W_DENSE", "1.0")),
            fusion_lex_mode=env.get("RR_FUSION_LEX_MODE", "ids"),
            fusion_w_graph=float(env.get("RR_FUSION_W_GRAPH", "1.0")),
            fusion_seeds=int(env.get("RR_FUSION_SEEDS", "20")),
            fusion_graph_hops=int(env.get("RR_FUSION_GRAPH_HOPS", "1")),
            fusion_w_trigram=float(env.get("RR_FUSION_W_TRIGRAM", "0.0")),
            fusion_w_prior=float(env.get("RR_FUSION_W_PRIOR", "0.3")),
            strict_arms=env.get("RR_STRICT_ARMS") == "1",
        )


def build_backend(config: RagdConfig):
    if config.backend == "srclight":
        from rainbow_retriever.srclight_backend import SrclightBackend
        reranker = None
        if config.rerank_endpoint:
            from rainbow_retriever.reranker import HttpReranker
            reranker = HttpReranker(config.rerank_endpoint,
                                    config.rerank_model or "reranker")
        return SrclightBackend(embed_model=config.embed_model, dims=config.dims,
                               reranker=reranker, query_instruct=config.query_instruct,
                               dedup_file=config.dedup_file)
    if config.backend == "fusion":
        from rainbow_retriever.fusion.fusion_backend import FusionBackend
        reranker = None
        if config.rerank_endpoint:
            from rainbow_retriever.reranker import HttpReranker
            reranker = HttpReranker(config.rerank_endpoint,
                                    config.rerank_model or "reranker")
        return FusionBackend(embed_model=config.embed_model, dims=config.dims,
                             reranker=reranker, query_instruct=config.query_instruct,
                             dedup_file=config.dedup_file,
                             w_lex=config.fusion_w_lex, w_dense=config.fusion_w_dense,
                             lex_mode=config.fusion_lex_mode, w_graph=config.fusion_w_graph,
                             seeds=config.fusion_seeds, graph_hops=config.fusion_graph_hops,
                             w_trigram=config.fusion_w_trigram, w_prior=config.fusion_w_prior,
                             strict_arms=config.strict_arms)
    from rainbow_retriever.fake_backend import FakeBackend
    return FakeBackend()


class EmbedQueue:
    """Simple bounded gate for semantic/hybrid embeds. Raises BackendBusy when full."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._inflight = 0

    def __enter__(self):
        if self._inflight >= self.maxsize:
            raise BackendBusy("embed queue saturated", details={"retryable": True})
        self._inflight += 1
        return self

    def __exit__(self, *exc):
        self._inflight -= 1
        return False
