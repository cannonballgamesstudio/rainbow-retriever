import pytest
from rainbow_retriever.ragd.config import RagdConfig, build_backend, EmbedQueue
from rainbow_retriever.ragd.errors import BackendBusy
from rainbow_retriever.fake_backend import FakeBackend
from rainbow_retriever.srclight_backend import SrclightBackend


def test_from_env_defaults():
    c = RagdConfig.from_env({})
    assert c.backend == "fake" and c.embed_queue_size == 4


def test_from_env_srclight():
    c = RagdConfig.from_env({"RR_BACKEND": "srclight", "RR_EMBED_MODEL": "qwen3-embedding",
                            "RR_DIMS": "4096", "RR_EMBED_QUEUE": "2"})
    assert c.backend == "srclight" and c.dims == 4096 and c.embed_queue_size == 2


def test_build_backend_selects_type():
    assert isinstance(build_backend(RagdConfig.from_env({})), FakeBackend)
    sc = build_backend(RagdConfig.from_env({"RR_BACKEND": "srclight"}))
    assert isinstance(sc, SrclightBackend)


def test_embed_queue_blocks_when_full():
    q = EmbedQueue(maxsize=0)
    with pytest.raises(BackendBusy):
        with q:
            pass
    q2 = EmbedQueue(maxsize=1)
    with q2:                       # one slot ok
        with pytest.raises(BackendBusy):
            with q2:               # nested -> full
                pass


def test_from_env_reads_rerank_settings():
    cfg = RagdConfig.from_env({
        "RR_BACKEND": "srclight",
        "RR_RERANK_ENDPOINT": "http://localhost:7997",
        "RR_RERANK_MODEL": "qwen3-reranker-4b",
    })
    assert cfg.rerank_endpoint == "http://localhost:7997"
    assert cfg.rerank_model == "qwen3-reranker-4b"


def test_build_backend_injects_reranker_when_endpoint_set():
    cfg = RagdConfig(backend="srclight", rerank_endpoint="http://localhost:7997",
                     rerank_model="m")
    be = build_backend(cfg)
    assert be.reranker is not None
    assert be.reranker.endpoint == "http://localhost:7997"


def test_build_backend_no_reranker_without_endpoint():
    cfg = RagdConfig(backend="srclight")
    be = build_backend(cfg)
    assert be.reranker is None


def test_from_env_reads_query_max_per_file():
    c = RagdConfig.from_env({"RR_QUERY_MAX_PER_FILE": "2"})
    assert c.query_max_per_file == 2
    assert RagdConfig.from_env({}).query_max_per_file is None          # default unset
    assert RagdConfig.from_env({"RR_QUERY_MAX_PER_FILE": "0"}).query_max_per_file is None
    assert RagdConfig.from_env({"RR_QUERY_MAX_PER_FILE": ""}).query_max_per_file is None
