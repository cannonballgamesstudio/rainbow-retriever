# tests/fusion/test_config_fusion.py
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.ragd.config import RagdConfig, build_backend


def test_from_env_reads_fusion_weights():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion",
                               "RR_FUSION_W_LEX": "2.5",
                               "RR_FUSION_W_DENSE": "0.5"})
    assert cfg.backend == "fusion"
    assert cfg.fusion_w_lex == 2.5 and cfg.fusion_w_dense == 0.5


def test_fusion_weights_default_to_one():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion"})
    assert cfg.fusion_w_lex == 1.0 and cfg.fusion_w_dense == 1.0


def test_build_backend_constructs_fusion():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion",
                               "RR_EMBED_MODEL": "qwen3-embedding",
                               "RR_DIMS": "4096",
                               "RR_FUSION_W_LEX": "2.0"})
    be = build_backend(cfg)
    assert isinstance(be, FusionBackend)
    assert be.embed_model == "qwen3-embedding" and be.dims == 4096
    assert be.w_lex == 2.0 and be.w_dense == 1.0
    assert be.query_instruct                    # validated default instruct ON
    assert be.dedup_file is True


def test_build_backend_wires_reranker_for_fusion():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion",
                               "RR_RERANK_ENDPOINT": "http://localhost:9000"})
    be = build_backend(cfg)
    assert be.reranker is not None


def test_srclight_and_fake_branches_unchanged():
    from rainbow_retriever.fake_backend import FakeBackend
    from rainbow_retriever.srclight_backend import SrclightBackend
    assert isinstance(build_backend(RagdConfig.from_env({})), FakeBackend)
    assert isinstance(
        build_backend(RagdConfig.from_env({"RR_BACKEND": "srclight"})),
        SrclightBackend)


def test_from_env_reads_p2_fields():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion",
                               "RR_FUSION_LEX_MODE": "full",
                               "RR_FUSION_W_GRAPH": "0.7",
                               "RR_FUSION_SEEDS": "30",
                               "RR_FUSION_GRAPH_HOPS": "2"})
    assert cfg.fusion_lex_mode == "full"
    assert cfg.fusion_w_graph == 0.7
    assert cfg.fusion_seeds == 30 and cfg.fusion_graph_hops == 2


def test_p2_defaults():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion"})
    assert cfg.fusion_lex_mode == "ids"
    assert cfg.fusion_w_graph == 1.0    # served default
    assert cfg.fusion_seeds == 20 and cfg.fusion_graph_hops == 1


def test_build_backend_threads_p2_fields():
    be = build_backend(RagdConfig.from_env({"RR_BACKEND": "fusion",
                                            "RR_FUSION_LEX_MODE": "off",
                                            "RR_FUSION_W_GRAPH": "1.0"}))
    assert be.lex_mode == "off" and be.w_graph == 1.0


def test_from_env_reads_p3_fields():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion",
                               "RR_FUSION_W_TRIGRAM": "0.8",
                               "RR_FUSION_W_PRIOR": "0.2"})
    assert cfg.fusion_w_trigram == 0.8 and cfg.fusion_w_prior == 0.2


def test_p3_defaults_off():
    cfg = RagdConfig.from_env({"RR_BACKEND": "fusion"})
    assert cfg.fusion_w_trigram == 0.0 and cfg.fusion_w_prior == 0.3  # served defaults


def test_build_backend_threads_all_arm_knobs():
    be = build_backend(RagdConfig.from_env({"RR_BACKEND": "fusion",
                                            "RR_FUSION_W_TRIGRAM": "0.8",
                                            "RR_FUSION_W_PRIOR": "0.2",
                                            "RR_FUSION_SEEDS": "30",
                                            "RR_FUSION_GRAPH_HOPS": "2"}))
    assert be.w_trigram == 0.8 and be.w_prior == 0.2
    assert be.seeds == 30 and be.graph_hops == 2


def test_strict_arms_env():
    cfg = RagdConfig.from_env({})
    assert cfg.strict_arms is False
    assert RagdConfig.from_env({"RR_STRICT_ARMS": "1"}).strict_arms is True


def test_build_backend_threads_strict_arms():
    be = build_backend(RagdConfig.from_env({"RR_BACKEND": "fusion",
                                            "RR_STRICT_ARMS": "1"}))
    assert be.strict_arms is True
