import pytest

from rainbow_retriever.fusion.fusion_backend import FusionBackend


@pytest.fixture(scope="module")
def indexed(srclight_index, tmp_path_factory):
    wt, _ = srclight_index
    idx = tmp_path_factory.mktemp("fusion_graph_idx")
    be = FusionBackend()                        # keyword-only is fine for graph ops
    be.index(str(wt), str(idx))
    return be, str(idx)


def test_file_overview_lists_symbols(indexed):
    be, idx = indexed
    syms = be.file_overview(idx, "r", "m", "sample.cs")
    assert {s.symbol for s in syms} >= {"PlayerInventory", "AddGoldCoins"}


def test_ctx_resolves_unique_symbol(indexed):
    be, idx = indexed
    g = be.ctx(idx, "r", "m", "AddGoldCoins")
    assert g is not None and g.defined_file.endswith("sample.cs")


def test_ctx_missing_symbol_returns_none(indexed):
    be, idx = indexed
    assert be.ctx(idx, "r", "m", "no_such_symbol_xyz") is None


def test_structure_returns_dirs_and_hotspots(indexed):
    be, idx = indexed
    st = be.structure(idx, "r", "m")
    assert st.dirs or st.hotspots


def test_deps_and_hierarchy_and_tests_return_lists(indexed):
    be, idx = indexed
    assert isinstance(be.deps(idx, "r", "m", "Greeter"), list)
    h = be.hierarchy(idx, "r", "m", "Greeter")
    assert hasattr(h, "base") and hasattr(h, "subclasses")
    assert isinstance(be.tests(idx, "r", "m", "greet_user"), list)


def test_satisfies_retrieval_backend_protocol(indexed):
    from rainbow_retriever.backend import RetrievalBackend
    be, _ = indexed
    assert isinstance(be, RetrievalBackend)     # runtime_checkable structural check
