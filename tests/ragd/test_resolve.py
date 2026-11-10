import os
import pytest
from rainbow_retriever.registry import Registry, RegistryRO
from rainbow_retriever.ragd.resolve import index_dir_for, resolve_ref
from rainbow_retriever.ragd.errors import RefNotFound


def _ro(tmp_path):
    reg = Registry.init_db(str(tmp_path / "r.db"))
    reg.upsert_ref("r", "main", "s_main", "/wt/main", "ready", 1, "v1", "c1")
    reg.upsert_ref("r", "feat", "s_feat", "/wt/feat", "ready", 1, "v1", "c1")
    reg.upsert_ref("r", "wip", "s_wip", "/wt/wip", "indexing", 1, "v1", "c1")  # not ready
    return RegistryRO(str(tmp_path / "r.db"))


def test_index_dir_for():
    from rainbow_retriever.registry import Ref
    ref = Ref("r", "main", "s", "/wt/main", "ready", 1, "srclight", "v1", "c1")
    assert index_dir_for(ref) == os.path.join("/wt/main", ".srclight")


def test_primary_hit_is_not_fallback(tmp_path):
    ref, fb = resolve_ref(_ro(tmp_path), "r", "main", [])
    assert ref.ref == "main" and fb is False


def test_walks_fallbacks_to_first_ready(tmp_path):
    ref, fb = resolve_ref(_ro(tmp_path), "r", "mybranch", ["wip", "feat", "main"])
    assert ref.ref == "feat" and fb is True       # wip not ready -> feat


def test_all_miss_raises(tmp_path):
    with pytest.raises(RefNotFound):
        resolve_ref(_ro(tmp_path), "r", "mybranch", ["wip", "nope"])
