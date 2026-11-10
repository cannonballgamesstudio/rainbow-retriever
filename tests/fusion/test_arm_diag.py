# tests/fusion/test_arm_diag.py
import logging
import subprocess

import pytest

from rainbow_retriever.fusion import store
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from tests.fusion.test_dense import FakeProvider


def _backend(**kw):
    return FusionBackend(embed_model="fake", dims=8, w_graph=1.0, w_prior=0.3,
                         provider_factory=lambda model: FakeProvider(), **kw)


@pytest.fixture()
def built_idx(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "diag_idx"
    _backend().index(str(wt), str(idx))
    return idx


def _git(cwd, *args):
    import os
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                   capture_output=True)


@pytest.fixture()
def git_idx(tmp_path):
    """A one-file git worktree indexed by fusion; returns (index_dir, HEAD sha)."""
    wt = tmp_path / "gwt"
    wt.mkdir()
    (wt / "mod.py").write_text("def fn():\n    return 1\n")
    _git(wt, "init", "-q"); _git(wt, "add", "."); _git(wt, "commit", "-q", "-m", "x")
    sha = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    idx = tmp_path / "gidx"
    _backend().index(str(wt), str(idx))
    return idx, sha


def _drop_adjacency(idx):
    conn = store.open_db(idx / "fusion.db")
    conn.execute("DROP TABLE adjacency")
    conn.commit()
    conn.close()


def test_diag_reports_active_arms(built_idx):
    be = _backend()
    diag = {}
    be.query(str(built_idx), "r", "m", "AddGoldCoins inventory", mode="hybrid",
             limit=5, arm_diag=diag)
    assert "lex" in diag["active"] and "dense" in diag["active"]
    assert "graph" in diag["active"] and "prior" in diag["active"]
    assert diag["dropped"] == {}


def test_dropped_table_reported_not_silent(built_idx, caplog):
    _drop_adjacency(built_idx)
    be = _backend()
    diag = {}
    with caplog.at_level(logging.WARNING, logger="rainbow_retriever.fusion"):
        hits = be.query(str(built_idx), "r", "m", "AddGoldCoins inventory",
                        mode="hybrid", limit=5, arm_diag=diag)
    assert hits                                          # fail-soft still serves
    assert "graph" in diag["dropped"] and "graph" not in diag["active"]
    assert "OperationalError" in diag["dropped"]["graph"]


def test_strict_mode_raises_on_dropped_arm(built_idx):
    _drop_adjacency(built_idx)
    be = _backend(strict_arms=True)
    with pytest.raises(Exception, match="adjacency"):
        be.query(str(built_idx), "r", "m", "AddGoldCoins inventory",
                 mode="hybrid", limit=5)


def test_warn_once_per_index_and_arm(built_idx, caplog):
    _drop_adjacency(built_idx)
    be = _backend()
    with caplog.at_level(logging.WARNING, logger="rainbow_retriever.fusion"):
        be.query(str(built_idx), "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
        be.query(str(built_idx), "r", "m", "AddGoldCoins", mode="hybrid", limit=5)
    warns = [r for r in caplog.records if "graph arm dropped" in r.getMessage()]
    assert len(warns) == 1


def test_index_commit_stamped_and_surfaced(git_idx):
    idx, sha = git_idx
    conn = store.open_db_ro(idx / "fusion.db")
    assert store.meta_get(conn, "index_commit") == sha
    conn.close()
    diag = {}
    _backend().query(str(idx), "r", "m", "fn", mode="hybrid", limit=5,
                     arm_diag=diag)
    assert diag["index_commit"] == sha
    assert diag["commit_mismatch"] is False


def test_commit_mismatch_warns_never_fails(git_idx, caplog):
    idx, sha = git_idx
    diag = {}
    be = _backend(strict_arms=True)      # strict governs arms, not the SHA guard
    with caplog.at_level(logging.WARNING, logger="rainbow_retriever.fusion"):
        hits = be.query(str(idx), "r", "m", "fn", mode="hybrid", limit=5,
                        arm_diag=diag, expected_commit="f" * 40)
    assert hits                                          # never a hard failure
    assert diag["commit_mismatch"] is True
    assert diag["expected_commit"] == "f" * 40
    assert any("commit drift" in r.getMessage() for r in caplog.records)


def test_no_git_worktree_stamps_nothing(built_idx):
    # conftest's srclight_index worktree has no .git -> fail-soft, no stamp
    conn = store.open_db_ro(built_idx / "fusion.db")
    assert store.meta_get(conn, "index_commit") is None
    conn.close()


def test_prior_join_rate_logged_once_per_index(git_idx, caplog):
    idx, _ = git_idx
    be = _backend()
    with caplog.at_level(logging.INFO, logger="rainbow_retriever.fusion"):
        be.query(str(idx), "r", "m", "fn", mode="hybrid", limit=5)
        be.query(str(idx), "r", "m", "fn", mode="hybrid", limit=5)
    msgs = [r for r in caplog.records if "priors join rate" in r.getMessage()]
    assert len(msgs) == 1
