# tests/fusion/test_priors.py
import subprocess
import time

import pytest

from rainbow_retriever.fusion import ingest, store
from rainbow_retriever.fusion.priors import build_priors, load


def _git(cwd, *args, env_extra=None):
    import os
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               **(env_extra or {}))
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                   capture_output=True)


@pytest.fixture()
def git_worktree(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "hot.py").write_text("def hot():\n    return 1\n")
    (wt / "cold.py").write_text("def cold():\n    return 2\n")
    _git(wt, "init", "-q")
    old = f"{int(time.time()) - 300 * 86400} +0000"   # ~10 months ago (epoch + tz -- git's raw date format)
    _git(wt, "add", "."); _git(wt, "commit", "-q", "-m", "init",
         env_extra={"GIT_AUTHOR_DATE": old, "GIT_COMMITTER_DATE": old})
    for i in range(3):                             # hot.py touched recently, 3x
        (wt / "hot.py").write_text(f"def hot():\n    return {i}\n")
        _git(wt, "add", "."); _git(wt, "commit", "-q", "-m", f"hot {i}")
    return wt


def test_hot_file_outranks_cold(git_worktree, srclight_index, tmp_path):
    idx = tmp_path / "pidx"
    idx.mkdir()
    from srclight.db import Database
    from srclight.indexer import Indexer, IndexConfig
    db = Database(idx / "index.db"); db.open(); db.initialize()
    Indexer(db, IndexConfig(root=git_worktree, embed_model=None)).index(git_worktree)
    db.close()
    ingest.ingest_srclight(idx)
    n = build_priors(idx, str(git_worktree))
    assert n >= 2
    conn = store.open_db(idx / "fusion.db")
    p = load(conn)
    conn.close()
    assert p["hot.py"] > p["cold.py"]
    assert 0.0 <= p["cold.py"] <= p["hot.py"] <= 1.0


def test_no_git_history_builds_empty_table(srclight_index, tmp_path):
    _, index_dir = srclight_index                  # fixture worktree has no .git
    ingest.ingest_srclight(index_dir)
    wt_no_git = tmp_path / "nogit"
    wt_no_git.mkdir()
    n = build_priors(index_dir, str(wt_no_git))
    assert n == 0
    conn = store.open_db(index_dir / "fusion.db")
    assert conn.execute("SELECT COUNT(*) FROM priors").fetchone()[0] == 0
    conn.close()


def test_prior_boost_reorders_final_ranking(git_worktree, tmp_path):
    from rainbow_retriever.fusion.fusion_backend import FusionBackend
    from tests.fusion.test_dense import FakeProvider
    idx = tmp_path / "bidx"
    be0 = FusionBackend(embed_model="fake", dims=8, w_prior=0.0,
                        provider_factory=lambda model: FakeProvider())
    be0.index(str(git_worktree), str(idx))
    beP = FusionBackend(embed_model="fake", dims=8, w_prior=5.0,   # exaggerated
                        provider_factory=lambda model: FakeProvider())
    q = "return value helper"                       # prose: dense-only, near-tie
    base = [h.file for h in be0.query(str(idx), "r", "m", q, mode="hybrid", limit=2)]
    boosted = [h.file for h in beP.query(str(idx), "r", "m", q, mode="hybrid", limit=2)]
    assert boosted[0].endswith("hot.py")            # prior pulls the hot file up
    # control documents the boost actually changed something OR hot was already first
    assert base != boosted or base[0].endswith("hot.py")


def test_dot_prefixed_paths_keep_their_keys(tmp_path):
    """.github/-style paths must round-trip: build key == query-time lookup key."""
    wt = tmp_path / "wt"
    (wt / ".github").mkdir(parents=True)
    (wt / ".github" / "ci.yml").write_text("name: ci\n")
    (wt / "app.py").write_text("def app():\n    return 1\n")
    _git(wt, "init", "-q")
    _git(wt, "add", ".")
    _git(wt, "commit", "-q", "-m", "init")

    idx = tmp_path / "idx"
    idx.mkdir()
    from srclight.db import Database
    from srclight.indexer import Indexer, IndexConfig
    db = Database(idx / "index.db"); db.open(); db.initialize()
    Indexer(db, IndexConfig(root=wt, embed_model=None)).index(wt)
    db.close()
    ingest.ingest_srclight(idx)
    build_priors(idx, str(wt))
    conn = store.open_db(idx / "fusion.db")
    p = load(conn)
    conn.close()
    assert ".github/ci.yml" in p          # dot preserved -- matches lookup normalization
    assert "github/ci.yml" not in p
