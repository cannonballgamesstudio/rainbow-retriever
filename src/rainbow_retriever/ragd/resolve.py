from __future__ import annotations

import os

from rainbow_retriever.ragd.errors import RefNotFound


def index_dir_for(ref) -> str:
    return os.path.join(ref.worktree, ".srclight")


def resolve_ref(ro, repo, branch, fallback_refs):
    primary = ro.resolve(repo, branch)
    if primary is not None:
        return primary, False
    for fb in fallback_refs:
        candidate = ro.resolve(repo, fb)
        if candidate is not None:
            return candidate, True
    raise RefNotFound(
        f"no indexed ref for {repo!r} branch {branch!r} (or its fallbacks)",
        details={"repo": repo, "branch": branch, "fallback_refs": list(fallback_refs)},
    )
