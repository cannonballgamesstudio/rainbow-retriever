from __future__ import annotations


def compute_staleness(indexed_sha, your_sha, served_is_fallback,
                      requested_branch, served_ref) -> dict:
    sha_mismatch = your_sha is not None and your_sha != indexed_sha
    stale = sha_mismatch or served_is_fallback
    push_hint = None
    if served_is_fallback:
        push_hint = (f"branch {requested_branch!r} is not indexed; served {served_ref!r} "
                     f"-- push to index this branch")
    elif sha_mismatch:
        push_hint = (f"indexed at {indexed_sha[:8]}, you are at {your_sha[:8]} "
                     f"-- push this branch to refresh the index")
    return {"indexed_sha": indexed_sha, "your_sha": your_sha,
            "stale": stale, "push_hint": push_hint}
