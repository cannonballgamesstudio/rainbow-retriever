from __future__ import annotations

# Bump when fusion.db schema / sidecar formats change incompatibly. Part of the
# stored backend_version, so the registry re-indexes instead of silently drifting.
FUSION_FORMAT_VERSION = 1


def srclight_format() -> str:
    """srclight's index-format marker (same fallback rule as SrclightBackend)."""
    import srclight
    fmt = getattr(srclight, "INDEX_FORMAT_VERSION", None)
    if fmt is None:
        fmt = "fmt-" + ".".join(srclight.__version__.split(".")[:2])
    return str(fmt)


def fusion_identity(embed_model: str | None, dims: int | None) -> str:
    return (f"fusion@{FUSION_FORMAT_VERSION}+{srclight_format()}"
            f"+{embed_model or 'none'}+{dims or 0}")
