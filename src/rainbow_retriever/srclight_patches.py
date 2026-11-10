"""Index-time patches to the pinned external `srclight` wheel, applied from code we own.

srclight (0.15.x) ships a tree-sitter C/C++ symbol query that only matches a *direct*
`function_declarator`:

    (function_definition
        declarator: (function_declarator
            declarator: (identifier) @fn.name)) @fn.def

A pointer-returning function -- `char *foo(...)`, `void *bar(...)` -- parses as
`function_definition -> pointer_declarator -> function_declarator`, so it never matches
and the function is silently dropped as a symbol (and thus as a call-graph node). On
C-heavy code bases this can drop a large share of the call graph.

srclight is a plain wheel with no local source, so we cannot fix it upstream here. Instead
we extend the in-memory `LanguageConfig.symbol_query` on our index path (the config is a
mutable dataclass; the indexer compiles the query per run via `Query(language,
config.symbol_query)`, so mutating the string before indexing takes effect). Idempotent.
"""
from __future__ import annotations

# Single- and double-pointer-returning function definitions. Same capture names
# (@fn.name / @fn.def) the base query uses, so srclight treats them identically.
_C_POINTER_PATTERNS = """
(function_definition
    declarator: (pointer_declarator
        declarator: (function_declarator
            declarator: (identifier) @fn.name))) @fn.def

(function_definition
    declarator: (pointer_declarator
        declarator: (pointer_declarator
            declarator: (function_declarator
                declarator: (identifier) @fn.name)))) @fn.def
"""

_MARKER = "pointer_declarator"  # idempotency sentinel


def apply() -> list[str]:
    """Extend srclight's C and C++ symbol queries to capture pointer-returning
    functions. Idempotent. Returns the list of language names patched this call
    (empty if already patched or srclight is unavailable)."""
    try:
        from srclight import languages as L
    except Exception:
        return []

    patched: list[str] = []
    for lang in ("c", "cpp"):
        cfg = L.LANGUAGES.get(lang)
        if cfg is None:
            continue
        if _MARKER in cfg.symbol_query:
            continue  # already has pointer patterns (patched, or a future upstream fix)
        cfg.symbol_query = cfg.symbol_query.rstrip() + "\n" + _C_POINTER_PATTERNS
        patched.append(lang)
    return patched
