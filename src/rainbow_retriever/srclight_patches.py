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

import importlib.util

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
    if register_lean():
        patched.append("lean")
    return patched


# ---------------------------------------------------------------------------
# Lean 4. srclight 0.15 has no Lean support; we register it from the
# tree-sitter-lean grammar (github.com/Julian/tree-sitter-lean, MIT, Python
# module `tree_sitter_lean`, installed via the `lean` extra). Declarations map
# onto srclight's existing capture prefixes so kinds, outlines and call-graph
# edges work unchanged: def/theorem/lemma/abbrev/opaque/axiom/instance ->
# function, structure/class -> struct, inductive -> enum, namespace -> namespace.
# Node and field names are the grammar's own (see its queries/locals.scm).
_LEAN_QUERY = """
(def name: (identifier) @fn.name) @fn.def
(theorem name: (identifier) @fn.name) @fn.def
(abbrev name: (identifier) @fn.name) @fn.def
(opaque name: (identifier) @fn.name) @fn.def
(axiom name: (identifier) @fn.name) @fn.def
(instance name: (identifier) @fn.name) @fn.def
(structure name: (identifier) @struct.name) @struct.def
(inductive name: (identifier) @enum.name) @enum.def
(namespace name: (identifier) @ns.name) @ns.def
"""
LEAN_GRAMMAR_MODULE = "tree_sitter_lean"


def register_lean() -> bool:
    """Register Lean 4 (`.lean`) with srclight if the grammar is installed.
    Idempotent. Returns True if Lean was registered by this call."""
    try:
        from srclight import languages as L
    except Exception:
        return False
    if "lean" in L.LANGUAGES:
        return False
    if importlib.util.find_spec(LEAN_GRAMMAR_MODULE) is None:
        return False  # grammar absent: leave .lean unindexed rather than half-registered
    L.LANGUAGES["lean"] = L.LanguageConfig(
        name="lean", extensions=(".lean",), loader=LEAN_GRAMMAR_MODULE, symbol_query=_LEAN_QUERY)
    L._EXT_TO_LANG.setdefault(".lean", "lean")  # detect_language reads this module-level map
    return True
