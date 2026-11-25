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

import re

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
    """Register Lean 4 (`.lean`) with srclight. Idempotent. Returns True if Lean
    was registered by this call. Uses the tree-sitter-lean grammar when it loads."""
    try:
        from srclight import languages as L
    except Exception:
        return False
    if "lean" in L.LANGUAGES:
        return False
    # The grammar is optional: without it (or with a build that lacks its
    # compiled parser) srclight gets no parser for "lean" and the line-scan
    # fallback below supplies every declaration on its own.
    L.LANGUAGES["lean"] = L.LanguageConfig(
        name="lean", extensions=(".lean",), loader=LEAN_GRAMMAR_MODULE, symbol_query=_LEAN_QUERY)
    L._EXT_TO_LANG.setdefault(".lean", "lean")  # detect_language reads this module-level map
    _install_lean_fallback()
    return True


# The grammar is experimental: on real Mathlib-style code (norm bars, big
# operators, `module` headers) most files contain ERROR nodes, and a
# declaration inside one is never captured (on the upstream RH ZeroFreeRegion
# slice it caught 272 of 350). Top-level Lean commands start at column 0, so a
# line scan recovers whatever the tree-sitter pass missed. It only ever ADDS
# symbols the query did not produce, so a better grammar later just means the
# fallback finds less to add.
_LEAN_KIND = {"theorem": "function", "lemma": "function", "def": "function", "abbrev": "function",
              "opaque": "function", "axiom": "function", "instance": "function",
              "structure": "struct", "class": "struct", "inductive": "enum"}
_LEAN_DECL = re.compile(
    r"^(?:@\[[^\n]*?\]\s*)?"
    r"(?:(?:private|protected|public|noncomputable|nonrec|partial|unsafe|scoped|local)\s+)*"
    r"(theorem|lemma|def|abbrev|opaque|axiom|instance|structure|inductive|class\s+inductive|class)\b"
    r"(?:\s*\([^)]*\))?"          # instance priority, e.g. `instance (priority := 100)`
    r"(?:\s+(?![:\[({])([^\s:({\[]+))?")


def lean_line_decls(text: str) -> list[tuple[str, str, int, int, str | None]]:
    """(kind, name, start_line, end_line, doc) for each column-0 Lean declaration
    (1-based, inclusive). A declaration runs until the next non-blank line that
    starts at column 0. Anonymous instances are skipped (no name to index)."""
    lines = text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        m = _LEAN_DECL.match(lines[i])
        if not m:
            i += 1
            continue
        kw = "inductive" if m.group(1).startswith("class") and "inductive" in m.group(1) else m.group(1)
        j = i + 1
        while j < n and (not lines[j] or lines[j][0].isspace()):
            j += 1
        end = j
        while end > i + 1 and not lines[end - 1].strip():
            end -= 1
        doc = None
        k = i - 1
        if k >= 0 and lines[k].rstrip().endswith("-/"):
            while k >= 0 and not lines[k].lstrip().startswith("/-"):
                k -= 1
            if k >= 0 and lines[k].lstrip().startswith("/--"):
                doc = "\n".join(lines[k:i]).strip()
        if m.group(2):
            out.append((_LEAN_KIND[kw], m.group(2), i + 1, end, doc))
        i = j
    return out


def _install_lean_fallback() -> None:
    import hashlib

    from srclight import indexer as I

    if getattr(I.Indexer._extract_symbols, "_rr_lean", False):
        return
    orig = I.Indexer._extract_symbols

    def _extract_symbols(self, file_id, rel_path, source, lang):
        count = orig(self, file_id, rel_path, source, lang)
        if lang != "lean":
            return count
        have = {(r[0], r[1]) for r in self.db.conn.execute(
            "SELECT name, start_line FROM symbols WHERE file_id = ?", (file_id,))}
        names = {nm for nm, _ in have}
        text = source.decode("utf-8", errors="replace")
        lines = text.splitlines()
        for kind, name, start, end, doc in lean_line_decls(text):
            if name in names or (name, start) in have:
                continue
            body = "\n".join(lines[start - 1:end])
            self.db.insert_symbol(I.SymbolRecord(
                file_id=file_id, kind=kind, name=name, qualified_name=name,
                signature=lines[start - 1].strip()[:300], start_line=start, end_line=end,
                content=body, doc_comment=doc,
                body_hash=hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
                line_count=end - start + 1, parent_symbol_id=None), rel_path)
            names.add(name)
            count += 1
        return count

    _extract_symbols._rr_lean = True
    I.Indexer._extract_symbols = _extract_symbols
