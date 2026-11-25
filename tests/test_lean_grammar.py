"""Lean 4 registration in srclight_patches (needs the `lean` extra for the parse tests)."""
import importlib.util
from pathlib import Path

import pytest
import srclight.languages as L

from rainbow_retriever import srclight_patches

HAVE_GRAMMAR = importlib.util.find_spec("tree_sitter_lean") is not None

SAMPLE = b"""import Mathlib

namespace Foo

/-- doc -/
theorem bar_pos (n : Nat) : 0 < n + 1 := Nat.succ_pos n

lemma baz : True := trivial

def double (n : Nat) : Nat := n + n

abbrev Nat2 := Nat

structure Point where
  x : Nat
  y : Nat

class Shape (a : Type) where
  area : a -> Nat

inductive Color
  | red
  | green

instance instShapePoint : Shape Point := { area := fun p => double p.x }

end Foo
"""


def _unregister():
    L.LANGUAGES.pop("lean", None)
    L._EXT_TO_LANG.pop(".lean", None)


def test_without_grammar_line_scan_still_indexes(monkeypatch, tmp_path):
    import sqlite3

    from srclight.db import Database
    from srclight.indexer import Indexer

    _unregister()
    monkeypatch.setattr(srclight_patches, "LEAN_GRAMMAR_MODULE", "no_such_grammar_module")
    L._LANGUAGES.pop("lean", None)
    assert srclight_patches.register_lean() is True
    assert L.detect_language(Path("A.lean")) == "lean"
    assert L.get_language("lean") is None
    (tmp_path / "Hard.lean").write_text(HARD, encoding="utf-8")
    (tmp_path / ".srclight").mkdir()
    db = Database(tmp_path / ".srclight" / "index.db")
    db.open()
    db.initialize()
    Indexer(db).index(tmp_path)
    names = {r[0] for r in sqlite3.connect(tmp_path / ".srclight" / "index.db").execute("select name from symbols")}
    assert {"norm_sum_le", "lamF", "uses_lamF", "instFoo"} <= names
    _unregister()
    L._LANGUAGES.pop("lean", None)


@pytest.mark.skipif(not HAVE_GRAMMAR, reason="tree-sitter-lean not installed (pip install '.[lean]')")
def test_register_is_idempotent_and_detects_extension():
    _unregister()
    assert srclight_patches.register_lean() is True
    assert srclight_patches.register_lean() is False
    assert L.detect_language(Path("src/Foo/Bar.lean")) == "lean"


@pytest.mark.skipif(not HAVE_GRAMMAR, reason="tree-sitter-lean not installed (pip install '.[lean]')")
def test_query_extracts_lean_declarations():
    from tree_sitter import Parser, Query, QueryCursor

    _unregister()
    srclight_patches.register_lean()
    lang = L.get_language("lean")
    assert lang is not None
    tree = Parser(lang).parse(SAMPLE)
    got = set()
    for _, caps in QueryCursor(Query(lang, L.LANGUAGES["lean"].symbol_query)).matches(tree.root_node):
        name = next(n for k, ns in caps.items() if k.endswith(".name") for n in ns).text.decode()
        kind = next(k for k in caps if k.endswith(".def")).split(".")[0]
        got.add((kind, name))
    expected = {("ns", "Foo"), ("fn", "bar_pos"), ("fn", "baz"), ("fn", "double"), ("fn", "Nat2"),
                ("struct", "Point"), ("struct", "Shape"), ("enum", "Color"), ("fn", "instShapePoint")}
    assert expected <= got, expected - got


# Mathlib notation the experimental grammar does not parse (norm bars, big
# operators, `module` header): the line-scan fallback must still find these.
HARD = """module

public import Mathlib

open Finset

namespace Formalism.ZF

/-- the bound -/
theorem norm_sum_le (Z : Finset ℂ) (m : ℂ → ℕ) (z : ℂ) :
    ‖∑ ρ ∈ Z, (m ρ : ℂ) / (z - ρ)‖ ≤ ∑ ρ ∈ Z, ‖(m ρ : ℂ) / (z - ρ)‖ :=
  norm_sum_le _ _

@[simp] noncomputable def lamF (σ : ℝ) : ℝ := ∑' n : ℕ, |σ| ^ n

lemma uses_lamF (σ : ℝ) : lamF σ = lamF σ := rfl

instance (priority := 100) instFoo : Inhabited ℕ := ⟨0⟩

end Formalism.ZF
"""


def test_line_scan_finds_declarations_with_docs_and_spans():
    got = {name: (kind, s, e, doc) for kind, name, s, e, doc in srclight_patches.lean_line_decls(HARD)}
    assert set(got) == {"norm_sum_le", "lamF", "uses_lamF", "instFoo"}
    kind, s, e, doc = got["norm_sum_le"]
    assert kind == "function" and (s, e) == (10, 12) and doc == "/-- the bound -/"
    assert got["lamF"][1:3] == (14, 14)


@pytest.mark.skipif(not HAVE_GRAMMAR, reason="tree-sitter-lean not installed (pip install '.[lean]')")
def test_index_recovers_declarations_the_grammar_misses(tmp_path):
    import sqlite3

    from srclight.db import Database
    from srclight.indexer import Indexer

    srclight_patches.apply()
    (tmp_path / "Hard.lean").write_text(HARD, encoding="utf-8")
    (tmp_path / ".srclight").mkdir()
    db = Database(tmp_path / ".srclight" / "index.db")
    db.open()
    db.initialize()
    Indexer(db).index(tmp_path)
    c = sqlite3.connect(tmp_path / ".srclight" / "index.db")
    names = {r[0] for r in c.execute("select name from symbols where kind != 'namespace'")}
    assert {"norm_sum_le", "lamF", "uses_lamF", "instFoo"} <= names
    edges = {(a, b) for a, b in c.execute(
        "select x.name, y.name from symbol_edges e join symbols x on x.id = e.source_id "
        "join symbols y on y.id = e.target_id")}
    assert ("uses_lamF", "lamF") in edges
