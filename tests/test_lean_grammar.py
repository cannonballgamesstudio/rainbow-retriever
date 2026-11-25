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


def test_without_grammar_lean_stays_unregistered(monkeypatch):
    _unregister()
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "tree_sitter_lean" else real(name, *a))
    assert srclight_patches.register_lean() is False
    assert "lean" not in L.LANGUAGES
    assert L.detect_language(Path("A.lean")) is None


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
