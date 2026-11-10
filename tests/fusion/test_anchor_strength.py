# tests/fusion/test_anchor_strength.py
import pytest

from rainbow_retriever.fusion.lex import anchor_strength


@pytest.mark.parametrize("text,expected", [
    # none: prose only, no identifier shapes
    ("How does the app avoid regenerating the planet mesh every frame?", "none"),
    ("", "none"),
    # strong: long snake_case / dotted / camelCase anchors
    ("crash in foo_bar handler", "strong"),                    # foo_bar: len 7, '_'
    ("Fix `PlayerInventory.AddGoldCoins` returning +1", "strong"),
    ("os.path joins break on UNC shares", "strong"),           # dotted len 7
    ("the parseQuestData call returns raw", "strong"),         # camel len 14
    # weak: identifier-ish but short / no strong shape
    ("see `my_fn` for details", "weak"),                       # len 5 despite '_'
    ("the HTTP layer times out", "weak"),                      # SCREAMING len 4
])
def test_anchor_strength_frozen_rule(text, expected):
    assert anchor_strength(text) == expected
