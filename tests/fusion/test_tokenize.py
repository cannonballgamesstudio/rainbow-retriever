from rainbow_retriever.fusion.lex import NOSPLIT_LANGUAGES, tokenize


def test_whole_identifiers_always_present():
    assert "playerinventory" in tokenize("PlayerInventory", "csharp")
    assert "take_damage" in tokenize("take_damage", "gdscript")


def test_camelcase_split_for_csharp_and_js():
    toks = tokenize("AddGoldCoins(amount)", "csharp")
    assert {"addgoldcoins", "add", "gold", "coins"} <= set(toks)
    toks = tokenize("parseQuestData", "javascript")
    assert {"parsequestdata", "parse", "quest", "data"} <= set(toks)


def test_snake_case_not_split_for_python_and_gdscript():
    assert set(tokenize("greet_user", "python")) == {"greet_user"}
    assert set(tokenize("take_damage", "gdscript")) == {"take_damage"}
    assert NOSPLIT_LANGUAGES == frozenset({"python", "gdscript"})


def test_snake_case_split_for_split_languages():
    toks = tokenize("gold_coin_count", "cpp")
    assert {"gold_coin_count", "gold", "coin", "count"} <= set(toks)


def test_query_mode_always_splits():
    toks = tokenize("PlayerInventory add_gold", query=True)
    assert {"playerinventory", "player", "inventory", "add_gold", "add", "gold"} <= set(toks)


def test_short_subtokens_dropped_but_whole_kept():
    toks = tokenize("QLog", "csharp")           # 'q' is len 1 -> dropped as subtoken
    assert "qlog" in toks and "q" not in toks


def test_non_identifier_noise_ignored():
    assert tokenize("):{}=+//", "python") == []
