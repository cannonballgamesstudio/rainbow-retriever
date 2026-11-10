from rainbow_retriever.fusion.lex import id_tokens


def test_extracts_all_identifier_shapes():
    text = ("Dynamically adding `xfail` marker: request.node.add_marker(pytest.mark.xfail) "
            "raised INTERNAL_ERROR in PlayerInventory via take_damage and parseQuestData")
    toks = id_tokens(text).split()
    assert "xfail" in toks
    assert "request.node.add_marker" in toks
    assert "pytest.mark.xfail" in toks
    assert "INTERNAL_ERROR" in toks
    assert "PlayerInventory" in toks
    assert "take_damage" in toks
    assert "parseQuestData" in toks


def test_plain_prose_yields_empty():
    assert id_tokens("Where is the arena boundary shape defined and drawn?") == ""
    assert id_tokens("") == ""


def test_dedup_case_insensitive_keeps_first():
    toks = id_tokens("take_damage then TAKE_DAMAGE again take_damage").split()
    assert toks.count("take_damage") == 1
    assert "TAKE_DAMAGE" not in toks          # deduped against first-seen (lowercased key)


def test_cap_limits_token_count():
    text = " ".join(f"snake_tok_{i}" for i in range(200))
    assert len(id_tokens(text, cap=10).split()) == 10


def test_first_seen_order_preserved():
    toks = id_tokens("zeta_fn then AlphaClass then `beta_var`").split()
    assert toks.index("zeta_fn") < toks.index("AlphaClass") < toks.index("beta_var")


def test_dotted_path_with_underscored_segments_stays_whole():
    toks = id_tokens("see django.db.models.query_utils and my_module.sub_name").split()
    assert "django.db.models.query_utils" in toks
    assert "my_module.sub_name" in toks
    assert "query_utils" not in toks           # not fragmented
