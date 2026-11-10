# tests/fusion/conftest.py
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def srclight_index(tmp_path_factory):
    """A tiny polyglot worktree indexed by srclight (keyword/graph only, no embeddings).

    NOTE: sample.gd is deliberately included even though srclight 0.15.1 does NOT
    index GDScript -- test_gdscript_gap_tripwire documents the gap.
    """
    from srclight.db import Database
    from srclight.indexer import Indexer, IndexConfig

    root = tmp_path_factory.mktemp("fusion_fixture")
    wt = root / "wt"
    wt.mkdir()
    (wt / "sample.py").write_text(
        "class Greeter:\n"
        "    def greet_user(self, name):\n"
        "        \"\"\"Return a greeting for the player.\"\"\"\n"
        "        return f'hi {name}'\n"
        "\n"
        "def top_level_fn():\n"
        "    g = Greeter()\n"
        "    return g.greet_user('x')\n")
    (wt / "caller.py").write_text(
        "from sample import top_level_fn\n\n"
        "def boot_sequence():\n"
        "    return top_level_fn()\n")
    (wt / "sample.cs").write_text(
        "namespace Demo {\n"
        "  public class PlayerInventory {\n"
        "    public int AddGoldCoins(int amount) { return amount + 1; }\n"
        "  }\n"
        "}\n")
    (wt / "sample.js").write_text(
        "class QuestLog {\n"
        "  addQuestEntry(entry) { return entry; }\n"
        "}\n"
        "function parseQuestData(raw) { return raw; }\n")
    (wt / "sample.gd").write_text(
        "extends Node\n\nfunc take_damage(amount):\n    health -= amount\n")
    (wt / "README.md").write_text(
        "# Demo\n\nThe inventory system lives in sample.cs.\n")

    index_dir = root / "idx"
    index_dir.mkdir()
    db = Database(index_dir / "index.db")
    db.open()
    db.initialize()
    Indexer(db, IndexConfig(root=wt, embed_model=None)).index(wt)
    db.close()
    return wt, index_dir
