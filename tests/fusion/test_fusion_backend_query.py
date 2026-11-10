# tests/fusion/test_fusion_backend_query.py
import pytest

from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.reranker import FakeReranker
from tests.fusion.test_dense import FakeProvider


@pytest.fixture(scope="module")
def indexed(srclight_index, tmp_path_factory):
    wt, _ = srclight_index
    idx = tmp_path_factory.mktemp("fusion_query_idx")
    be = FusionBackend(embed_model="fake", dims=8,
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    return be, str(idx)


def test_keyword_mode_finds_exact_symbol(indexed):
    be, idx = indexed
    hits = be.query(idx, "example/demo", "main", "AddGoldCoins", mode="keyword", limit=5)
    assert hits and hits[0].symbol == "AddGoldCoins"
    assert hits[0].repo == "example/demo" and hits[0].ref == "main"
    assert hits[0].file.endswith("sample.cs") and hits[0].line >= 1
    assert hits[0].snippet                     # self-contained hit


def test_hybrid_fuses_lexical_and_dense(indexed):
    be, idx = indexed
    # dedup_file=False on purpose: dedup-by-file is exercised by its own test
    # below (test_dedup_file_default_on) and, with this fixture's tiny corpus,
    # AddGoldCoins and PlayerInventory share sample.cs -- file-dedup would keep
    # only whichever of the two wins the fused rank (a razor-thin RRF margin
    # under the FakeProvider's 8-dim hash embedding), which is not what this
    # test is checking. Reuse the already-built index (dedup_file is
    # query-time-only; it doesn't affect the index artifacts).
    fusion_be = FusionBackend(embed_model="fake", dims=8, dedup_file=False,
                              provider_factory=lambda model: FakeProvider())
    hits = fusion_be.query(idx, "r", "m", "AddGoldCoins inventory", mode="hybrid", limit=5)
    names = {h.symbol for h in hits}
    assert "AddGoldCoins" in names


def test_semantic_mode_runs_dense_only(indexed):
    be, idx = indexed
    # `indexed` is module-scoped and its provider is created once at index() time,
    # so other tests may have already recorded queries on it -- assert growth by
    # exactly 1 across this call rather than an absolute queries-list length.
    provider = be._provider
    before = len(provider.queries)
    hits = be.query(idx, "r", "m", "quest data parsing", mode="semantic", limit=5)
    assert len(provider.queries) == before + 1  # exactly one embed_query call: dense arm only
    assert hits                                 # non-vacuous: dense arm actually retrieves
    for h in hits:
        assert h.file and h.snippet              # self-contained Hit contract


def test_unsupported_mode_raises():
    be = FusionBackend()                       # keyword-only backend
    with pytest.raises(ValueError, match="unsupported mode"):
        be.query("/nope", "r", "m", "x", mode="hybrid")


def test_dedup_file_default_on(indexed):
    be, idx = indexed
    # "QuestLog" reaches both the QuestLog class chunk and addQuestEntry (js splits
    # to add/quest/entry) -- same file, so dedup must collapse them to one hit.
    hits = be.query(idx, "r", "m", "QuestLog", mode="keyword", limit=10)
    assert hits                                # non-vacuous: the query matches
    files = [h.file for h in hits]
    assert len(files) == len(set(files))       # one hit per file


def test_code_only_demotes_markdown(indexed):
    be, idx = indexed
    hits = be.query(idx, "r", "m", "inventory system", mode="hybrid",
                    limit=10, code_only=True)
    kinds = [h.kind for h in hits]
    if "section" in kinds:                     # doc present -> must come after code
        assert kinds.index("section") > 0


class RecordingReranker(FakeReranker):
    """FakeReranker records nothing (verified src/rainbow_retriever/reranker.py:13-28) -- wrap it."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append(query)
        return super().rerank(query, documents)


def test_rerank_flag_routes_through_reranker(indexed, srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "rr_idx"
    rr = RecordingReranker()
    be = FusionBackend(embed_model="fake", dims=8, reranker=rr, lex_mode="full",
                       provider_factory=lambda model: FakeProvider())
    be.index(str(wt), str(idx))
    be.query(str(idx), "r", "m", "gold", mode="hybrid", limit=3, rerank=True)
    assert rr.calls == ["gold"]                # reranker saw the candidate pool


def test_scores_are_reciprocal_rank(indexed):
    # xref merges hits across refs by score; float(n - i) made a bigger index
    # systematically outrank a smaller one at equal rank. 1/(i+1) is comparable
    # across refs and order-preserving within one.
    be, idx = indexed
    hits = be.query(idx, "r", "m", "AddGoldCoins", mode="keyword", limit=5,
                    dedup=False)
    assert hits
    assert [h.score for h in hits] == [1.0 / (i + 1) for i in range(len(hits))]


def test_max_per_file_cap_overrides_binary_dedup(indexed):
    # Backend default dedup_file=True. "QuestLog" reaches BOTH sample.js chunks
    # (QuestLog class + addQuestEntry -- see test_dedup_file_default_on): binary
    # dedup collapses them to 1; max_per_file=2 must keep both.
    be, idx = indexed
    deduped = be.query(idx, "r", "m", "QuestLog", mode="keyword", limit=10)
    capped = be.query(idx, "r", "m", "QuestLog", mode="keyword", limit=10, max_per_file=2)
    from collections import Counter
    assert sum(1 for h in deduped if h.file.endswith("sample.js")) == 1
    assert sum(1 for h in capped if h.file.endswith("sample.js")) == 2   # cap=2 overrides dedup-ON
    assert max(Counter(h.file for h in capped).values()) <= 2


def test_max_per_file_1_equals_binary_dedup(indexed):
    be, idx = indexed
    files = [h.file for h in be.query(idx, "r", "m", "QuestLog", mode="keyword",
                                      limit=10, max_per_file=1)]
    assert len(files) == len(set(files))         # one hit per file, same as dedup-ON


def test_max_per_file_invalid_raises(indexed):
    be, idx = indexed
    with pytest.raises(ValueError, match="max_per_file"):
        be.query(idx, "r", "m", "QuestLog", mode="keyword", limit=5, max_per_file=0)
