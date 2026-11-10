# tests/fusion/test_parity_smoke.py
import time

from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.srclight_backend import SrclightBackend


def test_keyword_parity_on_fixture(srclight_index, tmp_path):
    wt, srclight_idx = srclight_index            # srclight's own index (from conftest)
    fusion_idx = tmp_path / "fidx"
    fusion = FusionBackend()
    fusion.index(str(wt), str(fusion_idx))
    srcl = SrclightBackend()

    for q in ["AddGoldCoins", "Greeter", "parseQuestData"]:
        f_files = {h.file.replace("\\", "/") for h in
                   fusion.query(str(fusion_idx), "r", "m", q, mode="keyword", limit=5)}
        s_files = {h.file.replace("\\", "/") for h in
                   srcl.query(str(srclight_idx), "r", "m", q, mode="keyword", limit=5)}
        assert s_files <= f_files or not s_files, (
            f"fusion keyword missed files srclight found for {q!r}: {s_files - f_files}")


def test_query_latency_tripwire(srclight_index, tmp_path):
    wt, _ = srclight_index
    idx = tmp_path / "lidx"
    be = FusionBackend()
    be.index(str(wt), str(idx))
    be.query(str(idx), "r", "m", "warmup", mode="keyword", limit=5)   # warm caches
    t0 = time.perf_counter()
    for _ in range(10):
        be.query(str(idx), "r", "m", "AddGoldCoins quest inventory",
                 mode="keyword", limit=10)
    per_query_ms = (time.perf_counter() - t0) / 10 * 1000
    # Fixture-sized corpus must be far under a 300ms budget; this catches
    # accidental O(n²) or per-query rebuild regressions, not real-world perf.
    assert per_query_ms < 150, f"keyword query {per_query_ms:.0f}ms on a toy index"
