from __future__ import annotations


def apply_budget(hits, budget):
    if budget is None:
        return hits, False
    kept = []
    used = 0
    for h in hits:
        cost = max(1, len(h["snippet"]) // 4)
        if kept and used + cost > budget:
            return kept, True
        kept.append(h)
        used += cost
    return kept, False
