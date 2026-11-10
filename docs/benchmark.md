# Benchmark: fusion vs srclight

Both backends were driven through the same `RetrievalBackend` interface with
matched settings: instruction-prefixed query embedding, file-dedup on,
`mode=hybrid`, k=10. Both arms were indexed fresh from the same worktrees with
the same srclight version (0.15.1). Deltas are paired per query; `*` means the
95% paired bootstrap confidence interval (10k resamples) excludes zero.

## SWE-bench Verified, n=78

File localization against the gold patch files. Two fusion candidates were
locked before the full run; no tuning happened after the lock.

| variant | r@1 | r@3 | r@5 | r@10 | MRR | median ms |
|---|---|---|---|---|---|---|
| srclight (baseline) | 0.181 | 0.475 | 0.578 | 0.690 | 0.381 | ~650 |
| A: dense-led | 0.215 `+.034*` | 0.507 | 0.608 | 0.741 `+.051` | 0.443 `+.062*` | 42 |
| B: precision-led ids | 0.245 `+.064` | 0.469 | 0.563 | 0.691 | 0.430 `+.049` | 47 |
| G: ids + graph w=1 (exploratory) | 0.276 `+.095` | 0.572 `+.096*` | 0.614 | 0.691 | 0.476 `+.095*` | 47 |

- A (dense-led: context-headered dense arm, lexical arm off) is the confirmed
  result: significant MRR (+0.062) and r@1 (+0.034) over srclight, at about 14x
  lower median query latency.
- B did not replicate an earlier small-sample result; there is no significant
  difference between A and B.
- G was added after the protocol was locked. It is one untuned exploratory look,
  not a headline claim.

## Held-out multilingual set, n=40

8 repositories (fmt, gin, gson, laravel, preact, redis, rubocop, tokio; C++, Go,
Java, PHP, JS, C, Ruby, Rust). The protocol and decision rule were committed
before any query ran.

| variant | r@1 | r@10 | MRR | verdict |
|---|---|---|---|---|
| srclight | 0.138 | 0.650 | 0.333 | |
| A: dense-led | 0.212 `+.075` | 0.725 `+.075` | 0.396 `+.064` | not significant (MRR CI [-.005, +.137]) |
| G: ids + graph | 0.325 `+.188*` | 0.725 | 0.476 `+.143*` | confirmed |
| G + trigram (exploratory) | 0.331 `+.194*` | 0.700 | 0.492 `+.160*` | marginal over G |
| G + prior 0.3 (exploratory) | 0.400 `+.263*` | 0.725 | 0.514 `+.182*` | see caveat |

The decision rule (G vs srclight MRR significant and G at least as good as A)
fired, so the server defaults are `lex_mode=ids, w_graph=1.0`.

Caveat on the prior arm: files that change often are more likely to be the
target of an issue, so part of the prior's gain reflects how the benchmark is
sampled rather than retrieval quality alone. The worktree histories predate the
issues, so there is no direct leakage.

## Notes

- Lexical arm design: OR-scored BM25 over the full text of a long natural-language
  query tends to rank prose documents (changelogs, contributing guides) above code.
  The `ids` lexical mode feeds BM25 only the identifier-shaped tokens of the query
  (backticked spans, snake_case, camelCase, PascalCase, dotted.paths, CONSTANTS)
  and restricts that arm to code chunks. On queries with no identifiers the arm
  stays silent and the dense arm carries the ranking.
- Embedding servers such as llama.cpp do not always re-embed identical text
  bit-identically (cosine about 0.9999). Observed run-to-run MRR jitter was about
  +/-0.002, small next to the effects above.
