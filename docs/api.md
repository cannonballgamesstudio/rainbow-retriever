# API reference

Backend id: `fusion`. Package: `rainbow_retriever`.

```
     lexical     trigram     dense      graph      priors     rerank
        \           \          |          /           /          |
         +----- weighted RRF (first fuse) ----+      /       (opt-in)
                        |                           /
              top-S seeds -> graph expansion -> final fuse
                        |
              x history-prior boost -> file-dedup -> top-K
```

## 1. The interface (`rainbow_retriever.backend.RetrievalBackend`)

Every backend (`FakeBackend`, `SrclightBackend`, `FusionBackend`) implements one
`Protocol` defined in `src/rainbow_retriever/backend.py`. The server and the
tests only use this interface.

```python
class RetrievalBackend(Protocol):
    def capabilities(self) -> Capabilities
    def index(self, worktree_path: str, index_dir: str) -> IndexResult
    def query(self, index_dir, repo, ref, text, mode="hybrid", limit=20,
              dims=None, code_only=False, rerank=False) -> list[Hit]
    def ctx(self, index_dir, repo, ref, symbol) -> SymbolGraph | None
    def deps(...) -> list[GraphEdge];  def file_overview(...) -> list[FileSymbol]
    def tests(...) -> list[GraphEdge]; def structure(...) -> Structure
    def hierarchy(...) -> Hierarchy;   def reachable(...) -> ReachResult
```

`Hit` is frozen: `repo, ref, file, line, symbol, score, snippet, kind`. The
score is rank-derived (reciprocal rank); list order is authoritative. Snippets
are self-contained.

## 2. `rainbow_retriever.fusion.fusion_backend.FusionBackend`

### Construction

```python
FusionBackend(
    embed_model=None,      # e.g. "openai:Qwen/Qwen3-Embedding-8B"; None => keyword-only
    dims=None,             # embedding dims (4096 for Qwen3-Embedding-8B)
    reranker=None,         # HttpReranker | None (opt-in per query)
    query_instruct=None,   # instruction prefixed to the embedded query only
    dedup_file=True,       # one hit per file, over-fetch backfills
    w_lex=1.0, w_dense=1.0,          # lexical / dense RRF weights
    lex_mode="ids",                  # "ids" | "full" | "off"
    w_graph=0.0,                     # graph arm (server default 1.0)
    seeds=20, graph_hops=1,          # graph expansion breadth / depth
    graph_damp=0.5,                  # hop-2 damping
    graph_edge_variant="spec",       # "spec" | "flat" | "calls_led"
    graph_seed_arm="fused",          # "fused" | "dense"
    graph_seed_exclude_docs=False,
    w_trigram=0.0,                   # trigram arm
    w_prior=0.0,                     # git-history prior (server default 0.3)
    header_imports=False,            # experimental: `uses:` line in embed headers
    strict_arms=False,               # raise instead of dropping a failed enhancer arm
    provider_factory=None,           # test seam: model -> embedding provider
)
```

Library defaults are conservative (enhancer arms off). The server defaults (via
`RagdConfig`) are the benchmark-selected configuration: `lex_mode=ids,
w_graph=1.0, w_prior=0.3`. See [benchmark.md](benchmark.md).

The embedding provider comes from `srclight.embeddings.get_provider(model)`.
`openai:<model>` uses any OpenAI-compatible `/v1/embeddings` endpoint, configured
with `OPENAI_BASE_URL` (for example `http://localhost:8080`) and, if required,
`OPENAI_API_KEY`.

### `index(worktree, index_dir)`: build once per ref

1. srclight's `Indexer` runs without an embedding model (parse, symbols, edges).
   A small in-memory patch to srclight's C/C++ symbol query also captures
   pointer-returning functions (`srclight_patches.py`).
2. **Ingest** (`fusion/ingest.py`): srclight's `index.db` becomes the
   `fusion.db` chunk store. Each chunk gets a context header,
   `relpath [language] EnclosingScope`, prepended to its embed text, plus
   fielded lexical text (name / signature / body and doc comment).
3. **Lexical** (`fusion/lex.py`): bm25s index. Identifier tokenization keeps the
   whole identifier and adds camelCase/PascalCase/snake_case subtokens, except for
   snake_case-conventional languages (Python, GDScript), where subtoken splitting
   is skipped. Field weighting by token duplication (name x4, signature x2, body x1).
4. **Adjacency** (`fusion/graph.py`): `adjacency(src, dst, w, edge_type)` in
   chunk-row space, both directions, `w = type_weight x srclight confidence`,
   max() aggregation. Type weights: contains 1.0, calls 0.9,
   inherits/implements 0.8, imports 0.7, other 0.6.
5. **Trigrams** (`fusion/trigram.py`): postings over identifier tokens of length
   4 or more, lowercase 3-grams.
6. **Priors** (`fusion/priors.py`): per file,
   `0.7 * exp(-age_days / 180) + 0.3 * min(1, log1p(commits) / log1p(50))`
   from `git log` (last 2000 commits, 30 s timeout). No git gives an empty table.
7. **Dense** (`fusion/dense.py`, only with `embed_model`): embeds each chunk's
   header and code text (batched, with per-item retry). More than 10% failed
   items raises instead of writing a partial index. Output is an L2-normalized
   float16 `fusion_emb.npy`.

All sidecars are written to a temporary path and swapped into place. The index
identity string is `fusion@<fmt>+<srclight-fmt>+<model>+<dims>`; a change in any
component means the index should be rebuilt. The adjacency, trigram and prior
tables are additive and can be rebuilt without re-embedding.

### `query(...)`

```
mode=keyword   -> lexical over the raw query text only (lex_mode ignored)
mode=semantic  -> dense only
mode=hybrid    -> arms by config:
   lexical, lex_mode=ids:  BM25 over id_tokens(query), the identifier-shaped
                           tokens only (backticks, snake, camel, Pascal, dotted,
                           CONSTANTS; dedup, cap 64), restricted to code chunks.
                           Silent when the query has no identifiers.
                           ("full" = raw text, "off" = disabled)
   trigram, w_trigram>0:   containment-scored trigram match over id tokens
                           (6+ chars); catches prefix/partial identifiers.
   dense, always:          instruction-prefixed query embedding, cosine over
                           the float16 matrix (cached per index).
-> first fuse: weighted RRF (k=60) over the present arms
-> graph, w_graph>0: top-`seeds` (row, score) -> deterministic expansion
   score(u) = sum over seeds v of seed_score(v) * w(v,u)
   (hop 2 damped x0.5, capped) -> final fuse
-> priors, w_prior>0: score x (1 + w_prior * prior(file)), final boost only
-> rerank=True: HttpReranker pass over the fused pool (opt-in)
-> code_only=True: docs demoted below code
-> file-dedup (best per file, over-fetch backfills) or max_per_file cap
-> truncate to limit
```

Optional keyword-only arguments: `arm_diag` (a dict filled with active and
dropped arms and the index commit) and `expected_commit` (warn once if the index
was built from a different commit).

**Failure behavior.** Advertised modes fail hard: if the embedding server is down
on a `semantic` or `hybrid` query, the query raises; it never silently serves
keyword results as "hybrid". Enhancer arms (graph, trigram, priors) fail soft:
the arm is dropped, a warning is logged once per index on the
`rainbow_retriever.fusion` logger, and the query continues. `strict_arms=True`
turns these into errors.

Graph operations (`ctx`, `deps`, `file_overview`, `tests`, `structure`,
`hierarchy`, `reachable`) delegate to the srclight adapter over the same
`index.db`.

## 3. Reranker (`rainbow_retriever.reranker`)

`HttpReranker(endpoint, model)` posts
`{"model", "query", "documents", "return_documents": false}` to
`<endpoint>/rerank` and reads `results[].index` / `relevance_score`. It is
fail-open: any error keeps the original order. `FakeReranker` is a deterministic
test double.

## 4. Server (`rainbow_retriever.ragd`)

```bash
RR_BACKEND=fusion RR_REGISTRY_DB=registry.db RR_DEFAULT_IDENTITY=local \
  uvicorn rainbow_retriever.ragd.server:build_app --factory --port 8081
```

The registry (`rainbow_retriever.registry`) maps `(repo, ref)` to a worktree,
head SHA and status. Only `ready` refs are served, and the index directory for a
ref is `<worktree>/.srclight`.

Request body for `POST /v1/query`:

| field | type | notes |
|---|---|---|
| `repo`, `branch` | str | required |
| `text` | str | required |
| `mode` | str | `keyword`, `semantic`, `hybrid` (default) |
| `limit` | int | 1..200, default 20 |
| `fallback_refs` | list[str] | refs to try if `branch` is not indexed |
| `local_head_sha` | str | 40-char hex; enables the staleness check |
| `code_only`, `rerank` | bool | opt-in |
| `dedup` | bool | null = server default |
| `max_chunks_per_file` | int | overrides `dedup` |
| `budget` | int | approximate token budget for returned snippets |
| `query_dims` | int | must match a supported dimension |

The response carries `served_ref`, `hits`, `staleness`, `capabilities`,
`truncated`, `anchored` (`strong` / `weak` / `none`: how identifier-bearing the
query is) and, for the fusion backend, `arms` (diagnostics).

`POST /v1/xref` takes `refs` and merges hits across them by reciprocal rank.
`POST /v1/ctx` and the `GET` endpoints `/v1/deps`, `/v1/file`, `/v1/tests`,
`/v1/map`, `/v1/hierarchy`, `/v1/reachable` expose the graph operations.

Errors are JSON `{"error": {"code", "message", "request_id", "details"?}}` with
codes such as `invalid_request` (400), `identity_required` (401),
`ref_not_found` (404), `ambiguous_symbol` (409), `unsupported_mode` (422),
`backend_degraded` / `backend_busy` (503).

Environment variables are listed in the README.

## 5. Command-line client (`cli/`)

`rr` is a Rust client for the server. It derives the repository id and branch
from the current git checkout and calls the endpoints above. See the README for
the command table, client environment variables, and the optional Claude Code
hooks.
