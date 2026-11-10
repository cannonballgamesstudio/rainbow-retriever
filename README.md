# rainbow-retriever

Multi-signal code retrieval. Given a natural-language or identifier-bearing query
and an indexed repository, it returns a ranked list of code locations (file, line,
symbol, snippet).

Several retrieval signals ("arms") are computed independently and combined with
weighted reciprocal rank fusion (RRF):

- **lexical**: BM25 (bm25s) over fielded symbol text, with code-aware identifier tokenization
- **trigram**: identifier trigram containment, for partial and prefix identifier matches
- **dense**: cosine similarity over embeddings from any OpenAI-compatible embedding endpoint (optional)
- **graph**: expansion over the call / contains / inherits / imports graph, seeded from the fused ranking
- **priors**: a per-file git-history score (recency and change frequency), applied as a final boost
- **rerank**: an optional cross-encoder pass through an HTTP `/rerank` endpoint

Parsing, symbol extraction and edge extraction are done by
[srclight](https://pypi.org/project/srclight/) (MIT), used unmodified as a library
dependency. rainbow-retriever reads srclight's `index.db` and builds its own
sidecar indexes next to it.

With no embedding model configured the backend is keyword-only and fully
deterministic (no network calls, same input gives the same output).

## Architecture

```
     lexical     trigram     dense      graph      priors     rerank
        \           \          |          /           /          |
         +----- weighted RRF (first fuse) ----+      /       (opt-in)
                        |                           /
              top-S seeds -> graph expansion -> final fuse
                        |
              x history-prior boost -> file-dedup -> top-K
```

Index layout (one directory per indexed ref):

| file | contents |
|---|---|
| `index.db` | srclight symbols and edges |
| `fusion.db` | chunk store, adjacency, trigram postings, priors, metadata |
| `fusion_bm25/` | bm25s lexical index |
| `fusion_emb.npy` | L2-normalized float16 embeddings (dense mode only) |

See [docs/api.md](docs/api.md) for the full API reference.

## Install

Requires Python 3.11 or newer.

```bash
cd rainbow-retriever           # a checkout of this repository
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick usage

### Keyword-only (deterministic)

```python
from rainbow_retriever.fusion.fusion_backend import FusionBackend

be = FusionBackend()                       # no embed_model: keyword mode only
be.index("path/to/repo", "path/to/repo/.srclight")

hits = be.query("path/to/repo/.srclight", repo="my/repo", ref="main",
                text="parse_config", mode="keyword", limit=10)
for h in hits:
    print(f"{h.file}:{h.line}  {h.symbol}  ({h.kind})")
```

`index()` runs srclight's indexer on the worktree (no embeddings), then builds the
fusion sidecars. Graph operations are also available on the same index:

```python
graph = be.ctx(idx, "my/repo", "main", "parse_config")       # callers and callees
edges = be.deps(idx, "my/repo", "main", "parse_config", transitive=True)
```

### Dense / hybrid

Dense mode needs an embedding server that speaks the OpenAI `/v1/embeddings`
API (for example llama.cpp `llama-server --embedding`, vLLM, or a hosted
provider). The model string is passed to srclight's provider factory; the
`openai:` prefix selects the OpenAI-compatible client, which reads
`OPENAI_BASE_URL` (and `OPENAI_API_KEY` if the server needs one).

```bash
export OPENAI_BASE_URL=http://localhost:8080
```

```python
from rainbow_retriever.fusion.dense import DEFAULT_QUERY_INSTRUCT
from rainbow_retriever.fusion.fusion_backend import FusionBackend

be = FusionBackend(embed_model="openai:Qwen/Qwen3-Embedding-8B", dims=4096,
                   query_instruct=DEFAULT_QUERY_INSTRUCT,
                   lex_mode="ids", w_graph=1.0, w_prior=0.3)
be.index("path/to/repo", "path/to/repo/.srclight")
hits = be.query("path/to/repo/.srclight", "my/repo", "main",
                "where are retries handled when the upload times out?",
                mode="hybrid", limit=20)
```

Notes:

- Embedding the same text twice on some servers is not bit-identical, so dense
  and hybrid rankings can vary slightly between runs. Keyword mode does not.
- If more than 10% of chunks fail to embed, `index()` raises instead of writing a
  partial dense index.
- If the embedding server is down at query time, `semantic` and `hybrid` queries
  raise. They do not silently fall back to keyword search. Enhancer arms
  (graph, trigram, priors) fail soft: the arm is dropped and a warning is logged.

## Running the server

`rainbow_retriever.ragd` is a FastAPI app that serves one backend over HTTP. It
resolves `(repo, branch)` to an indexed worktree through a small SQLite
registry. The index for a registered worktree is expected at
`<worktree>/.srclight`.

Register an indexed ref:

```python
import subprocess, time
from rainbow_retriever.fusion.fusion_backend import FusionBackend
from rainbow_retriever.registry import Registry

wt = "path/to/repo"
be = FusionBackend()
res = be.index(wt, f"{wt}/.srclight")
sha = subprocess.check_output(["git", "-C", wt, "rev-parse", "HEAD"], text=True).strip()

reg = Registry.init_db("registry.db")
reg.upsert_ref("my/repo", "main", sha, wt, "ready", int(time.time()),
               res.backend_version, "default", backend_id="fusion")
```

Start the server:

```bash
export RR_BACKEND=fusion
export RR_REGISTRY_DB=registry.db
export RR_DEFAULT_IDENTITY=local        # see "Identity" below
uvicorn rainbow_retriever.ragd.server:build_app --factory --host 127.0.0.1 --port 8081
```

Query it:

```bash
curl -s localhost:8081/v1/query -H 'content-type: application/json' \
  -d '{"repo": "my/repo", "branch": "main", "text": "parse_config", "mode": "keyword"}'
```

Without `RR_EMBED_MODEL` the server supports only `"mode": "keyword"`; the
default mode `hybrid` returns 422 `unsupported_mode`.

Endpoints: `POST /v1/query`, `POST /v1/xref` (query across several refs),
`POST /v1/ctx`, `GET /v1/deps`, `/v1/file`, `/v1/tests`, `/v1/map`,
`/v1/hierarchy`, `/v1/reachable`, `/v1/resolve`, `/v1/status`,
`/v1/capabilities`, `/healthz`.

**Identity.** Every `/v1` request needs an `X-User-Login` header, which an
authenticating reverse proxy is expected to set. For local use without a proxy,
`RR_DEFAULT_IDENTITY` supplies a fallback login. Query text is logged only as a
hash unless `RR_LOG_RAW_TEXT=1`.

### Configuration (environment)

| variable | default | meaning |
|---|---|---|
| `RR_BACKEND` | `fake` | `fusion`, `srclight`, or `fake` (test double) |
| `RR_REGISTRY_DB` | (required) | path to the registry SQLite file |
| `RR_EMBED_MODEL` | unset | e.g. `openai:Qwen/Qwen3-Embedding-8B`; unset = keyword only |
| `RR_DIMS` | unset | embedding dimensions |
| `RR_QUERY_INSTRUCT` | on | instruction prefixed to the embedded query (`0` disables, other text is used verbatim) |
| `RR_QUERY_DEDUP_FILE` | on | at most one hit per file |
| `RR_QUERY_MAX_PER_FILE` | unset | integer per-file cap (overrides dedup) |
| `RR_FUSION_LEX_MODE` | `ids` | `ids` (identifier tokens only), `full`, or `off` |
| `RR_FUSION_W_LEX` / `RR_FUSION_W_DENSE` | `1.0` | RRF weights |
| `RR_FUSION_W_GRAPH` | `1.0` | graph arm weight |
| `RR_FUSION_SEEDS` / `RR_FUSION_GRAPH_HOPS` | `20` / `1` | graph expansion breadth and depth |
| `RR_FUSION_W_TRIGRAM` | `0.0` | trigram arm weight |
| `RR_FUSION_W_PRIOR` | `0.3` | git-history prior boost |
| `RR_RERANK_ENDPOINT` / `RR_RERANK_MODEL` | unset | optional `/rerank` server |
| `RR_EMBED_QUEUE` | `4` | max concurrent embedding queries before 503 |
| `RR_STRICT_ARMS` | off | `1` raises on a failed enhancer arm instead of dropping it |

The library constructor defaults are conservative (graph and priors off). The
server defaults above are the configuration selected by the benchmarks below.

## Command-line tool

`cli/` is a Rust client for the server, with a single binary, `rr`. It needs a
Rust toolchain (stable).

```bash
cargo install --path cli
```

Run it from inside a git checkout. It works out the repository id from the
`origin` remote (`owner/repo`), or uses `local/<directory name>` when there is no
remote. The branch comes from `HEAD`. Override either with `--repo` and `--ref`.
These must match a ref registered on the server.

| command | old name (still accepted) | what it does |
|---|---|---|
| `rr fetch "<text>"` | `query` | ranked search (`--mode keyword\|semantic\|hybrid`, `--limit N`) |
| `rr sniff [PATH]` | `map` | directory structure and hotspot files |
| `rr dig <FILE>` | `file` | symbols defined in a file |
| `rr track <symbol>` | `deps` | dependency edges (`--transitive`, `--depth N`) |
| `rr point <symbol>` | `ctx` | definition, signature, callers and callees |
| `rr pedigree <Class>` | `hierarchy` | base classes, subclasses, implementors |
| `rr watchdog <symbol>` | `tests` | tests that cover a symbol |
| `rr trace <a,b>` | `reach` | transitive reachability (`--direction callees\|callers`) |
| `rr xref "<text>" --refs a,b` | | search several refs and merge |
| `rr speak` | `capabilities` | server capabilities |
| `rr status` / `rr ls` | | indexed refs on the server |
| `rr resolve` / `rr health` | | registry row for this branch / health check |
| `rr goodboy` | | prints the band legend and version |

Example, against the server started above:

```bash
$ export RR_URL=http://localhost:8081
$ rr fetch parse_config --mode keyword
cfg.py:1 parse_config [function] 1.0000  def parse_config(path):
[1 hits | fusion@1+fmt-0.15+none+0]
```

On a terminal, results are compact text. When stdout is not a terminal (piped),
most commands print JSON; `--json` forces it. `fetch` stays compact unless
`--json` or `RR_FULL=1` is given.

Client environment:

| variable | default | meaning |
|---|---|---|
| `RR_URL` | `http://localhost:8081` | server URL (same as `--endpoint`) |
| `RR_USER` | `USER` / `USERNAME` | login sent in the `X-User-Login` header |
| `RR_LEAN` / `RR_SIG` / `RR_FULL` | off | `fetch` output: no def line / def line / full JSON |
| `RR_CODE_ONLY` / `RR_RERANK` | off | defaults for `--code-only` / `--rerank` |

When the server sits behind an authenticating reverse proxy, the proxy should
set or overwrite `X-User-Login`. Without a proxy, the server accepts the login
the client sends.

### Optional: Claude Code hooks

`rr hook <event>` implements three [Claude Code hooks](https://docs.claude.com/en/docs/claude-code/hooks).
Each reads the hook event JSON on stdin. None are required.

- `pretooluse`: denies broad, tree-wide `grep`/`rg`/`ag` searches (the `Grep`
  tool, or a Bash command that greps a directory, a glob, or recursively) and
  suggests `rr fetch` / `rr point` / `rr track` instead. A grep of one named
  file, or a grep fed by a pipe, is allowed. A command can opt out with a quoted
  reason, `RR_FALLBACK='why' grep ...`.
- `posttooluse`: logs those `RR_FALLBACK` reasons to stderr, or to the file
  named by `RR_FALLBACK_LOG`.
- `userpromptsubmit`: runs a hybrid query with the prompt text (12 hits,
  code only, no file-dedup) and adds the hits to the model's context before its
  first turn. Hybrid mode needs a server with an embedding model configured.
  It skips short prompts and slash commands, and it does nothing (it never
  blocks the prompt) when the server is unreachable, the query fails, or the
  working directory is not a git checkout. Set `RR_INJECT=0` to turn it off.

Example `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Grep|Bash", "hooks": [{ "type": "command", "command": "rr hook pretooluse" }] }
    ],
    "PostToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "rr hook posttooluse" }] }
    ],
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "rr hook userpromptsubmit" }] }
    ]
  }
}
```

## Results

Measured against srclight's own hybrid search as the baseline, both driven
through the same interface with matched settings (instruction-prefixed query,
file-dedup, hybrid mode, k=10). Deltas are paired; `*` means the 95% paired
bootstrap confidence interval (10k resamples) excludes zero.

**SWE-bench Verified, file localization, n=78 (protocol locked before the run).**
Dense-led configuration (context-headered dense arm, lexical arm off):

| | r@1 | r@10 | MRR | median query latency |
|---|---|---|---|---|
| srclight | 0.181 | 0.690 | 0.381 | ~650 ms |
| rainbow-retriever (dense-led) | 0.215 (+0.034*) | 0.741 (+0.051) | 0.443 (+0.062*) | 42 ms |

About 14x lower median query latency.

**Held-out multilingual set, n=40, 8 repositories (C++, Go, Java, PHP, JS, C,
Ruby, Rust).** Identifier-token lexical arm plus graph arm (`lex_mode=ids`,
`w_graph=1.0`), decision rule committed before any query ran:

| | r@1 | r@10 | MRR |
|---|---|---|---|
| srclight | 0.138 | 0.650 | 0.333 |
| rainbow-retriever (ids + graph) | 0.325 (+0.188*) | 0.725 | 0.476 (+0.143*) |

Details: [docs/benchmark.md](docs/benchmark.md).

## Citation

This retriever was used in:

> The Recall Trap. arXiv:2608.14838.
> Replication package: https://github.com/cannonballgamesstudio/recall-trap-replication

## Development

```bash
pip install -e ".[dev]"
pytest
cargo test --manifest-path cli/Cargo.toml
```

The test suite builds tiny fixture repositories with srclight and uses a fake
embedding provider, so it needs no embedding server and no network.

## License

MIT. See [LICENSE](LICENSE).
