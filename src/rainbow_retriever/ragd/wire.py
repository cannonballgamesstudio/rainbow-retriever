from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repo: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    local_head_sha: str | None = None
    fallback_refs: list[str] = Field(default_factory=list)
    mode: str = "hybrid"
    limit: int = Field(default=20, ge=1, le=200)
    query_dims: int | None = None
    budget: int | None = None

    @field_validator("local_head_sha")
    @classmethod
    def _check_sha(cls, v):
        if v is not None and not _SHA_RE.match(v):
            raise ValueError("local_head_sha must be 40-char lowercase hex")
        return v


class QueryRequest(_Base):
    text: str = Field(min_length=1)
    code_only: bool = False  # opt-in: demote doc/markdown hits below code (default: keep interleaved)
    rerank: bool = False     # opt-in: cross-encoder rerank of the top hits (default off)
    dedup: bool | None = None  # None -> backend default; False -> depth pack (keep multiple chunks
                               # per file).
    max_chunks_per_file: int | None = Field(default=None, ge=1)
    # Integer cap on chunks per file: 1 == binary dedup-ON, higher ==
    # middle arms between breadth (dedup-ON) and depth (dedup-OFF). When set it
    # overrides `dedup`. Unset -> server default (RR_QUERY_MAX_PER_FILE, normally unset).


class CtxRequest(_Base):
    symbol: str = Field(min_length=1)
    file: str | None = None  # disambiguate when the symbol name is defined in multiple files


class XrefRequest(QueryRequest):
    refs: list[str] = Field(min_length=1)
