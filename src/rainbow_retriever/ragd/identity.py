from __future__ import annotations

import hashlib
import json

from rainbow_retriever.ragd.errors import IdentityRequired


class CallerIdentity:
    def __init__(self, login: str, name: str | None):
        self.login = login
        self.name = name


def extract_identity(headers, default_login=None) -> CallerIdentity:
    """Identify the caller from the `X-User-Login` header, which an
    authenticating reverse proxy in front of the server is expected to set.

    `default_login` is an opt-in fallback (RR_DEFAULT_IDENTITY) for running
    the server without such a proxy (local use). When it is None, a missing
    header returns 401.
    """
    login = headers.get("X-User-Login") or default_login
    if not login:
        raise IdentityRequired("missing X-User-Login header")
    return CallerIdentity(login, headers.get("X-User-Name"))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def audit_log(logger, identity, request_id, repo, ref, op, stale,
              text=None, symbol=None, log_raw=False) -> None:
    rec = {"caller": identity.login, "request_id": request_id,
           "repo": repo, "ref": ref, "op": op, "stale": stale}
    if text is not None:
        rec.update({"text": text} if log_raw else {"text_sha256": _h(text), "text_len": len(text)})
    if symbol is not None:
        rec.update({"symbol": symbol} if log_raw
                   else {"symbol_sha256": _h(symbol), "symbol_len": len(symbol)})
    logger.info(json.dumps(rec))
