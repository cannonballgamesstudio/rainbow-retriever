import json, logging
import pytest
from rainbow_retriever.ragd.identity import CallerIdentity, extract_identity, audit_log
from rainbow_retriever.ragd.errors import IdentityRequired


def test_extract_requires_login():
    with pytest.raises(IdentityRequired):
        extract_identity({})
    ident = extract_identity({"X-User-Login": "dev@example.com", "X-User-Name": "Dev"})
    assert (ident.login, ident.name) == ("dev@example.com", "Dev")


def test_audit_hashes_text_by_default(caplog):
    ident = CallerIdentity("dev@example.com", "Dev")
    with caplog.at_level(logging.INFO, logger="ragd.audit"):
        audit_log(logging.getLogger("ragd.audit"), ident, "req-1", "r", "main", "query",
                  stale=False, text="secret query")
    rec = json.loads(caplog.records[-1].getMessage())
    assert "text" not in rec and rec["text_len"] == len("secret query")
    assert rec["caller"] == "dev@example.com" and rec["request_id"] == "req-1" and rec["op"] == "query"


def test_audit_raw_opt_in(caplog):
    ident = CallerIdentity("a", None)
    with caplog.at_level(logging.INFO, logger="ragd.audit"):
        audit_log(logging.getLogger("ragd.audit"), ident, "r2", "r", "main", "query",
                  stale=True, text="hi", log_raw=True)
    rec = json.loads(caplog.records[-1].getMessage())
    assert rec["text"] == "hi"
