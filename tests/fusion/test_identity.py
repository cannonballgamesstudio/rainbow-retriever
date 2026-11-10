import sys
import types

import srclight

from rainbow_retriever.fusion.identity import FUSION_FORMAT_VERSION, fusion_identity, srclight_format


def _fake_srclight(version, fmt=None):
    m = types.SimpleNamespace(__version__=version)
    if fmt is not None:
        m.INDEX_FORMAT_VERSION = fmt
    return m


def test_identity_shape_without_embeddings():
    ident = fusion_identity(None, None)
    assert ident == f"fusion@{FUSION_FORMAT_VERSION}+{srclight_format()}+none+0"


def test_identity_with_embeddings():
    ident = fusion_identity("qwen3-embedding", 4096)
    assert ident.startswith(f"fusion@{FUSION_FORMAT_VERSION}+")
    assert ident.endswith("+qwen3-embedding+4096")


def test_srclight_fmt_stable_across_patch(monkeypatch):
    monkeypatch.setitem(sys.modules, "srclight", _fake_srclight("0.7.2"))
    a = srclight_format()
    monkeypatch.setitem(sys.modules, "srclight", _fake_srclight("0.7.3"))
    assert a == srclight_format() == "fmt-0.7"


def test_srclight_fmt_prefers_marker(monkeypatch):
    monkeypatch.setitem(sys.modules, "srclight", _fake_srclight("0.7.2", fmt="v3"))
    assert srclight_format() == "v3"
