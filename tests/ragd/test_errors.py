import pytest
from rainbow_retriever.ragd.errors import (
    RagdError, InvalidRequest, IdentityRequired, RefNotFound, UnsupportedMode,
    BackendBusy, make_error_response,
)


def test_subclasses_carry_code_and_status():
    assert (InvalidRequest("x").code, InvalidRequest("x").http_status) == ("invalid_request", 400)
    assert (IdentityRequired("x").http_status, RefNotFound("x").http_status) == (401, 404)
    assert UnsupportedMode("x").http_status == 422 and BackendBusy("x").http_status == 503


def test_make_error_response_shape():
    err = UnsupportedMode("nope", details={"requested": "semantic", "supported": ["keyword"]})
    body = make_error_response(err, "req-123")
    assert body == {"error": {"code": "unsupported_mode", "message": "nope",
                              "details": {"requested": "semantic", "supported": ["keyword"]},
                              "request_id": "req-123"}}


def test_make_error_response_omits_details_when_none():
    body = make_error_response(InvalidRequest("bad"), "r1")
    assert "details" not in body["error"] and body["error"]["request_id"] == "r1"
    assert isinstance(InvalidRequest("bad"), RagdError)
