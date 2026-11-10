from __future__ import annotations


class RagdError(Exception):
    code: str = "internal"
    http_status: int = 500

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidRequest(RagdError):       code = "invalid_request";       http_status = 400
class IdentityRequired(RagdError):     code = "identity_required";     http_status = 401
class RefNotFound(RagdError):          code = "ref_not_found";         http_status = 404
class CtxUnsupported(RagdError):       code = "ctx_unsupported";       http_status = 404
class NotAcceptable(RagdError):        code = "not_acceptable";        http_status = 406
class UnsupportedMediaType(RagdError): code = "unsupported_media_type"; http_status = 415
class UnsupportedMode(RagdError):      code = "unsupported_mode";      http_status = 422
class UnsupportedDims(RagdError):      code = "unsupported_dims";      http_status = 422
class BackendDegraded(RagdError):      code = "backend_degraded";      http_status = 503
class BackendBusy(RagdError):          code = "backend_busy";          http_status = 503
class Internal(RagdError):             code = "internal";              http_status = 500
class AmbiguousSymbol(RagdError):      code = "ambiguous_symbol";      http_status = 409
class UnsupportedOp(RagdError):        code = "unsupported_op";        http_status = 400


def make_error_response(err: RagdError, request_id: str) -> dict:
    body = {"code": err.code, "message": err.message, "request_id": request_id}
    if err.details is not None:
        body["details"] = err.details
    return {"error": body}
