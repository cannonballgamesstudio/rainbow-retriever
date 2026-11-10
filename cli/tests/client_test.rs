use httpmock::prelude::*;
use rr::client::{ClientError, RagClient};
use rr::model::{QueryRequest, QueryResponse};

fn req() -> QueryRequest {
    QueryRequest { repo: "example/repo".into(), branch: "main".into(), local_head_sha: None,
        fallback_refs: vec![], text: "hi".into(), mode: "keyword".into(),
        limit: 20, query_dims: None, budget: None, code_only: false, rerank: false, dedup: None }
}

#[test]
fn post_sends_headers_and_parses_200() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST).path("/v1/query")
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .header_exists("x-request-id");
        then.status(200).json_body(serde_json::json!({
            "served_ref":"main","hits":[],
            "staleness":{"indexed_sha":"a","your_sha":null,"stale":false,"push_hint":null},
            "capabilities":{"backend_id":"fake","backend_version":"v","modes":["keyword"],
                            "supports_callgraph":true,"supports_dims":[]},
            "truncated":false}));
    });
    let client = RagClient::new(server.base_url());
    let resp: QueryResponse = client.post("/v1/query", &req()).unwrap();
    m.assert();
    assert_eq!(resp.served_ref, "main");
}

#[test]
fn non_2xx_maps_to_api_error_with_code_and_exit_2() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/query");
        then.status(422).json_body(serde_json::json!({
            "error":{"code":"unsupported_mode","message":"no","request_id":"r1",
                     "details":{"supported":["keyword"]}}}));
    });
    let client = RagClient::new(server.base_url());
    let err = client.post::<_, QueryResponse>("/v1/query", &req()).unwrap_err();
    match &err {
        ClientError::Api(a) => {
            assert_eq!(a.status, 422);
            assert_eq!(a.code, "unsupported_mode");
            assert_eq!(err.exit_code(), 2);
        }
        _ => panic!("expected Api error"),
    }
}

#[test]
fn sends_user_login_header_when_user_is_set() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(GET).path("/v1/capabilities").header("x-user-login", "dev");
        then.status(200).json_body(serde_json::json!({"backend_id": "fake"}));
    });
    let client = RagClient::with_user(server.base_url(), Some("dev".into()));
    let v: serde_json::Value = client.get("/v1/capabilities", &[]).unwrap();
    m.assert();
    assert_eq!(v["backend_id"], "fake");
}
