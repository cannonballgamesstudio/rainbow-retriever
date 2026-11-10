use httpmock::prelude::*;
use std::process::Command;

fn rr(args: &[&str], endpoint: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rr"))
        .args(["--endpoint", endpoint, "--repo", "example/repo", "--ref", "main", "--json"])
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn deps_renders_edges() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/v1/deps");
        then.status(200).header("content-type", "application/json")
            .json_body(serde_json::json!({"served_ref": "main",
                "edges": [{"symbol":"greet","file":"m.py","line":1,"kind":"function","edge_type":"calls","confidence":1.0}]}));
    });
    let out = rr(&["deps", "hello"], &server.base_url());
    assert!(out.status.success());
    let body: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(body["edges"][0]["symbol"], "greet");
}

#[test]
fn file_renders_symbols() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/v1/file");
        then.status(200).json_body(serde_json::json!({"served_ref":"main",
            "symbols":[{"symbol":"foo","kind":"function","line":2,"signature":"def foo()"}]}));
    });
    let out = rr(&["file", "m.py"], &server.base_url());
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("foo"));
}

#[test]
fn reach_renders_reachable_nodes_json() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/v1/reachable")
            .query_param("symbol", "alpha,delta")
            .query_param("direction", "callees")
            .query_param("depth", "2");
        then.status(200).header("content-type", "application/json")
            .json_body(serde_json::json!({
                "served_ref": "main",
                "entry": ["alpha", "delta"],
                "reachable": [
                    {"symbol":"beta","file":"m.py","line":3,"kind":"function","depth":1},
                    {"symbol":"gamma","file":"m.py","line":9,"kind":"function","depth":2}
                ],
                "truncated": false
            }));
    });
    let out = rr(&["reach", "alpha,delta", "--depth", "2"], &server.base_url());
    assert!(out.status.success());
    let body: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(body["reachable"][0]["symbol"], "beta");
    assert_eq!(body["reachable"][1]["depth"], 2);
    assert_eq!(body["truncated"], false);
}

#[test]
fn ambiguous_symbol_exits_2() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/v1/hierarchy");
        then.status(409).json_body(serde_json::json!({"error":{"code":"ambiguous_symbol",
            "message":"ambiguous symbol","request_id":"x",
            "details":{"candidates":[{"symbol":"C","file":"a.py","line":1},{"symbol":"C","file":"b.py","line":2}]}}}));
    });
    let out = rr(&["hierarchy", "C"], &server.base_url());
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("ambiguous"));
}
