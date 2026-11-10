use clap::Parser;
use httpmock::prelude::*;
use rr::cli::Cli;

#[test]
fn query_e2e_renders_json_and_warns_staleness() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/query");
        then.status(200).json_body(serde_json::json!({
            "served_ref":"main",
            "hits":[{"repo":"example/repo","ref":"main","file":"a.py","line":7,
                     "symbol":"f","score":-1.0,"snippet":"x","kind":"function"}],
            "staleness":{"indexed_sha":"a","your_sha":"b","stale":true,"push_hint":"push it"},
            "capabilities":{"backend_id":"fake","backend_version":"v","modes":["keyword"],
                            "supports_callgraph":true,"supports_dims":[]},
            "truncated":false}));
    });
    // explicit --repo/--ref so dispatch does not need a git repo; --json forces JSON
    let cli = Cli::try_parse_from([
        "rr", "--endpoint", &server.base_url(), "--repo", "example/repo", "--ref", "main",
        "--json", "query", "find",
    ]).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = rr::dispatch(cli, &mut out, &mut err);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();   // stdout is clean JSON
    assert_eq!(v["served_ref"], "main");
    assert!(String::from_utf8(err).unwrap().contains("push it"));        // staleness -> stderr
}

// Fix 2: ctx end-to-end httpmock test
#[test]
fn ctx_e2e_renders_json_for_found_symbol() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/ctx");
        then.status(200).json_body(serde_json::json!({
            "served_ref": "main",
            "found": true,
            "graph": {
                "symbol": "greet",
                "defined_file": "src/hello.rs",
                "defined_line": 42,
                "signature": "fn greet(name: &str)",
                "callers": [{"symbol":"main","file":"src/main.rs","line":10,
                              "kind":"call","edge_type":"direct","confidence":1.0}],
                "callees": [{"symbol":"format","file":"src/hello.rs","line":43,
                              "kind":"call","edge_type":"direct","confidence":0.9}]
            },
            "staleness": {"indexed_sha":"a","your_sha":"b","stale":false,"push_hint":null},
            "capabilities": {"backend_id":"fake","backend_version":"v","modes":["keyword"],
                             "supports_callgraph":true,"supports_dims":[]}
        }));
    });
    // explicit --repo/--ref so dispatch does not need a git repo; --json forces JSON
    let cli = Cli::try_parse_from([
        "rr", "--endpoint", &server.base_url(), "--repo", "example/repo", "--ref", "main",
        "--json", "ctx", "greet",
    ]).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = rr::dispatch(cli, &mut out, &mut err);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap(); // stdout is clean JSON
    assert_eq!(v["served_ref"], "main");
    assert_eq!(v["found"], true);
}

// `speak` (alias `capabilities`) GETs the server's capabilities and pretty-prints the
// JSON -- same in-process httpmock pattern as query/ctx above.
#[test]
fn speak_e2e_pretty_prints_capabilities_json() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/v1/capabilities");
        then.status(200).json_body(serde_json::json!({
            "backend_id": "fusion", "backend_version": "v1",
            "modes": ["hybrid", "keyword", "semantic"],
            "supports_callgraph": true, "supports_dims": [4096],
            "supports_ops": ["deps", "file", "tests", "map", "hierarchy"]
        }));
    });
    let cli = Cli::try_parse_from([
        "rr", "--endpoint", &server.base_url(), "--repo", "example/repo", "--ref", "main",
        "speak",
    ]).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = rr::dispatch(cli, &mut out, &mut err);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap(); // pretty JSON still parses
    assert_eq!(v["backend_id"], "fusion");
    assert_eq!(v["supports_ops"][0], "deps");
}

// goodboy needs no endpoint/mock at all: it's a pure client-side easter egg.
#[test]
fn goodboy_prints_and_exits_0() {
    let cli = Cli::try_parse_from(["rr", "goodboy"]).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = rr::dispatch(cli, &mut out, &mut err);
    assert_eq!(code, 0);
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("Red") && s.contains("Violet"));
    assert!(err.is_empty());
}

#[test]
fn api_error_returns_exit_2() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/query");
        then.status(404).json_body(serde_json::json!({
            "error":{"code":"ref_not_found","message":"no","request_id":"r1"}}));
    });
    let cli = Cli::try_parse_from([
        "rr", "--endpoint", &server.base_url(), "--repo", "r", "--ref", "ghost", "query", "x",
    ]).unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = rr::dispatch(cli, &mut out, &mut err);
    assert_eq!(code, 2);
    assert!(String::from_utf8(err).unwrap().contains("ref_not_found"));
}
