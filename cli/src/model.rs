use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct QueryRequest {
    pub repo: String,
    pub branch: String,
    pub local_head_sha: Option<String>,      // always serialized (incl. null)
    pub fallback_refs: Vec<String>,
    pub text: String,
    pub mode: String,
    pub limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_dims: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<i64>,
    pub code_only: bool,   // demote doc/markdown hits below code (default: keep interleaved)
    pub rerank: bool,      // cross-encoder rerank of the top hits (default: off)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dedup: Option<bool>, // None -> backend default; Some(false) -> depth pack (prompt hook)
}

#[derive(Serialize)]
pub struct CtxRequest {
    pub repo: String,
    pub branch: String,
    pub local_head_sha: Option<String>,
    pub fallback_refs: Vec<String>,
    pub symbol: String,
    pub file: Option<String>,
}

#[derive(Serialize)]
pub struct XrefRequest {
    pub repo: String,
    pub branch: String,
    pub local_head_sha: Option<String>,
    pub fallback_refs: Vec<String>,
    pub text: String,
    pub refs: Vec<String>,
    pub mode: String,
    pub limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_dims: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Hit {
    pub repo: String,
    #[serde(rename = "ref")]
    pub ref_: String,
    pub file: String,
    pub line: i64,
    pub symbol: String,
    pub score: f64,
    pub snippet: String,
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Staleness {
    pub indexed_sha: Option<String>,
    pub your_sha: Option<String>,
    pub stale: bool,
    pub push_hint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub backend_id: String,
    pub backend_version: String,
    pub modes: Vec<String>,
    pub supports_callgraph: bool,
    pub supports_dims: Vec<i64>,
}

#[derive(Serialize, Deserialize)]
pub struct GraphEdge {
    pub symbol: String,
    pub file: String,
    pub line: i64,
    pub kind: String,
    pub edge_type: String,
    pub confidence: f64,
}

#[derive(Serialize, Deserialize)]
pub struct SymbolGraph {
    pub symbol: String,
    pub defined_file: String,
    pub defined_line: i64,
    pub signature: Option<String>,
    pub callers: Vec<GraphEdge>,
    pub callees: Vec<GraphEdge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub served_ref: String,
    pub hits: Vec<Hit>,
    pub staleness: Staleness,
    pub capabilities: Capabilities,
    pub truncated: bool,
}

#[derive(Serialize, Deserialize)]
pub struct XrefResponse {
    pub served_refs: Vec<String>,
    pub unresolved_refs: Vec<String>,
    pub hits: Vec<Hit>,
    pub staleness: Staleness,
    pub capabilities: Capabilities,
    pub truncated: bool,
}

#[derive(Serialize, Deserialize)]
pub struct CtxResponse {
    pub served_ref: String,
    pub found: bool,
    pub graph: Option<SymbolGraph>,
    pub staleness: Staleness,
    pub capabilities: Capabilities,
}

#[derive(Serialize, Deserialize)]
pub struct ResolveResponse {
    pub repo: String,
    #[serde(rename = "ref")]
    pub ref_: String,
    pub head_sha: String,
    pub worktree: String,
    pub status: String,
    pub indexed_at: i64,
    pub backend_id: String,
    pub backend_version: String,
    pub config_hash: String,
}

#[derive(Serialize, Deserialize)]
pub struct StatusResponse {
    pub refs: Vec<ResolveResponse>,
}

// ── Comprehension response types (deps / file / tests / map / hierarchy) ──

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Edge {
    pub symbol: String,
    pub file: String,
    pub line: i64,
    pub kind: String,
    pub edge_type: String,
    pub confidence: f64,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct FileSym {
    pub symbol: String,
    pub kind: String,
    pub line: i64,
    pub signature: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct DirEntry {
    pub path: String,
    pub files: i64,
    pub symbols: i64,
    pub languages: Vec<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct HotspotFile {
    pub path: String,
    pub language: Option<String>,
    pub lines: i64,
    pub symbols: i64,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct EdgesResp {
    pub served_ref: String,
    pub edges: Vec<Edge>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct FileResp {
    pub served_ref: String,
    pub symbols: Vec<FileSym>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct MapResp {
    pub served_ref: String,
    pub dirs: Vec<DirEntry>,
    pub hotspots: Vec<HotspotFile>,
}

// ── Reachability (`reach`/`trace`) response types ──

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct ReachNode {
    pub symbol: String,
    pub file: String,
    pub line: u32,
    pub depth: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct ReachResponse {
    pub served_ref: String,
    pub entry: Vec<String>,
    pub reachable: Vec<ReachNode>,
    pub truncated: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct HierarchyResp {
    pub served_ref: String,
    pub base: Vec<Edge>,
    pub subclasses: Vec<Edge>,
    pub implementors: Vec<Edge>,
}

#[derive(Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub request_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_response_round_trips_with_ref_rename_and_null_indexed_sha() {
        let json = r#"{"served_ref":"main",
          "hits":[{"repo":"example/repo","ref":"main","file":"a.py","line":12,
                   "symbol":"f","score":-1.5,"snippet":">>>f<<<","kind":"function"}],
          "staleness":{"indexed_sha":null,"your_sha":null,"stale":true,"push_hint":"x"},
          "capabilities":{"backend_id":"srclight","backend_version":"v1","modes":["keyword"],
                          "supports_callgraph":true,"supports_dims":[4096]},
          "truncated":false,"future_field":"ignored"}"#;
        let r: QueryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.hits[0].ref_, "main");           // serde rename "ref" -> ref_
        assert_eq!(r.hits[0].file, "a.py");
        assert!(r.staleness.indexed_sha.is_none());   // null indexed_sha
        assert!(r.staleness.stale);
    }

    #[test]
    fn request_serializes_local_head_sha_as_null_and_omits_query_dims() {
        let req = QueryRequest {
            repo: "r".into(), branch: "main".into(), local_head_sha: None,
            fallback_refs: vec![], text: "hi".into(), mode: "hybrid".into(),
            limit: 20, query_dims: None, budget: None, code_only: false, rerank: false,
            dedup: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("local_head_sha").unwrap().is_null());   // present, null
        assert!(v.get("query_dims").is_none());                // omitted when None
        assert!(v.get("budget").is_none());
        assert!(v.get("dedup").is_none());                     // omitted when None
    }

    #[test]
    fn error_envelope_parses() {
        let json = r#"{"error":{"code":"unsupported_mode","message":"nope",
          "details":{"supported":["keyword"]},"request_id":"r1"}}"#;
        let e: ErrorEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(e.error.code, "unsupported_mode");
        assert_eq!(e.error.request_id.as_deref(), Some("r1"));
    }
}
