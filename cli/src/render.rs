use crate::model::{CtxResponse, QueryResponse, Staleness};
use std::io::{self, Write};

/// The seven retrieval signals mapped to ROYGBIV bands.
pub const BANDS: [(&str, &str, &str); 7] = [
    ("Red", "lexical", "identifier-token BM25"),
    ("Orange", "trigram", "identifier-trigram anchor"),
    ("Yellow", "dense", "instructed dense embeddings"),
    ("Green", "graph", "deterministic call-graph expansion"),
    ("Blue", "priors", "git recency/frequency boost"),
    ("Indigo", "reranker", "cross-encoder rerank (opt-in)"),
    ("Violet", "enrichment", "index-time context headers"),
];

/// `goodboy`: client-side easter egg. Prints a small ASCII rainbow, the 7-band
/// arm table, and the binary version. No network, no git context -- always exits 0.
pub fn goodboy(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "        .--\"\"\"--.")?;
    writeln!(out, "     .-'  R O Y   '-.")?;
    writeln!(out, "    (   G B I V      )    /^ ^\\")?;
    writeln!(out, "     '-.          .-'    ( o.o )  woof!")?;
    writeln!(out, "        '--...--'         > ^ <")?;
    writeln!(out)?;
    writeln!(out, "  band     arm         what it does")?;
    for (band, arm, desc) in BANDS {
        writeln!(out, "  {band:<8} {arm:<11} {desc}")?;
    }
    writeln!(out)?;
    writeln!(out, "rainbow-retriever rr {} (good dog, good retrieval)", env!("CARGO_PKG_VERSION"))
}

pub fn use_json(force_json: bool, is_tty: bool) -> bool {
    force_json || !is_tty
}

pub fn warn_staleness(err: &mut impl Write, st: &Staleness) {
    if st.stale {
        if let Some(hint) = &st.push_hint {
            let _ = writeln!(err, "warning: {hint}");
        }
    }
}

pub fn render_query(out: &mut impl Write, resp: &QueryResponse, json: bool, sig: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(resp).unwrap());
    }
    for h in &resp.hits {
        if sig {
            // Self-contained, grep-equivalent: append the symbol's def/first line so the
            // code is visible in place.
            let def: String = h.snippet.lines().next().unwrap_or("").trim().chars().take(120).collect();
            writeln!(out, "{}:{} {} [{}] {:.4}  {}", h.file, h.line, h.symbol, h.kind, h.score, def)?;
        } else {
            writeln!(out, "{}:{} {} [{}] {:.4}", h.file, h.line, h.symbol, h.kind, h.score)?;
        }
    }
    writeln!(out, "[{} hits{} | {}]",
             resp.hits.len(),
             if resp.truncated { " | truncated" } else { "" },
             resp.capabilities.backend_version)
}

pub fn render_ctx(out: &mut impl Write, resp: &CtxResponse, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(resp).unwrap());
    }
    match &resp.graph {
        None => writeln!(out, "symbol not found")?,
        Some(g) => {
            writeln!(out, "{} ({}:{})", g.symbol, g.defined_file, g.defined_line)?;
            if let Some(sig) = &g.signature {
                writeln!(out, "  {sig}")?;
            }
            writeln!(out, "  callers:")?;
            for e in &g.callers {
                writeln!(out, "    {} ({}:{})", e.symbol, e.file, e.line)?;
            }
            writeln!(out, "  callees:")?;
            for e in &g.callees {
                writeln!(out, "    {} ({}:{})", e.symbol, e.file, e.line)?;
            }
        }
    }
    Ok(())
}

pub fn render_xref(out: &mut impl Write, resp: &crate::model::XrefResponse, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(resp).unwrap());
    }
    for h in &resp.hits {
        let def: String = h.snippet.lines().next().unwrap_or("").trim().chars().take(120).collect();
        writeln!(out, "{}@{}:{} {} [{}] {:.4}  {}", h.ref_, h.file, h.line, h.symbol, h.kind, h.score, def)?;
    }
    if !resp.unresolved_refs.is_empty() {
        writeln!(out, "[unresolved refs: {}]", resp.unresolved_refs.join(", "))?;
    }
    writeln!(out, "[{} hits{} | {}]",
             resp.hits.len(),
             if resp.truncated { " | truncated" } else { "" },
             resp.capabilities.backend_version)
}

pub fn status(out: &mut impl Write, resp: &crate::model::StatusResponse, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(resp).unwrap());
    }
    for r in &resp.refs {
        let sha: String = r.head_sha.chars().take(8).collect();
        writeln!(out, "{:<30} {:<20} {:<9} {}  {}", r.repo, r.ref_, r.status, sha, r.backend_version)?;
    }
    Ok(())
}

// ── Comprehension render functions (deps / file / tests / map / hierarchy) ──

pub fn edges(out: &mut impl Write, edges: &[crate::model::Edge], json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(&serde_json::json!({"edges": edges})).unwrap());
    }
    for e in edges {
        writeln!(out, "{}  {}:{}  [{}]", e.symbol, e.file, e.line, e.edge_type)?;
    }
    Ok(())
}

pub fn file_symbols(out: &mut impl Write, syms: &[crate::model::FileSym], json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(&serde_json::json!({"symbols": syms})).unwrap());
    }
    for s in syms {
        writeln!(out, "{}:{}  {}  {}", s.kind, s.line, s.symbol, s.signature.as_deref().unwrap_or_default())?;
    }
    Ok(())
}

pub fn structure(out: &mut impl Write, m: &crate::model::MapResp, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(m).unwrap());
    }
    for d in &m.dirs {
        writeln!(out, "{:<40} {} files {} symbols", d.path, d.files, d.symbols)?;
    }
    writeln!(out, "-- hotspots --")?;
    for h in &m.hotspots {
        writeln!(out, "{:<40} {} symbols", h.path, h.symbols)?;
    }
    Ok(())
}

pub fn hierarchy(out: &mut impl Write, h: &crate::model::HierarchyResp, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(h).unwrap());
    }
    for (label, v) in [("base", &h.base), ("subclasses", &h.subclasses), ("implementors", &h.implementors)] {
        for e in v {
            writeln!(out, "{:<12} {}  {}:{}", label, e.symbol, e.file, e.line)?;
        }
    }
    Ok(())
}

// Reachability render: depth-indented (two spaces per depth level) -- mirrors
// `edges` above, but the shape carries a depth per node (BFS from N entry symbols)
// instead of a flat edge_type list.
pub fn reach(out: &mut impl Write, resp: &crate::model::ReachResponse, json: bool) -> io::Result<()> {
    if json {
        return writeln!(out, "{}", serde_json::to_string_pretty(resp).unwrap());
    }
    for n in &resp.reachable {
        // depth is 1-based (immediate callees/callers of an entry symbol); the first
        // BFS level renders flush-left, each level deeper adds two more spaces.
        let indent = "  ".repeat(n.depth.saturating_sub(1) as usize);
        writeln!(out, "{indent}{}  {}:{}", n.symbol, n.file, n.line)?;
    }
    if resp.truncated {
        writeln!(out, "[truncated]")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    fn reach_resp() -> ReachResponse {
        ReachResponse {
            served_ref: "main".into(),
            entry: vec!["alpha".into()],
            reachable: vec![
                ReachNode { symbol: "beta".into(), file: "m.py".into(), line: 3, depth: 1 },
                ReachNode { symbol: "gamma".into(), file: "m.py".into(), line: 9, depth: 2 },
            ],
            truncated: false,
        }
    }

    #[test]
    fn reach_terse_render_indents_by_depth() {
        let mut buf = Vec::new();
        reach(&mut buf, &reach_resp(), false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], "beta  m.py:3");
        assert_eq!(lines[1], "  gamma  m.py:9");
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn reach_terse_render_notes_truncation() {
        let mut r = reach_resp();
        r.truncated = true;
        let mut buf = Vec::new();
        reach(&mut buf, &r, false).unwrap();
        assert!(String::from_utf8(buf).unwrap().contains("truncated"));
    }

    #[test]
    fn reach_json_render_emits_full_response() {
        let mut buf = Vec::new();
        reach(&mut buf, &reach_resp(), true).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v["served_ref"], "main");
        assert_eq!(v["entry"][0], "alpha");
        assert_eq!(v["reachable"][0]["symbol"], "beta");
        assert_eq!(v["reachable"][1]["depth"], 2);
        assert_eq!(v["truncated"], false);
    }

    #[test]
    fn goodboy_prints_all_seven_bands_and_the_version() {
        let mut buf = Vec::new();
        goodboy(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        for (band, arm, _) in BANDS {
            assert!(s.contains(band), "missing band {band}");
            assert!(s.contains(arm), "missing arm {arm}");
        }
        assert!(s.contains(env!("CARGO_PKG_VERSION")), "missing binary version");
    }

    fn resp() -> QueryResponse {
        QueryResponse {
            served_ref: "main".into(),
            hits: vec![Hit { repo: "r".into(), ref_: "main".into(), file: "a.py".into(),
                             line: 12, symbol: "greet".into(), score: -1.0,
                             snippet: "x".into(), kind: "function".into() }],
            staleness: Staleness { indexed_sha: Some("a".into()), your_sha: None,
                                   stale: true, push_hint: Some("push it".into()) },
            capabilities: Capabilities { backend_id: "fake".into(), backend_version: "v".into(),
                modes: vec!["keyword".into()], supports_callgraph: true, supports_dims: vec![] },
            truncated: false,
        }
    }

    #[test]
    fn use_json_is_true_when_forced_or_not_tty() {
        assert!(use_json(true, true));
        assert!(use_json(false, false));
        assert!(!use_json(false, true));
    }

    #[test]
    fn warn_staleness_writes_hint_only_when_stale() {
        let mut buf = Vec::new();
        warn_staleness(&mut buf, &resp().staleness);
        assert!(String::from_utf8(buf).unwrap().contains("push it"));

        let mut buf2 = Vec::new();
        let fresh = Staleness { indexed_sha: Some("a".into()), your_sha: None,
                                stale: false, push_hint: None };
        warn_staleness(&mut buf2, &fresh);
        assert!(buf2.is_empty());
    }

    #[test]
    fn json_mode_emits_parseable_json_tty_mode_has_file_line() {
        let mut j = Vec::new();
        render_query(&mut j, &resp(), true, false).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&j).unwrap();
        assert_eq!(parsed["served_ref"], "main");

        let mut t = Vec::new();
        render_query(&mut t, &resp(), false, false).unwrap();
        let s = String::from_utf8(t).unwrap();
        assert!(s.contains("a.py:12") && s.contains("greet"));

        // --sig appends the snippet's def/first line inline (self-contained like grep)
        let mut g = Vec::new();
        render_query(&mut g, &resp(), false, true).unwrap();
        let gs = String::from_utf8(g).unwrap();
        assert!(gs.contains("a.py:12") && gs.contains("greet") && gs.contains("x"));
    }

    // Fix 3 (Minor): guards stale=true, push_hint=None → no output regression
    #[test]
    fn warn_staleness_stale_without_hint_emits_nothing() {
        let mut buf = Vec::new();
        let stale_no_hint = Staleness {
            indexed_sha: None, your_sha: None,
            stale: true, push_hint: None,
        };
        warn_staleness(&mut buf, &stale_no_hint);
        assert!(buf.is_empty());
    }

    fn ctx_resp_found() -> CtxResponse {
        CtxResponse {
            served_ref: "main".into(),
            found: true,
            graph: Some(SymbolGraph {
                symbol: "greet".into(),
                defined_file: "src/hello.rs".into(),
                defined_line: 42,
                signature: Some("fn greet(name: &str)".into()),
                callers: vec![GraphEdge {
                    symbol: "main".into(),
                    file: "src/main.rs".into(),
                    line: 10,
                    kind: "call".into(),
                    edge_type: "direct".into(),
                    confidence: 1.0,
                }],
                callees: vec![GraphEdge {
                    symbol: "format".into(),
                    file: "src/hello.rs".into(),
                    line: 43,
                    kind: "call".into(),
                    edge_type: "direct".into(),
                    confidence: 0.9,
                }],
            }),
            staleness: Staleness { indexed_sha: None, your_sha: None, stale: false, push_hint: None },
            capabilities: Capabilities {
                backend_id: "fake".into(), backend_version: "v".into(),
                modes: vec![], supports_callgraph: true, supports_dims: vec![],
            },
        }
    }

    // Fix 1: render_ctx found, TTY mode
    #[test]
    fn render_ctx_found_tty_mode_shows_symbol_file_and_caller_callee() {
        let r = ctx_resp_found();
        let mut buf = Vec::new();
        render_ctx(&mut buf, &r, false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        // symbol + defined location on first line
        assert!(s.contains("greet"), "missing symbol name");
        assert!(s.contains("src/hello.rs"), "missing defined_file");
        assert!(s.contains("42"), "missing defined_line");
        // signature
        assert!(s.contains("fn greet(name: &str)"), "missing signature");
        // caller and callee symbol names
        assert!(s.contains("main"), "missing caller symbol");
        assert!(s.contains("format"), "missing callee symbol");
    }

    // Fix 1: render_ctx not-found, TTY mode
    #[test]
    fn render_ctx_not_found_tty_mode_prints_not_found() {
        let r = CtxResponse {
            served_ref: "main".into(),
            found: false,
            graph: None,
            staleness: Staleness { indexed_sha: None, your_sha: None, stale: false, push_hint: None },
            capabilities: Capabilities {
                backend_id: "fake".into(), backend_version: "v".into(),
                modes: vec![], supports_callgraph: true, supports_dims: vec![],
            },
        };
        let mut buf = Vec::new();
        render_ctx(&mut buf, &r, false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("symbol not found"));
    }

    // Fix 1 (optional): render_ctx JSON mode emits parseable JSON with served_ref
    #[test]
    fn render_ctx_json_mode_emits_parseable_json() {
        let r = ctx_resp_found();
        let mut buf = Vec::new();
        render_ctx(&mut buf, &r, true).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(parsed["served_ref"], "main");
        assert_eq!(parsed["found"], true);
    }
}
