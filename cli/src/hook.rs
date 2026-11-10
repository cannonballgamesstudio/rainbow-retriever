use serde::Deserialize;
use std::io::Write;

#[derive(Debug, PartialEq)]
pub enum Decision {
    Allow,
    Deny(String),
}

#[derive(Deserialize)]
pub struct HookEvent {
    pub tool_name: String,
    #[serde(default)]
    pub tool_input: serde_json::Value,
}

const GREP_FAMILY: &[&str] = &["grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack", "ack-grep"];

pub const DENY_REASON: &str = "rainbow-retriever: use `rr fetch`/`rr point`/`rr track` for broad or \
repo-wide code search instead of grepping the tree (a targeted grep of a specific named file \
is allowed). If rr's results are genuinely too poor (or the index is stale / the server is \
down), re-run the grep with a quoted reason: RR_FALLBACK='rr fetch returned 0 hits for X' <your grep>.";

pub fn decide(tool_name: &str, tool_input: &serde_json::Value) -> Decision {
    match tool_name {
        "Grep" => Decision::Deny(DENY_REASON.to_string()),
        "Bash" => {
            let command = tool_input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            analyze_bash(command)
        }
        _ => Decision::Allow,
    }
}

pub fn analyze_bash(command: &str) -> Decision {
    if extract_fallback_reason(command).is_some() {
        return Decision::Allow; // explicit override: a quoted RR_FALLBACK='<reason>'
    }
    for (preceded_by_pipe, segment) in command_starts(command) {
        if preceded_by_pipe {
            continue; // grep fed by a pipe is a pipe-filter, not a content search
        }
        if let Some(word) = leading_command_word(segment) {
            let base = basename(word);
            if GREP_FAMILY.contains(&base) && grep_is_broad(segment) {
                return Decision::Deny(DENY_REASON.to_string());
            }
        }
    }
    Decision::Allow
}

/// Split into (preceded_by_single_pipe, segment), breaking on top-level `|`, `||`, `&&`,
/// `;`, and newlines. Quote-naive -- an accepted heuristic limit.
fn command_starts(command: &str) -> Vec<(bool, &str)> {
    let bytes = command.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut preceded_by_pipe = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let (is_sep, len, is_pipe) = match bytes[i] {
            b'|' if bytes.get(i + 1) == Some(&b'|') => (true, 2, false),
            b'|' => (true, 1, true),
            b'&' if bytes.get(i + 1) == Some(&b'&') => (true, 2, false),
            b';' | b'\n' => (true, 1, false),
            _ => (false, 0, false),
        };
        if is_sep {
            out.push((preceded_by_pipe, &command[start..i]));
            preceded_by_pipe = is_pipe;
            i += len;
            start = i;
        } else {
            i += 1;
        }
    }
    out.push((preceded_by_pipe, &command[start..]));
    out
}

/// First command word of a segment, skipping leading `VAR=value` env assignments.
fn leading_command_word(segment: &str) -> Option<&str> {
    segment.split_whitespace().find(|t| !is_env_assignment(t))
}

fn is_env_assignment(token: &str) -> bool {
    match token.find('=') {
        Some(eq) if eq > 0 => token[..eq].chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        _ => false,
    }
}

fn basename(word: &str) -> &str {
    word.rsplit(['/', '\\']).next().unwrap_or(word)
}

/// A grep-family command-start is "broad" (→ redirect to rr) when it scans the tree
/// rather than one specific named file: a recursive flag, a directory/glob/`.` target,
/// or no file target at all. A targeted grep of a concrete file is allowed -- blocking it
/// only forces a read of the whole file instead, which costs more context, not less.
fn grep_is_broad(segment: &str) -> bool {
    let mut toks = segment.split_whitespace().skip_while(|t| is_env_assignment(t));
    let _cmd = toks.next(); // the grep-family word itself
    let mut recursive = false;
    let mut seen_pattern = false;
    let mut targets: Vec<&str> = Vec::new();
    for t in toks {
        if t.len() > 1 && t.starts_with('-') {
            if t == "--recursive"
                || (!t.starts_with("--") && t[1..].chars().any(|c| c == 'r' || c == 'R'))
            {
                recursive = true;
            }
            continue;
        }
        if !seen_pattern {
            seen_pattern = true; // first non-flag token is the search pattern
            continue;
        }
        targets.push(t);
    }
    recursive || targets.is_empty() || targets.iter().any(|t| target_is_broad(t))
}

fn target_is_broad(t: &str) -> bool {
    t == "." || t == ".." || t.ends_with('/') || t.ends_with('\\')
        || t.contains('*') || t.contains('?')
        || !basename(t).contains('.') // no extension → treat as a directory
}

pub fn extract_fallback_reason(command: &str) -> Option<String> {
    let idx = command.find("RR_FALLBACK=")?;
    let rest = &command[idx + "RR_FALLBACK=".len()..];
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None; // require a quoted reason
    }
    let tail = &rest[quote.len_utf8()..];
    tail.find(quote).map(|end| tail[..end].to_string())
}

fn log_fallback(reason: &str) {
    // Deployment sink: a file at $RR_FALLBACK_LOG, else stderr. Not unit-tested.
    let line = format!("rr-fallback: {reason}\n");
    match std::env::var("RR_FALLBACK_LOG") {
        Ok(path) if !path.is_empty() => {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = f.write_all(line.as_bytes());
            }
        }
        _ => eprint!("{line}"),
    }
}

// ---------------------------------------------------------------- UserPromptSubmit
//
// Retrieves BEFORE the model's first turn and adds the top hits to its context as
// `additionalContext`, instead of waiting for the model to call a search tool. The
// model's own search tools stay available to verify and expand on the pack.
//
// FAILS SILENT, BY DESIGN: a degraded retriever must never break a prompt. No output
// => Claude Code proceeds with the prompt unmodified, exactly as if the hook were not
// installed.

#[derive(Deserialize)]
pub struct PromptEvent {
    #[serde(default)]
    pub prompt: String,
}

/// Injecting on "ok", "yes", "continue", "run it" wastes a retrieval round-trip and
/// pollutes the context with hits for a prompt that isn't a code-location question.
/// Deliberately permissive otherwise: a needless pack costs a few thousand tokens.
pub fn should_inject(prompt: &str) -> bool {
    let p = prompt.trim();
    if p.len() < 25 {
        return false;
    }
    if p.starts_with('/') {
        return false; // slash command, not a task
    }
    true
}

/// The `## Retrieved context` block injected by the UserPromptSubmit hook.
pub fn format_pack(hits: &[crate::model::Hit], served_ref: &str) -> String {
    let mut s = format!(
        "## Retrieved context (top-{}, from this repo's retrieval index @ {})\n\
         These are the code locations most relevant to the request above, retrieved before \
         you started. Start here rather than searching from scratch; your search tools remain \
         available to verify and expand.\n",
        hits.len(),
        served_ref
    );
    for h in hits {
        s.push_str(&format!(
            "\n### {}:{} -- {} [{}]\n```\n{}\n```\n",
            h.file,
            h.line,
            h.symbol,
            h.kind,
            h.snippet.trim_end()
        ));
    }
    s
}

pub fn emit_additional_context(context: &str, out: &mut impl Write) {
    let payload = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    });
    let _ = writeln!(out, "{payload}");
}

pub fn run_hook(event: &str, stdin_json: &str, out: &mut impl Write) -> i32 {
    match event {
        "pretooluse" => {
            let decision = match serde_json::from_str::<HookEvent>(stdin_json) {
                Ok(ev) => decide(&ev.tool_name, &ev.tool_input),
                Err(_) => Decision::Allow, // never block on a malformed event
            };
            if let Decision::Deny(reason) = decision {
                let payload = serde_json::json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                });
                let _ = writeln!(out, "{payload}");
            }
            0
        }
        "posttooluse" => {
            if let Ok(ev) = serde_json::from_str::<HookEvent>(stdin_json) {
                if ev.tool_name == "Bash" {
                    let command = ev.tool_input.get("command").and_then(|v| v.as_str()).unwrap_or("");
                    if let Some(reason) = extract_fallback_reason(command) {
                        log_fallback(&reason);
                    }
                }
            }
            0
        }
        other => {
            eprintln!("rr hook: unknown event {other:?} (expected pretooluse|posttooluse|userpromptsubmit)");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bash(cmd: &str) -> Decision {
        decide("Bash", &json!({ "command": cmd }))
    }

    #[test]
    fn grep_tool_denied() {
        assert!(matches!(decide("Grep", &json!({"pattern": "foo"})), Decision::Deny(_)));
    }

    #[test]
    fn read_and_other_tools_allowed() {
        assert_eq!(decide("Read", &json!({"file_path": "a.rs"})), Decision::Allow);
        assert_eq!(decide("Edit", &json!({})), Decision::Allow);
    }

    #[test]
    fn source_grep_denied() {
        assert!(matches!(bash("rg foo src/"), Decision::Deny(_)));
        assert!(matches!(bash("grep -r foo ."), Decision::Deny(_)));
        assert!(matches!(bash("/usr/bin/grep x ."), Decision::Deny(_)));   // basename match
        assert!(matches!(bash("ag pattern"), Decision::Deny(_)));
    }

    #[test]
    fn non_grep_bash_allowed() {
        assert_eq!(bash("ls -la"), Decision::Allow);
        assert_eq!(bash("cargo build"), Decision::Allow);
        assert_eq!(bash("echo hi"), Decision::Allow);
    }

    #[test]
    fn pipe_filter_grep_allowed() {
        assert_eq!(bash("ps aux | grep node"), Decision::Allow);
        assert_eq!(bash("cat file.txt | rg pattern"), Decision::Allow);
    }

    #[test]
    fn sequenced_grep_denied() {
        assert!(matches!(bash("grep foo a.txt && rg bar src/"), Decision::Deny(_)));
        assert!(matches!(bash("ls ; grep foo ."), Decision::Deny(_)));
    }

    #[test]
    fn override_token_allows() {
        assert_eq!(bash("RR_FALLBACK='rr fetch returned 0 hits' rg foo src/"), Decision::Allow);
    }

    #[test]
    fn quoted_override_allows_but_bare_does_not() {
        // quoted reason -> override (already covered, re-assert for clarity)
        assert_eq!(bash("RR_FALLBACK='rr fetch returned 0 hits' rg foo src/"), Decision::Allow);
        // bare / unquoted RR_FALLBACK does NOT override (no loggable reason) -> still denied
        assert!(matches!(bash("RR_FALLBACK= rg foo src/"), Decision::Deny(_)));
        assert!(matches!(bash("RR_FALLBACK=bare rg foo src/"), Decision::Deny(_)));
    }

    #[test]
    fn targeted_single_file_grep_allowed() {
        // Blocking these forces a whole-file read instead -- worse, not better.
        assert_eq!(bash("grep -n check_cv sklearn/model_selection/__init__.py"), Decision::Allow);
        assert_eq!(bash("grep foo a.txt"), Decision::Allow);
        assert_eq!(bash("rg pattern path/to/file.py"), Decision::Allow);
    }

    #[test]
    fn broad_search_still_denied() {
        assert!(matches!(bash("rg foo"), Decision::Deny(_)));             // no target -> tree scan
        assert!(matches!(bash("grep foo src"), Decision::Deny(_)));       // extensionless -> dir
        assert!(matches!(bash("grep -rn foo a.txt"), Decision::Deny(_))); // recursive flag
        assert!(matches!(bash("rg foo src/*.py"), Decision::Deny(_)));    // glob
    }

    #[test]
    fn honest_limit_gaps_allowed() {
        assert_eq!(bash("find . | xargs grep foo"), Decision::Allow);   // leading word is xargs
        assert_eq!(bash("python -c \"print(1)\""), Decision::Allow);
    }

    #[test]
    fn extract_reason_quoted() {
        assert_eq!(extract_fallback_reason("RR_FALLBACK='why it failed' rg x"),
                   Some("why it failed".to_string()));
        assert_eq!(extract_fallback_reason("RR_FALLBACK=\"dq reason\" grep x ."),
                   Some("dq reason".to_string()));
    }

    #[test]
    fn extract_reason_absent_or_unquoted() {
        assert_eq!(extract_fallback_reason("rg x src/"), None);
        assert_eq!(extract_fallback_reason("RR_FALLBACK=bare rg x"), None);
    }

    #[test]
    fn should_inject_skips_trivia_and_slash_commands() {
        // A retrieval round-trip + a context pack is pure waste on these.
        assert!(!should_inject("ok"));
        assert!(!should_inject("yes, continue"));
        assert!(!should_inject("   "));
        assert!(!should_inject("/compact and keep the plan in mind"));
        // ...and useful on these.
        assert!(should_inject("separability_matrix returns the wrong result for nested CompoundModels"));
        assert!(should_inject("the graph expansion drops edges for inherited symbols, fix it"));
    }

    #[test]
    fn format_pack_shape() {
        let hit = crate::model::Hit {
            repo: "a/b".into(), ref_: "main".into(), file: "src/x.py".into(), line: 42,
            symbol: "frob".into(), score: 1.0, snippet: "def frob():\n    pass\n".into(),
            kind: "function".into(),
        };
        let p = format_pack(std::slice::from_ref(&hit), "main");
        assert!(p.starts_with("## Retrieved context (top-1,"));
        assert!(p.contains("### src/x.py:42 -- frob [function]"));
        assert!(p.contains("def frob():"));
        // The pack tells the model its search tools remain available.
        assert!(p.contains("your search tools remain\navailable") || p.contains("search tools remain"));
    }

    #[test]
    fn emit_additional_context_is_the_userpromptsubmit_shape() {
        let mut out = Vec::new();
        emit_additional_context("PACK", &mut out);
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "PACK");
    }

    #[test]
    fn run_hook_pretooluse_deny_emits_decision() {
        let mut out = Vec::new();
        let stdin = r#"{"tool_name":"Bash","tool_input":{"command":"rg foo src/"}}"#;
        assert_eq!(run_hook("pretooluse", stdin, &mut out), 0);
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["permissionDecision"], "deny");
    }

    #[test]
    fn run_hook_pretooluse_allow_is_silent() {
        let mut out = Vec::new();
        let stdin = r#"{"tool_name":"Bash","tool_input":{"command":"ls -la"}}"#;
        assert_eq!(run_hook("pretooluse", stdin, &mut out), 0);
        assert!(out.is_empty());
    }

    #[test]
    fn run_hook_malformed_event_allows() {
        let mut out = Vec::new();
        assert_eq!(run_hook("pretooluse", "not json", &mut out), 0);
        assert!(out.is_empty()); // never block on garbage
    }

    #[test]
    fn run_hook_posttooluse_exits_zero() {
        let mut out = Vec::new();
        let stdin = r#"{"tool_name":"Bash","tool_input":{"command":"RR_FALLBACK='r' rg x"}}"#;
        assert_eq!(run_hook("posttooluse", stdin, &mut out), 0);
    }

    #[test]
    fn run_hook_unknown_event_errors() {
        let mut out = Vec::new();
        assert_eq!(run_hook("bogus", "{}", &mut out), 2);
    }
}
