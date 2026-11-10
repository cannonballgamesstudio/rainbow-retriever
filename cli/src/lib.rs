pub mod cli;
pub mod client;
pub mod model;
pub mod gitctx;
pub mod render;
pub mod hook;

use std::io::{self, IsTerminal, Read, Write};

pub fn run() -> i32 {
    let cli = <cli::Cli as clap::Parser>::parse();
    if let cli::Command::Hook { event } = &cli.command {
        let mut stdin_json = String::new();
        let _ = io::stdin().read_to_string(&mut stdin_json);
        let mut out = io::stdout().lock();
        if event == "userpromptsubmit" {
            // Needs the endpoint + git context that run_hook's other events don't.
            // Every failure path here returns 0 with no output: the user's prompt goes
            // through unmodified rather than being blocked by a degraded retriever.
            return run_prompt_hook(&cli, &stdin_json, &mut out);
        }
        return hook::run_hook(event, &stdin_json, &mut out);
    }
    let json = render::use_json(cli.json, io::stdout().is_terminal());
    let mut out = io::stdout().lock();
    let mut err = io::stderr().lock();
    dispatch_inner(cli, &mut out, &mut err, json)
}

/// Context-pack settings for the UserPromptSubmit hook: 12 hits, hybrid, code-only.
const INJECT_LIMIT: u32 = 12;
const INJECT_MODE: &str = "hybrid";

/// Pre-inject a context pack before the model's first turn (the UserPromptSubmit hook).
/// Silent on every failure: a broken retriever must degrade to "no hook installed",
/// never to a blocked prompt.
fn run_prompt_hook(cli: &cli::Cli, stdin_json: &str, out: &mut impl Write) -> i32 {
    if std::env::var("RR_INJECT").map(|v| v == "0" || v == "off").unwrap_or(false) {
        return 0;
    }
    let ev: hook::PromptEvent = match serde_json::from_str(stdin_json) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    if !hook::should_inject(&ev.prompt) {
        return 0;
    }
    let endpoint = match cli.endpoint.clone() {
        Some(ep) => ep,
        None => return 0,
    };
    let ctx = match gitctx::gather(cli.repo.clone(), cli.r#ref.clone(), cli.pr) {
        Ok(c) => c,
        Err(_) => return 0, // not a git repo: nothing to retrieve against
    };
    let req = model::QueryRequest {
        repo: ctx.repo.clone(),
        branch: ctx.branch.clone(),
        local_head_sha: ctx.local_head_sha.clone(),
        fallback_refs: gitctx::fallback_refs(),
        text: ev.prompt.chars().take(6000).collect(),
        mode: INJECT_MODE.to_string(),
        limit: INJECT_LIMIT,
        query_dims: None,
        budget: None,
        code_only: true,
        rerank: false,
        // Request a depth pack (no file-dedup): a fixed number of slots is better spent
        // on several chunks of the few files that matter than on one chunk each of many.
        dedup: Some(false),
    };
    let rc = client::RagClient::new(endpoint);
    match rc.post::<_, model::QueryResponse>("/v1/query", &req) {
        Ok(resp) if !resp.hits.is_empty() => {
            hook::emit_additional_context(&hook::format_pack(&resp.hits, &resp.served_ref), out);
            0
        }
        _ => 0, // no hits, or the server is down: proceed with the bare prompt
    }
}

/// Test entrypoint: forces JSON if cli.json, else assumes non-TTY (json=true) for capture.
pub fn dispatch(cli: cli::Cli, out: &mut impl Write, err: &mut impl Write) -> i32 {
    let json = render::use_json(cli.json, false);
    dispatch_inner(cli, out, err, json)
}

fn print_value(out: &mut impl Write, v: &serde_json::Value) {
    let _ = writeln!(out, "{}", serde_json::to_string_pretty(v).unwrap());
}

fn dispatch_inner(cli: cli::Cli, out: &mut impl Write, err: &mut impl Write, json: bool) -> i32 {
    // `goodboy` is a pure client-side easter egg: no endpoint, no git context needed.
    if let cli::Command::Goodboy = &cli.command {
        let _ = render::goodboy(out);
        return 0;
    }

    let endpoint = match cli.endpoint {
        Some(ep) => ep,
        None => {
            let _ = writeln!(err, "error: no endpoint configured; pass --endpoint or set RR_URL");
            return 1;
        }
    };
    let rc = client::RagClient::new(endpoint);

    // Commands that need no repo/branch context.
    match &cli.command {
        cli::Command::Capabilities => {
            // `speak`: print the server's capabilities as pretty JSON. A raw
            // serde_json::Value round-trip keeps fields the Rust model doesn't know.
            return match rc.get::<serde_json::Value>("/v1/capabilities", &[]) {
                Ok(v) => { print_value(out, &v); 0 }
                Err(e) => report_err(err, &e),
            };
        }
        cli::Command::Health => {
            return match rc.get::<serde_json::Value>("/healthz", &[]) {
                Ok(v) => { print_value(out, &v); 0 }
                Err(e) => report_err(err, &e),
            };
        }
        cli::Command::Status | cli::Command::Ls => {
            let q: Vec<(&str, &str)> = match &cli.repo {
                Some(r) => vec![("repo", r.as_str())],
                None => vec![],
            };
            return match rc.get::<model::StatusResponse>("/v1/status", &q) {
                Ok(resp) => {
                    let _ = render::status(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            };
        }
        _ => {}
    }

    let ctx = match gitctx::gather(cli.repo.clone(), cli.r#ref.clone(), cli.pr) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "error: {e}");
            return 1;
        }
    };

    // fallback_refs returns Vec<String> directly (best-effort, no Result).
    let fallbacks = gitctx::fallback_refs();

    match cli.command {
        cli::Command::Query { text, mode, limit, query_dims, budget, lean, sig, code_only, rerank } => {
            // --sig is the default: terse, self-contained hits (file:line symbol [kind] score +
            // def line). Opt into the full JSON response with explicit --json (or RR_FULL);
            // --lean drops the def line.
            let envflag = |k: &str| std::env::var(k).map(|v| !v.is_empty() && v != "0").unwrap_or(false);
            let lean = lean || envflag("RR_LEAN");
            let want_full = cli.json || envflag("RR_FULL");
            let sig = sig || envflag("RR_SIG") || (!lean && !want_full);
            let as_json = want_full && !lean && !sig;
            let mut req = cli::build_query_request(text, mode, limit, query_dims, budget, &ctx, fallbacks);
            req.code_only = code_only || envflag("RR_CODE_ONLY");
            req.rerank = rerank || envflag("RR_RERANK");
            match rc.post::<_, model::QueryResponse>("/v1/query", &req) {
                Ok(resp) => {
                    render::warn_staleness(err, &resp.staleness);
                    let _ = render::render_query(out, &resp, as_json, sig);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Xref { text, refs, mode, limit, query_dims, budget } => {
            let req = model::XrefRequest {
                repo: ctx.repo.clone(),
                branch: ctx.branch.clone(),
                local_head_sha: ctx.local_head_sha.clone(),
                fallback_refs: fallbacks,
                text,
                refs,
                mode,
                limit,
                query_dims,
                budget,
            };
            match rc.post::<_, model::XrefResponse>("/v1/xref", &req) {
                Ok(resp) => {
                    render::warn_staleness(err, &resp.staleness);
                    let _ = render::render_xref(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Resolve => {
            match rc.get::<serde_json::Value>("/v1/resolve",
                &[("repo", ctx.repo.as_str()), ("branch", ctx.branch.as_str())]) {
                Ok(v) => { print_value(out, &v); 0 }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Ctx { symbol, file } => {
            let req = model::CtxRequest {
                repo: ctx.repo.clone(),
                branch: ctx.branch.clone(),
                local_head_sha: ctx.local_head_sha.clone(),
                fallback_refs: fallbacks,
                symbol,
                file,
            };
            match rc.post::<_, model::CtxResponse>("/v1/ctx", &req) {
                Ok(resp) => {
                    render::warn_staleness(err, &resp.staleness);
                    let _ = render::render_ctx(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Deps { symbol, transitive, depth, file } => {
            let mut q: Vec<(&str, String)> = vec![
                ("repo", ctx.repo.clone()),
                ("branch", ctx.branch.clone()),
                ("symbol", symbol),
                ("transitive", transitive.to_string()),
                ("depth", depth.to_string()),
            ];
            if let Some(f) = file { q.push(("file", f)); }
            let qref: Vec<(&str, &str)> = q.iter().map(|(k, v)| (*k, v.as_str())).collect();
            match rc.get::<model::EdgesResp>("/v1/deps", &qref) {
                Ok(resp) => {
                    let _ = render::edges(out, &resp.edges, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::File { path } => {
            match rc.get::<model::FileResp>("/v1/file",
                &[("repo", ctx.repo.as_str()), ("branch", ctx.branch.as_str()), ("path", path.as_str())]) {
                Ok(resp) => {
                    let _ = render::file_symbols(out, &resp.symbols, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Tests { symbol } => {
            match rc.get::<model::EdgesResp>("/v1/tests",
                &[("repo", ctx.repo.as_str()), ("branch", ctx.branch.as_str()), ("symbol", symbol.as_str())]) {
                Ok(resp) => {
                    let _ = render::edges(out, &resp.edges, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Map { path, depth } => {
            let d = depth.to_string();
            let mut q: Vec<(&str, &str)> = vec![
                ("repo", ctx.repo.as_str()),
                ("branch", ctx.branch.as_str()),
                ("depth", d.as_str()),
            ];
            if let Some(ref p) = path { q.push(("path", p.as_str())); }
            match rc.get::<model::MapResp>("/v1/map", &q) {
                Ok(resp) => {
                    let _ = render::structure(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Reach { symbols, depth, direction, file } => {
            let mut q: Vec<(&str, String)> = vec![
                ("repo", ctx.repo.clone()),
                ("branch", ctx.branch.clone()),
                ("symbol", symbols.join(",")),
                ("direction", direction),
                ("depth", depth.to_string()),
            ];
            if let Some(f) = file { q.push(("file", f)); }
            let qref: Vec<(&str, &str)> = q.iter().map(|(k, v)| (*k, v.as_str())).collect();
            match rc.get::<model::ReachResponse>("/v1/reachable", &qref) {
                Ok(resp) => {
                    let _ = render::reach(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        cli::Command::Hierarchy { class, file } => {
            let mut q: Vec<(&str, &str)> = vec![
                ("repo", ctx.repo.as_str()),
                ("branch", ctx.branch.as_str()),
                ("symbol", class.as_str()),
            ];
            if let Some(ref f) = file { q.push(("file", f.as_str())); }
            match rc.get::<model::HierarchyResp>("/v1/hierarchy", &q) {
                Ok(resp) => {
                    let _ = render::hierarchy(out, &resp, json);
                    0
                }
                Err(e) => report_err(err, &e),
            }
        }
        // Handled above (goodboy, hook, speak, health, status/ls).
        _ => 0,
    }
}

fn report_err(err: &mut impl Write, e: &client::ClientError) -> i32 {
    match e {
        client::ClientError::Api(a) => {
            let _ = writeln!(err, "error [{}]: {} ({})", a.status, a.code, a.message);
            if let Some(d) = &a.details {
                let _ = writeln!(err, "  details: {d}");
            }
        }
        client::ClientError::Transport(m) => {
            let _ = writeln!(err, "error: cannot reach the server: {m}");
        }
    }
    e.exit_code()
}
