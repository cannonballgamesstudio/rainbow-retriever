use crate::gitctx::GitCtx;
use crate::model::QueryRequest;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rr", version, about = "rainbow-retriever command-line client")]
pub struct Cli {
    /// Server URL
    #[arg(long, env = "RR_URL", default_value = "http://localhost:8081", global = true)]
    pub endpoint: Option<String>,
    /// Repository id as registered on the server (default: from the `origin` remote,
    /// else `local/<directory name>`)
    #[arg(long, global = true)]
    pub repo: Option<String>,
    /// Branch / ref to query (default: the current branch)
    #[arg(long = "ref", global = true)]
    pub r#ref: Option<String>,
    #[arg(long, global = true)]
    pub pr: Option<u64>,
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Ranked hybrid search (semantic+keyword) -- go get it.
    #[command(name = "fetch", visible_alias = "query")]
    Query {
        text: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
        #[arg(long = "query-dims")]
        query_dims: Option<i64>,
        #[arg(long)]
        budget: Option<i64>,
        /// Terse output: one line per hit (file:line symbol [kind] score), no snippets.
        /// Forces the compact render even when stdout is not a TTY.
        #[arg(long)]
        lean: bool,
        /// Self-contained output: like --lean but appends each symbol's def/first line inline
        /// (file:line symbol [kind] score  <def line>), so a follow-up file read is rarely
        /// needed. Also forces the compact render in non-TTY.
        #[arg(long)]
        sig: bool,
        /// Demote doc/markdown hits below code (default: keep them interleaved)
        #[arg(long = "code-only")]
        code_only: bool,
        /// Cross-encoder rerank of the top hits (opt-in; needs a reranker configured on the server)
        #[arg(long)]
        rerank: bool,
    },
    /// A symbol's definition, signature, callers and callees -- point straight at it.
    #[command(name = "point", visible_alias = "ctx")]
    Ctx {
        symbol: String,
        /// Disambiguate when the symbol is defined in multiple files
        #[arg(long)]
        file: Option<String>,
    },
    /// Search across several refs and merge the results.
    Xref {
        text: String,
        #[arg(long, value_delimiter = ',', required = true)]
        refs: Vec<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
        #[arg(long = "query-dims")]
        query_dims: Option<i64>,
        #[arg(long)]
        budget: Option<i64>,
    },
    /// List indexed refs on the server.
    Status,
    /// Alias of status.
    Ls,
    /// Show the registry row for the current repo/branch.
    Resolve,
    /// Ask the server what it can do -- print its capabilities as pretty JSON.
    #[command(name = "speak", visible_alias = "capabilities")]
    Capabilities,
    /// Server health check.
    Health,
    /// Claude Code hook: `rr hook pretooluse|posttooluse|userpromptsubmit` (reads the event on stdin).
    #[command(hide = true)]
    Hook {
        event: String,
    },
    /// Dependency edges for a symbol -- trace who calls what, and what it calls.
    #[command(name = "track", visible_alias = "deps")]
    Deps {
        symbol: String,
        #[arg(long)]
        transitive: bool,
        #[arg(long, default_value_t = 5)]
        depth: i64,
        #[arg(long)]
        file: Option<String>,
    },
    /// List symbols defined in a file -- dig up what's inside.
    #[command(name = "dig", visible_alias = "file")]
    File {
        path: String,
    },
    /// Show tests that cover a symbol -- the watchdog keeps an eye on it.
    #[command(name = "watchdog", visible_alias = "tests")]
    Tests {
        symbol: String,
    },
    /// Repo directory structure and hotspots -- sniff out the lay of the land.
    #[command(name = "sniff", visible_alias = "map")]
    Map {
        path: Option<String>,
        #[arg(long, default_value_t = 3)]
        depth: i64,
    },
    /// Inheritance hierarchy for a class -- trace its pedigree.
    #[command(name = "pedigree", visible_alias = "hierarchy")]
    Hierarchy {
        class: String,
        #[arg(long)]
        file: Option<String>,
    },
    /// Transitive reachability from one or more entry symbols -- follow the trail.
    #[command(name = "trace", visible_alias = "reach")]
    Reach {
        #[arg(value_delimiter = ',')]
        symbols: Vec<String>,
        #[arg(long, default_value_t = 5)]
        depth: u32,
        #[arg(long, default_value = "callees")]
        direction: String,
        #[arg(long)]
        file: Option<String>,
    },
    /// Print a small ASCII rainbow + the seven retrieval bands, and the binary version.
    Goodboy,
}

pub fn build_query_request(
    text: String,
    mode: String,
    limit: u32,
    query_dims: Option<i64>,
    budget: Option<i64>,
    ctx: &GitCtx,
    fallback_refs: Vec<String>,
) -> QueryRequest {
    QueryRequest {
        repo: ctx.repo.clone(),
        branch: ctx.branch.clone(),
        local_head_sha: ctx.local_head_sha.clone(),
        fallback_refs,
        text,
        mode,
        limit,
        query_dims,
        budget,
        code_only: false,
        rerank: false,
        dedup: None,  // interactive `rr fetch` keeps the backend default (dedup on: terse, one line per file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_query_with_flags() {
        let cli = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "--json", "query", "find total",
            "--mode", "keyword", "--limit", "5",
        ]).unwrap();
        assert!(cli.json);
        match cli.command {
            Command::Query { text, mode, limit, .. } => {
                assert_eq!(text, "find total");
                assert_eq!(mode, "keyword");
                assert_eq!(limit, 5);
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn parses_xref_comma_refs() {
        let cli = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "xref", "T", "--refs", "main,pr-1",
        ]).unwrap();
        match cli.command {
            Command::Xref { refs, .. } => assert_eq!(refs, vec!["main", "pr-1"]),
            _ => panic!("expected Xref"),
        }
    }

    #[test]
    fn parses_query_with_rerank() {
        let cli = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "query", "find total", "--rerank",
        ]).unwrap();
        match cli.command {
            Command::Query { rerank, .. } => assert!(rerank),
            _ => panic!("expected Query"),
        }
    }

    // ── Every old verb still parses, and parses to the exact same command as its
    // new name. ──

    /// Structural tag for a Command, ignoring field values -- lets us assert two
    /// parses landed on "the same subcommand" without deriving PartialEq on Command.
    fn tag(c: &Command) -> &'static str {
        match c {
            Command::Query { .. } => "query",
            Command::Ctx { .. } => "ctx",
            Command::Xref { .. } => "xref",
            Command::Status => "status",
            Command::Ls => "ls",
            Command::Resolve => "resolve",
            Command::Capabilities => "capabilities",
            Command::Health => "health",
            Command::Hook { .. } => "hook",
            Command::Deps { .. } => "deps",
            Command::File { .. } => "file",
            Command::Tests { .. } => "tests",
            Command::Map { .. } => "map",
            Command::Hierarchy { .. } => "hierarchy",
            Command::Reach { .. } => "reach",
            Command::Goodboy => "goodboy",
        }
    }

    fn parse(args: &[&str]) -> Cli {
        let mut full = vec!["rr", "--endpoint", "https://x"];
        full.extend_from_slice(args);
        Cli::try_parse_from(full).unwrap()
    }

    #[test]
    fn dog_name_aliases_parse_to_the_same_command_as_old_verbs() {
        let pairs: &[(&[&str], &[&str])] = &[
            (&["fetch", "hi"], &["query", "hi"]),
            (&["point", "greet"], &["ctx", "greet"]),
            (&["track", "greet"], &["deps", "greet"]),
            (&["dig", "a.py"], &["file", "a.py"]),
            (&["watchdog", "greet"], &["tests", "greet"]),
            (&["sniff"], &["map"]),
            (&["pedigree", "C"], &["hierarchy", "C"]),
            (&["speak"], &["capabilities"]),
        ];
        for (new, old) in pairs {
            let a = parse(new);
            let b = parse(old);
            assert_eq!(tag(&a.command), tag(&b.command),
                       "expected {new:?} and {old:?} to parse to the same command");
        }
    }

    #[test]
    fn old_verb_flags_still_parse_identically_under_the_alias() {
        // A representative flag-bearing pair: the old `deps` verb keeps its exact
        // flag surface (--transitive/--depth/--file) once `track` is primary.
        let cli = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "deps", "greet",
            "--transitive", "--depth", "2", "--file", "a.py",
        ]).unwrap();
        match cli.command {
            Command::Deps { symbol, transitive, depth, file } => {
                assert_eq!(symbol, "greet");
                assert!(transitive);
                assert_eq!(depth, 2);
                assert_eq!(file.as_deref(), Some("a.py"));
            }
            _ => panic!("expected Deps"),
        }
        // And the new `track` name accepts the identical flags.
        let cli2 = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "track", "greet",
            "--transitive", "--depth", "2", "--file", "a.py",
        ]).unwrap();
        match cli2.command {
            Command::Deps { symbol, transitive, depth, file } => {
                assert_eq!(symbol, "greet");
                assert!(transitive);
                assert_eq!(depth, 2);
                assert_eq!(file.as_deref(), Some("a.py"));
            }
            _ => panic!("expected Deps"),
        }
    }

    #[test]
    fn parses_goodboy() {
        let cli = Cli::try_parse_from(["rr", "--endpoint", "https://x", "goodboy"]).unwrap();
        assert_eq!(tag(&cli.command), "goodboy");
    }

    #[test]
    fn parses_reach_with_comma_symbols_and_flags() {
        let cli = Cli::try_parse_from([
            "rr", "--endpoint", "https://x", "reach", "A,B",
            "--direction", "callers", "--depth", "3", "--file", "a.py",
        ]).unwrap();
        match cli.command {
            Command::Reach { symbols, depth, direction, file } => {
                assert_eq!(symbols, vec!["A".to_string(), "B".to_string()]);
                assert_eq!(depth, 3);
                assert_eq!(direction, "callers");
                assert_eq!(file.as_deref(), Some("a.py"));
            }
            _ => panic!("expected Reach"),
        }
    }

    #[test]
    fn parses_reach_defaults() {
        let cli = Cli::try_parse_from(["rr", "--endpoint", "https://x", "reach", "A"]).unwrap();
        match cli.command {
            Command::Reach { symbols, depth, direction, file } => {
                assert_eq!(symbols, vec!["A".to_string()]);
                assert_eq!(depth, 5);
                assert_eq!(direction, "callees");
                assert_eq!(file, None);
            }
            _ => panic!("expected Reach"),
        }
    }

    #[test]
    fn trace_alias_parses_to_the_same_command_as_reach() {
        let a = parse(&["trace", "A"]);
        let b = parse(&["reach", "A"]);
        assert_eq!(tag(&a.command), "reach");
        assert_eq!(tag(&b.command), "reach");
    }

    #[test]
    fn build_query_request_assembles_from_ctx() {
        use crate::gitctx::GitCtx;
        let ctx = GitCtx { repo: "example/repo".into(), branch: "feat".into(),
                           local_head_sha: Some("a".repeat(40)) };
        let req = build_query_request("hi".into(), "hybrid".into(), 20, None, None, &ctx, vec!["main".into()]);
        assert_eq!(req.repo, "example/repo");
        assert_eq!(req.branch, "feat");
        assert_eq!(req.fallback_refs, vec!["main".to_string()]);
        assert_eq!(req.local_head_sha.unwrap().len(), 40);
    }
}
