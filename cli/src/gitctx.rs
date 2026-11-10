use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct GitCtx {
    pub repo: String,
    pub branch: String,
    pub local_head_sha: Option<String>,
}

pub fn is_valid_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn branch_for_pr(n: u64) -> String {
    format!("pr-{n}")
}

pub fn parse_repo_from_remote_url(url: &str) -> Option<String> {
    let s = url.trim();
    let tail = if let Some(rest) = s.strip_prefix("git@") {
        // git@host:owner/repo(.git)
        rest.split_once(':').map(|(_, p)| p)?
    } else if let Some(rest) = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")) {
        // host/owner/repo(.git)
        rest.split_once('/').map(|(_, p)| p)?
    } else {
        return None;
    };
    let tail = tail.strip_suffix(".git").unwrap_or(tail);
    let parts: Vec<&str> = tail.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() >= 2 {
        Some(format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]))
    } else {
        None
    }
}

fn git_out(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Like `git_out`, but scoped to an arbitrary directory via `git -C <cwd>` rather than the
/// process's own cwd.
fn git_out_at(cwd: &Path, args: &[&str]) -> Option<String> {
    let cwd_str = cwd.to_str()?;
    let mut full: Vec<&str> = vec!["-C", cwd_str];
    full.extend_from_slice(args);
    let out = Command::new("git").args(&full).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// A stable repo id for workspaces with no `origin` remote (or where the remote can't be
/// parsed): `local/<name>` where `<name>` is the file name of the parent directory of the
/// *main* working tree's `.git` dir -- i.e. `git -C cwd rev-parse --git-common-dir`, resolved
/// against `cwd` when git reports it as a relative path. Because `--git-common-dir` always
/// points at the one shared `.git` regardless of which of a repo's worktrees/branches you're
/// standing in, this id is stable across all of them. Returns `None` when `cwd` isn't inside a
/// git repo (or git can't be run).
pub fn local_repo_id(cwd: &Path) -> Option<String> {
    let common_dir = git_out_at(cwd, &["rev-parse", "--git-common-dir"])?;
    let common_path = PathBuf::from(&common_dir);
    let abs = if common_path.is_absolute() { common_path } else { cwd.join(&common_path) };
    let parent = abs.parent()?;
    let name = parent.file_name()?.to_str()?;
    Some(format!("local/{name}"))
}

pub fn fallback_refs() -> Vec<String> {
    // The repo's REAL default branch, e.g. origin/main -> "main". Best-effort.
    match git_out(&["rev-parse", "--abbrev-ref", "origin/HEAD"]) {
        Some(s) => match s.strip_prefix("origin/") {
            Some(b) => vec![b.to_string()],
            None => vec![s],
        },
        None => vec![],
    }
}

pub fn gather(repo: Option<String>, r#ref: Option<String>, pr: Option<u64>) -> Result<GitCtx> {
    let repo = match repo {
        Some(r) => r,
        None => match git_out(&["remote", "get-url", "origin"]) {
            Some(url) => match parse_repo_from_remote_url(&url) {
                Some(r) => r,
                None => std::env::current_dir().ok().and_then(|d| local_repo_id(&d))
                    .ok_or_else(|| anyhow!("could not parse repo from remote {url:?}; pass --repo"))?,
            },
            None => std::env::current_dir().ok().and_then(|d| local_repo_id(&d))
                .ok_or_else(|| anyhow!("not in a git repo; pass --repo owner/repo"))?,
        },
    };
    let branch = if let Some(n) = pr {
        branch_for_pr(n)
    } else if let Some(r) = r#ref {
        r
    } else {
        let b = git_out(&["rev-parse", "--abbrev-ref", "HEAD"])
            .context("not in a git repo; pass --repo and --ref")?;
        if b == "HEAD" {
            return Err(anyhow!("detached HEAD; pass --ref <branch>"));
        }
        b
    };
    let local_head_sha = git_out(&["rev-parse", "HEAD"]).filter(|s| is_valid_sha(s));
    Ok(GitCtx { repo, branch, local_head_sha })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique scratch dir under the system temp dir (no `tempfile` dev-dependency in this
    /// crate; deleted at the end of each test that uses it).
    fn scratch_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("rr-gitctx-test-{tag}-{pid}-{n}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn run_git(cwd: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?} failed in {cwd:?}");
    }

    fn init_repo(dir: &Path) {
        run_git(dir, &["init", "-q"]);
        run_git(dir, &["config", "user.email", "test@example.com"]);
        run_git(dir, &["config", "user.name", "test"]);
        fs::write(dir.join("f.txt"), "x").expect("write file");
        run_git(dir, &["add", "f.txt"]);
        run_git(dir, &["commit", "-q", "-m", "init"]);
    }

    #[test]
    fn repo_with_origin_remote_still_resolves_owner_repo_via_remote_url() {
        let root = scratch_dir("with-remote");
        let main = root.join("main");
        fs::create_dir_all(&main).expect("mkdir main");
        init_repo(&main);
        run_git(&main, &["remote", "add", "origin", "https://github.com/example/repo.git"]);

        let url = git_out_at(&main, &["remote", "get-url", "origin"]).expect("remote url");
        assert_eq!(parse_repo_from_remote_url(&url).as_deref(), Some("example/repo"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn no_remote_repo_yields_local_slash_dir_name() {
        let root = scratch_dir("no-remote");
        let main = root.join("myrepo");
        fs::create_dir_all(&main).expect("mkdir main");
        init_repo(&main);

        assert_eq!(git_out_at(&main, &["remote", "get-url", "origin"]), None);
        assert_eq!(local_repo_id(&main).as_deref(), Some("local/myrepo"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn second_worktree_yields_same_local_repo_id_as_main() {
        let root = scratch_dir("worktree");
        let main = root.join("myrepo");
        fs::create_dir_all(&main).expect("mkdir main");
        init_repo(&main);

        let wt = root.join("myrepo-wt2");
        run_git(
            &main,
            &["worktree", "add", wt.to_str().expect("wt path"), "-b", "wt2branch", "-q"],
        );

        let main_id = local_repo_id(&main);
        let wt_id = local_repo_id(&wt);
        assert_eq!(main_id.as_deref(), Some("local/myrepo"));
        assert_eq!(wt_id, main_id, "worktree repo-id must match the main tree's");

        // Clean up the worktree registration before removing its directory, else git's
        // metadata under main/.git/worktrees goes stale (harmless here, but tidy).
        let _ = Command::new("git")
            .args(["-C", main.to_str().unwrap(), "worktree", "remove", "--force", wt.to_str().unwrap()])
            .status();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn non_git_dir_yields_none() {
        let root = scratch_dir("non-git");
        assert_eq!(local_repo_id(&root), None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parses_ssh_and_https_remotes_stripping_dot_git() {
        assert_eq!(parse_repo_from_remote_url("git@github.com:example/repo.git").as_deref(),
                   Some("example/repo"));
        assert_eq!(parse_repo_from_remote_url("https://github.com/example/repo.git").as_deref(),
                   Some("example/repo"));
        assert_eq!(parse_repo_from_remote_url("https://github.com/example/repo").as_deref(),
                   Some("example/repo"));
        assert_eq!(parse_repo_from_remote_url("not a url"), None);
    }

    #[test]
    fn validates_lowercase_40_hex_sha() {
        assert!(is_valid_sha("9f3c1a2b4d5e6f7081920aabbccddeeff0011223"));
        assert!(!is_valid_sha("9F3C1A2B4D5E6F7081920AABBCCDDEEFF0011223")); // uppercase
        assert!(!is_valid_sha("abc"));                                       // short
    }

    #[test]
    fn pr_maps_to_pr_branch() {
        assert_eq!(branch_for_pr(123), "pr-123");
    }
}
