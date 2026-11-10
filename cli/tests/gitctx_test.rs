use std::process::Command;

fn git(dir: &std::path::Path, args: &[&str]) {
    let ok = Command::new("git").args(args).current_dir(dir).status().unwrap().success();
    assert!(ok, "git {:?} failed", args);
}

struct CwdGuard(std::path::PathBuf);
impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn gather_reads_repo_branch_and_sha_from_cwd_git() {
    let dir = std::env::temp_dir().join(format!("rrtest_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init", "-q"]);
    git(&dir, &["remote", "add", "origin", "git@github.com:example/repo.git"]);
    git(&dir, &["config", "user.email", "t@t"]);
    git(&dir, &["config", "user.name", "t"]);
    std::fs::write(dir.join("f.txt"), "x").unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "c"]);
    let _guard = CwdGuard(std::env::current_dir().unwrap());
    std::env::set_current_dir(&dir).unwrap();
    let ctx = rr::gitctx::gather(None, None, None).unwrap();
    // _guard's Drop restores the original cwd even if the line above panics
    assert_eq!(ctx.repo, "example/repo");
    assert_eq!(ctx.local_head_sha.as_ref().unwrap().len(), 40);
    assert!(rr::gitctx::is_valid_sha(ctx.local_head_sha.as_ref().unwrap()));
}
