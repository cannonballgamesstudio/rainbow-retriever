use std::io::Write;
use std::process::{Command, Stdio};

fn run_hook_bin(event: &str, stdin: &str) -> (i32, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rr"))
        .args(["hook", event])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).to_string())
}

#[test]
fn hook_pretooluse_denies_source_grep_via_binary() {
    let (code, stdout) = run_hook_bin(
        "pretooluse",
        r#"{"tool_name":"Bash","tool_input":{"command":"rg foo src/"}}"#,
    );
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["hookSpecificOutput"]["permissionDecision"], "deny");
}

#[test]
fn hook_pretooluse_allows_normal_command_via_binary() {
    let (code, stdout) = run_hook_bin(
        "pretooluse",
        r#"{"tool_name":"Bash","tool_input":{"command":"cargo build"}}"#,
    );
    assert_eq!(code, 0);
    assert!(stdout.trim().is_empty());
}
