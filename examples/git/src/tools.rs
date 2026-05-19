use agentsdk::Tool;
use agentsdk::tool;
use std::process::Command;

fn run_git(cmd: &mut Command) -> Result<String, String> {
    match cmd.output() {
        Ok(output) if output.status.success() => {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        Ok(output) => Err(String::from_utf8_lossy(&output.stderr).to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Returns the git diff from two commits or branches
#[tool]
pub fn diff(left: String, right: String) -> Tool {
    run_git(Command::new("git").arg("diff").arg(&left).arg(&right))
}

/// Returns the git status for the current repository
#[tool]
pub fn status() -> Tool {
    run_git(Command::new("git").arg("status").arg("--porcelain"))
}

/// Returns the git log for the current repository
#[tool]
pub fn log(n: Option<i32>) -> Tool {
    let n = n.unwrap_or(1);
    run_git(
        Command::new("git")
            .arg("log")
            .arg(format!("-{n}"))
            .arg("--format=%B"),
    )
}
