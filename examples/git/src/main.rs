use agentsdk::{Agent, AgentListener, Message, Messages, OpenAI, messages};
use async_trait::async_trait;
use std::error::Error;
use tracing::Level;
use tracing_subscriber::EnvFilter;

mod git_tools {
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
}

struct GitHandler;

#[async_trait]
impl AgentListener for GitHandler {
    async fn prepare_system_prompt(
        &mut self,
        _history: &Messages,
    ) -> Option<std::borrow::Cow<'static, str>> {
        Some(std::borrow::Cow::Borrowed(PROMPT))
    }

    async fn on_text_delta(&mut self, text: &str) {
        print!("{text}");
    }

    async fn on_tool_pre_execute(
        &mut self,
        id: &str,
        name: &str,
        arguments: &serde_json::Value,
    ) -> agentsdk::PreToolAction {
        println!("\n[Executing Tool] ID: {id}, Name: {name}, Args: {arguments}");
        agentsdk::PreToolAction::Continue(None)
    }

    async fn on_tool_post_execute(
        &mut self,
        id: &str,
        name: &str,
        result: &serde_json::Value,
    ) -> agentsdk::PostToolAction {
        let result_len = match result {
            serde_json::Value::String(s) => s.len(),
            other => other.to_string().len(),
        };
        println!("[Tool {name} Finished] ID: {id}, Result length: {result_len}");
        agentsdk::PostToolAction::Continue(None)
    }

    async fn on_tool_error(
        &mut self,
        _id: &str,
        name: &str,
        error: &str,
    ) -> agentsdk::ToolErrorAction {
        eprintln!("Tool {name} failed: {error}");
        agentsdk::ToolErrorAction::Continue(None)
    }
}

const PROMPT: &str = r"
You are a simple Git summary tool. As per the user's request, you will summarize
sections of the repository version history. You have access to the following tools:

- `diff`: This tool returns the git diff between two commits or branches.
- `log`: This tool returns the git log for a given repository and branch.
- `status`: This tool returns the git status for a given repository.

The user will provide you with a request in the form of a question. and you will
respond with a valid text response. You can make as many tool calls as you want.

Here are some examples of valid requests:

- What is the latest commit message?
- How many commits are there in the current branch?
- Summarize the latest changes in the repository.
- Write a PR description for the current branch.
- what is the dev branch working on.
- et cetera.

You should assume that the tools return git results from a pre configured git repository.
";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .with_target(false)
        .without_time()
        .init();

    tracing::info!("Starting git-summary example");

    let config = agentsdk::ModelConfig::from_env()?;
    let client = OpenAI::new(config);

    let agent = Agent::builder()
        .client(client)
        .options(
            agentsdk::AgentOptions::builder()
                .messages(std::sync::Arc::new(vec![messages::user("Show me the current git status and the last 3 commit messages using the tools provided.")]))
                .with_tool(&git_tools::diff())
                .with_tool(&git_tools::status())
                .with_tool(&git_tools::log())
                .build()?
        )
        .build()?;

    let mut handler = GitHandler;
    let history = agent.run(&mut handler).await?;

    if let Some(content) = history.last().and_then(|m| {
        let Message::AssistantMessage(assistant) = m else {
            return None;
        };
        assistant.content.as_ref()
    }) {
        println!("\nFinal Response: {content}");
    }

    Ok(())
}
