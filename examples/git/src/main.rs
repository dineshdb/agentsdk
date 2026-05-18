use agentsdk::Message;
use agentsdk::{Agent, AgentEvent, OpenAI, messages};
use futures::StreamExt;
use std::env;
use std::error::Error;
use tracing::Level;
use tracing_subscriber::EnvFilter;

mod git_tools {
    use agentsdk::Tool;
    use agentsdk::tool;
    use std::process::Command;

    /// Returns the git diff from two commits or branches
    /// # Arguments
    /// * `left` - The left side of the diff
    /// * `right` - The right side of the diff
    ///
    /// # Returns
    /// The git diff between the two commits or branches
    #[tool]
    pub fn diff(left: String, right: String) -> Tool {
        let output = Command::new("git")
            .arg("diff")
            .arg(&left)
            .arg(&right)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    Err(String::from_utf8_lossy(&output.stderr).to_string())
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Returns the git status for the current repository
    ///
    /// # Returns
    /// The git status for the current repository
    #[tool]
    pub fn status() -> Tool {
        let output = Command::new("git")
            .arg("status")
            .arg("--porcelain")
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    Err(String::from_utf8_lossy(&output.stderr).to_string())
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Returns the git log for the current repository
    #[tool]
    pub fn log(n: Option<i32>) -> Tool {
        let n = n.unwrap_or(1);
        let output = Command::new("git")
            .arg("log")
            .arg(format!("-{n}"))
            .arg("--format=%B")
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    Err(String::from_utf8_lossy(&output.stderr).to_string())
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

const PROMPT: &str = r"
You are a simple Git summary tool. As per the user's request, you will summarize
sections of the repository version history. You have access to the following tools:

- `diff`: This tool returns the git log for a given repository and branch.
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

    let endpoint =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY must be set")?;
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

    let client = OpenAI::builder()
        .base_url(endpoint)
        .api_key(api_key)
        .model(model)
        .build()?;

    let agent = Agent::builder()
        .client(client)
        .options(
            agentsdk::AgentOptions::builder()
                .system(Some(PROMPT.into()))
                .messages(std::sync::Arc::new(vec![messages::user("Show me the current git status and the last 3 commit messages using the tools provided.")]))
                .with_tool(&git_tools::diff())
                .with_tool(&git_tools::status())
                .with_tool(&git_tools::log())
                .build()?
        )
        .build()?;

    let mut stream = agent.stream();
    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentEvent::TextDelta(text)) => print!("{text}"),
            Ok(AgentEvent::PreToolExecute {
                id,
                name,
                arguments,
            }) => {
                println!("[Executing Tool] ID: {id}, Name: {name}, Args: {arguments}");
            }
            Ok(AgentEvent::PostToolExecute { id, name, result }) => {
                let result_len = match &result {
                    serde_json::Value::String(s) => s.len(),
                    other => other.to_string().len(),
                };
                println!("[Tool {name} Finished] ID: {id}, Result length: {result_len}");
            }
            Ok(AgentEvent::Finished(history)) => {
                if let Some(content) = history.last().and_then(|m| {
                    let Message::AssistantMessage(assistant) = m else {
                        return None;
                    };
                    assistant.content.as_ref()
                }) {
                    println!("\nFinal Response: {content}");
                }
            }
            Ok(AgentEvent::ToolExecuteError { name, error, .. }) => {
                eprintln!("Tool {name} failed: {error}");
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
            _ => {}
        }
    }

    Ok(())
}
