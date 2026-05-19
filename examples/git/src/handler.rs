use agentsdk::core::retry::RetryAction;
use agentsdk::error::AgentSdkError;
use agentsdk::{AgentListener, Messages};
use async_trait::async_trait;
use std::time::Duration;

pub const PROMPT: &str = r"
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

pub struct GitHandler {
    pub total_errors: u32,
    pub rate_limit_errors: u32,
}

impl GitHandler {
    pub fn new() -> Self {
        Self {
            total_errors: 0,
            rate_limit_errors: 0,
        }
    }
}

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

    async fn on_api_error(&mut self, error: &AgentSdkError) -> RetryAction {
        self.total_errors += 1;

        if let Some(status) = error.status_code() {
            if status == 429 {
                self.rate_limit_errors += 1;
                if self.rate_limit_errors > 5 {
                    println!("Too many rate limits, aborting");
                    return RetryAction::DoNotRetry;
                }
                // Fast backoff for 429
                return RetryAction::Retry(Duration::from_secs(2));
            }

            if status.is_server_error() {
                if self.total_errors > 10 {
                    println!("Too many total errors, aborting");
                    return RetryAction::DoNotRetry;
                }
                // Slow fixed backoff for server errors
                return RetryAction::Retry(Duration::from_secs(10));
            }
        }

        RetryAction::DoNotRetry
    }
}
