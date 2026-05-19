use agentsdk::core::retry::RetryAction;
use agentsdk::error::AgentSdkError;
use agentsdk::{AgentPlugin, Messages, PluginContext};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use crate::config::PROMPT;

#[derive(Default, Debug)]
pub struct GitMetrics {
    pub total_errors: AtomicU32,
    pub rate_limit_errors: AtomicU32,
}

pub struct GitHandler {
    pub metrics: Arc<GitMetrics>,
}

impl GitHandler {
    pub fn new(metrics: Arc<GitMetrics>) -> Self {
        Self { metrics }
    }
}

#[async_trait]
impl AgentPlugin for GitHandler {
    fn name(&self) -> &'static str {
        "git_handler"
    }

    async fn prepare_system_prompt(
        &mut self,
        _ctx: &PluginContext,
        _history: &Messages,
    ) -> Option<std::borrow::Cow<'static, str>> {
        Some(std::borrow::Cow::Borrowed(PROMPT))
    }

    async fn on_text_delta(&mut self, _ctx: &PluginContext, text: &str) {
        print!("{text}");
    }

    async fn on_tool_pre_execute(
        &mut self,
        _ctx: &PluginContext,
        id: &str,
        name: &str,
        arguments: &serde_json::Value,
    ) -> agentsdk::PreToolAction {
        println!("\n[Executing Tool] ID: {id}, Name: {name}, Args: {arguments}");
        agentsdk::PreToolAction::Continue(None)
    }

    async fn on_tool_post_execute(
        &mut self,
        _ctx: &PluginContext,
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
        _ctx: &PluginContext,
        _id: &str,
        name: &str,
        error: &str,
    ) -> agentsdk::ToolErrorAction {
        eprintln!("Tool {name} failed: {error}");
        agentsdk::ToolErrorAction::Continue(None)
    }

    async fn on_api_error(&mut self, _ctx: &PluginContext, error: &AgentSdkError) -> RetryAction {
        let metrics = &self.metrics;
        metrics.total_errors.fetch_add(1, Ordering::Relaxed);

        if let Some(status) = error.status_code() {
            if status == 429 {
                metrics.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
                if metrics.rate_limit_errors.load(Ordering::Relaxed) > 5 {
                    println!("Too many rate limits, aborting");
                    return RetryAction::DoNotRetry;
                }
                return RetryAction::Retry(Duration::from_secs(2));
            }

            if status.is_server_error() {
                if metrics.total_errors.load(Ordering::Relaxed) > 10 {
                    println!("Too many total errors, aborting");
                    return RetryAction::DoNotRetry;
                }
                return RetryAction::Retry(Duration::from_secs(10));
            }
        }

        RetryAction::DoNotRetry
    }
}
