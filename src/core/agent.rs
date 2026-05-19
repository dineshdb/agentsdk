use crate::core::history::HistoryStore;
use crate::core::messages::{self, Message, Messages, ToolCall, ToolFunction};
use crate::core::tools::{Tool, ToolContext, ToolDefinition, ToolExecute};
use crate::error::{AgentSdkError, Result};
use crate::openai::OpenAI;
use crate::openai::api::types;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::{Stream, StreamExt};
use o3gen_openai::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageRole,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

const DEFAULT_MAX_ITERATIONS: usize = 25;

/// What the agent should do with a final text completion.
///
/// Returned by [`AgentListener::on_completion`] to control whether the agent
/// accepts the output, transforms it, or rejects it and retries.
#[derive(Debug, Clone)]
pub enum CompletionAction {
    /// Accept the completion as-is, or replace it with a transformed version.
    Accept(Option<String>),
    /// Reject the completion and retry. The agent appends the rejected
    /// assistant message and a correction prompt to history, then continues
    /// the loop.
    Reject { reason: String },
}

/// Action to take before executing a tool.
#[derive(Debug, Clone)]
pub enum PreToolAction {
    /// Proceed with execution. Optionally provide transformed arguments.
    Continue(Option<Value>),
    /// Skip execution and return this text to the model as the tool's result.
    /// Useful for safety rejections, caching, or mocks.
    Abort(String),
}

impl PreToolAction {
    pub(crate) fn resolve(self, original: Value) -> std::result::Result<Value, String> {
        match self {
            Self::Continue(transformed) => Ok(transformed.unwrap_or(original)),
            Self::Abort(reason) => Err(reason),
        }
    }
}

/// Action to take after a tool executes successfully.
#[derive(Debug, Clone)]
pub enum PostToolAction {
    /// Use the result. Optionally provide a transformed version.
    Continue(Option<Value>),
    /// Send this string back to the model instead of the result.
    /// Useful for providing feedback or corrections to the model.
    Retry(String),
}

impl PostToolAction {
    pub(crate) fn resolve(self, original: Value) -> String {
        match self {
            Self::Continue(transformed) => {
                let val = transformed.unwrap_or(original);
                if let Value::String(s) = val {
                    s
                } else {
                    val.to_string()
                }
            }
            Self::Retry(feedback) => feedback,
        }
    }
}

/// Action to take when a tool execution fails.
#[derive(Debug, Clone)]
pub enum ToolErrorAction {
    /// Pass the error through to the model. Optionally transform the error message.
    Continue(Option<String>),
    /// Provide a fallback result instead of the error.
    Retry(String),
}

impl ToolErrorAction {
    pub(crate) fn resolve(self, original: String) -> String {
        match self {
            Self::Continue(transformed) => transformed.unwrap_or(original),
            Self::Retry(fallback) => fallback,
        }
    }
}

/// Trait for listening to agent lifecycle events and influencing behavior.
///
/// Implement this trait to receive updates during the agent's execution loop.
/// The methods can return actions to control whether the agent continues,
/// transforms data, or retries with corrections.
#[async_trait]
pub trait AgentListener: Send + Sync {
    /// Fired when a chunk of text is received from the LLM.
    async fn on_text_delta(&mut self, _text: &str) {}

    /// Fired when a full model response (turn) is completed.
    /// This includes responses that contain tool calls.
    async fn on_model_response_completed(&mut self, _msg: &Message) {}

    /// Fired before each model response iteration.
    /// Returns an optional system prompt override for this turn.
    /// Use this for declarative prompting or injecting dynamic context.
    async fn prepare_system_prompt(
        &mut self,
        _history: &Messages,
    ) -> Option<std::borrow::Cow<'static, str>> {
        None
    }

    /// Fired before a tool is executed.
    /// Returns a [`PreToolAction`] to control execution.
    async fn on_tool_pre_execute(
        &mut self,
        _id: &str,
        _name: &str,
        _args: &Value,
    ) -> PreToolAction {
        PreToolAction::Continue(None)
    }

    /// Fired after a tool executes successfully.
    /// Returns a [`PostToolAction`] to control how the result is used.
    async fn on_tool_post_execute(
        &mut self,
        _id: &str,
        _name: &str,
        _result: &Value,
    ) -> PostToolAction {
        PostToolAction::Continue(None)
    }

    /// Fired when a tool execution fails.
    /// Returns a [`ToolErrorAction`] to control how the error is handled.
    async fn on_tool_error(&mut self, _id: &str, _name: &str, _error: &str) -> ToolErrorAction {
        ToolErrorAction::Continue(None)
    }

    /// Fired when the agent produces a final text completion (no tool calls).
    /// Returns a [`CompletionAction`] to accept or reject the completion.
    async fn on_completion(&mut self, _text: String) -> CompletionAction {
        CompletionAction::Accept(None)
    }
    /// Fired when an API or network error occurs.
    /// Returns a [`RetryAction`] to control whether the agent should retry the request.
    async fn on_api_error(&mut self, _error: &AgentSdkError) -> crate::core::retry::RetryAction {
        crate::core::retry::RetryAction::DoNotRetry
    }
}

/// Configuration for an [`Agent`], including model parameters, tools,
/// and messages.
#[derive(Clone, Default, Builder)]
#[builder(pattern = "owned", setter(into, strip_option))]
pub struct AgentOptions {
    // ── Model configuration ───────────────────────────────────────
    #[builder(default)]
    pub extensions: crate::core::extensions::Extensions,
    #[builder(default)]
    pub model: Option<String>,
    #[builder(default)]
    pub temperature: Option<f32>,
    #[builder(default)]
    pub max_tokens: Option<u32>,
    #[builder(default)]
    pub top_p: Option<f32>,
    #[builder(default)]
    pub stop: Option<Vec<String>>,
    #[builder(default)]
    pub max_iterations: Option<usize>,

    // ── History and state management ──────────────────────────────
    #[builder(default)]
    pub history_store: Option<Arc<dyn HistoryStore>>,
    #[builder(default)]
    pub session_id: Option<String>,

    // ── Tools ─────────────────────────────────────────────────────
    #[builder(default)]
    pub tool_definitions: Option<Arc<Vec<ToolDefinition>>>,
    #[builder(default)]
    pub tool_executors: Option<Arc<HashMap<String, ToolExecute>>>,

    // ── Structured Output ─────────────────────────────────────────
    #[builder(default)]
    pub response_schema: Option<schemars::Schema>,
}

impl fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentOptions")
            .field("model", &self.model)
            .field("max_iterations", &self.max_iterations)
            .field("has_history_store", &self.history_store.is_some())
            .field("session_id", &self.session_id)
            .field("has_tool_definitions", &self.tool_definitions.is_some())
            .field("has_tool_executors", &self.tool_executors.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentOptions {
    #[must_use]
    pub fn builder() -> AgentOptionsBuilder {
        AgentOptionsBuilder::default()
    }
}

// ── Model Response Accumulator ───────────────────────────────────────

#[derive(Default)]
struct ModelResponseAccumulator {
    content: String,
    tool_calls: BTreeMap<i64, ToolCall>,
}

impl ModelResponseAccumulator {
    async fn push<H: AgentListener>(
        &mut self,
        chunk: &types::CreateChatCompletionStreamResponse,
        handler: &mut H,
    ) -> Option<Message> {
        let choice = chunk.choices.first()?;
        if let Some(content) = &choice.delta.content {
            self.content.push_str(content);
            handler.on_text_delta(content).await;
        }

        if let Some(deltas) = &choice.delta.tool_calls {
            for delta in deltas {
                let entry = self
                    .tool_calls
                    .entry(delta.index)
                    .or_insert_with(|| ToolCall {
                        id: String::new(),
                        r#type: types::ToolCallType::Function,
                        function: ToolFunction {
                            name: String::new(),
                            arguments: String::new(),
                        },
                    });

                if let Some(id) = &delta.id {
                    entry.id.clone_from(id);
                }

                if let Some(f) = &delta.function {
                    if let Some(name) = &f.name {
                        entry.function.name.clone_from(name);
                    }
                    if let Some(args) = &f.arguments {
                        entry.function.arguments.push_str(args);
                    }
                }
            }
        }

        if choice.finish_reason.is_some() {
            Some(self.finish())
        } else {
            None
        }
    }

    fn finish(&mut self) -> Message {
        let tool_calls = if self.tool_calls.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.tool_calls).into_values().collect())
        };

        // Ensure we always have content if there are no tool calls
        let content = if tool_calls.is_none() && self.content.is_empty() {
            Some(String::new())
        } else if !self.content.is_empty() {
            Some(std::mem::take(&mut self.content))
        } else {
            None
        };

        Message::AssistantMessage(ChatCompletionRequestAssistantMessage {
            content,
            name: None,
            tool_calls,
            role: ChatCompletionRequestAssistantMessageRole::Assistant,
            function_call: None,
        })
    }
}

// ── Builder ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct AgentBuilder {
    client: Option<OpenAI>,
    options: AgentOptions,
}

impl AgentBuilder {
    #[must_use]
    pub fn client(mut self, client: OpenAI) -> Self {
        self.client = Some(client);
        self
    }

    #[must_use]
    pub fn options(mut self, options: AgentOptions) -> Self {
        self.options = options;
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build(self) -> Result<Agent> {
        let client = self
            .client
            .ok_or_else(|| AgentSdkError::ConfigError("Client required".into()))?;
        Ok(Agent {
            client,
            options: self.options,
        })
    }
}

impl AgentOptionsBuilder {
    #[must_use]
    pub fn with_tool(mut self, tool: &Tool) -> Self {
        let mut defs = self
            .tool_definitions
            .take()
            .flatten()
            .map_or_else(Vec::new, |arc| {
                Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
            });
        let mut execs = self
            .tool_executors
            .take()
            .flatten()
            .map_or_else(HashMap::new, |arc| {
                Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
            });

        defs.push(tool.definition.clone());
        execs.insert(tool.definition.name.clone(), tool.execute.clone());

        self.tool_definitions = Some(Some(Arc::new(defs)));
        self.tool_executors = Some(Some(Arc::new(execs)));
        self
    }
}

// ── Agent ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Agent {
    client: OpenAI,
    options: AgentOptions,
}

// ── Agent implementation ──────────────────────────────────────────────

impl Agent {
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }
    /// Run the agent to completion and extract a structured JSON response.
    ///
    /// This method is similar to [`run`](Self::run), but it ensures the final
    /// response conforms to the schema of type `T`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The agent loop fails
    /// - History store is not configured
    /// - Session ID is missing
    /// - The final response cannot be parsed into `T`
    #[tracing::instrument(skip(self, handler), fields(model = %self.client.config.model))]
    pub async fn run_json<T, H>(&self, handler: &mut H) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
        H: AgentListener,
    {
        // First, run the normal agent loop to handle tool calls and reasoning.
        self.run(handler).await?;

        // Now, perform a final turn to extract the structured data.
        let store = self
            .options
            .history_store
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("History store required".into()))?;
        let session_id = self
            .options
            .session_id
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("Session ID required".into()))?;

        let mut history = store.load(session_id).await?;
        self.prepare_prompt(handler, &mut history).await;

        let schema = schemars::schema_for!(T);
        let schema_val = serde_json::to_value(schema)?;

        let val = self
            .client
            .get_json(&self.options, &history, &schema_val)
            .await?;

        let result: T = serde_json::from_value(val)?;
        Ok(result)
    }

    /// Run the agent to completion.
    ///
    /// The agent execution loop:
    /// 1. Load history from `history_store`.
    /// 2. Send messages to the LLM, receive response chunks.
    /// 3. If a full assistant message is received, push it to `history_store`.
    /// 4. If the response contains tool calls, execute each one and push
    ///    results to `history_store`, then go to step 1.
    /// 5. If the response is text-only, call `handler.on_completion`. If accepted,
    ///    return. If rejected, push a correction to `history_store` and go to step 1.
    ///
    /// Lifecycle events are delivered to the provided `handler`.
    ///
    /// # Errors
    /// Returns an error if the LLM API request fails, if history cannot be
    /// loaded/appended, or if there is a configuration issue.
    #[tracing::instrument(skip(self, handler), fields(model = %self.client.config.model))]
    pub async fn run<H: AgentListener>(&self, handler: &mut H) -> Result<()> {
        let store = self
            .options
            .history_store
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("History store required".into()))?;
        let session_id = self
            .options
            .session_id
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("Session ID required".into()))?;

        let max_iterations = self
            .options
            .max_iterations
            .unwrap_or(DEFAULT_MAX_ITERATIONS);
        let ctx_options = Arc::new(self.options.clone());

        for _ in 0..max_iterations {
            let mut history = store.load(session_id).await?;
            self.prepare_prompt(handler, &mut history).await;

            let mut upstream = self.stream_step_with_retry(handler, &history).await?;
            let mut acc = ModelResponseAccumulator::default();
            let mut assistant_msg = None;

            while let Some(chunk) = upstream.next().await {
                if let Some(msg) = acc.push(&chunk?, handler).await {
                    handler.on_model_response_completed(&msg).await;
                    store.push(session_id, msg.clone()).await?;
                    assistant_msg = Some(msg);
                }
            }

            let Some(Message::AssistantMessage(a)) = assistant_msg else {
                break;
            };

            if let Some(calls) = a.tool_calls {
                let msgs = self
                    .execute_parallel_tool_calls(handler, &calls, &ctx_options)
                    .await;
                for msg in msgs {
                    store.push(session_id, msg).await?;
                }
            } else {
                let final_text = a.content.unwrap_or_default();

                match handler.on_completion(final_text).await {
                    CompletionAction::Accept(_) => return Ok(()),
                    CompletionAction::Reject { reason } => {
                        store
                            .push(
                                session_id,
                                messages::user(format!(
                                    "Your previous response was rejected:\n\
                                     {reason}\n\nPlease fix and retry."
                                )),
                            )
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn prepare_prompt<H: AgentListener>(&self, handler: &mut H, history: &mut Messages) {
        if let Some(sys) = handler.prepare_system_prompt(history).await {
            let content = sys.into_owned();
            if let Some(Message::SystemMessage(s)) = history.first_mut() {
                s.content = Some(content);
            } else {
                history.insert(0, messages::system(content));
            }
        }
    }

    async fn stream_step_with_retry<H: AgentListener>(
        &self,
        handler: &mut H,
        history: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<types::CreateChatCompletionStreamResponse>> + Send>>>
    {
        loop {
            match self.client.stream_step(&self.options, history).await {
                Ok(stream) => return Ok(stream),
                Err(e) => match handler.on_api_error(&e).await {
                    crate::core::retry::RetryAction::Retry(delay) => {
                        tracing::warn!(error = %e, "API call failed, retrying in {:?}", delay);
                        tokio::time::sleep(delay).await;
                    }
                    crate::core::retry::RetryAction::DoNotRetry => return Err(e),
                },
            }
        }
    }

    #[tracing::instrument(skip(self, handler, calls, ctx_options), fields(tools_count = calls.len()))]
    async fn execute_parallel_tool_calls<H: AgentListener>(
        &self,
        handler: &mut H,
        calls: &[ToolCall],
        ctx_options: &Arc<AgentOptions>,
    ) -> Vec<Message> {
        let mut pre_results = Vec::new();
        let mut futures = Vec::new();

        for call in calls {
            let args = serde_json::from_str::<Value>(&call.function.arguments)
                .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));

            let pre_action = handler
                .on_tool_pre_execute(&call.id, &call.function.name, &args)
                .await;

            match pre_action.resolve(args) {
                Ok(exec_args) => {
                    futures.push(self.execute_tool(&call.function.name, exec_args, ctx_options));
                    pre_results.push(None);
                }
                Err(reason) => {
                    pre_results.push(Some(messages::tool(reason, &call.id)));
                }
            }
        }

        let exec_results = futures::future::join_all(futures).await;
        let mut exec_iter = exec_results.into_iter();
        let mut messages = Vec::with_capacity(calls.len());

        for (call, pre_res) in calls.iter().zip(pre_results) {
            if let Some(msg) = pre_res {
                messages.push(msg);
            } else if let Some(result) = exec_iter.next() {
                let content = match result {
                    Ok(res) => handler
                        .on_tool_post_execute(&call.id, &call.function.name, &res)
                        .await
                        .resolve(res),
                    Err(err) => handler
                        .on_tool_error(&call.id, &call.function.name, &err)
                        .await
                        .resolve(err),
                };
                messages.push(messages::tool(content, &call.id));
            }
        }

        messages
    }

    #[tracing::instrument(skip(self, args, ctx_options), fields(tool = %name))]
    async fn execute_tool(
        &self,
        name: &str,
        args: Value,
        ctx_options: &Arc<AgentOptions>,
    ) -> std::result::Result<Value, String> {
        let Some(executor) = self
            .options
            .tool_executors
            .as_ref()
            .and_then(|m| m.get(name))
        else {
            return Err(format!("Tool {name} not found"));
        };

        let ctx = ToolContext {
            options: Arc::clone(ctx_options),
        };

        executor.call(ctx, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockListener;
    #[async_trait]
    impl AgentListener for MockListener {}

    #[tokio::test]
    async fn test_execute_parallel_tool_calls_handles_errors() -> Result<()> {
        let config = crate::ModelConfig {
            base_url: "test".into(),
            api_key: "test".into(),
            model: "test".into(),
        };
        let client = OpenAI::new(config);

        // Setup tools: one succeeds, one fails, one missing
        let mut tool_executors = HashMap::new();
        tool_executors.insert(
            "success".to_string(),
            ToolExecute::from_sync(|_, _| Ok(serde_json::json!("ok"))),
        );
        tool_executors.insert(
            "fail".to_string(),
            ToolExecute::from_sync(|_, _| Err("error".to_string())),
        );

        let options = AgentOptions::builder()
            .tool_executors(Arc::new(tool_executors))
            .build()
            .map_err(|e| AgentSdkError::ConfigError(e.to_string()))?;

        let agent = Agent::builder().client(client).options(options).build()?;

        let calls = vec![
            ToolCall {
                id: "1".into(),
                r#type: types::ToolCallType::Function,
                function: ToolFunction {
                    name: "success".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: "2".into(),
                r#type: types::ToolCallType::Function,
                function: ToolFunction {
                    name: "fail".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: "3".into(),
                r#type: types::ToolCallType::Function,
                function: ToolFunction {
                    name: "missing".into(),
                    arguments: "{}".into(),
                },
            },
        ];

        let mut handler = MockListener;
        let ctx_options = Arc::new(agent.options.clone());
        let messages = agent
            .execute_parallel_tool_calls(&mut handler, &calls, &ctx_options)
            .await;

        assert_eq!(messages.len(), 3);

        // Success
        match messages.first() {
            Some(Message::ToolMessage(m)) => {
                assert_eq!(m.tool_call_id, "1");
                assert_eq!(m.content, Some("ok".into()));
            }
            _ => return Err(AgentSdkError::ConfigError("Expected ToolMessage".into())),
        }

        // Fail
        match messages.get(1) {
            Some(Message::ToolMessage(m)) => {
                assert_eq!(m.tool_call_id, "2");
                assert_eq!(m.content, Some("error".into()));
            }
            _ => return Err(AgentSdkError::ConfigError("Expected ToolMessage".into())),
        }

        // Missing
        match messages.get(2) {
            Some(Message::ToolMessage(m)) => {
                assert_eq!(m.tool_call_id, "3");
                assert_eq!(m.content, Some("Tool missing not found".into()));
            }
            _ => return Err(AgentSdkError::ConfigError("Expected ToolMessage".into())),
        }

        Ok(())
    }
}
