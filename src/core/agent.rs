use crate::core::messages::{self, Message, Messages, ToolCall, ToolFunction};
use crate::core::tools::{Tool, ToolContext, ToolDefinition, ToolExecute};
use crate::error::{AgentSdkError, Result};
use crate::openai::OpenAI;
use crate::openai::api::types;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::StreamExt;
use o3gen_openai::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageRole,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
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
}

/// Configuration for an [`Agent`], including model parameters, tools,
/// and messages.
#[derive(Clone, Default, Builder)]
#[builder(pattern = "owned", setter(into))]
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

    // ── Messages and tools ────────────────────────────────────────
    #[builder(default)]
    pub messages: Option<Arc<Messages>>,
    #[builder(default)]
    pub tool_definitions: Option<Arc<Vec<ToolDefinition>>>,
    #[builder(default)]
    pub tool_executors: Option<Arc<HashMap<String, ToolExecute>>>,
}

impl fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentOptions")
            .field("model", &self.model)
            .field("max_iterations", &self.max_iterations)
            .field("has_messages", &self.messages.is_some())
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
        let content = (!self.content.is_empty()).then_some(std::mem::take(&mut self.content));

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

    /// Run the agent to completion.
    ///
    /// The agent execution loop:
    /// 1. Send messages to the LLM, receive response chunks.
    /// 2. If the response contains tool calls, execute each one and append
    ///    results to history, then go to step 1.
    /// 3. If the response is text-only, call `handler.on_completion`. If accepted,
    ///    return the final history. If rejected, append a correction and go to step 1.
    ///
    /// Lifecycle events are delivered to the provided `handler`.
    ///
    /// # Errors
    /// Returns an error if the LLM API request fails or if there is a
    /// configuration issue.
    pub async fn run<H: AgentListener>(&self, handler: &mut H) -> Result<Arc<Messages>> {
        let mut history = self
            .options
            .messages
            .as_ref()
            .map(|m| (**m).clone())
            .unwrap_or_default();
        let max_iterations = self
            .options
            .max_iterations
            .unwrap_or(DEFAULT_MAX_ITERATIONS);
        let ctx_options = Arc::new(self.options.clone());

        for _ in 0..max_iterations {
            let sys_injected = match handler.prepare_system_prompt(&history).await {
                Some(sys) => {
                    let content = sys.into_owned();
                    if let Some(Message::SystemMessage(s)) = history.first_mut() {
                        s.content = Some(content);
                        false
                    } else {
                        history.insert(0, messages::system(content));
                        true
                    }
                }
                None => false,
            };

            let mut upstream = self.client.stream_step(&self.options, &history).await?;
            let mut acc = ModelResponseAccumulator::default();

            while let Some(chunk) = upstream.next().await {
                let chunk = chunk?;
                if let Some(msg) = acc.push(&chunk, handler).await {
                    handler.on_model_response_completed(&msg).await;
                    history.push(msg);
                }
            }

            if sys_injected {
                history.remove(0);
            }

            let calls = match history.last() {
                Some(Message::AssistantMessage(a)) => a.tool_calls.clone(),
                _ => None,
            };

            if let Some(calls) = calls {
                for call in calls {
                    let msg = self.handle_tool_call(handler, &call, &ctx_options).await;
                    history.push(msg);
                }
            } else {
                let final_text = match history.last() {
                    Some(Message::AssistantMessage(a)) => a.content.clone().unwrap_or_default(),
                    _ => String::new(),
                };

                match handler.on_completion(final_text).await {
                    CompletionAction::Accept(transformed) => {
                        if let Some(t) = transformed
                            && let Some(Message::AssistantMessage(a)) = history.last_mut()
                        {
                            a.content = Some(t);
                        }
                        return Ok(Arc::new(history));
                    }
                    CompletionAction::Reject { reason } => {
                        history.push(messages::user(format!(
                            "Your previous response was rejected:\n\
                             {reason}\n\nPlease fix and retry."
                        )));
                    }
                }
            }
        }

        Ok(Arc::new(history))
    }

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

    async fn handle_tool_call<H: AgentListener>(
        &self,
        handler: &mut H,
        call: &ToolCall,
        ctx_options: &Arc<AgentOptions>,
    ) -> Message {
        let args = serde_json::from_str::<Value>(&call.function.arguments)
            .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));

        let pre_action = handler
            .on_tool_pre_execute(&call.id, &call.function.name, &args)
            .await;

        let exec_args = match pre_action.resolve(args) {
            Ok(a) => a,
            Err(reason) => return messages::tool(reason, &call.id),
        };

        let result = self
            .execute_tool(&call.function.name, exec_args, ctx_options)
            .await;

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

        messages::tool(content, &call.id)
    }
}
