use crate::core::history::History;
use crate::core::messages::{self, Message, Messages, ToolCall, ToolFunction};
use crate::core::plugin::{AgentPlugin, PluginContext};
use crate::core::retry::RetryAction;
use crate::core::tools::{Tool, ToolContext, ToolDefinition, ToolExecute};
use crate::error::{AgentSdkError, Result};
use crate::openai::OpenAI;
use derive_builder::Builder;
use futures::{Stream, StreamExt};
use o3gen_openai::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageRole,
    CreateChatCompletionStreamResponse,
};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

const DEFAULT_MAX_ITERATIONS: usize = 25;

// ── Action enums ──────────────────────────────────────────────────────

/// What the agent should do with a final text completion.
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

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for an [`Agent`], including model parameters, tools,
/// and messages.
#[derive(Clone, Default, Builder)]
#[builder(pattern = "owned", setter(into, strip_option))]
pub struct AgentOptions {
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

// ── Model Response Accumulator ────────────────────────────────────────

#[derive(Default)]
pub(crate) struct ModelResponseAccumulator {
    content: String,
    tool_calls: BTreeMap<i64, ToolCall>,
}

impl ModelResponseAccumulator {
    async fn push(
        &mut self,
        chunk: &CreateChatCompletionStreamResponse,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
    ) -> Option<Message> {
        let choice = chunk.choices.first()?;
        if let Some(content) = &choice.delta.content {
            self.content.push_str(content);
            for p in plugins.iter_mut() {
                p.on_text_delta(ctx, content).await;
            }
        }

        if let Some(deltas) = &choice.delta.tool_calls {
            for delta in deltas {
                let entry = self
                    .tool_calls
                    .entry(delta.index)
                    .or_insert_with(|| ToolCall {
                        id: String::new(),
                        r#type: o3gen_openai::types::ToolCallType::Function,
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

        let content = if self.content.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.content))
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

// ── Agent Run Output ──────────────────────────────────────────────────

/// Output of a completed [`Agent::run`] call.
///
/// Contains the [`hecs::World`] with the agent entity and all components
/// that plugins wrote during execution.  Useful for inspecting state
/// (e.g. reading [`History`]) after the agent finishes.
pub struct AgentRunOutput {
    pub world: hecs::World,
    pub entity: hecs::Entity,
}

impl fmt::Debug for AgentRunOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentRunOutput")
            .field("entity", &self.entity)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
pub struct AgentBuilder {
    client: Option<OpenAI>,
    options: AgentOptions,
    plugins: Vec<Box<dyn AgentPlugin>>,
}

impl fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("client", &self.client)
            .field("options", &self.options)
            .field("plugins", &format!("[{} plugins]", self.plugins.len()))
            .finish()
    }
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

    #[must_use]
    pub fn plugin<P: AgentPlugin + 'static>(mut self, plugin: P) -> Self {
        self.plugins.push(Box::new(plugin));
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
            plugins: self.plugins,
        })
    }
}

// ── Agent ─────────────────────────────────────────────────────────────

pub struct Agent {
    client: OpenAI,
    options: AgentOptions,
    plugins: Vec<Box<dyn AgentPlugin>>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("client", &self.client)
            .field("options", &self.options)
            .field("plugins", &format!("[{} plugins]", self.plugins.len()))
            .finish()
    }
}

// ── Dispatch helpers ──────────────────────────────────────────────────

impl Agent {
    async fn dispatch_model_response_completed(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        msg: &Message,
    ) {
        for p in plugins.iter_mut() {
            p.on_model_response_completed(ctx, msg).await;
        }
    }

    async fn dispatch_prepare_system_prompt(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        history: &Messages,
    ) -> Option<Cow<'static, str>> {
        let mut parts: Vec<String> = Vec::new();
        for p in plugins.iter_mut() {
            if let Some(prompt) = p.prepare_system_prompt(ctx, history).await {
                parts.push(prompt.into_owned());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(Cow::Owned(parts.join("\n\n")))
        }
    }

    async fn dispatch_tool_pre_execute(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        id: &str,
        name: &str,
        args: &Value,
    ) -> PreToolAction {
        for p in plugins.iter_mut() {
            match p.on_tool_pre_execute(ctx, id, name, args).await {
                PreToolAction::Continue(None) => {}
                decisive => return decisive,
            }
        }
        PreToolAction::Continue(None)
    }

    async fn dispatch_tool_post_execute(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        id: &str,
        name: &str,
        result: &Value,
    ) -> PostToolAction {
        for p in plugins.iter_mut() {
            match p.on_tool_post_execute(ctx, id, name, result).await {
                PostToolAction::Continue(None) => {}
                decisive => return decisive,
            }
        }
        PostToolAction::Continue(None)
    }

    async fn dispatch_tool_error(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        id: &str,
        name: &str,
        error: &str,
    ) -> ToolErrorAction {
        for p in plugins.iter_mut() {
            match p.on_tool_error(ctx, id, name, error).await {
                ToolErrorAction::Continue(None) => {}
                decisive => return decisive,
            }
        }
        ToolErrorAction::Continue(None)
    }

    async fn dispatch_api_error(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        error: &AgentSdkError,
    ) -> RetryAction {
        for p in plugins.iter_mut() {
            match p.on_api_error(ctx, error).await {
                RetryAction::DoNotRetry => {}
                decisive @ RetryAction::Retry(_) => return decisive,
            }
        }
        RetryAction::DoNotRetry
    }
}

// ── Agent implementation ──────────────────────────────────────────────

impl Agent {
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Run the agent to completion and return the final [`AgentRunOutput`],
    /// which contains the full [`hecs::World`] with all components.
    ///
    /// # Errors
    /// Returns an error if the LLM API call fails and no plugin handles it.
    #[tracing::instrument(skip(self), fields(model = %self.client.config.model))]
    pub async fn run(&mut self) -> Result<AgentRunOutput> {
        // Borrow fields individually to avoid borrow-conflicts with `self` inside the loop.
        let client = &self.client;
        let options = &self.options;
        let plugins = &mut self.plugins;

        let mut world = hecs::World::new();
        let entity = world.spawn((History::default(),));
        let mut ctx = PluginContext { world, entity };

        for p in plugins.iter_mut() {
            p.init(&mut ctx).await;
        }

        let max_iterations = options.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);

        for _ in 0..max_iterations {
            let mut history: Messages = ctx
                .get_mut::<History>()
                .map(|mut h| std::mem::take(&mut h.0))
                .unwrap_or_default();

            Self::prepare_prompt(plugins, &ctx, &mut history).await;

            let mut upstream =
                Self::stream_with_retry(client, options, plugins, &ctx, &history).await?;

            // Return history to the component — no clone needed, we still own it.
            if let Some(mut h) = ctx.get_mut::<History>() {
                h.0 = history;
            }
            let mut acc = ModelResponseAccumulator::default();
            let mut assistant_msg = None;

            while let Some(chunk) = upstream.next().await {
                if let Some(msg) = acc.push(&chunk?, plugins, &ctx).await {
                    Self::dispatch_model_response_completed(plugins, &ctx, &msg).await;

                    // Append assistant message to History component
                    if let Some(mut h) = ctx.get_mut::<History>() {
                        h.0.push(msg.clone());
                    }

                    assistant_msg = Some(msg);
                }
            }

            let Some(Message::AssistantMessage(a)) = assistant_msg else {
                break;
            };

            if let Some(calls) = a.tool_calls {
                let msgs =
                    Self::execute_parallel_tool_calls(options, plugins, &mut ctx, &calls).await;

                // Append tool result messages to History component
                if let Some(mut h) = ctx.get_mut::<History>() {
                    for msg in msgs {
                        h.0.push(msg);
                    }
                }
            } else {
                let final_text = a.content.unwrap_or_default();

                let mut action = CompletionAction::Accept(None);
                for p in plugins.iter_mut() {
                    match p.on_completion(&ctx, final_text.clone()).await {
                        CompletionAction::Accept(None) => {}
                        decisive => {
                            action = decisive;
                            break;
                        }
                    }
                }

                match action {
                    CompletionAction::Accept(_) => {
                        break;
                    }
                    CompletionAction::Reject { reason } => {
                        let correction = messages::user(format!(
                            "Your previous response was rejected:\n\
                             {reason}\n\nPlease fix and retry."
                        ));
                        if let Some(mut h) = ctx.get_mut::<History>() {
                            h.0.push(correction);
                        }
                    }
                }
            }
        }

        for p in plugins.iter_mut() {
            p.shutdown(&mut ctx).await;
        }

        Ok(AgentRunOutput {
            world: ctx.world,
            entity: ctx.entity,
        })
    }

    /// Run the agent to completion and extract a structured JSON response.
    ///
    /// This is a convenience wrapper around [`run`](Self::run) that performs
    /// an extra LLM call with a response schema after the agent loop finishes.
    ///
    /// # Errors
    /// Returns an error if the agent loop fails or JSON deserialization fails.
    #[tracing::instrument(skip(self), fields(model = %self.client.config.model))]
    pub async fn run_json<T>(&mut self) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let output = self.run().await?;

        let history: Messages = output
            .world
            .get::<&History>(output.entity)
            .map(|h| h.0.clone())
            .unwrap_or_default();

        let schema = schemars::schema_for!(T);
        let schema_val = serde_json::to_value(schema)?;

        let val = self
            .client
            .get_json(&self.options, &history, &schema_val)
            .await?;

        let result: T = serde_json::from_value(val)?;
        Ok(result)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    async fn prepare_prompt(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        history: &mut Messages,
    ) {
        if let Some(sys) = Self::dispatch_prepare_system_prompt(plugins, ctx, history).await {
            let content = sys.into_owned();
            if let Some(Message::SystemMessage(s)) = history.first_mut() {
                s.content = Some(content);
            } else {
                history.insert(0, messages::system(content));
            }
        }
    }

    async fn stream_with_retry(
        client: &OpenAI,
        options: &AgentOptions,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &PluginContext,
        history: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse>> + Send>>>
    {
        loop {
            match client.stream(options, history).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    let action = Self::dispatch_api_error(plugins, ctx, &e).await;
                    match action {
                        RetryAction::Retry(delay) => {
                            tracing::warn!(error = %e, "API call failed, retrying in {:?}", delay);
                            tokio::time::sleep(delay).await;
                        }
                        RetryAction::DoNotRetry => return Err(e),
                    }
                }
            }
        }
    }

    #[tracing::instrument(skip(plugins, ctx, calls), fields(tools_count = calls.len()))]
    async fn execute_parallel_tool_calls(
        options: &AgentOptions,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        calls: &[ToolCall],
    ) -> Vec<Message> {
        let options_arc = Arc::new(options.clone());
        let mut pre_results = Vec::new();
        let mut futures = Vec::new();

        for call in calls {
            let args = serde_json::from_str::<Value>(&call.function.arguments)
                .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));

            let pre_action =
                Self::dispatch_tool_pre_execute(plugins, ctx, &call.id, &call.function.name, &args)
                    .await;

            match pre_action.resolve(args) {
                Ok(exec_args) => {
                    futures.push(Self::execute_tool(
                        &options_arc,
                        &call.function.name,
                        exec_args,
                    ));
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
                    Ok(res) => Self::dispatch_tool_post_execute(
                        plugins,
                        ctx,
                        &call.id,
                        &call.function.name,
                        &res,
                    )
                    .await
                    .resolve(res),
                    Err(err) => {
                        Self::dispatch_tool_error(plugins, ctx, &call.id, &call.function.name, &err)
                            .await
                            .resolve(err)
                    }
                };
                messages.push(messages::tool(content, &call.id));
            }
        }

        messages
    }

    #[tracing::instrument(skip(args), fields(tool = %name))]
    async fn execute_tool(
        options: &Arc<AgentOptions>,
        name: &str,
        args: Value,
    ) -> std::result::Result<Value, String> {
        let Some(executor) = options.tool_executors.as_ref().and_then(|m| m.get(name)) else {
            return Err(format!("Tool {name} not found"));
        };

        let ctx = ToolContext {
            options: Arc::clone(options),
        };

        executor.call(ctx, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use o3gen_openai::ToolCallType;

    struct NoopPlugin;

    #[async_trait]
    impl AgentPlugin for NoopPlugin {
        fn name(&self) -> &'static str {
            "noop"
        }
    }

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

        // The old test called execute_parallel_tool_calls directly.
        // With the new dispatch API we simulate the same world/plugin setup.
        let mut world = hecs::World::new();
        let entity = world.spawn((History::default(),));
        let mut ctx = PluginContext { world, entity };

        let calls = vec![
            ToolCall {
                id: "1".into(),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "success".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: "2".into(),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "fail".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: "3".into(),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "missing".into(),
                    arguments: "{}".into(),
                },
            },
        ];

        let mut plugins: Vec<Box<dyn AgentPlugin>> = vec![Box::new(NoopPlugin)];
        let messages =
            Agent::execute_parallel_tool_calls(&agent.options, &mut plugins, &mut ctx, &calls)
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
