use crate::core::history::History;
use crate::core::messages::{self, Message, ToolCall, ToolFunction};
use crate::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use crate::core::retry::RetryAction;
use crate::core::tools::{Tool, ToolContext, ToolDefinition, ToolExecute};
use crate::error::{AgentSdkError, Result};
use crate::openai::OpenAI;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::{FutureExt, Stream, StreamExt};
use o3gen_openai::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageRole,
    CreateChatCompletionStreamResponse,
};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_MAX_ITERATIONS: usize = 250;
const PLUGIN_HOOK_TIMEOUT: Duration = Duration::from_secs(5);
const PLUGIN_INIT_TIMEOUT: Duration = Duration::from_secs(10);

async fn plugin_hook<F, T>(name: &'static str, hook: &str, fut: F) -> Option<T>
where
    F: Future<Output = T>,
{
    let result =
        tokio::time::timeout(PLUGIN_HOOK_TIMEOUT, AssertUnwindSafe(fut).catch_unwind()).await;
    match result {
        Ok(Ok(val)) => Some(val),
        Ok(Err(e)) => {
            tracing::error!(plugin = name, "{hook} panicked: {e:?}");
            None
        }
        Err(_) => {
            tracing::error!(plugin = name, "{hook} timed out");
            None
        }
    }
}

// ── LLM Backend trait ────────────────────────────────────────────────

/// A pluggable LLM backend.
#[async_trait]
pub trait LLMBackend: Send + Sync {
    async fn stream(
        &self,
        options: &AgentOptions,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse>> + Send>>>;

    async fn get_json(
        &self,
        options: &AgentOptions,
        messages: &[Message],
        schema: &Value,
    ) -> Result<Value>;
}

#[async_trait]
impl LLMBackend for OpenAI {
    async fn stream(
        &self,
        options: &AgentOptions,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse>> + Send>>>
    {
        OpenAI::stream(self, options, messages).await
    }

    async fn get_json(
        &self,
        options: &AgentOptions,
        messages: &[Message],
        schema: &Value,
    ) -> Result<Value> {
        OpenAI::get_json(self, options, messages, schema).await
    }
}

/// A closure that injects a component into the [`hecs::World`].
type ComponentInjector = Box<dyn FnOnce(&mut hecs::World, hecs::Entity) + Send + Sync>;

// ── Action enums ──────────────────────────────────────────────────────

/// What the agent should do with a final text completion.
#[derive(Debug, Clone)]
pub enum CompletionAction {
    /// Accept the completion as-is.
    Accept,
    /// Reject the completion and retry. The agent appends the rejected
    /// assistant message and a correction prompt to history, then continues
    /// the loop.
    Reject { reason: String },
}

/// Action to take before executing a tool.
#[derive(Debug, Clone)]
pub enum PreToolAction {
    /// Proceed with execution. Optionally provide transformed arguments.
    Proceed(Option<Value>),
    /// Skip execution and return this text to the model as the tool's result.
    Abort(String),
    /// Stop the agent entirely with an exit reason.
    Stop(String),
}

/// Action to take after a tool executes (success or failure).
#[derive(Debug, Clone)]
pub enum PostToolAction {
    /// Use the result. Optionally provide a transformed version.
    /// - On `Ok`: `None` uses the tool's return value; `Some(v)` uses `v` instead.
    /// - On `Err`: `None` passes the error message through; `Some(v)` provides
    ///   a fallback result.
    Proceed(Option<Value>),
    /// Send this string back to the model instead of the result/error.
    Override(String),
}

impl PostToolAction {
    pub(crate) fn resolve(self, original: std::result::Result<Value, String>) -> String {
        match self {
            Self::Proceed(transformed) => match transformed {
                Some(val) => {
                    if let Value::String(s) = val {
                        s
                    } else {
                        val.to_string()
                    }
                }
                None => match original {
                    Ok(val) => {
                        if let Value::String(s) = val {
                            s
                        } else {
                            val.to_string()
                        }
                    }
                    Err(err) => err,
                },
            },
            Self::Override(feedback) => feedback,
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
        if let Some(Some(arc)) = &mut self.tool_definitions {
            Arc::make_mut(arc).push(tool.definition.clone());
        } else {
            self.tool_definitions = Some(Some(Arc::new(vec![tool.definition.clone()])));
        }

        if let Some(Some(arc)) = &mut self.tool_executors {
            Arc::make_mut(arc).insert(tool.definition.name.clone(), tool.execute.clone());
        } else {
            let mut h = HashMap::new();
            h.insert(tool.definition.name.clone(), tool.execute.clone());
            self.tool_executors = Some(Some(Arc::new(h)));
        }

        self
    }
}

// ── Model Response Accumulator ────────────────────────────────────────

#[derive(Default)]
pub(crate) struct ModelResponseAccumulator {
    content: String,
    tool_calls: BTreeMap<i64, ToolCall>,
    /// A response finishes exactly once: some providers (e.g.
    /// `OpenRouter`)
    /// emit multiple chunks carrying `finish_reason` (a terminal
    /// usage/stats chunk repeats it). Finishing twice would emit a
    /// second, empty assistant message that overwrites the real one.
    finished: bool,
}

impl ModelResponseAccumulator {
    fn push(
        &mut self,
        chunk: &CreateChatCompletionStreamResponse,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
    ) -> Option<Message> {
        if self.finished {
            return None;
        }
        let choice = chunk.choices.first()?;
        if let Some(content) = &choice.delta.content {
            self.content.push_str(content);
            for p in plugins.iter_mut() {
                let name = p.name();
                if let Err(e) = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    p.on_text_delta(ctx, content);
                })) {
                    tracing::error!(plugin = name, "on_text_delta panicked: {:?}", e);
                }
            }
        }

        if let Some(deltas) = &choice.delta.tool_calls {
            for delta in deltas {
                let entry = self
                    .tool_calls
                    .entry(delta.index)
                    .or_insert_with(|| ToolCall {
                        id: Some(String::new()),
                        r#type: o3gen_openai::types::ToolCallType::Function,
                        function: ToolFunction {
                            name: String::new(),
                            arguments: String::new(),
                        },
                    });

                if let Some(id) = &delta.id {
                    entry.id = Some(id.clone());
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
            self.finished = true;
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
pub struct Agent<T: LLMBackend = OpenAI> {
    backend: T,
    options: Arc<AgentOptions>,
    plugins: Vec<Box<dyn AgentPlugin>>,
    tool_plugin: HashMap<String, usize>,
    pub world: Option<hecs::World>,
    pub entity: Option<hecs::Entity>,
}

pub struct AgentBuilder<T: LLMBackend = OpenAI> {
    backend: Option<T>,
    options: AgentOptions,
    plugins: Vec<Box<dyn AgentPlugin>>,
    component_injectors: Vec<ComponentInjector>,
}

impl<T: LLMBackend> Default for AgentBuilder<T> {
    fn default() -> Self {
        Self {
            backend: None,
            options: AgentOptions::default(),
            plugins: Vec::new(),
            component_injectors: Vec::new(),
        }
    }
}

impl<T: LLMBackend> fmt::Debug for AgentBuilder<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("options", &self.options)
            .field("plugins", &format!("[{} plugins]", self.plugins.len()))
            .finish_non_exhaustive()
    }
}

impl<T: LLMBackend> AgentBuilder<T> {
    #[must_use]
    pub fn client(mut self, backend: T) -> Self {
        self.backend = Some(backend);
        self
    }

    #[must_use]
    pub fn options(mut self, options: AgentOptions) -> Self {
        self.options = options;
        self
    }

    #[must_use]
    pub fn component<C: Send + Sync + 'static>(mut self, component: C) -> Self {
        self.component_injectors
            .push(Box::new(move |world, entity| {
                let _ = world.insert_one(entity, component);
            }));
        self
    }

    #[must_use]
    pub fn plugin<P: AgentPlugin + 'static>(mut self, plugin: P) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build(mut self) -> Result<Agent<T>> {
        let backend = self.backend.ok_or_else(|| {
            AgentSdkError::ConfigError(
                "No LLM backend registered. Use .client() to register one.".into(),
            )
        })?;

        let mut options = self.options;
        let mut plugin_tool_map = HashMap::new();

        for (i, plugin) in self.plugins.iter().enumerate() {
            for def in plugin.tools() {
                plugin_tool_map.insert(def.name.clone(), i);
                options = Self::add_tool_definition(options, def);
            }
        }

        let mut world = hecs::World::new();
        let entity = world.spawn((History::default(),));

        for injector in std::mem::take(&mut self.component_injectors) {
            injector(&mut world, entity);
        }

        Ok(Agent {
            backend,
            options: Arc::new(options),
            plugins: self.plugins,
            tool_plugin: plugin_tool_map,
            world: Some(world),
            entity: Some(entity),
        })
    }

    fn add_tool_definition(mut options: AgentOptions, def: ToolDefinition) -> AgentOptions {
        if let Some(arc) = &mut options.tool_definitions {
            Arc::make_mut(arc).push(def);
        } else {
            options.tool_definitions = Some(Arc::new(vec![def]));
        }
        options
    }
}

// ── Agent ─────────────────────────────────────────────────────────────

impl<T: LLMBackend> fmt::Debug for Agent<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("options", &self.options)
            .field("plugins", &format!("[{} plugins]", self.plugins.len()))
            .field(
                "plugin_tool_map",
                &format!("[{} entries]", self.tool_plugin.len()),
            )
            .finish_non_exhaustive()
    }
}

// ── Dispatch helpers ──────────────────────────────────────────────────

impl<T: LLMBackend> Agent<T> {
    fn dispatch_assistant_message(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        msg: &Message,
    ) {
        for p in plugins.iter_mut() {
            let name = p.name();
            if let Err(e) = std::panic::catch_unwind(AssertUnwindSafe(|| {
                p.on_assistant_message(ctx, msg);
            })) {
                tracing::error!(plugin = name, "on_assistant_message panicked: {:?}", e);
            }
        }
    }

    async fn dispatch_prepare_system_prompt(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
    ) -> Option<Cow<'static, str>> {
        let mut parts: Vec<String> = Vec::new();
        for p in plugins.iter_mut() {
            if let Some(Some(prompt)) = plugin_hook(
                p.name(),
                "prepare_system_prompt",
                p.prepare_system_prompt(ctx),
            )
            .await
            {
                parts.push(prompt.into_owned());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(Cow::Owned(parts.join("\n\n")))
        }
    }

    async fn dispatch_iteration_start(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        iteration: usize,
    ) {
        for p in plugins.iter_mut() {
            plugin_hook(
                p.name(),
                "on_iteration_start",
                p.on_iteration_start(ctx, iteration),
            )
            .await;
        }
    }

    async fn dispatch_iteration_end(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        iteration: usize,
        had_tool_calls: bool,
    ) {
        for p in plugins.iter_mut() {
            plugin_hook(
                p.name(),
                "on_iteration_end",
                p.on_iteration_end(ctx, iteration, had_tool_calls),
            )
            .await;
        }
    }

    async fn dispatch_tool_pre_execute(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        id: &str,
        name: &str,
        args: &Value,
    ) -> PreToolAction {
        for p in plugins.iter_mut() {
            let action = plugin_hook(
                p.name(),
                "on_tool_pre_execute",
                p.on_tool_pre_execute(ctx, id, name, args),
            )
            .await
            .unwrap_or(PreToolAction::Proceed(None));
            match action {
                PreToolAction::Proceed(None) => {}
                decisive => return decisive,
            }
        }
        PreToolAction::Proceed(None)
    }

    async fn dispatch_tool_post_execute(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        id: &str,
        name: &str,
        result: &std::result::Result<Value, String>,
    ) -> PostToolAction {
        for p in plugins.iter_mut() {
            let action = plugin_hook(
                p.name(),
                "on_tool_post_execute",
                p.on_tool_post_execute(ctx, id, name, result),
            )
            .await
            .unwrap_or(PostToolAction::Proceed(None));
            match action {
                PostToolAction::Proceed(None) => {}
                decisive => return decisive,
            }
        }
        PostToolAction::Proceed(None)
    }

    async fn dispatch_api_error(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        error: &AgentSdkError,
    ) -> RetryAction {
        for p in plugins.iter_mut() {
            let action = plugin_hook(p.name(), "on_api_error", p.on_api_error(ctx, error))
                .await
                .unwrap_or(RetryAction::GiveUp);
            match action {
                RetryAction::GiveUp => {}
                decisive @ RetryAction::RetryAfter(_) => return decisive,
            }
        }
        RetryAction::GiveUp
    }

    async fn dispatch_init(
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
    ) -> Result<()> {
        for p in plugins.iter_mut() {
            let name = p.name();
            let result = tokio::time::timeout(
                PLUGIN_INIT_TIMEOUT,
                AssertUnwindSafe(p.init(ctx)).catch_unwind(),
            )
            .await;
            match result {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(e))) => {
                    tracing::error!(plugin = name, error = %e, "init failed");
                    return Err(e);
                }
                Ok(Err(e)) => {
                    return Err(AgentSdkError::ConfigError(format!(
                        "Plugin {name} panicked during init: {e:?}"
                    )));
                }
                Err(_) => {
                    return Err(AgentSdkError::ConfigError(format!(
                        "Plugin {name} init timed out"
                    )));
                }
            }
        }
        Ok(())
    }
}

// ── Agent implementation ──────────────────────────────────────────────

struct ScopedPluginContext<'a> {
    agent_world: &'a mut Option<hecs::World>,
    agent_entity: &'a mut Option<hecs::Entity>,
    ctx: Option<PluginContext>,
}

impl<'a> ScopedPluginContext<'a> {
    fn new(
        agent_world: &'a mut Option<hecs::World>,
        agent_entity: &'a mut Option<hecs::Entity>,
        spawn_history: bool,
    ) -> Self {
        let mut world = agent_world.take().unwrap_or_default();
        let entity = agent_entity.unwrap_or_else(|| {
            if spawn_history {
                world.spawn((History::default(),))
            } else {
                hecs::Entity::DANGLING
            }
        });
        Self {
            agent_world,
            agent_entity,
            ctx: Some(PluginContext::new(world, entity)),
        }
    }

    #[allow(clippy::expect_used)]
    fn into_inner(mut self) -> PluginContext {
        self.ctx
            .take()
            .expect("ctx is always Some until dropped or into_inner is called")
    }
}

impl Drop for ScopedPluginContext<'_> {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            let (world, entity) = ctx.into_parts();
            *self.agent_world = Some(world);
            *self.agent_entity = Some(entity);
        }
    }
}

impl std::ops::Deref for ScopedPluginContext<'_> {
    type Target = PluginContext;

    #[allow(clippy::expect_used)]
    fn deref(&self) -> &Self::Target {
        self.ctx.as_ref().expect("ctx is always Some until dropped")
    }
}

impl std::ops::DerefMut for ScopedPluginContext<'_> {
    #[allow(clippy::expect_used)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx.as_mut().expect("ctx is always Some until dropped")
    }
}

impl<T: LLMBackend> Agent<T> {
    #[must_use]
    pub fn builder() -> AgentBuilder<T> {
        AgentBuilder::default()
    }
    pub async fn dispatch_user_message(&mut self, text: &str) -> String {
        let mut ctx = ScopedPluginContext::new(&mut self.world, &mut self.entity, false);
        let mut current = text.to_string();
        for p in &mut self.plugins {
            let input = current.clone();
            current = plugin_hook(
                p.name(),
                "on_user_message",
                p.on_user_message(&mut ctx, input),
            )
            .await
            .unwrap_or(current);
        }
        current
    }

    /// Run the agent to completion and return the final [`AgentRunOutput`],
    /// which contains the full [`hecs::World`] with all components.
    ///
    /// # Errors
    /// Returns an error if the LLM API call fails and no plugin handles it.
    /// Append a completion-rejection correction to history. A rejected
    /// completion may be completely empty (no content, no tool calls) —
    /// providers reject such assistant messages at template render
    /// ("must contain 'content' or '`tool_calls`'"), so it is dropped and the
    /// correction message carries the context on its own.
    fn push_rejection(history: &mut Vec<Message>, reason: &str) {
        let drop_last = matches!(
            history.last(),
            Some(Message::AssistantMessage(a))
                if a.content.as_deref().is_none_or(str::is_empty) && a.tool_calls.is_none()
        );
        if drop_last {
            history.pop();
        }
        history.push(messages::user(format!(
            "Your previous response was rejected:\n{reason}\n\nPlease fix and retry."
        )));
    }

    /// Run the agent loop until a final completion or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails, a plugin aborts the run, or
    /// tool execution fails fatally.
    #[tracing::instrument(skip(self))]
    pub async fn run(&mut self) -> Result<AgentRunOutput> {
        // Borrow fields individually to avoid borrow-conflicts with `self` inside the loop.
        let options = &self.options;
        let plugins = &mut self.plugins;
        let plugin_tool_map = &self.tool_plugin;

        let mut ctx = ScopedPluginContext::new(&mut self.world, &mut self.entity, true);

        Self::dispatch_init(plugins, &mut ctx).await?;

        let max_iterations = options.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);

        for i in 0..max_iterations {
            Self::dispatch_iteration_start(plugins, &mut ctx, i).await;
            Self::prepare_prompt(plugins, &mut ctx).await;

            let mut upstream =
                Self::stream_with_retry(&self.backend, options, plugins, &mut ctx).await?;
            let mut acc = ModelResponseAccumulator::default();
            let mut assistant_msg = None;

            while let Some(chunk) = upstream.next().await {
                if let Some(msg) = acc.push(&chunk?, plugins, &mut ctx) {
                    Self::dispatch_assistant_message(plugins, &mut ctx, &msg);

                    // Append assistant message to History component
                    if let Some(mut h) = ctx.get_mut::<History>() {
                        h.0.push(msg.clone());
                    }

                    assistant_msg = Some(msg);
                }
            }

            let Some(Message::AssistantMessage(a)) = assistant_msg else {
                tracing::warn!("No assistant message produced, restarting iteration");
                continue;
            };

            if let Some(calls) = a.tool_calls {
                let msgs =
                    match Self::execute_tools(options, plugins, &mut ctx, &calls, plugin_tool_map)
                        .await
                    {
                        Ok(msgs) => msgs,
                        Err(reason) => {
                            if let Some(mut h) = ctx.get_mut::<History>() {
                                h.0.push(messages::tool(reason.clone(), "error"));
                            }
                            return Err(AgentSdkError::ConfigError(reason));
                        }
                    };

                // Append tool result messages to History component
                if let Some(mut h) = ctx.get_mut::<History>() {
                    for msg in &msgs {
                        h.0.push(msg.clone());
                    }
                }

                Self::dispatch_iteration_end(plugins, &mut ctx, i, true).await;
            } else {
                let final_text = a.content.unwrap_or_default();

                let mut action = CompletionAction::Accept;
                for p in plugins.iter_mut() {
                    let plugin_action = plugin_hook(
                        p.name(),
                        "on_completion",
                        p.on_completion(&mut ctx, &final_text),
                    )
                    .await
                    .unwrap_or(CompletionAction::Accept);
                    match plugin_action {
                        CompletionAction::Accept => {}
                        r @ CompletionAction::Reject { .. } => {
                            action = r;
                            break;
                        }
                    }
                }

                Self::dispatch_iteration_end(plugins, &mut ctx, i, false).await;

                match action {
                    CompletionAction::Accept => {
                        break;
                    }
                    CompletionAction::Reject { reason } => {
                        if let Some(mut h) = ctx.get_mut::<History>() {
                            Self::push_rejection(&mut h.0, &reason);
                        }
                    }
                }
            }
        }

        for p in plugins.iter_mut() {
            let name = p.name();
            let result = plugin_hook(name, "shutdown", p.shutdown(&mut ctx)).await;
            if let Some(Err(e)) = result {
                tracing::error!(plugin = name, error = %e, "shutdown failed");
            }
        }

        let final_ctx = ctx.into_inner();
        let (world, entity) = final_ctx.into_parts();

        Ok(AgentRunOutput { world, entity })
    }

    /// Run the agent to completion and extract a structured JSON response.
    ///
    /// This is a convenience wrapper around [`run`](Self::run) that performs
    /// an extra LLM call with a response schema after the agent loop finishes.
    ///
    /// # Errors
    /// Returns an error if the agent loop fails or JSON deserialization fails.
    #[tracing::instrument(skip(self))]
    pub async fn run_json<R>(&mut self) -> Result<R>
    where
        R: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let output = self.run().await?;

        let schema = schemars::schema_for!(R);
        let schema_val = serde_json::to_value(schema)?;

        let val = {
            let history_ref = output.world.get::<&History>(output.entity);
            let history = history_ref
                .as_ref()
                .ok()
                .map(|h| h.0.as_slice())
                .unwrap_or_default();
            self.backend
                .get_json(&self.options, history, &schema_val)
                .await?
        };

        let result: R = serde_json::from_value(val)?;
        Ok(result)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    async fn prepare_prompt(plugins: &mut [Box<dyn AgentPlugin>], ctx: &mut PluginContext) {
        if let Some(sys) = Self::dispatch_prepare_system_prompt(plugins, ctx).await {
            let content = sys.into_owned();
            if let Some(mut h) = ctx.get_mut::<History>() {
                if let Some(Message::SystemMessage(s)) = h.0.first_mut() {
                    s.content = Some(content);
                } else {
                    h.0.insert(0, messages::system(content));
                }
            }
        }
    }

    async fn stream_with_retry(
        backend: &T,
        options: &AgentOptions,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse>> + Send>>>
    {
        loop {
            let stream_result = {
                let history_ref = ctx.get::<History>();
                let history = history_ref
                    .as_ref()
                    .map(|h| h.0.as_slice())
                    .unwrap_or_default();
                LLMBackend::stream(backend, options, history).await
            };
            match stream_result {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    let action = Self::dispatch_api_error(plugins, ctx, &e).await;
                    match action {
                        RetryAction::RetryAfter(delay) => {
                            tracing::warn!(error = %e, "API call failed, retrying in {:?}", delay);
                            tokio::time::sleep(delay).await;
                        }
                        RetryAction::GiveUp => return Err(e),
                    }
                }
            }
        }
    }

    #[tracing::instrument(skip(plugins, ctx, calls, plugin_tool_map, options), fields(tools_count = calls.len()))]
    async fn execute_tools(
        options: &Arc<AgentOptions>,
        plugins: &mut [Box<dyn AgentPlugin>],
        ctx: &mut PluginContext,
        calls: &[ToolCall],
        plugin_tool_map: &HashMap<String, usize>,
    ) -> std::result::Result<Vec<Message>, String> {
        let options_arc = Arc::clone(options);
        let mut pre_results: Vec<Option<Message>> = Vec::new();
        let mut static_futures = Vec::new();

        for call in calls {
            let args = serde_json::from_str::<Value>(&call.function.arguments)
                .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));

            let call_id = call.id.clone().unwrap_or_default();
            let pre_action =
                Self::dispatch_tool_pre_execute(plugins, ctx, &call_id, &call.function.name, &args)
                    .await;

            match pre_action {
                PreToolAction::Proceed(transformed) => {
                    let exec_args = transformed.unwrap_or(args);
                    if let Some(&plugin_idx) = plugin_tool_map.get(&call.function.name) {
                        // Plugin-owned tool — run sequentially (needs &mut plugin)
                        let tool_call = PluginToolCall {
                            id: call_id.clone(),
                            name: call.function.name.clone(),
                            arguments: exec_args,
                        };
                        let Some(plugin) = plugins.get_mut(plugin_idx) else {
                            continue;
                        };
                        let result = plugin.run_tool(ctx, &tool_call).await;
                        let action = Self::dispatch_tool_post_execute(
                            plugins,
                            ctx,
                            &call_id,
                            &call.function.name,
                            &result,
                        )
                        .await;
                        let content = action.resolve(result);
                        pre_results.push(Some(messages::tool(content, call_id.clone())));
                    } else {
                        // Static tool — queue for parallel execution
                        static_futures.push(Self::execute_tool(
                            &options_arc,
                            &call.function.name,
                            exec_args,
                        ));
                        pre_results.push(None);
                    }
                }
                PreToolAction::Abort(reason) => {
                    pre_results.push(Some(messages::tool(reason, call_id.clone())));
                }
                PreToolAction::Stop(reason) => {
                    return Err(reason);
                }
            }
        }

        // Run static tools in parallel
        let exec_results = futures::future::join_all(static_futures).await;
        let mut exec_iter = exec_results.into_iter();
        let mut messages = Vec::with_capacity(calls.len());

        for (call, pre_res) in calls.iter().zip(pre_results) {
            let call_id = call.id.clone().unwrap_or_default();
            if let Some(msg) = pre_res {
                messages.push(msg);
            } else if let Some(result) = exec_iter.next() {
                let action = Self::dispatch_tool_post_execute(
                    plugins,
                    ctx,
                    &call_id,
                    &call.function.name,
                    &result,
                )
                .await;
                let content = action.resolve(result);
                messages.push(messages::tool(content, call_id));
            }
        }

        Ok(messages)
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

    fn stream_chunk(
        delta: o3gen_openai::types::ChatCompletionStreamResponseDelta,
        finish_reason: Option<
            o3gen_openai::types::CreateChatCompletionStreamResponseChoicesFinishReason,
        >,
    ) -> CreateChatCompletionStreamResponse {
        use o3gen_openai::types::CreateChatCompletionStreamResponseChoices;
        CreateChatCompletionStreamResponse {
            id: "chatcmpl-test".into(),
            object:
                o3gen_openai::types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 1,
            model: "test".into(),
            system_fingerprint: None,
            choices: vec![CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta,
                finish_reason,
            }],
        }
    }

    // OpenRouter (and some proxies) repeat `finish_reason` on a terminal
    // chunk. The accumulator must finish exactly once — a second finish
    // previously emitted an empty assistant message that overwrote the
    // real one, losing tool calls and producing phantom "empty
    // completion" retries.
    #[tokio::test]
    async fn accumulator_finishes_once_on_duplicate_finish_chunks() -> Result<()> {
        use o3gen_openai::types::{
            ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCallChunkFunction,
            ChatCompletionMessageToolCallChunkType, ChatCompletionStreamResponseDelta as Delta,
            CreateChatCompletionStreamResponseChoicesFinishReason as Finish,
        };

        let mut acc = ModelResponseAccumulator::default();
        let mut plugins: Vec<Box<dyn AgentPlugin>> = vec![Box::new(NoopPlugin)];
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);

        // tool-call delta
        let tool_chunk = stream_chunk(
            Delta {
                content: Some(String::new()),
                role: None,
                function_call: None,
                tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                    index: 0,
                    id: Some("call_1".into()),
                    r#type: Some(ChatCompletionMessageToolCallChunkType::Function),
                    function: Some(ChatCompletionMessageToolCallChunkFunction {
                        name: Some("Bash".into()),
                        arguments: Some("{\"command\":\"ls\"}".into()),
                    }),
                }]),
            },
            None,
        );
        assert!(
            acc.push(&tool_chunk, &mut plugins, &mut ctx).is_none(),
            "non-finish chunk must not emit a message"
        );

        // first finish chunk -> emits the accumulated assistant message
        let finish_chunk = stream_chunk(Delta::default(), Some(Finish::ToolCalls));
        let first = acc.push(&finish_chunk, &mut plugins, &mut ctx);
        let Message::AssistantMessage(a) = first.as_ref().ok_or_else(|| {
            AgentSdkError::ConfigError("first finish chunk must emit the assistant message".into())
        })?
        else {
            return Err(AgentSdkError::ConfigError(
                "expected assistant message".into(),
            ));
        };
        let calls = a
            .tool_calls
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("tool calls must survive".into()))?;
        let name = calls
            .first()
            .map(|c| c.function.name.as_str())
            .ok_or_else(|| AgentSdkError::ConfigError("expected a tool call".into()))?;
        assert_eq!(name, "Bash");

        // duplicate finish chunk (OpenRouter terminal stats chunk) -> ignored
        let duplicate = acc.push(&finish_chunk, &mut plugins, &mut ctx);
        assert!(
            duplicate.is_none(),
            "duplicate finish chunk must not emit a second, empty message"
        );
        Ok(())
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

        let mut agent = Agent::builder().client(client).options(options).build()?;

        // The old test called execute_parallel_tool_calls directly.
        // With the new dispatch API we simulate the same world/plugin setup.
        let world = agent
            .world
            .take()
            .ok_or_else(|| AgentSdkError::ConfigError("world missing".into()))?;
        let entity = agent
            .entity
            .ok_or_else(|| AgentSdkError::ConfigError("entity missing".into()))?;

        let mut ctx = PluginContext::new(world, entity);

        let calls = vec![
            ToolCall {
                id: Some("1".into()),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "success".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: Some("2".into()),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "fail".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: Some("3".into()),
                r#type: ToolCallType::Function,
                function: ToolFunction {
                    name: "missing".into(),
                    arguments: "{}".into(),
                },
            },
        ];

        let mut plugins: Vec<Box<dyn AgentPlugin>> = vec![Box::new(NoopPlugin)];
        let empty_map = HashMap::new();
        let messages = Agent::<OpenAI>::execute_tools(
            &agent.options,
            &mut plugins,
            &mut ctx,
            &calls,
            &empty_map,
        )
        .await
        .map_err(AgentSdkError::ConfigError)?;

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

    struct SearchPlugin;

    #[async_trait]
    impl AgentPlugin for SearchPlugin {
        fn name(&self) -> &'static str {
            "search"
        }

        fn tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "query".into(),
                description: "Search for documents".into(),
                input_schema: schemars::schema_for!(Value),
            }]
        }

        async fn run_tool(
            &mut self,
            _ctx: &mut PluginContext,
            call: &PluginToolCall,
        ) -> std::result::Result<Value, String> {
            Ok(serde_json::json!({
                "tool": call.name,
                "args": call.arguments,
                "id": call.id,
            }))
        }
    }

    #[test]
    fn test_plugin_tools_registered() -> Result<()> {
        let config = crate::ModelConfig {
            base_url: "test".into(),
            api_key: "test".into(),
            model: "test".into(),
        };
        let client = OpenAI::new(config);
        let agent = Agent::builder()
            .client(client)
            .plugin(SearchPlugin)
            .build()?;

        assert!(agent.tool_plugin.contains_key("query"));
        assert_eq!(*agent.tool_plugin.get("query").unwrap_or(&999), 0);

        let defs = agent
            .options
            .tool_definitions
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("no tool_definitions".into()))?;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs.first().map(|d| d.name.as_str()), Some("query"));

        Ok(())
    }

    #[tokio::test]
    async fn test_plugin_tool_dispatch() -> Result<()> {
        let config = crate::ModelConfig {
            base_url: "test".into(),
            api_key: "test".into(),
            model: "test".into(),
        };
        let client = OpenAI::new(config);
        let mut agent = Agent::builder()
            .client(client)
            .plugin(SearchPlugin)
            .build()?;

        let world = agent
            .world
            .take()
            .ok_or_else(|| AgentSdkError::ConfigError("world missing".into()))?;
        let entity = agent
            .entity
            .ok_or_else(|| AgentSdkError::ConfigError("entity missing".into()))?;

        let mut ctx = PluginContext::new(world, entity);

        let calls = vec![ToolCall {
            id: Some("tc_1".into()),
            r#type: ToolCallType::Function,
            function: ToolFunction {
                name: "query".into(),
                arguments: "{\"q\":\"hello\"}".into(),
            },
        }];

        let mut plugins: Vec<Box<dyn AgentPlugin>> = vec![Box::new(SearchPlugin)];
        let messages = Agent::<OpenAI>::execute_tools(
            &agent.options,
            &mut plugins,
            &mut ctx,
            &calls,
            &agent.tool_plugin,
        )
        .await
        .map_err(AgentSdkError::ConfigError)?;

        assert_eq!(messages.len(), 1);
        match messages.first() {
            Some(Message::ToolMessage(m)) => {
                assert_eq!(m.tool_call_id, "tc_1");
                let content = m
                    .content
                    .as_ref()
                    .ok_or_else(|| AgentSdkError::ConfigError("no content".into()))?;
                let val: Value = serde_json::from_str(content)
                    .map_err(|e| AgentSdkError::ConfigError(e.to_string()))?;
                assert_eq!(val.get("tool").and_then(Value::as_str), Some("query"));
                assert_eq!(
                    val.get("args")
                        .and_then(|a| a.get("q"))
                        .and_then(Value::as_str),
                    Some("hello")
                );
                assert_eq!(val.get("id").and_then(Value::as_str), Some("tc_1"));
            }
            _ => return Err(AgentSdkError::ConfigError("Expected ToolMessage".into())),
        }

        Ok(())
    }

    struct DbPlugin;

    #[async_trait]
    impl AgentPlugin for DbPlugin {
        fn name(&self) -> &'static str {
            "db"
        }

        fn tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "db_query".into(),
                description: "Query the database".into(),
                input_schema: schemars::schema_for!(Value),
            }]
        }

        async fn run_tool(
            &mut self,
            _ctx: &mut PluginContext,
            call: &PluginToolCall,
        ) -> std::result::Result<Value, String> {
            Ok(serde_json::json!({"source": "db", "tool": call.name}))
        }
    }

    #[test]
    fn test_multiple_plugins_unique_tools() -> Result<()> {
        let config = crate::ModelConfig {
            base_url: "test".into(),
            api_key: "test".into(),
            model: "test".into(),
        };
        let client = OpenAI::new(config);
        let agent = Agent::builder()
            .client(client)
            .plugin(SearchPlugin)
            .plugin(DbPlugin)
            .build()?;

        assert!(agent.tool_plugin.contains_key("query"));
        assert!(agent.tool_plugin.contains_key("db_query"));
        assert_eq!(*agent.tool_plugin.get("query").unwrap_or(&999), 0);
        assert_eq!(*agent.tool_plugin.get("db_query").unwrap_or(&999), 1);

        let defs = agent
            .options
            .tool_definitions
            .as_ref()
            .ok_or_else(|| AgentSdkError::ConfigError("no tool_definitions".into()))?;
        assert_eq!(defs.len(), 2);

        Ok(())
    }
}
