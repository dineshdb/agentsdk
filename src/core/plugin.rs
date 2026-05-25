use crate::core::agent::{CompletionAction, PostToolAction, PreToolAction};
use crate::core::messages::Message;
use crate::core::retry::RetryAction;
use crate::core::tools::ToolDefinition;
use crate::error::AgentSdkError;
use async_trait::async_trait;
use serde_json::Value;
use std::borrow::Cow;

/// Shared context passed to every plugin lifecycle hook.
///
/// Wraps a [`hecs::World`] with a dedicated entity for the agent session.
/// Plugins can read/write typed components on this entity to share state
/// with other plugins and with the agent loop.
///
/// # ECS change detection
/// Use [`hecs::Changed`] / [`hecs::Added`] to detect which components
/// changed since the last [`hecs::World::clear_tracked_data()`] call.
pub struct PluginContext {
    world: hecs::World,
    entity: hecs::Entity,
}

impl std::fmt::Debug for PluginContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginContext")
            .field("entity", &self.entity)
            .finish_non_exhaustive()
    }
}

impl PluginContext {
    /// Internal method to consume the context and return its parts.
    pub(crate) fn into_parts(self) -> (hecs::World, hecs::Entity) {
        (self.world, self.entity)
    }

    /// Method to construct a context from parts.
    pub fn new(world: hecs::World, entity: hecs::Entity) -> Self {
        Self { world, entity }
    }

    /// Borrow a component by type.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<hecs::Ref<'_, T>> {
        self.world.get::<&T>(self.entity).ok()
    }

    /// Mutably borrow a component by type.
    /// The component is marked as changed for change-detection purposes
    /// via [`hecs::RefMut`] once the borrow is dropped.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<hecs::RefMut<'_, T>> {
        self.world.get::<&mut T>(self.entity).ok()
    }

    /// Insert a component onto the agent entity.
    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        if let Err(e) = self.world.insert_one(self.entity, val) {
            tracing::warn!("Failed to insert component: {e}");
        }
    }

    /// Read-only access to the underlying [`hecs::World`] for custom queries.
    pub fn world(&self) -> &hecs::World {
        &self.world
    }

    /// Mutable access to the underlying [`hecs::World`] for advanced use.
    pub fn world_mut(&mut self) -> &mut hecs::World {
        &mut self.world
    }
}

/// A plugin extends the agent with custom behavior.
///
/// Plugins receive lifecycle events in registration order.
/// For control-flow hooks the *first decisive return value wins*;
/// subsequent plugins are skipped.  Observability hooks fire for *all* plugins.
///
/// Every method has a default no-op implementation so you only
/// override what you need.
#[async_trait]
pub trait AgentPlugin: Send + Sync {
    /// Human-readable plugin name (used in logs / diagnostics).
    fn name(&self) -> &'static str;

    // ── Lifecycle ──────────────────────────────────────────────────

    /// Called once when the agent starts.
    /// Use this to load state, spawn components, etc.
    async fn init(&mut self, _ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        Ok(())
    }

    /// Called once when the agent finishes (or errors out).
    async fn shutdown(&mut self, _ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        Ok(())
    }

    // ── Observability (all plugins fire) ───────────────────────────

    /// A chunk of text was streamed from the LLM.
    fn on_text_delta(&mut self, _ctx: &mut PluginContext, _text: &str) {}

    /// A full model response (turn) was completed.
    /// This includes responses that contain tool calls.
    fn on_assistant_message(&mut self, _ctx: &mut PluginContext, _msg: &Message) {}

    // ── Iteration lifecycle ────────────────────────────────────────

    /// Called at the start of each agent loop iteration, before prompt preparation.
    async fn on_iteration_start(&mut self, _ctx: &mut PluginContext, _iteration: usize) {}

    /// Called at the end of each agent loop iteration, after tool execution or
    /// completion handling. `had_tool_calls` indicates whether this iteration
    /// produced tool calls (vs a final text completion).
    async fn on_iteration_end(
        &mut self,
        _ctx: &mut PluginContext,
        _iteration: usize,
        _had_tool_calls: bool,
    ) {
    }

    // ── Control flow (first decisive return wins) ──────────────────

    /// Before each model iteration.  Return a system prompt override.
    /// Use [`PluginContext::get::<History>()`] to inspect conversation history.
    async fn prepare_system_prompt(
        &mut self,
        _ctx: &mut PluginContext,
    ) -> Option<Cow<'static, str>> {
        None
    }

    /// Before a tool executes.  Return `Abort` to skip or `Proceed` with
    /// transformed arguments.
    async fn on_tool_pre_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        _name: &str,
        _args: &Value,
    ) -> PreToolAction {
        PreToolAction::Proceed(None)
    }

    /// After a tool executes (success or failure).
    ///
    /// `result` is `Ok(value)` on success or `Err(message)` on failure.
    /// Return `Proceed(None)` to pass through, `Proceed(Some(v))` to transform
    /// the result or provide a fallback, or `Override(s)` to replace the output.
    async fn on_tool_post_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        _name: &str,
        _result: &Result<Value, String>,
    ) -> PostToolAction {
        PostToolAction::Proceed(None)
    }

    /// When the agent produces a final text completion (no tool calls).
    async fn on_completion(&mut self, _ctx: &mut PluginContext, _text: &str) -> CompletionAction {
        CompletionAction::Accept
    }

    /// When an API / network error occurs.
    async fn on_api_error(
        &mut self,
        _ctx: &mut PluginContext,
        _error: &AgentSdkError,
    ) -> RetryAction {
        RetryAction::GiveUp
    }

    // ── Plugin-owned tools ─────────────────────────────────────────

    /// Tool definitions this plugin provides.
    /// Names will be automatically scoped as `{plugin_name}__{tool_name}`.
    fn tools(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    // ── User Input (transformation) ───────────────────────────────

    /// Transform user input before it is displayed or persisted.
    async fn on_user_message(&mut self, _ctx: &mut PluginContext, text: String) -> String {
        text
    }

    /// Execute a plugin-owned tool. Called only for tools returned by [`tools()`](Self::tools).
    async fn run_tool(
        &mut self,
        _ctx: &mut PluginContext,
        _call: &PluginToolCall,
    ) -> Result<Value, String> {
        Err(format!("run_tool not implemented for {}", self.name()))
    }
}

/// Context passed to [`AgentPlugin::run_tool()`] with all info from the LLM's tool call.
#[derive(Debug, Clone)]
pub struct PluginToolCall {
    /// The tool call ID from the LLM.
    pub id: String,
    /// Unscoped tool name (e.g. `"search"`, not `"my_plugin__search"`).
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: Value,
}

/// A trait for enums that represent a set of tools for a plugin.
///
/// This trait is typically derived using `#[derive(PluginTools)]`.
pub trait PluginTools: Sized {
    /// Returns the tool definitions for this set of tools.
    fn definitions() -> Vec<ToolDefinition>;

    /// Parses a [`PluginToolCall`] into this enum.
    ///
    /// # Errors
    /// Returns an error if the tool name is unknown or if the arguments
    /// fail to deserialize into the variant's payload.
    fn from_call(call: &PluginToolCall) -> Result<Self, String>;
}
