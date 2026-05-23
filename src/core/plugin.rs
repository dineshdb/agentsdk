use crate::core::agent::{CompletionAction, PostToolAction, PreToolAction, ToolErrorAction};
use crate::core::messages::{Message, Messages};
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
    pub world: hecs::World,
    pub entity: hecs::Entity,
}

impl std::fmt::Debug for PluginContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginContext")
            .field("entity", &self.entity)
            .finish_non_exhaustive()
    }
}

impl PluginContext {
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
    async fn init(&mut self, _ctx: &mut PluginContext) {}

    /// Called once when the agent finishes (or errors out).
    async fn shutdown(&mut self, _ctx: &mut PluginContext) {}

    // ── Observability (all plugins fire) ───────────────────────────

    /// A chunk of text was streamed from the LLM.
    async fn on_text_delta(&mut self, _ctx: &mut PluginContext, _text: &str) {}

    /// A full model response (turn) was completed.
    /// This includes responses that contain tool calls.
    async fn on_model_response_completed(&mut self, _ctx: &mut PluginContext, _msg: &Message) {}

    // ── Control flow (first decisive return wins) ──────────────────

    /// Before each model iteration.  Return a system prompt override.
    async fn prepare_system_prompt(
        &mut self,
        _ctx: &mut PluginContext,
        _history: &Messages,
    ) -> Option<Cow<'static, str>> {
        None
    }

    /// Before a tool executes.  Return `Abort` to skip or `Continue` with
    /// transformed arguments.
    async fn on_tool_pre_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        _name: &str,
        _args: &Value,
    ) -> PreToolAction {
        PreToolAction::Continue(None)
    }

    /// After a tool executes successfully.
    async fn on_tool_post_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        _name: &str,
        _result: &Value,
    ) -> PostToolAction {
        PostToolAction::Continue(None)
    }

    /// When a tool execution fails.
    async fn on_tool_error(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        _name: &str,
        _error: &str,
    ) -> ToolErrorAction {
        ToolErrorAction::Continue(None)
    }

    /// When the agent produces a final text completion (no tool calls).
    async fn on_completion(&mut self, _ctx: &mut PluginContext, _text: String) -> CompletionAction {
        CompletionAction::Accept(None)
    }

    /// When an API / network error occurs.
    async fn on_api_error(
        &mut self,
        _ctx: &mut PluginContext,
        _error: &AgentSdkError,
    ) -> RetryAction {
        RetryAction::DoNotRetry
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
