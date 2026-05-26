use crate::core::agent::CompletionAction;
use crate::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use crate::core::tools::ToolDefinition;
use crate::error::AgentSdkError;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

// ── HilState ──────────────────────────────────────────────────────
// ECS component.  Inserted by [`HilPlugin::init`], read/written by
// [`HilPlugin`] and by other plugins via [`PluginContext::get::<HilState>()`].

/// HIL (Human-in-the-Loop) state shared across plugins and the consumer.
///
/// Plugins access this via [`PluginContext::get::<HilState>()`] or
/// [`PluginContext::get_mut::<HilState>()`] to create interaction items
/// when they need user input.  **If absent, plugins proceed without
/// prompting** — no error, no stall.
#[derive(Debug, Clone)]
pub struct HilState {
    items: VecDeque<HilItem>,
}

impl HilState {
    /// Any items still waiting for a response?
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.items.iter().any(|i| i.response.is_none())
    }

    /// Items that have not yet been answered.
    pub fn pending(&self) -> impl Iterator<Item = &HilItem> {
        self.items.iter().filter(|i| i.response.is_none())
    }

    /// All items, answered or not.
    pub fn iter(&self) -> impl Iterator<Item = &HilItem> {
        self.items.iter()
    }

    /// Record a response for a pending item.
    ///
    /// Returns `false` if no item with `id` exists.
    pub fn respond(&mut self, id: &str, value: Value) -> bool {
        for item in &mut self.items {
            if item.id == id {
                item.response = Some(value);
                return true;
            }
        }
        false
    }

    /// Check whether a specific item has been answered.
    #[must_use]
    pub fn has_response(&self, id: &str) -> bool {
        self.items
            .iter()
            .any(|i| i.id == id && i.response.is_some())
    }

    /// The response value for an item, if available.
    #[must_use]
    pub fn response(&self, id: &str) -> Option<&Value> {
        self.items
            .iter()
            .find(|i| i.id == id)
            .and_then(|i| i.response.as_ref())
    }

    /// Remove an item (answered or expired).
    pub fn remove(&mut self, id: &str) -> Option<HilItem> {
        let idx = self.items.iter().position(|i| i.id == id)?;
        self.items.remove(idx)
    }

    // ── Plugin-facing push helpers ───────────────────────────────

    /// Push a free-text question.
    pub fn push_question(
        &mut self,
        id: impl Into<String>,
        message: impl Into<String>,
        placeholder: Option<String>,
    ) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::Question { placeholder },
            message: message.into(),
            response: None,
        });
    }

    /// Push a yes/no confirmation.
    pub fn push_confirm(
        &mut self,
        id: impl Into<String>,
        message: impl Into<String>,
        default: Option<bool>,
    ) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::Confirm { default },
            message: message.into(),
            response: None,
        });
    }

    /// Push a selection prompt.
    pub fn push_select(
        &mut self,
        id: impl Into<String>,
        message: impl Into<String>,
        options: Vec<SelectOption>,
        multiple: bool,
    ) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::Select { options, multiple },
            message: message.into(),
            response: None,
        });
    }

    /// Push an informational message (no response expected).
    pub fn push_message(&mut self, id: impl Into<String>, message: impl Into<String>) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::Message,
            message: message.into(),
            response: None,
        });
    }

    /// Push a table.
    pub fn push_table(
        &mut self,
        id: impl Into<String>,
        message: impl Into<String>,
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    ) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::Table { headers, rows },
            message: message.into(),
            response: None,
        });
    }

    /// Push a list.
    pub fn push_list(
        &mut self,
        id: impl Into<String>,
        message: impl Into<String>,
        items: Vec<String>,
        ordered: bool,
    ) {
        self.items.push_back(HilItem {
            id: id.into(),
            r#type: HilType::List { items, ordered },
            message: message.into(),
            response: None,
        });
    }

    // ── Internal ────────────────────────────────────────────────

    /// Check if there is a response for the given question text, take it,
    /// and remove the item.
    fn take_response_for(&mut self, question: &str) -> Option<Value> {
        let idx = self
            .items
            .iter()
            .position(|i| i.message == question && i.response.is_some())?;
        let item = self.items.remove(idx)?;
        item.response
    }
}

// ── HilItem ───────────────────────────────────────────────────────

/// A single user-interaction item.
#[derive(Debug, Clone, Serialize)]
pub struct HilItem {
    /// Unique identifier (scoped to a single agent run).
    pub id: String,
    /// The kind of interaction.
    pub r#type: HilType,
    /// Human-readable prompt text.
    pub message: String,
    /// Set by the consumer via [`HilState::respond`] or [`HilPlugin::respond`].
    pub response: Option<Value>,
}

// ── HilType ───────────────────────────────────────────────────────

/// Variants of user-facing prompts.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum HilType {
    /// Free-text input.
    Question { placeholder: Option<String> },
    /// Yes / No.
    Confirm { default: Option<bool> },
    /// Choose one or more from a list.
    Select {
        options: Vec<SelectOption>,
        multiple: bool,
    },
    /// Informational (no response expected).
    Message,
    /// Tabular data.
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    /// Ordered or unordered list.
    List { items: Vec<String>, ordered: bool },
}

// ── SelectOption ──────────────────────────────────────────────────

/// A choice in a [`HilType::Select`].
#[derive(Debug, Clone, Serialize)]
pub struct SelectOption {
    pub label: String,
    pub value: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl SelectOption {
    pub fn new(label: impl Into<String>, value: impl Into<Value>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            description: None,
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

// ── Ask tool input ────────────────────────────────────────────────

/// Input for the `ask` tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct AskInput {
    /// The question or message to show the user. Be clear and specific.
    question: String,
    /// Type of interaction: "question" (free text), "confirm" (yes/no),
    /// "select" (choose one), "`multi_select`" (choose multiple).
    #[schemars(with = "String")]
    interaction_type: String,
    /// Options for select / `multi_select`.
    #[serde(default)]
    options: Vec<AskOptionInput>,
    /// Placeholder text (question type only).
    placeholder: Option<String>,
    /// Default value (confirm type only).
    default: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AskOptionInput {
    /// Display label shown to the user.
    label: String,
    /// The value returned when this option is chosen.
    value: String,
    /// Optional longer description.
    description: Option<String>,
}

// ── HilPlugin ─────────────────────────────────────────────────────

/// Human-in-the-Loop plugin.
///
/// Provides a single [`ask`](https://) tool that the LLM can call when
/// it needs user input.  Other plugins can also push items directly via
/// [`HilState`].
///
/// The consumer (CLI / web app) reads pending items between agent runs
/// and records responses through the plugin's [`respond`](Self::respond)
/// method.
///
/// # Example
/// ```ignore
/// use agentsdk::hil::HilPlugin;
///
/// let hil = HilPlugin::new();
/// Agent::builder()
///     .plugin(hil.clone())
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct HilPlugin {
    inner: Arc<RwLock<HilState>>,
}

impl Default for HilPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl HilPlugin {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HilState {
                items: VecDeque::new(),
            })),
        }
    }

    // ── Consumer API (accessible between agent runs) ─────────────

    /// Items waiting for user input.
    pub async fn pending_items(&self) -> Vec<HilItem> {
        self.inner
            .read()
            .await
            .items
            .iter()
            .filter(|i| i.response.is_none())
            .cloned()
            .collect()
    }

    /// Record the user's response to a pending item.
    ///
    /// Returns `false` if no item with `id` exists.
    pub async fn respond(&self, id: &str, value: Value) -> bool {
        self.inner.write().await.respond(id, value)
    }
}

#[async_trait]
impl AgentPlugin for HilPlugin {
    fn name(&self) -> &'static str {
        "hil"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "ask".into(),
            description: "Ask the user for input. Use this when you need \
                          clarification, a decision, or any information \
                          from the user."
                .into(),
            input_schema: schemars::schema_for!(AskInput),
        }]
    }

    async fn init(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        let state = self.inner.read().await.clone();
        ctx.insert(state);
        Ok(())
    }

    async fn shutdown(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        if let Some(state) = ctx.get::<HilState>() {
            *self.inner.write().await = (*state).clone();
        }
        Ok(())
    }

    async fn prepare_system_prompt(
        &mut self,
        ctx: &mut PluginContext,
    ) -> Option<std::borrow::Cow<'static, str>> {
        let state = ctx.get::<HilState>()?;
        if state.has_pending() {
            Some(std::borrow::Cow::Borrowed(
                "You have asked the user a question and are waiting for their response. \
                 Call the `ask` tool again to check if the user has responded. \
                 Do NOT proceed without the user's input.",
            ))
        } else {
            None
        }
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        let input: AskInput = serde_json::from_value(call.arguments.clone())
            .map_err(|e| format!("Invalid ask input: {e}"))?;

        let mut state = ctx
            .get_mut::<HilState>()
            .ok_or_else(|| "HilState not found in world".to_string())?;

        // If there's already a response for this question, return it.
        if let Some(value) = state.take_response_for(&input.question) {
            return Ok(json!({"status": "ready", "value": value}));
        }

        // Generate a unique ID and create the item.
        let id = format!("hil_{}", uuid());

        match input.interaction_type.as_str() {
            "question" => {
                state.push_question(&id, &input.question, input.placeholder);
            }
            "confirm" => {
                state.push_confirm(&id, &input.question, input.default);
            }
            "select" | "multi_select" => {
                let multiple = input.interaction_type == "multi_select";
                let options: Vec<SelectOption> = input
                    .options
                    .into_iter()
                    .map(|o| SelectOption {
                        label: o.label,
                        value: Value::String(o.value),
                        description: o.description,
                    })
                    .collect();
                state.push_select(&id, &input.question, options, multiple);
            }
            other => {
                return Err(format!("Unknown interaction_type: {other}"));
            }
        }

        Ok(json!({"status": "pending", "id": id}))
    }

    async fn on_completion(&mut self, ctx: &mut PluginContext, _text: &str) -> CompletionAction {
        if let Some(state) = ctx.get::<HilState>()
            && state.has_pending()
        {
            return CompletionAction::Reject {
                reason: "Waiting for user input on pending questions. \
                     Call the `ask` tool again to check for a response."
                    .into(),
            };
        }
        CompletionAction::Accept
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // use last 12 hex digits for a compact id
    format!("{:012x}", nanos & 0xffff_ffff_ffff)
}
