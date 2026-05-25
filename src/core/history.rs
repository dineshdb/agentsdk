use crate::core::messages::{self, Message, Messages};
use crate::core::plugin::{AgentPlugin, PluginContext};
use crate::error::AgentSdkError;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;

/// Conversation history stored as a component on the agent entity.
///
/// Use [`PluginContext::get_mut::<History>()`] to append messages, or
/// query [`hecs::Changed<History>`] for change detection.
#[derive(Debug, Clone, Default)]
pub struct History(pub Vec<Message>);

impl History {
    /// Inject a synthetic tool call + result pair into history.
    ///
    /// This tells the model that a tool was called and its result is available,
    /// useful for auto-injection (e.g. auto-searching skills before the model
    /// responds).
    pub fn inject_tool_call(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        result: &serde_json::Value,
    ) {
        let call_id = format!(
            "auto_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros()
        );

        self.0
            .push(messages::assistant_tool_call(name, &call_id, arguments));
        self.0.push(messages::tool(result.to_string(), &call_id));
    }
}

/// A file-backed history plugin.
///
/// Loads conversation history from a JSON file on [`init`](AgentPlugin::init)
/// and saves on [`shutdown`](AgentPlugin::shutdown).  Across runs the file
/// acts as the single source of truth.
///
/// # Example
/// ```ignore
/// // Before the agent loop, push user messages:
/// let mut plugin = FileHistoryPlugin::new(".session.json")?;
/// plugin.push(messages::user("Hello")).await?;
///
/// // The agent reads the file on init, runs, and saves on shutdown:
/// Agent::builder().plugin(plugin).build()?.run().await?;
/// ```
pub struct FileHistoryPlugin {
    inner: Arc<RwLock<FileHistoryInner>>,
}

impl std::fmt::Debug for FileHistoryPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHistoryPlugin").finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct FileHistoryInner {
    path: PathBuf,
    messages: Messages,
}

impl FileHistoryPlugin {
    /// Creates a new file-based history plugin.
    ///
    /// # Errors
    /// Returns an error if the parent directory cannot be created.
    pub fn new(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(FileHistoryInner {
                path,
                messages: Vec::new(),
            })),
        })
    }

    /// Append a message to the history and persist to disk immediately.
    ///
    /// # Errors
    /// Returns an error if serialization or the file write fails.
    pub async fn push(&self, msg: Message) -> crate::error::Result<()> {
        let mut inner = self.inner.write().await;
        inner.messages.push(msg);
        let data = serde_json::to_string_pretty(&inner.messages)?;
        fs::write(&inner.path, data).await?;
        Ok(())
    }

    /// Load the current persisted messages (without running the agent).
    ///
    /// # Errors
    /// Returns an error if the underlying read lock is poisoned.
    pub async fn load(&self) -> crate::error::Result<Messages> {
        let inner = self.inner.read().await;
        Ok(inner.messages.clone())
    }
}

impl Clone for FileHistoryPlugin {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait]
impl AgentPlugin for FileHistoryPlugin {
    fn name(&self) -> &'static str {
        "file_history"
    }

    async fn init(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        let inner = self.inner.read().await;
        ctx.insert(History(inner.messages.clone()));
        Ok(())
    }

    async fn shutdown(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        if let Some(h) = ctx.get::<History>() {
            let mut inner = self.inner.write().await;
            inner.messages.clone_from(&h.0);
            let data = serde_json::to_string_pretty(&inner.messages)?;
            let path = inner.path.clone();
            fs::write(&path, data).await?;
        }
        Ok(())
    }
}

/// An in-memory history plugin (no persistence across runs).
///
/// Useful for tests or single-turn agents where persistence is not needed.
#[derive(Clone, Default)]
pub struct MemoryHistoryPlugin {
    inner: Arc<RwLock<Messages>>,
}

impl std::fmt::Debug for MemoryHistoryPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHistoryPlugin")
            .finish_non_exhaustive()
    }
}

impl MemoryHistoryPlugin {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a message to the in-memory buffer.
    pub async fn push(&self, msg: Message) {
        self.inner.write().await.push(msg);
    }

    /// Read the current message buffer.
    pub async fn messages(&self) -> Messages {
        self.inner.read().await.clone()
    }
}

#[async_trait]
impl AgentPlugin for MemoryHistoryPlugin {
    fn name(&self) -> &'static str {
        "memory_history"
    }

    async fn init(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        let msgs = self.inner.read().await.clone();
        if let Some(mut existing) = ctx.get_mut::<History>() {
            // Prepend memory history so injected messages (from on_user_message) come after
            let mut combined = msgs;
            combined.append(&mut existing.0);
            existing.0 = combined;
        } else {
            ctx.insert(History(msgs));
        }
        Ok(())
    }

    async fn shutdown(&mut self, ctx: &mut PluginContext) -> Result<(), AgentSdkError> {
        if let Some(h) = ctx.get::<History>() {
            *self.inner.write().await = h.0.clone();
        }
        Ok(())
    }
}
