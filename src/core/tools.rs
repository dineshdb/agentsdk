use crate::core::agent::AgentOptions;
use crate::error::AgentSdkError;
use derive_builder::Builder;
use schemars::Schema;
use serde_json::Value;
use std::fmt::{self, Debug, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ToolContext {
    pub options: Arc<AgentOptions>,
}

pub type ToolOutput = Result<Value, String>;
pub type ToolFuture = Pin<Box<dyn Future<Output = ToolOutput> + Send>>;

type SyncToolFn = dyn Fn(ToolContext, Value) -> ToolOutput + Send + Sync;
type AsyncToolFn = dyn Fn(ToolContext, Value) -> ToolFuture + Send + Sync;

#[derive(Clone)]
pub enum ToolExecute {
    Sync(Arc<SyncToolFn>),
    Async(Arc<AsyncToolFn>),
}

impl Debug for ToolExecute {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sync(_) => f.debug_tuple("Sync").finish(),
            Self::Async(_) => f.debug_tuple("Async").finish(),
        }
    }
}

impl ToolExecute {
    pub fn from_sync<F>(f: F) -> Self
    where
        F: Fn(ToolContext, Value) -> ToolOutput + Send + Sync + 'static,
    {
        Self::Sync(Arc::new(f))
    }

    pub fn from_async<F, Fut>(f: F) -> Self
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolOutput> + Send + 'static,
    {
        Self::Async(Arc::new(move |ctx, val| Box::pin(f(ctx, val))))
    }

    /// Executes the tool with the given context and input.
    ///
    /// # Errors
    /// Returns the tool's error string if execution fails.
    pub async fn call(&self, ctx: ToolContext, input: Value) -> ToolOutput {
        match self {
            Self::Sync(f) => f(ctx, input),
            Self::Async(f) => f(ctx, input).await,
        }
    }
}

#[derive(Debug, Builder, Clone)]
#[builder(pattern = "owned", setter(into))]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Schema,
}

impl ToolDefinition {
    #[must_use]
    pub fn builder() -> ToolDefinitionBuilder {
        ToolDefinitionBuilder::default()
    }
}

#[derive(Builder, Clone)]
#[builder(pattern = "owned", setter(into), build_fn(error = "AgentSdkError"))]
pub struct Tool {
    pub definition: ToolDefinition,
    pub execute: ToolExecute,
}

impl Debug for Tool {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("definition", &self.definition)
            .field("execute", &self.execute)
            .finish()
    }
}

impl Tool {
    #[must_use]
    pub fn builder() -> ToolBuilder {
        ToolBuilder::default()
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    #[must_use]
    pub fn description(&self) -> &str {
        &self.definition.description
    }

    #[must_use]
    pub fn input_schema(&self) -> &Schema {
        &self.definition.input_schema
    }

    /// Executes the tool with the given context and input.
    ///
    /// # Errors
    /// Returns `AgentSdkError::ToolCallError` if execution fails.
    pub async fn call(&self, ctx: ToolContext, input: Value) -> crate::error::Result<Value> {
        self.execute
            .call(ctx, input)
            .await
            .map_err(AgentSdkError::ToolCallError)
    }
}
