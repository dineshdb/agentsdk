pub mod core;
pub mod error;
pub mod openai;

pub use agentsdk_macros::{PluginTools, tool};
pub use core::agent::{
    Agent, AgentBuilder, AgentOptions, CompletionAction, LLMBackend, PostToolAction, PreToolAction,
};
pub use core::hil::{HilItem, HilPlugin, HilState, HilType, SelectOption};
pub use core::history::{FileHistoryPlugin, History, MemoryHistoryPlugin};
pub use core::messages::{self, Message, Messages};
pub use core::plugin::{AgentPlugin, PluginContext, PluginToolCall, PluginTools};
pub use core::retry::RetryAction;
pub use core::tools::{Tool, ToolContext, ToolDefinition};
pub use core::usage::Usage;
pub use error::{AgentSdkError, Result};
/// Re-exported because [`core::plugin::PluginContext::new`] takes its types:
/// plugin crates need it to build a context (tests, embedders).
pub use hecs;
pub use openai::{ModelConfig, OpenAI};

pub mod __private {
    pub use schemars;
    pub use serde_json;
}
