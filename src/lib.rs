pub mod core;
pub mod error;
pub mod openai;
mod utils;

pub use agentsdk_macros::tool;
pub use error::{AgentSdkError, Result};

pub use core::agent::{Agent, AgentBuilder, AgentEvent, AgentOptions, AgentStream};
pub use core::extensions::Extensions;
pub use core::messages::{self, Message, Messages};
pub use core::tools::{Tool, ToolContext};
pub use openai::OpenAI;

pub mod __private {
    pub use schemars;
    pub use serde_json;
}
