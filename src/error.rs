use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentSdkError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Tool call error: {0}")]
    ToolCallError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Builder error: {0}")]
    BuilderError(#[from] derive_builder::UninitializedFieldError),
}

pub type Result<T> = std::result::Result<T, AgentSdkError>;
