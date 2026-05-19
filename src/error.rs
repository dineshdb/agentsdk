use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentSdkError {
    #[error("API error: {0}")]
    ApiError(#[from] o3gen_openai::ApiError),

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

impl AgentSdkError {
    /// Returns the HTTP status code if the error was caused by a non-success HTTP response.
    #[must_use]
    pub fn status_code(&self) -> Option<reqwest::StatusCode> {
        match self {
            Self::ApiError(o3gen_openai::ApiError::Status { status, .. }) => Some(*status),
            Self::NetworkError(e) => e.status(),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, AgentSdkError>;
