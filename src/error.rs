use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentSdkError {
    #[error("API error: {0}")]
    ApiError(#[from] o3gen_openai::ApiError),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

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
            // A mid-stream error chunk carries the provider's status code;
            // classify it like an HTTP-level error so retry budgets apply.
            Self::ApiError(o3gen_openai::ApiError::Stream {
                code: Some(code), ..
            }) => reqwest::StatusCode::from_u16(*code).ok(),
            Self::NetworkError(e) => e.status(),
            _ => None,
        }
    }

    /// True for transport-level failures — dropped/truncated response bodies,
    /// connection and timeout errors — as opposed to errors the API itself
    /// reported. The request's outcome is unknown, so these are the candidates
    /// for re-issuing it.
    #[must_use]
    pub fn is_transport(&self) -> bool {
        let (Self::ApiError(o3gen_openai::ApiError::Reqwest(e)) | Self::NetworkError(e)) = self
        else {
            return false;
        };
        e.is_connect() || e.is_request() || e.is_body() || e.is_decode() || e.is_timeout()
    }
}

pub type Result<T> = std::result::Result<T, AgentSdkError>;
