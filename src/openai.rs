use crate::core::agent::AgentOptions;
use crate::core::messages::Message;
use crate::error::{AgentSdkError, Result};
use api::OpenAIApiClient;
use api::types;
use futures::{Stream, StreamExt};
pub use o3gen_openai as api;
use o3gen_openai::ChatCompletionTool;
use std::pin::Pin;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Configuration for an AI model provider.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// e.g. `https://api.openai.com/v1`
    pub base_url: String,
    pub api_key: String,
    /// e.g. `"gpt-4o"`
    pub model: String,
}

impl ModelConfig {
    /// Create a new `ModelConfig` from environment variables.
    ///
    /// Reads:
    /// - `OPENAI_API_KEY` (required)
    /// - `OPENAI_MODEL` (required)
    /// - `OPENAI_BASE_URL` (optional, defaults to `OpenAI`)
    ///
    /// # Errors
    /// Returns an error if required environment variables are missing.
    pub fn from_env() -> Result<Self> {
        let base_url =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| AgentSdkError::ConfigError("OPENAI_API_KEY not set".into()))?;
        let model = std::env::var("OPENAI_MODEL")
            .map_err(|_| AgentSdkError::ConfigError("OPENAI_MODEL not set".into()))?;

        Ok(Self {
            base_url,
            api_key,
            model,
        })
    }
}

#[derive(Clone)]
pub struct OpenAI {
    pub config: ModelConfig,
    client: std::sync::Arc<OpenAIApiClient>,
}

impl std::fmt::Debug for OpenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAI")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl OpenAI {
    /// Creates a new `OpenAI` client from a [`ModelConfig`].
    #[must_use]
    pub fn new(config: ModelConfig) -> Self {
        let client =
            OpenAIApiClient::new(config.base_url.clone()).with_api_key(config.api_key.clone());
        Self {
            config,
            client: std::sync::Arc::new(client),
        }
    }

    #[must_use]
    pub fn builder() -> OpenAIBuilder {
        OpenAIBuilder::default()
    }

    fn convert_tools(options: &AgentOptions) -> Result<Vec<ChatCompletionTool>> {
        let Some(tools) = options.tool_definitions.as_ref() else {
            return Ok(Vec::new());
        };

        tools
            .iter()
            .map(|t| {
                Ok(ChatCompletionTool {
                    function: types::FunctionObject {
                        description: Some(t.description.clone()),
                        name: t.name.clone(),
                        parameters: serde_json::to_value(&t.input_schema)?,
                    },
                    r#type: types::ChatCompletionToolType::Function,
                })
            })
            .collect()
    }

    fn build_request(
        &self,
        options: &AgentOptions,
        messages: &[Message],
    ) -> Result<types::ChatCompletionRequest> {
        let tools = Self::convert_tools(options)?;
        let model = options.model.as_deref().unwrap_or(&self.config.model);

        Ok(types::ChatCompletionRequest {
            messages: messages.to_vec(),
            model: types::ChatCompletionRequestModel::String(model.to_owned()),
            tools: if tools.is_empty() { None } else { Some(tools) },
            temperature: options.temperature.map(f64::from),
            max_tokens: options.max_tokens.map(i64::from),
            top_p: options.top_p.map(f64::from),
            stop: options
                .stop
                .as_ref()
                .map(|s| types::ChatCompletionRequestStop::Array(s.clone())),
            stream: Some(true),
            ..Default::default()
        })
    }

    #[allow(clippy::missing_errors_doc)]
    #[tracing::instrument(skip(self, options, messages), fields(model = %self.config.model))]
    pub async fn stream_step(
        &self,
        options: &AgentOptions,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<types::CreateChatCompletionStreamResponse>> + Send>>>
    {
        let req = self.build_request(options, messages)?;
        let stream = self.client.stream_chat(req).await?;
        Ok(Box::pin(stream.map(|res| res.map_err(Into::into))))
    }
}

#[derive(Debug, Default)]
pub struct OpenAIBuilder {
    config: Option<ModelConfig>,
}

impl OpenAIBuilder {
    #[must_use]
    pub fn config(mut self, config: ModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build(self) -> Result<OpenAI> {
        let config = self
            .config
            .ok_or_else(|| AgentSdkError::ConfigError("config required".into()))?;
        Ok(OpenAI::new(config))
    }
}
