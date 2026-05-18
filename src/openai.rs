use crate::core::agent::AgentOptions;
use crate::error::{AgentSdkError, Result};
use crate::utils::to_terse_json;
use api::OpenAIApiClient;
use api::types;
use futures::{Stream, StreamExt};
pub use o3gen_openai as api;
use o3gen_openai::ChatCompletionTool;
use std::pin::Pin;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Clone)]
pub struct OpenAI {
    pub model: String,
    client: std::sync::Arc<OpenAIApiClient>,
}

impl std::fmt::Debug for OpenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAI")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl OpenAI {
    /// Creates a new `OpenAI` client.
    ///
    /// # Errors
    /// Returns an error if the base URL is invalid.
    pub fn new(api_key: String, base_url: String, model: String) -> Result<Self> {
        let client = OpenAIApiClient::new(base_url).with_api_key(api_key);
        Ok(Self {
            model,
            client: std::sync::Arc::new(client),
        })
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
                        parameters: to_terse_json(&serde_json::to_value(&t.input_schema)?),
                    },
                    r#type: types::ChatCompletionToolType::Function,
                })
            })
            .collect()
    }

    fn build_request(&self, options: &AgentOptions) -> Result<types::ChatCompletionRequest> {
        let tools = Self::convert_tools(options)?;
        dbg!(&tools);
        let model = if options.model.is_empty() {
            &self.model
        } else {
            &options.model
        };

        Ok(types::ChatCompletionRequest {
            messages: options
                .messages
                .as_ref()
                .map(|m| (**m).clone())
                .unwrap_or_default(),
            model: types::ChatCompletionRequestModel::String(model.clone()),
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
    pub async fn stream_step(
        &self,
        options: &AgentOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<types::CreateChatCompletionStreamResponse>> + Send>>>
    {
        let req = self.build_request(options)?;
        let stream = self
            .client
            .stream_chat(req)
            .await
            .map_err(|e| AgentSdkError::ApiError(e.to_string()))?;

        Ok(Box::pin(stream.map(|res| {
            res.map_err(|e| AgentSdkError::ApiError(e.to_string()))
        })))
    }
}

#[derive(Debug, Default)]
pub struct OpenAIBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
}

impl OpenAIBuilder {
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build(self) -> Result<OpenAI> {
        OpenAI::new(
            self.api_key
                .ok_or_else(|| AgentSdkError::ConfigError("api_key required".into()))?,
            self.base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            self.model
                .ok_or_else(|| AgentSdkError::ConfigError("model required".into()))?,
        )
    }
}
