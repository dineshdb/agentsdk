use crate::core::agent::AgentOptions;
use crate::core::messages::Message;
use crate::error::{AgentSdkError, Result};
use api::OpenAIApi;
use api::OpenAIApiClient;
use api::types;
use futures::{Stream, StreamExt};
pub use o3gen_openai as api;
use o3gen_openai::{ApiError, ChatCompletionTool};
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
    /// Send a non-streaming chat completion request and return the content of the first choice.
    pub async fn text(&self, options: &AgentOptions, messages: &[Message]) -> Result<String> {
        let mut req = self.build_request(options, messages)?;
        req.stream = Some(false);

        let resp = OpenAIApi::create_chat_completion(&*self.client, req).await?;
        let content = resp
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| {
                AgentSdkError::ApiError(ApiError::Builder("No content in response".to_string()))
            })?;
        Ok(content)
    }

    #[allow(clippy::missing_errors_doc)]
    /// Call the model and deserialize the response into T based on its JSON schema.
    pub async fn json<T>(&self, options: &AgentOptions, messages: &[Message]) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let mut req = self.build_request(options, messages)?;
        req.stream = Some(false);
        let val = self.client.json::<T>(req).await?;
        Ok(val)
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn get_json(
        &self,
        options: &AgentOptions,
        messages: &[Message],
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut req = self.build_request(options, messages)?;
        req.stream = Some(false);
        // We use the underlying client's get_json which handles the schema injection.
        let val = self.client.get_json(req, schema).await?;
        Ok(val)
    }

    #[allow(clippy::missing_errors_doc)]
    #[tracing::instrument(skip(self, options, messages), fields(model = %self.config.model))]
    pub async fn stream(
        &self,
        options: &AgentOptions,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<types::CreateChatCompletionStreamResponse>> + Send>>>
    {
        let req = self.build_request(options, messages)?;
        let stream = self.client.stream_chat(req).await?;
        Ok(Box::pin(stream.map(|res| res.map_err(Into::into))))
    }

    #[allow(clippy::missing_errors_doc)]
    /// Fetch available models from the provider.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let resp = OpenAIApi::list_models(&*self.client).await?;
        Ok(resp.data.into_iter().map(|m| m.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use api::types;
    use o3gen_openai::test_helpers::mock::MockServer;

    fn openai(mock: &MockServer) -> OpenAI {
        let config = ModelConfig {
            base_url: mock.url(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
        };
        OpenAI::new(config)
    }

    // ── Builder ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_openai_builder_with_config() -> Result<()> {
        let mut mock = MockServer::new().await;
        let config = ModelConfig {
            base_url: mock.url(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
        };
        let client = OpenAI::builder().config(config).build()?;
        let _m = mock
            .server
            .mock("GET", "/models")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::to_string(&types::ListModelsResponse {
                object: types::ListModelsResponseObject::List,
                data: vec![types::Model {
                    id: "gpt-4o".into(),
                    object: types::ModelObject::Model,
                    created: 1_661_989_079,
                    owned_by: "openai".into(),
                }],
            })?)
            .create();
        let models = client.list_models().await?;
        assert_eq!(models.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_openai_builder_missing_config() {
        let err = OpenAI::builder().build();
        assert!(err.is_err());
    }

    // ── Stream ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_stream_returns_text_content() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let mut sse = String::new();
        let resp = types::CreateChatCompletionStreamResponse {
            id: "chatcmpl-abc123".into(),
            object: types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 1_677_610_605,
            model: "gpt-4o".into(),
            system_fingerprint: None,
            choices: vec![types::CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta: types::ChatCompletionStreamResponseDelta {
                    content: Some("Hello! How can I help?".into()),
                    role: Some(types::ChatCompletionStreamResponseDeltaRole::Assistant),
                    function_call: None,
                    tool_calls: None,
                },
                finish_reason: Some(
                    types::CreateChatCompletionStreamResponseChoicesFinishReason::Stop,
                ),
            }],
        };
        sse.push_str("data: ");
        sse.push_str(&serde_json::to_string(&resp)?);
        sse.push_str("\n\n");
        sse.push_str("data: [DONE]\n\n");

        let _m = mock
            .server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create();

        let options = AgentOptions::default();
        let messages = vec![crate::messages::user("Hi")];
        let mut stream = client.stream(&options, &messages).await?;

        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for choice in &chunk.choices {
                if let Some(content) = &choice.delta.content {
                    full.push_str(content);
                }
            }
        }
        assert_eq!(full, "Hello! How can I help?");
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_api_error() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);
        let _m = mock
            .server
            .mock("POST", "/chat/completions")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"error": {"message": "Invalid model", "type": "invalid_request_error"}}"#,
            )
            .create();

        let options = AgentOptions::default();
        let messages = vec![crate::messages::user("Hi")];
        let Err(err) = client.stream(&options, &messages).await else {
            return Err(AgentSdkError::ConfigError("expected error".into()));
        };
        assert!(matches!(
            err,
            AgentSdkError::ApiError(ApiError::Status { status, .. }) if status == reqwest::StatusCode::BAD_REQUEST
        ));
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_returns_tool_calls() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let mut sse = String::new();

        let tool_call_start = types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_abc".into()),
            r#type: Some(types::ChatCompletionMessageToolCallChunkType::Function),
            function: Some(types::ChatCompletionMessageToolCallChunkFunction {
                name: Some("get_weather".into()),
                arguments: Some(String::new()),
            }),
        };
        let chunk1 = types::CreateChatCompletionStreamResponse {
            id: "chatcmpl-abc123".into(),
            object: types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 1_677_610_605,
            model: "gpt-4o".into(),
            system_fingerprint: None,
            choices: vec![types::CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta: types::ChatCompletionStreamResponseDelta {
                    content: Some(String::new()),
                    role: Some(types::ChatCompletionStreamResponseDeltaRole::Assistant),
                    function_call: None,
                    tool_calls: Some(vec![tool_call_start]),
                },
                finish_reason: None,
            }],
        };
        sse.push_str("data: ");
        sse.push_str(&serde_json::to_string(&chunk1)?);
        sse.push_str("\n\n");

        let tool_call_arg = types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(types::ChatCompletionMessageToolCallChunkFunction {
                name: None,
                arguments: Some("{\"location\":\"NYC\"}".into()),
            }),
        };
        let chunk2 = types::CreateChatCompletionStreamResponse {
            id: "chatcmpl-abc123".into(),
            object: types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 1_677_610_605,
            model: "gpt-4o".into(),
            system_fingerprint: None,
            choices: vec![types::CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta: types::ChatCompletionStreamResponseDelta {
                    content: None,
                    role: Some(types::ChatCompletionStreamResponseDeltaRole::Assistant),
                    function_call: None,
                    tool_calls: Some(vec![tool_call_arg]),
                },
                finish_reason: Some(
                    types::CreateChatCompletionStreamResponseChoicesFinishReason::ToolCalls,
                ),
            }],
        };
        sse.push_str("data: ");
        sse.push_str(&serde_json::to_string(&chunk2)?);
        sse.push_str("\n\n");
        sse.push_str("data: [DONE]\n\n");

        let _m = mock
            .server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create();

        let options = AgentOptions::default();
        let messages = vec![crate::messages::user("What's the weather?")];
        let mut stream = client.stream(&options, &messages).await?;

        let mut tool_calls_found = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for choice in &chunk.choices {
                if let Some(tcs) = &choice.delta.tool_calls {
                    for tc in tcs {
                        let name = tc.function.as_ref().and_then(|f| f.name.as_ref());
                        if let Some(name) = name {
                            assert_eq!(name, "get_weather");
                            tool_calls_found = true;
                        }
                    }
                }
            }
        }
        assert!(tool_calls_found);
        Ok(())
    }

    // Regression: verbatim OpenRouter LFM chunks (reasoning fields, split
    // tool deltas) must parse through client.stream().
    #[tokio::test]
    async fn openrouter_stream_parses_tool_calls_and_finish() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let raw = concat!(
            "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"lfm\",\"provider\":\"Liquid\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\",\"reasoning\":\"The\",\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"The\",\"format\":\"unknown\",\"index\":0}]},\"finish_reason\":null,\"native_finish_reason\":null}]}\n\n",
            "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"lfm\",\"provider\":\"Liquid\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-abc\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"lfm\",\"provider\":\"Liquid\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\": \\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"lfm\",\"provider\":\"Liquid\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let _m = mock
            .server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(raw)
            .create();

        let options = AgentOptions::default();
        let messages = vec![crate::messages::user("probe")];
        let mut stream = client.stream(&options, &messages).await?;

        let mut n_chunks = 0usize;
        let mut n_tool_deltas = 0usize;
        let mut finish: Option<String> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            n_chunks += 1;
            for choice in &chunk.choices {
                if let Some(tcs) = &choice.delta.tool_calls {
                    n_tool_deltas += tcs.len();
                }
                if let Some(f) = &choice.finish_reason {
                    finish = Some(f.to_string());
                }
            }
        }
        eprintln!("PROBE chunks={n_chunks} tool_deltas={n_tool_deltas} finish={finish:?}");
        assert!(n_tool_deltas > 0, "tool deltas lost in stream parsing");
        assert_eq!(
            finish,
            Some("ToolCalls".to_string()),
            "finish_reason must parse (strum Display form)"
        );
        Ok(())
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
