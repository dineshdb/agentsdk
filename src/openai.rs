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
        let mut req = self.build_request(options, messages)?;
        // Ask the provider for a terminal usage chunk (token counts,
        // cache/reasoning breakdown). Ignored by providers that don't
        // support it.
        req.stream_options = Some(types::StreamOptions {
            include_usage: Some(true),
        });
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
            usage: None,
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

    // Providers honoring `stream_options.include_usage` terminate the
    // stream with a usage-only chunk (empty choices, `usage` set). The
    // generated chunk type must deserialize it, nulls included.
    #[tokio::test]
    async fn test_stream_parses_terminal_usage_chunk() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let sse = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150,\"prompt_tokens_details\":{\"cached_tokens\":80},\"completion_tokens_details\":{\"reasoning_tokens\":20}}}\n\n",
            "data: [DONE]\n\n",
        );

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

        let mut usage = None;
        while let Some(chunk) = stream.next().await {
            if let Some(u) = chunk?.usage {
                usage = Some(u);
            }
        }
        let usage = usage
            .ok_or_else(|| AgentSdkError::ConfigError("terminal usage chunk must parse".into()))?;
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
        assert_eq!(usage.total_tokens, Some(150));
        assert_eq!(
            usage.prompt_tokens_details.map(|d| d.cached_tokens),
            Some(Some(80))
        );
        assert_eq!(
            usage.completion_tokens_details.map(|d| d.reasoning_tokens),
            Some(Some(20))
        );
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

    // A gateway that hangs up mid-stream (body truncated, connection closed
    // early) must surface as a transport error — ApiError::Reqwest with a
    // decode cause — so callers can classify and retry it, instead of the
    // opaque Builder string every SSE failure used to collapse into.
    #[tokio::test]
    async fn test_stream_truncated_body_is_transport_error() -> Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
            // Lie about Content-Length, then hang up mid-body.
            let body = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";
            let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 600\r\nconnection: close\r\n\r\n";
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(body.as_bytes()).await;
            let _ = sock.shutdown().await;
        });

        let client = OpenAI::new(ModelConfig {
            base_url: format!("http://{addr}"),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
        });
        let options = AgentOptions::default();
        let messages = vec![crate::messages::user("Hi")];
        let mut stream = client
            .stream(&options, &messages)
            .await
            .map_err(|e| AgentSdkError::ConfigError(format!("headers must arrive: {e}")))?;

        let mut saw_partial = false;
        let mut transport_err = None;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(_) => saw_partial = true,
                Err(e) => {
                    transport_err = Some(e);
                    break;
                }
            }
        }
        let Some(err) = transport_err else {
            return Err(AgentSdkError::ConfigError(
                "truncated body must yield an error".into(),
            ));
        };
        assert!(saw_partial, "delivered bytes should stream before the drop");
        assert!(err.is_transport(), "expected a transport error, got {err}");
        assert!(
            matches!(
                &err,
                AgentSdkError::ApiError(ApiError::Reqwest(e)) if e.is_decode()
            ),
            "expected a body decode error, got {err}"
        );
        Ok(())
    }

    // OpenRouter-style gateways commit 200 and report later failures (rate
    // limits, provider drops) as an SSE chunk carrying a top-level `error`
    // payload. The provider's message and code must survive instead of dying
    // as a missing-field serde error — and the code must classify so retry
    // budgets apply.
    #[tokio::test]
    async fn test_stream_surfaces_midstream_error_chunk() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let sse = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"error\":{\"message\":\"Rate limit exceeded: free models per day\",\"code\":429}}\n\n",
            "data: [DONE]\n\n",
        );
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

        let Some(Ok(first)) = stream.next().await else {
            return Err(AgentSdkError::ConfigError(
                "the chunk before the error must parse".into(),
            ));
        };
        let Some(choice) = first.choices.first() else {
            return Err(AgentSdkError::ConfigError("first chunk has choices".into()));
        };
        assert_eq!(choice.delta.content.as_deref(), Some("hi"));

        let Some(Err(err)) = stream.next().await else {
            return Err(AgentSdkError::ConfigError(
                "the error chunk must surface an error".into(),
            ));
        };
        let AgentSdkError::ApiError(ApiError::Stream { message, code }) = &err else {
            return Err(AgentSdkError::ConfigError(format!(
                "expected a Stream error, got {err}"
            )));
        };
        assert!(message.contains("Rate limit exceeded"), "got {message}");
        assert_eq!(*code, Some(429));
        assert_eq!(
            err.status_code(),
            Some(reqwest::StatusCode::TOO_MANY_REQUESTS)
        );
        Ok(())
    }

    // Some gateways signal failure with a well-formed chunk whose
    // `finish_reason` is `"error"` — a variant the typed enum rejects. That
    // must surface as a Stream error, not a serde error.
    #[tokio::test]
    async fn test_stream_surfaces_finish_reason_error() -> Result<()> {
        let mut mock = MockServer::new().await;
        let client = openai(&mock);

        let sse = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n",
            "data: [DONE]\n\n",
        );
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

        let Some(Err(err)) = stream.next().await else {
            return Err(AgentSdkError::ConfigError(
                "the finish_reason error chunk must surface an error".into(),
            ));
        };
        let AgentSdkError::ApiError(ApiError::Stream { message, code }) = &err else {
            return Err(AgentSdkError::ConfigError(format!(
                "expected a Stream error, got {err}"
            )));
        };
        assert!(message.contains("finish_reason"), "got {message}");
        assert_eq!(*code, None);
        assert_eq!(err.status_code(), None);
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
            usage: None,
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
            usage: None,
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
