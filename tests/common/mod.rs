use async_trait::async_trait;
use futures::stream::{self};
use o3gen_openai::CreateChatCompletionStreamResponse;
use o3gen_openai::types;
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use agentsdk::core::agent::{AgentOptions, LLMBackend};
use agentsdk::core::messages::Message;
use agentsdk::error::Result as SdkResult;

/// A test LLM backend that returns pre-configured responses.
///
/// Each call to `stream()` consumes one response group from the queue,
/// enabling multi-turn testing.
#[allow(dead_code)]
pub struct TestLLMBackend {
    responses: Arc<Mutex<VecDeque<Vec<CreateChatCompletionStreamResponse>>>>,
}

#[allow(dead_code)]
impl TestLLMBackend {
    pub fn new(responses: Vec<Vec<CreateChatCompletionStreamResponse>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }

    pub fn text_chunk(content: &str) -> Vec<CreateChatCompletionStreamResponse> {
        vec![CreateChatCompletionStreamResponse {
            id: "test_id".into(),
            object: types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 0,
            model: "test".into(),
            system_fingerprint: None,
            choices: vec![types::CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta: types::ChatCompletionStreamResponseDelta {
                    content: Some(content.into()),
                    role: Some(types::ChatCompletionStreamResponseDeltaRole::Assistant),
                    function_call: None,
                    tool_calls: None,
                },
                finish_reason: Some(
                    types::CreateChatCompletionStreamResponseChoicesFinishReason::Stop,
                ),
            }],
        }]
    }

    pub fn tool_call_chunk(
        name: &str,
        arguments: &str,
        tool_id: &str,
    ) -> Vec<CreateChatCompletionStreamResponse> {
        vec![CreateChatCompletionStreamResponse {
            id: "test_id".into(),
            object: types::CreateChatCompletionStreamResponseObject::ChatCompletionChunk,
            created: 0,
            model: "test".into(),
            system_fingerprint: None,
            choices: vec![types::CreateChatCompletionStreamResponseChoices {
                index: 0,
                delta: types::ChatCompletionStreamResponseDelta {
                    content: None,
                    role: Some(types::ChatCompletionStreamResponseDeltaRole::Assistant),
                    function_call: None,
                    tool_calls: Some(vec![types::ChatCompletionMessageToolCallChunk {
                        index: 0,
                        id: Some(tool_id.into()),
                        r#type: Some(types::ChatCompletionMessageToolCallChunkType::Function),
                        function: Some(types::ChatCompletionMessageToolCallChunkFunction {
                            name: Some(name.into()),
                            arguments: Some(arguments.into()),
                        }),
                    }]),
                },
                finish_reason: Some(
                    types::CreateChatCompletionStreamResponseChoicesFinishReason::ToolCalls,
                ),
            }],
        }]
    }
}

#[async_trait]
impl LLMBackend for TestLLMBackend {
    async fn stream(
        &self,
        _options: &AgentOptions,
        _messages: &[Message],
    ) -> SdkResult<
        Pin<Box<dyn futures::Stream<Item = SdkResult<CreateChatCompletionStreamResponse>> + Send>>,
    > {
        let chunks = match self.responses.lock() {
            Ok(mut guard) => guard.pop_front(),
            Err(poisoned) => poisoned.into_inner().pop_front(),
        }
        .ok_or_else(|| {
            agentsdk::error::AgentSdkError::ConfigError("No more test responses".into())
        })?;
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn get_json(
        &self,
        _options: &AgentOptions,
        _messages: &[Message],
        _schema: &Value,
    ) -> SdkResult<Value> {
        Ok(serde_json::json!({"test": true}))
    }
}

pub fn init_llm_test() -> Option<agentsdk::OpenAI> {
    dotenv::dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Skipping LLM test: OPENAI_API_KEY not set");
        return None;
    }

    let config = agentsdk::ModelConfig::from_env().ok()?;
    Some(agentsdk::OpenAI::new(config))
}
