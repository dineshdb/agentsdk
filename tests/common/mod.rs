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
            usage: None,
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
            usage: None,
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

/// The Citadel gateway: it holds the provider credentials and routes by model
/// id, so these tests carry no key of their own.
const GATEWAY: &str = "http://citadel.lvh.me/v1";

/// A local llama-swap model, reached through the gateway. Free and offline,
/// and it passes this suite — which is the bar, since an eval is only worth
/// running on a model that can actually do the task.
///
/// Measured 2026-09-06 against `test_cpp_joke_full_workflow`:
/// - `@llama-swap/lfm8a1` — passes (3/3)
/// - `lfm2.5-8A1B` — passes (local too)
/// - `@openrouter/deepseek/deepseek-v4-flash-0731` (alias `dsv4f`) — FAILS:
///   answers after `FindSkills` without loading the reference
/// - `@llama-swap/apple-foundationmodel` — FAILS
const MODEL: &str = "@llama-swap/lfm8a1";

/// A live-LLM client for the evals, or `None` to skip them.
///
/// Defaults to a local model behind Citadel; override any part with
/// `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` (or a `.env`) to point
/// at `OpenRouter`, a direct provider, or a model you are evaluating.
///
/// Skipping happens only when nothing is listening on the gateway — a machine
/// without it still gets a green suite, but a reachable gateway always runs
/// the eval rather than quietly passing.
pub fn init_llm_test() -> Option<agentsdk::OpenAI> {
    dotenv::dotenv().ok();

    let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| GATEWAY.to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| MODEL.to_string());
    // The gateway authenticates upstream; the header just has to be present.
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "citadel".to_string());

    if !is_listening(&base_url) {
        eprintln!(
            "skipping LLM eval: nothing listening at {base_url} \
             (point OPENAI_BASE_URL/OPENAI_MODEL somewhere else to run it)"
        );
        return None;
    }

    eprintln!("LLM eval: {model} via {base_url}");
    Some(agentsdk::OpenAI::new(agentsdk::ModelConfig {
        base_url,
        api_key,
        model,
    }))
}

/// Can we open a TCP connection to the endpoint's host:port? Cheaper and more
/// honest than a request — it answers "is the gateway there", nothing more.
fn is_listening(base_url: &str) -> bool {
    use std::net::{TcpStream, ToSocketAddrs};

    let rest = base_url.split_once("://").map_or(base_url, |(_, r)| r);
    let authority = rest.split('/').next().unwrap_or(rest);
    let default_port = if base_url.starts_with("https") {
        443
    } else {
        80
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse().unwrap_or(default_port)),
        None => (authority, default_port),
    };

    (host, port)
        .to_socket_addrs()
        .into_iter()
        .flatten()
        .any(|addr| {
            TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(750)).is_ok()
        })
}
