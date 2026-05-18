use crate::core::messages::{self, Message, Messages, ToolCall, ToolFunction};
use crate::core::tools::{Tool, ToolContext, ToolDefinition, ToolExecute};
use crate::error::{AgentSdkError, Result};
use crate::openai::OpenAI;
use crate::openai::api::types;
use async_stream::stream;
use derive_builder::Builder;
use futures::{Stream, StreamExt};
use o3gen_openai::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageRole,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

const DEFAULT_MAX_STEPS: usize = 25;

#[derive(Clone, Default, Builder, Debug)]
#[builder(pattern = "owned", setter(into))]
pub struct AgentOptions {
    #[builder(default)]
    pub extensions: crate::core::extensions::Extensions,
    #[builder(default)]
    pub model: String,
    #[builder(default)]
    pub system: Option<String>,
    #[builder(default)]
    pub temperature: Option<f32>,
    #[builder(default)]
    pub max_tokens: Option<u32>,
    #[builder(default)]
    pub top_p: Option<f32>,
    #[builder(default)]
    pub stop: Option<Vec<String>>,
    #[builder(default)]
    pub max_steps: Option<usize>,
    #[builder(default)]
    pub messages: Option<Arc<Messages>>,
    #[builder(default)]
    pub tool_definitions: Option<Arc<Vec<ToolDefinition>>>,
    #[builder(default)]
    pub tool_executors: Option<Arc<HashMap<String, ToolExecute>>>,
}

impl AgentOptions {
    #[must_use]
    pub fn builder() -> AgentOptionsBuilder {
        AgentOptionsBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TextDelta(String),
    PreToolExecute {
        id: String,
        name: String,
        arguments: Value,
    },
    PostToolExecute {
        id: String,
        name: String,
        result: Value,
    },
    ToolExecuteError {
        id: String,
        name: String,
        error: String,
    },
    StepComplete(Message),
    Finished(Arc<Messages>),
}

pub type AgentStream = Pin<Box<dyn Stream<Item = Result<AgentEvent>> + Send>>;

#[derive(Debug, Default)]
pub struct AgentBuilder {
    client: Option<OpenAI>,
    options: AgentOptions,
}

impl AgentBuilder {
    #[must_use]
    pub fn client(mut self, client: OpenAI) -> Self {
        self.client = Some(client);
        self
    }

    #[must_use]
    pub fn options(mut self, options: AgentOptions) -> Self {
        self.options = options;
        self
    }

    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.options.system = Some(system.into());
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build(self) -> Result<Agent> {
        let client = self
            .client
            .ok_or_else(|| AgentSdkError::ConfigError("Client required".into()))?;
        Ok(Agent {
            client,
            options: self.options,
        })
    }
}

impl AgentOptionsBuilder {
    #[must_use]
    pub fn with_tool(mut self, tool: &Tool) -> Self {
        let mut defs = self
            .tool_definitions
            .take()
            .flatten()
            .map_or_else(Vec::new, |arc| {
                Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
            });
        let mut execs = self
            .tool_executors
            .take()
            .flatten()
            .map_or_else(HashMap::new, |arc| {
                Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
            });

        defs.push(tool.definition.clone());
        execs.insert(tool.definition.name.clone(), tool.execute.clone());

        self.tool_definitions = Some(Some(Arc::new(defs)));
        self.tool_executors = Some(Some(Arc::new(execs)));
        self
    }
}

#[derive(Debug)]
pub struct Agent {
    client: OpenAI,
    options: AgentOptions,
}

#[derive(Default)]
struct StepAccumulator {
    content: String,
    tool_calls: BTreeMap<i64, ToolCall>,
}

impl StepAccumulator {
    fn push(&mut self, chunk: &types::CreateChatCompletionStreamResponse) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        let Some(choice) = chunk.choices.first() else {
            return events;
        };

        if let Some(content) = &choice.delta.content {
            self.content.push_str(content);
            events.push(AgentEvent::TextDelta(content.clone()));
        }

        if let Some(deltas) = &choice.delta.tool_calls {
            for delta in deltas {
                let entry = self
                    .tool_calls
                    .entry(delta.index)
                    .or_insert_with(|| ToolCall {
                        id: String::new(),
                        r#type: types::ToolCallType::Function,
                        function: ToolFunction {
                            name: String::new(),
                            arguments: String::new(),
                        },
                    });

                if let Some(id) = &delta.id {
                    entry.id.clone_from(id);
                }

                if let Some(f) = &delta.function {
                    if let Some(name) = &f.name {
                        entry.function.name.clone_from(name);
                    }
                    if let Some(args) = &f.arguments {
                        entry.function.arguments.push_str(args);
                    }
                }
            }
        }

        if choice.finish_reason.is_some() {
            events.push(AgentEvent::StepComplete(self.finish()));
        }

        events
    }

    fn finish(&self) -> Message {
        let tool_calls = if self.tool_calls.is_empty() {
            None
        } else {
            Some(self.tool_calls.values().cloned().collect())
        };
        let content = (!self.content.is_empty()).then_some(self.content.clone());

        Message::AssistantMessage(ChatCompletionRequestAssistantMessage {
            content,
            name: None,
            tool_calls,
            role: ChatCompletionRequestAssistantMessageRole::Assistant,
            function_call: None,
        })
    }
}

fn tool_error(id: String, name: String, error: String) -> (AgentEvent, Value) {
    (
        AgentEvent::ToolExecuteError {
            id,
            name,
            error: error.clone(),
        },
        Value::String(error),
    )
}

fn tool_success(id: String, name: String, result: Value) -> (AgentEvent, Value) {
    (
        AgentEvent::PostToolExecute {
            id,
            name,
            result: result.clone(),
        },
        result,
    )
}

impl Agent {
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    #[must_use]
    #[allow(tail_expr_drop_order)]
    pub fn stream(self) -> AgentStream {
        let mut history = self
            .options
            .messages
            .as_ref()
            .map(|m| (**m).clone())
            .unwrap_or_default();
        if let Some(system) = &self.options.system {
            history.insert(0, messages::system(system));
        }
        let max_steps = self.options.max_steps.unwrap_or(DEFAULT_MAX_STEPS);

        let s = stream! {
            let mut current_history = history;
            for _ in 0..max_steps {
                let opts = AgentOptions {
                    messages: Some(Arc::new(current_history.clone())),
                    ..self.options.clone()
                };

                let mut upstream = match self.client.stream_step(&opts).await {
                    Ok(s) => s,
                    Err(e) => { yield Err(e); break; }
                };

                let mut acc = StepAccumulator::default();
                while let Some(chunk) = upstream.next().await {
                    match chunk {
                        Ok(chunk) => {
                            for event in acc.push(&chunk) {
                                if let AgentEvent::StepComplete(ref msg) = event {
                                    current_history.push(msg.clone());
                                }
                                yield Ok(event);
                            }
                        }
                        Err(e) => { yield Err(AgentSdkError::ApiError(e.to_string())); return; }
                    }
                }

                let calls = if let Some(Message::AssistantMessage(assistant)) = current_history.last() {
                    assistant.tool_calls.clone()
                } else {
                    None
                };

                if let Some(calls) = calls {
                    for call in calls {
                        let args = serde_json::from_str::<Value>(&call.function.arguments)
                            .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));

                        yield Ok(AgentEvent::PreToolExecute {
                            id: call.id.clone(),
                            name: call.function.name.clone(),
                            arguments: args.clone(),
                        });

                        let (event, result) = self.execute_tool(&call.id, &call.function.name, args).await;
                        yield Ok(event);

                        let content = match &result {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        current_history.push(messages::tool(content, &call.id));
                    }
                } else {
                    yield Ok(AgentEvent::Finished(Arc::new(current_history)));
                    return;
                }
            }

            yield Ok(AgentEvent::Finished(Arc::new(current_history)));
        };
        Box::pin(s)
    }

    async fn execute_tool(&self, id: &str, name: &str, args: Value) -> (AgentEvent, Value) {
        let Some(executor) = self
            .options
            .tool_executors
            .as_ref()
            .and_then(|m| m.get(name))
        else {
            return tool_error(id.into(), name.into(), format!("Tool {name} not found"));
        };

        let ctx = ToolContext {
            options: Arc::new(self.options.clone()),
        };

        match executor.call(ctx, args).await {
            Ok(r) => tool_success(id.into(), name.into(), r),
            Err(e) => tool_error(id.into(), name.into(), e),
        }
    }
}
