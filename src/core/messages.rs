use o3gen_openai::types;

pub use o3gen_openai::types::{
    ChatCompletionRequestMessage as Message, ChatCompletionRole as Role, ToolCall,
    ToolCallFunction as ToolFunction,
};

pub type Messages = Vec<Message>;

pub fn system(content: impl Into<String>) -> Message {
    Message::SystemMessage(types::ChatCompletionRequestSystemMessage {
        content: Some(content.into()),
        name: None,
        role: types::ChatCompletionRequestSystemMessageRole::System,
    })
}

pub fn user(content: impl Into<String>) -> Message {
    Message::UserMessage(types::ChatCompletionRequestUserMessage {
        content: Some(types::ChatCompletionRequestUserMessageContent::String(
            content.into(),
        )),
        name: None,
        role: types::ChatCompletionRequestUserMessageRole::User,
    })
}

pub fn assistant(content: impl Into<String>) -> Message {
    Message::AssistantMessage(types::ChatCompletionRequestAssistantMessage {
        content: Some(content.into()),
        name: None,
        tool_calls: None,
        role: types::ChatCompletionRequestAssistantMessageRole::Assistant,
        function_call: None,
    })
}

pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Message {
    Message::ToolMessage(types::ChatCompletionRequestToolMessage {
        content: Some(content.into()),
        tool_call_id: tool_call_id.into(),
        role: types::ChatCompletionRequestToolMessageRole::Tool,
    })
}

pub fn assistant_tool_call(
    name: impl Into<String>,
    call_id: impl Into<String>,
    arguments: &serde_json::Value,
) -> Message {
    Message::AssistantMessage(types::ChatCompletionRequestAssistantMessage {
        content: None,
        name: None,
        tool_calls: Some(vec![ToolCall {
            id: call_id.into(),
            r#type: types::ToolCallType::Function,
            function: ToolFunction {
                name: name.into(),
                arguments: arguments.to_string(),
            },
        }]),
        role: types::ChatCompletionRequestAssistantMessageRole::Assistant,
        function_call: None,
    })
}
