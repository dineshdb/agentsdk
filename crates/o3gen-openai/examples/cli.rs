use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use futures_util::StreamExt;
use o3gen_openai::OpenAIApiClient;
use o3gen_openai::types::*;

const SYSTEM: &str = r"
- Fix the root cause, not the symptoms. Think before reaching a conclusion: are you solving the root cause or the symptoms?
- Be terse. Small and useful response only.
- Follow through and verify your output against the user's goal.
- Solve tasks by invoking available tools whenever they might help. Prefer tools over direct answers when in doubt.
- load skills to know more about a topic and follow the instructions.
";

fn tool_def(name: &str, description: &str) -> ChatCompletionTool {
    ChatCompletionTool::builder()
        .r#type(ChatCompletionToolType::Function)
        .function(
            FunctionObject::builder()
                .name(name.to_string())
                .description(description.to_string())
                .parameters(serde_json::json!({}))
                .build(),
        )
        .build()
}

fn tools() -> Vec<ChatCompletionTool> {
    vec![
        tool_def(
            "ls",
            "List files and directories at a given path. \
             Parameters: path (string) - directory path to list",
        ),
        tool_def(
            "read",
            "Read the contents of a file at a given path. \
             Parameters: path (string) - file path to read",
        ),
        tool_def(
            "grep",
            "Search for a pattern in a file. \
             Parameters: pattern (string) - search pattern, \
             path (string) - file path to search",
        ),
    ]
}

struct AccTool {
    id: String,
    name: String,
    arguments: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into());

    let client = OpenAIApiClient::new(base_url).with_api_key(api_key);
    let mut msgs: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessage::builder()
            .content(SYSTEM)
            .role(ChatCompletionRequestSystemMessageRole::System)
            .build()
            .into(),
    ];

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n>>> ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.is_empty() || input == "exit" || input == "quit" {
            break;
        }

        msgs.push(
            ChatCompletionRequestUserMessage::builder()
                .role(ChatCompletionRequestUserMessageRole::User)
                .content(ChatCompletionRequestUserMessageContent::String(input))
                .build()
                .into(),
        );

        loop {
            let req = ChatCompletionRequest::builder()
                .model(model.clone())
                .messages(msgs.clone())
                .tools(tools())
                .build()?;

            let mut stream = client.stream_chat(req).await?;

            let mut content = String::new();
            let mut acc: HashMap<usize, AccTool> = HashMap::new();
            let mut finish_reason = None;

            println!();

            while let Some(item) = stream.next().await {
                let chunk = item?;
                for choice in &chunk.choices {
                    if let Some(delta) = &choice.delta.content {
                        content.push_str(delta);
                        print!("{delta}");
                        stdout.flush()?;
                    }
                    if let Some(tcs) = &choice.delta.tool_calls {
                        for tc in tcs {
                            let idx = tc.index as usize;
                            let entry = acc.entry(idx).or_insert_with(|| AccTool {
                                id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            });
                            if let Some(id) = &tc.id {
                                entry.id.push_str(id);
                            }
                            if let Some(func) = &tc.function {
                                if let Some(n) = &func.name {
                                    entry.name.push_str(n);
                                }
                                if let Some(a) = &func.arguments {
                                    entry.arguments.push_str(a);
                                }
                            }
                        }
                    }
                    if let Some(reason) = &choice.finish_reason {
                        finish_reason = Some(reason.clone());
                    }
                }
            }

            let assistant_content = if content.is_empty() {
                None
            } else {
                Some(content)
            };

            let tool_calls: Vec<ToolCall> = if acc.is_empty() {
                Vec::new()
            } else {
                let mut sorted: Vec<_> = acc.into_iter().collect();
                sorted.sort_by_key(|(idx, _)| *idx);
                sorted
                    .into_iter()
                    .map(|(_, t)| {
                        ToolCall::builder()
                            .id(t.id)
                            .r#type(ToolCallType::Function)
                            .function(
                                ToolCallFunction::builder()
                                    .name(t.name)
                                    .arguments(t.arguments)
                                    .build(),
                            )
                            .build()
                    })
                    .collect::<Vec<_>>()
            };

            {
                let mut b = ChatCompletionRequestAssistantMessage::builder();
                b.role(ChatCompletionRequestAssistantMessageRole::Assistant);
                if let Some(c) = assistant_content {
                    b.content(c);
                }
                if !tool_calls.is_empty() {
                    b.tool_calls(tool_calls.clone());
                }
                msgs.push(b.build().into());
            }

            match finish_reason {
                Some(CreateChatCompletionStreamResponseChoicesFinishReason::Stop) => break,
                Some(CreateChatCompletionStreamResponseChoicesFinishReason::ToolCalls) => {
                    for tc in &tool_calls {
                        let name = tc.function.name.clone();
                        let args = tc.function.arguments.clone();
                        print!(
                            "\n  └─ {} tool: {name}({args})\n",
                            tc.id.as_deref().unwrap_or("?")
                        );

                        let result = match name.as_str() {
                            "ls" => exec_tool(
                                "ls",
                                &[
                                    "-la",
                                    &extract(&args, "path").unwrap_or_else(|| ".".to_string()),
                                ],
                            ),
                            "read" => {
                                let p = extract(&args, "path").unwrap_or_default();
                                exec_tool("cat", &["--", &p])
                            }
                            "grep" => {
                                let p = extract(&args, "pattern").unwrap_or_default();
                                let f = extract(&args, "path").unwrap_or_default();
                                exec_tool("rg", &["-n", "--", &p, &f])
                            }
                            _ => Err(format!("unknown tool: {name}")),
                        };

                        let result_str = match result {
                            Ok(s) => s,
                            Err(e) => format!("Error: {e}\n"),
                        };

                        {
                            let mut b = ChatCompletionRequestToolMessage::builder();
                            b.role(ChatCompletionRequestToolMessageRole::Tool);
                            b.tool_call_id(tc.id.clone().unwrap_or_default());
                            if !result_str.is_empty() {
                                b.content(result_str);
                            }
                            msgs.push(b.build().into());
                        }
                    }
                }
                _ => break,
            }
        }
    }

    Ok(())
}

fn exec_tool(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run {cmd}: {e}"))?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        Ok(if s.is_empty() {
            "(empty)\n".to_string()
        } else {
            s
        })
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

fn extract(json: &str, key: &str) -> Option<String> {
    let needle = format!(r#""{key}":"#);
    let start = json.find(&needle)? + needle.len();
    let bytes = &json.as_bytes()[start..];
    if bytes.first()? != &b'"' {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            out.push(bytes[i + 1]);
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            break;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).ok()
}
