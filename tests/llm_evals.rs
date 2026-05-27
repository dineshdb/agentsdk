#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

use agentsdk::core::sandbox::{Sandbox, Unsandboxed};
use agentsdk::{Agent, AgentOptions, MemoryHistoryPlugin, Message, messages};
use agentsdk_plugin_skills::SkillsPlugin;
use serde_json::Value;
use std::path::PathBuf;

mod common;

struct ToolCall {
    name: String,
    arguments: Value,
}

struct Trace {
    tool_calls: Vec<ToolCall>,
    final_text: Option<String>,
}

impl Trace {
    fn from_messages(msgs: &[Message]) -> Self {
        let mut tool_calls = Vec::new();
        let mut final_text = None;

        for msg in msgs {
            if let Message::AssistantMessage(a) = msg {
                if let Some(calls) = &a.tool_calls {
                    for call in calls {
                        let args: Value =
                            serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
                        tool_calls.push(ToolCall {
                            name: call.function.name.clone(),
                            arguments: args,
                        });
                    }
                }
                if let Some(content) = &a.content {
                    final_text = Some(content.clone());
                }
            }
        }

        Self {
            tool_calls,
            final_text,
        }
    }

    fn tool_names(&self) -> Vec<&str> {
        self.tool_calls.iter().map(|c| c.name.as_str()).collect()
    }

    fn find_call(&self, name: &str) -> Option<&ToolCall> {
        self.tool_calls.iter().find(|c| c.name == name)
    }

    #[allow(dead_code)]
    fn has_tool(&self, name: &str) -> bool {
        self.tool_calls.iter().any(|c| c.name == name)
    }

    fn assert_tool(&self, name: &str) -> &ToolCall {
        self.find_call(name).unwrap_or_else(|| {
            panic!(
                "Expected tool call '{name}' but got: {:?}",
                self.tool_names()
            )
        })
    }

    fn assert_called(&self, name: &str) -> &Self {
        self.assert_tool(name);
        self
    }

    #[allow(dead_code)]
    fn assert_not_called(&self, name: &str) -> &Self {
        assert!(
            !self.has_tool(name),
            "Tool '{name}' should NOT have been called. Calls: {:?}",
            self.tool_names()
        );
        self
    }

    fn assert_arg(&self, tool_name: &str, key: &str, expected: &Value) -> &Self {
        let call = self.assert_tool(tool_name);
        let actual = call.arguments.get(key).unwrap_or_else(|| {
            panic!(
                "Tool '{tool_name}' args have no key '{key}'. Args: {}",
                serde_json::to_string_pretty(&call.arguments).unwrap_or_default()
            )
        });
        assert_eq!(
            actual, expected,
            "Tool '{tool_name}' arg '{key}': expected {expected}, got {actual}"
        );
        self
    }

    #[allow(dead_code)]
    fn assert_arg_contains(&self, tool_name: &str, key: &str, substring: &str) -> &Self {
        let call = self.assert_tool(tool_name);
        let actual = call
            .arguments
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                panic!(
                    "Tool '{tool_name}' arg '{key}' is not a string. Args: {}",
                    serde_json::to_string_pretty(&call.arguments).unwrap_or_default()
                )
            });
        assert!(
            actual.contains(substring),
            "Tool '{tool_name}' arg '{key}': expected to contain '{substring}', got '{actual}'"
        );
        self
    }

    fn assert_sequence(&self, names: &[&str]) -> &Self {
        let actual = self.tool_names();
        let mut actual_iter = actual.iter();
        for expected in names {
            let found = actual_iter.find(|&&a| a == *expected);
            assert!(
                found.is_some(),
                "Expected tool '{expected}' in sequence.\nExpected sequence: {names:?}\nActual calls: {actual:?}"
            );
        }
        self
    }

    fn assert_final_contains(&self, substring: &str) -> &Self {
        let text = self.final_text.as_deref().unwrap_or_else(|| {
            panic!(
                "No final text in trace. Tool calls: {:?}",
                self.tool_names()
            )
        });
        assert!(
            text.contains(substring),
            "Final text should contain '{substring}'. Got: {text}"
        );
        self
    }
}

async fn build_agent(
    client: agentsdk::OpenAI,
    prompt: &str,
) -> Result<(Agent, MemoryHistoryPlugin), Box<dyn std::error::Error>> {
    let skills = SkillsPlugin::builder()
        .search_paths(vec![PathBuf::from("examples/agent/.agentsdk/skills")])
        .build()?;

    let history = MemoryHistoryPlugin::new();
    history.push(messages::user(prompt)).await;

    let agent = Agent::builder()
        .client(client)
        .options(AgentOptions::builder().temperature(0.0).build()?)
        .component(Sandbox::new(Unsandboxed))
        .plugin(history.clone())
        .plugin(skills)
        .build()?;

    Ok((agent, history))
}

async fn run_trace(agent: &mut Agent, history: &MemoryHistoryPlugin) -> Trace {
    agent.run().await.expect("agent run failed");
    let msgs = history.messages().await;
    Trace::from_messages(&msgs)
}

/// Full workflow: find skills → load skill → load reference → answer with punchline.
#[tokio::test]
async fn test_cpp_joke_full_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let Some(client) = common::init_llm_test() else {
        return Ok(());
    };
    let (mut agent, history) = build_agent(client, "tell me a cpp joke").await?;
    let trace = run_trace(&mut agent, &history).await;

    trace
        .assert_sequence(&["FindSkills", "LoadSkillReference"])
        .assert_called("FindSkills")
        .assert_arg(
            "FindSkills",
            "query",
            &serde_json::json!("tell me a cpp joke"),
        )
        .assert_called("LoadSkillReference")
        .assert_arg("LoadSkillReference", "skill", &serde_json::json!("joke"))
        .assert_arg_contains("LoadSkillReference", "reference", "CPP")
        .assert_final_contains("C#");

    Ok(())
}
