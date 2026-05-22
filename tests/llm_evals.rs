use agentsdk::{Agent, AgentOptions, MemoryHistoryPlugin, Message, messages};
use agentsdk_plugin_skills::SkillsPlugin;
use std::path::PathBuf;

mod common;

#[tokio::test]
async fn test_cpp_joke_loading() -> Result<(), Box<dyn std::error::Error>> {
    let Some(client) = common::init_llm_test() else {
        return Ok(());
    };

    // Setup Skills Plugin with the joke skill from examples
    let skills = SkillsPlugin::builder()
        .search_paths(vec![PathBuf::from("examples/agent/.agentsdk/skills")])
        .build()?;

    let history = MemoryHistoryPlugin::new();
    history.push(messages::user("tell me a cpp joke")).await;

    let mut agent = Agent::builder()
        .client(client)
        .options(AgentOptions::builder().temperature(0.0).build()?)
        .plugin(history.clone())
        .plugin(skills)
        .build()?;

    let _ = agent.run().await?;

    let msgs = history.messages().await;

    // Find if skills__load_reference was called with joke/CPP.md
    let mut found_tool_call = false;
    for msg in &msgs {
        if let Message::AssistantMessage(a) = msg
            && let Some(tool_calls) = &a.tool_calls
        {
            for call in tool_calls {
                if call.function.name == "skills__load_reference"
                    && call.function.arguments.contains("joke/CPP.md")
                {
                    found_tool_call = true;
                }
            }
        }
    }

    assert!(
        found_tool_call,
        "Agent should have called load_reference for joke/CPP.md"
    );
    Ok(())
}

#[tokio::test]
async fn test_cpp_joke_content() -> Result<(), Box<dyn std::error::Error>> {
    let Some(client) = common::init_llm_test() else {
        return Ok(());
    };

    let skills = SkillsPlugin::builder()
        .search_paths(vec![PathBuf::from("examples/agent/.agentsdk/skills")])
        .build()?;

    let history = MemoryHistoryPlugin::new();
    history.push(messages::user("tell me a cpp joke")).await;

    let mut agent = Agent::builder()
        .client(client)
        .options(AgentOptions::builder().temperature(0.0).build()?)
        .plugin(history.clone())
        .plugin(skills)
        .build()?;

    let _ = agent.run().await?;

    let msgs = history.messages().await;

    // Check if the last assistant message contains the punchline from CPP.md
    // Punchline: "Because they can't C#"
    let mut found_punchline = false;
    if let Some(Message::AssistantMessage(a)) = msgs.last()
        && let Some(content) = &a.content
        && content.contains("C#")
    {
        found_punchline = true;
    }

    assert!(
        found_punchline,
        "Assistant response should contain the punchline from the reference file"
    );
    Ok(())
}
