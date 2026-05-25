use agentsdk::core::{Sandbox, Unsandboxed};
use agentsdk::{
    Agent, AgentOptions, AgentPlugin, MemoryHistoryPlugin, Message, OpenAI, PluginContext, messages,
};
use agentsdk_plugin_agentsmd::AgentsMdPlugin;
use agentsdk_plugin_fs::FileSystemPlugin;
use agentsdk_plugin_skills::SkillsPlugin;
use async_trait::async_trait;
use std::error::Error;
use std::io::{self, Write};
use tracing::Level;
use tracing_subscriber::EnvFilter;

/// A simple plugin to log tool calls and responses for observability.
#[derive(Debug, Clone, Default)]
struct LoggingPlugin;

#[async_trait]
impl AgentPlugin for LoggingPlugin {
    fn name(&self) -> &'static str {
        "logger"
    }

    async fn on_tool_pre_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> agentsdk::PreToolAction {
        tracing::info!(tool = %name, args = ?args, "Executing tool");
        agentsdk::PreToolAction::Proceed(None)
    }

    async fn on_tool_post_execute(
        &mut self,
        _ctx: &mut PluginContext,
        _id: &str,
        name: &str,
        result: &Result<serde_json::Value, String>,
    ) -> agentsdk::PostToolAction {
        match result {
            Ok(val) => {
                tracing::info!(tool = %name, "Tool execution successful");
                tracing::debug!(tool = %name, result = ?val, "Tool result");
            }
            Err(err) => {
                tracing::error!(tool = %name, error = %err, "Tool execution failed");
            }
        }
        agentsdk::PostToolAction::Proceed(None)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    // Initialize high-observability logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(Level::INFO.into())
                .add_directive("agentsdk=debug".parse()?)
                .add_directive("agent_example=debug".parse()?),
        )
        .with_target(true)
        .without_time()
        .init();

    tracing::info!("Starting Advanced Agent Example with Skills and FileSystem");

    let config = agentsdk::ModelConfig::from_env()?;
    let client = OpenAI::new(config);

    // 1. Initialize System Prompt Plugin (AgentsMd)
    // Resolves prompts from global config, project root, and local PWD
    let agentsmd = AgentsMdPlugin::builder()
        .search_paths(vec!["examples/agent/.agentsdk/AGENTS.md".into()])
        // In a real app, you might find the project root via git or env
        .project_root("examples/agent")
        .build()?;

    // 2. Initialize Skills Plugin
    // Scans and caches skills from global and local directories
    let skills = SkillsPlugin::builder()
        .search_paths(vec![std::path::PathBuf::from(
            "examples/agent/.agentsdk/skills",
        )])
        .build()?;

    // 3. Initialize FileSystem Plugin
    // Provides atomic file operations
    let fs = FileSystemPlugin::new();

    // 4. Memory History Plugin
    let history = MemoryHistoryPlugin::new();

    println!("--- Advanced Skills Agent ---");
    println!("Type 'exit' to quit. Try asking to load a skill or list files.");

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        if input == "/skills" {
            println!("\nAvailable Skills:");
            let available = skills.available_skills();
            if available.is_empty() {
                println!("  No skills found in search paths.");
            } else {
                for (name, desc, refs) in available {
                    println!("  - {name}: {desc}");
                    for r in refs {
                        println!("      - Reference: {} ({})", r.title, r.path);
                    }
                }
            }
            continue;
        }

        history.push(messages::user(input)).await;

        // Build the agent with all plugins
        let mut agent = Agent::builder()
            .client(client.clone())
            .options(AgentOptions::builder().temperature(0.0).build()?)
            .component(Sandbox::new(Unsandboxed))
            .plugin(history.clone())
            .plugin(agentsmd.clone())
            .plugin(skills.clone())
            .plugin(fs.clone())
            .plugin(LoggingPlugin) // High-observability via local plugin
            .build()?;

        // Run the agent loop
        let _output = agent.run().await?;

        // The last message is in history, let's find the content of the last assistant message
        let msgs = history.messages().await;
        if let Some(Message::AssistantMessage(a)) = msgs.last()
            && let Some(content) = &a.content
        {
            println!("\nAssistant: {content}");
        }
    }

    Ok(())
}
