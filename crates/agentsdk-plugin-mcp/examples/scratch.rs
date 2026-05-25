use agentsdk::{Agent, AgentPlugin, MemoryHistoryPlugin, OpenAI, PluginContext, messages};
use agentsdk_plugin_mcp::McpPlugin;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::process::Command;

struct PrinterPlugin;

#[async_trait]
impl AgentPlugin for PrinterPlugin {
    fn name(&self) -> &'static str {
        "printer"
    }

    fn on_text_delta(&mut self, _ctx: &mut PluginContext, text: &str) {
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    println!("--- Setting up MCP Plugin ---");
    let mut mcp_plugin = McpPlugin::new();

    // 1. Add 'everything' server (local)
    println!("Connecting to 'everything' server...");
    let mut cmd1 = Command::new("npx");
    cmd1.args(["-y", "@modelcontextprotocol/server-everything"]);
    mcp_plugin.add_server("everything", cmd1).await?;

    // 2. Add 'fetch' server (local)
    println!("Connecting to 'fetch' server...");
    let mut cmd2 = Command::new("npx");
    cmd2.args(["-y", "mcp-fetch-server"]);
    mcp_plugin.add_server("fetch", cmd2).await?;

    // 3. Add 'context7' server (remote)
    println!("Connecting to 'context7' remote server...");
    let mut headers = HashMap::new();
    if let Ok(api_key) = std::env::var("CONTEXT7_API_KEY") {
        headers.insert("CONTEXT7_API_KEY".to_string(), api_key);
    }
    // Basic connection check
    match mcp_plugin
        .add_remote_server("context7", "https://mcp.context7.com/mcp", headers)
        .await
    {
        Ok(_) => println!("Connected to Context7!"),
        Err(e) => eprintln!("Context7 connection failed: {}", e),
    }

    println!("\n--- Initializing Agent ---");
    let client = OpenAI::new(agentsdk::ModelConfig::from_env()?);
    let history = MemoryHistoryPlugin::new();

    let mut agent = Agent::builder()
        .client(client)
        .plugin(history.clone())
        .plugin(mcp_plugin)
        .plugin(PrinterPlugin)
        .build()?;

    let prompt = "Use the everything__get-sum tool to add 123 and 456, then fetch the content of 'https://example.com' using the fetch__fetch_markdown tool and summarize it briefly.";

    println!("Prompt: {}\n", prompt);
    history.push(messages::user(prompt)).await;

    print!("Assistant: ");
    std::io::Write::flush(&mut std::io::stdout())?;

    agent.run().await?;
    println!("\n\n--- Done ---");

    Ok(())
}
