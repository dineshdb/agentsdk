//! Live check: pie's MCP client against mem.qreta.io's skills face.
//!
//! ```text
//! MEM_TOKEN=... cargo run -p agentsdk-plugin-mcp --example mem_skills -- https://mem.qreta.io/mcp
//! ```
use agentsdk::AgentPlugin;
use agentsdk_plugin_mcp::McpPlugin;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let url = std::env::args().nth(1).unwrap_or_else(|| "https://mem.qreta.io/mcp".to_owned());
    let token = std::env::var("MEM_TOKEN").expect("MEM_TOKEN");

    let mut headers = HashMap::new();
    headers.insert("authorization".to_owned(), format!("Bearer {token}"));
    headers.insert("user-agent".to_owned(), "pie-mcp/0.1".to_owned());

    let mut plugin = McpPlugin::new();
    plugin.add_remote_server("mem", &url, headers).await?;

    println!("tools exposed to the agent:");
    for tool in plugin.tools() {
        println!("  - {}", tool.name);
    }

    let catalog = run(&mut plugin, "find_mcp_skills", serde_json::json!({})).await?;
    println!("catalog: {catalog:#}");

    let loaded = run(
        &mut plugin,
        "load_mcp_skill",
        serde_json::json!({ "name": "mem-worker-operations" }),
    )
    .await?;
    let text = loaded["content"].as_str().unwrap_or_default();
    println!("loaded mem-worker-operations: {} bytes via resources/read", text.len());
    println!("first line: {}", text.lines().next().unwrap_or_default());

    Ok(())
}

async fn run(
    plugin: &mut McpPlugin,
    name: &str,
    arguments: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut world = agentsdk::hecs::World::new();
    let entity = world.spawn(());
    let mut ctx = agentsdk::core::plugin::PluginContext::new(world, entity);
    plugin
        .run_tool(
            &mut ctx,
            &agentsdk::core::plugin::PluginToolCall {
                id: "live-1".into(),
                name: name.into(),
                arguments,
            },
        )
        .await
}
