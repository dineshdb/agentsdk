//! Live check: one skills surface (FindSkills / LoadSkills) spanning
//! filesystem skills and skills served over MCP by mem.qreta.io.
//!
//! ```text
//! MEM_TOKEN=... cargo run -p agentsdk-plugin-mcp --example mem_skills -- https://mem.qreta.io/mcp
//! ```
use agentsdk::AgentPlugin;
use agentsdk::core::plugin::PluginToolCall;
use agentsdk_plugin_mcp::McpPlugin;
use agentsdk_plugin_skills::SkillsPlugin;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://mem.qreta.io/mcp".to_owned());
    let token = std::env::var("MEM_TOKEN").expect("MEM_TOKEN");

    let mut headers = HashMap::new();
    headers.insert("authorization".to_owned(), format!("Bearer {token}"));
    headers.insert("user-agent".to_owned(), "pie-mcp/0.1".to_owned());

    // One sink: the MCP plugin fills it, the skills plugin serves it.
    let remote_skills = agentsdk_plugin_skills::remote_skill_sink();

    let mut mcp = McpPlugin::new().with_remote_skills(remote_skills.clone());
    mcp.add_remote_server("mem", &url, headers).await?;

    let mut skills = SkillsPlugin::builder()
        .remote_skills(remote_skills)
        .build()
        .map_err(|e| e.to_string())?;

    let catalog = run(
        &mut skills,
        "FindSkills",
        serde_json::json!({
            "query": "worker operations deploy",
            "max_results": 5,
        }),
    )
    .await?;
    println!("FindSkills -> {catalog:#}");

    let loaded = run(
        &mut skills,
        "LoadSkills",
        serde_json::json!({ "skills": ["mem/mem-worker-operations"] }),
    )
    .await?;
    let text = loaded.as_str().unwrap_or_default();
    println!("LoadSkills -> {} bytes", text.len());
    println!("{}", text.lines().next().unwrap_or_default());

    Ok(())
}

async fn run(
    plugin: &mut dyn AgentPlugin,
    name: &str,
    arguments: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut world = agentsdk::hecs::World::new();
    let entity = world.spawn(());
    let mut ctx = agentsdk::core::plugin::PluginContext::new(world, entity);
    plugin
        .run_tool(
            &mut ctx,
            &PluginToolCall {
                id: "live-1".into(),
                name: name.into(),
                arguments,
            },
        )
        .await
}
