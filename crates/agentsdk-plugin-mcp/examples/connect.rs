//! Live probe: connect to a remote MCP server exactly like pie does.
//! Usage: cargo run -p agentsdk-plugin-mcp --example connect -- <url> [k=v k=v]

use agentsdk::core::plugin::AgentPlugin as _;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let url = args.next().expect("usage: connect <url> [k=v ...]");
    let mut headers = HashMap::new();
    for pair in args {
        let (k, v) = pair.split_once('=').expect("header as k=v");
        headers.insert(k.to_string(), v.to_string());
    }

    let mut plugin = agentsdk_plugin_mcp::McpPlugin::new();
    plugin.add_remote_server("probe", &url, headers).await?;
    for tool in plugin.tools() {
        println!("tool: {}", tool.name);
    }
    println!("OK: connected and listed {} tools", plugin.tools().len());
    Ok(())
}
