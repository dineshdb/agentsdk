# agentsdk-plugin-mcp

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) plugin for [agentsdk](https://github.com/dineshdb/agentsdk).

---

## 🥧 Part of the Pie Ecosystem

`agentsdk-plugin-mcp` is a core component of the **[Pie](https://github.com/dineshdb/pie)** ecosystem.

**Pie** is a fast, minimal AI coding agent written in Rust. While Pie provides the interactive CLI experience and persistent sessions, **AgentSDK** (and this plugin) provides the modular orchestration layer that makes features like MCP support possible.

[**Explore Pie →**](https://github.com/dineshdb/pie)

---

This crate allows you to bridge any MCP server (local or remote) into your `agentsdk` agents. It automatically handles tool discovery, name disambiguation, and request routing.

## Features

- **Local Servers**: Connect to MCP servers via standard I/O (stdio).
- **Remote Servers**: Connect to remote MCP servers via HTTP/SSE.
- **Aggregation**: Manage multiple MCP servers within a single plugin.
- **Namespacing**: Automatically prefixes tools with `{server_name}__` to prevent name collisions.
- **Seamless Integration**: Implements `AgentPlugin` for direct use with `Agent::builder()`.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agentsdk = "0.10.0"
agentsdk-plugin-mcp = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## Usage

### Connecting to Local and Remote Servers

```rust
use agentsdk::Agent;
use agentsdk_plugin_mcp::McpPlugin;
use tokio::process::Command;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut mcp_plugin = McpPlugin::new();

    // Add a local server via npx (stdio)
    let mut sqlite_cmd = Command::new("npx");
    sqlite_cmd.args(["-y", "@modelcontextprotocol/server-sqlite", "--db", "my.db"]);
    mcp_plugin.add_server("sqlite", sqlite_cmd).await?;

    // Add a remote server (SSE)
    let mut headers = HashMap::new();
    headers.insert("CONTEXT7_API_KEY".to_string(), "your_api_key".to_string());
    mcp_plugin.add_remote_server("context7", "https://mcp.context7.com/mcp", headers).await?;

    // Build the agent
    let agent = Agent::builder()
        .plugin(mcp_plugin)
        .build()?;
    
    // Tools are now available as `mcp__sqlite__query`, `mcp__context7__query-docs`, etc.
    
    Ok(())
}
```

## License

MIT
