# agentsdk-plugin-agentsmd

A System Prompt plugin for [agentsdk](https://github.com/dineshdb/agentsdk) that manages system prompts by loading and merging files from a search path.

---

## 🥧 Part of the Pie Ecosystem

`agentsdk-plugin-agentsmd` is a core component of the **[Pie](https://github.com/dineshdb/pie)** ecosystem.

**Pie** is a fast, minimal AI coding agent written in Rust. While Pie provides the interactive CLI experience and persistent sessions, **AgentSDK** (and this plugin) provides the modular orchestration layer that makes features like dynamic system prompts possible.

[**Explore Pie →**](https://github.com/dineshdb/pie)

---

## Features

- **Placeholder Resolution**: Automatically resolves `${HOME}`, `${PROJECT_ROOT}`, and `${PWD}` in search paths.
- **Dynamic Merging**: Merges multiple system prompt files into a single prompt, separated by `---`.
- **Environment Aware**: Automatically initializes `${HOME}` and `${PWD}` from the environment.
- **Optional Project Root**: Explicitly set the project root for `${PROJECT_ROOT}` resolution.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agentsdk = "0.10.0"
agentsdk-plugin-agentsmd = "0.1.0"
```

## Usage

```rust
use agentsdk::Agent;
use agentsdk_plugin_agentsmd::AgentsMdPlugin;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let agentsmd_plugin = AgentsMdPlugin::builder()
        .paths(vec![
            "${HOME}/.config/pie/SYSTEM.md".into(),
            "${PROJECT_ROOT}/.agentsdk/SYSTEM.md".into(),
            "${PWD}/SYSTEM.md".into(),
        ])
        .project_root("/path/to/project")
        .build()?;

    let agent = Agent::builder()
        .plugin(agentsmd_plugin)
        .build()?;
    
    Ok(())
}
```

## License

MIT
