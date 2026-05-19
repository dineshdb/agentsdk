# AGENTSDK

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/agentsdk/latest)
[![Build Status](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lean Rust SDK for building AI agents with OpenAI-compatible APIs.
Type-safe tools, streaming, ECS-based plugin system, and agentic loops out of the box.

## Features

- **ECS-based plugin system** — lifecycle hooks with shared state via `hecs::World`; compose plugins, detect changes, inspect state post-run
- **Type-safe tools** — derive tools from plain Rust functions with the `#[tool]` macro
- **OpenAI-compatible** — works with OpenAI, OpenRouter, and any compatible endpoint
- **History plugins** — `FileHistoryPlugin` for persistence, `MemoryHistoryPlugin` for in-memory history
- **JSON Schema generation** — automatic input/output schemas via `schemars`
- **Retry policy** — configurable retry with backoff for API errors
- **Parallel tool calls** — execute multiple tool calls concurrently

## Installation

```bash
cargo add agentsdk
```

## Quick Start

```rust
use agentsdk::{Agent, AgentPlugin, MemoryHistoryPlugin, OpenAI, PluginContext, messages, tool, Tool};
use async_trait::async_trait;

// Define a tool from a plain function
#[tool]
/// Get the current weather for a location
fn get_weather(location: String) -> Tool {
    let temp = match location.as_str() {
        "Tokyo" => 22,
        "London" => 14,
        _ => 20,
    };
    Ok(format!("{temp}°C"))
}

// A plugin that streams text deltas to stdout
struct PrinterPlugin;

#[async_trait]
impl AgentPlugin for PrinterPlugin {
    fn name(&self) -> &'static str {
        "printer"
    }

    async fn on_text_delta(&mut self, _ctx: &PluginContext, text: &str) {
        print!("{text}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAI::new(agentsdk::ModelConfig::from_env()?);

    // Push the initial user message into in-memory history
    let history = MemoryHistoryPlugin::new();
    history.push(messages::user("What's the weather in Tokyo?")).await;

    let mut agent = Agent::builder()
        .client(client)
        .options(
            agentsdk::AgentOptions::builder()
                .with_tool(&get_weather())
                .build()?,
        )
        .plugin(history.clone())
        .plugin(PrinterPlugin)
        .build()?;

    let _output = agent.run().await?;
    Ok(())
}
```

## Defining Tools

Use the `#[tool]` macro to turn any function into a callable tool:

```rust
use agentsdk::{tool, Tool};

#[tool]
/// Calculate the sum of two numbers
fn add(a: i32, b: i32) -> Tool {
    Ok((a + b).to_string())
}
```

The macro automatically:
- Uses the function name as the tool name
- Extracts the description from doc comments
- Generates a JSON Schema from the parameters
- Supports both sync and async functions

### Struct parameters

```rust
use agentsdk::{tool, Tool};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize, Default)]
struct SearchQuery {
    query: String,
    limit: Option<i32>,
}

#[tool]
/// Search for documents
fn search(req: SearchQuery) -> Tool {
    Ok(format!("Found results for '{}'", req.query))
}
```

### Tool context

Access runtime context (model name, extensions) via `ToolContext`:

```rust
use agentsdk::{tool, Tool, ToolContext};

#[tool]
fn list_files(ctx: ToolContext, pattern: String) -> Tool {
    let model = ctx.options.model.as_deref().unwrap_or("unknown");
    Ok(format!("Model {model} asked for files matching '{pattern}'"))
}
```

## Plugin System

Plugins extend the agent with custom behavior through lifecycle hooks. Every method has a default no-op so you only implement what you need.

### Lifecycle

| Hook | Timing | Type |
|------|--------|------|
| `init` | Once when the agent starts | Setup |
| `shutdown` | Once when the agent finishes | Cleanup |
| `on_text_delta` | Each streaming chunk | Observability |
| `on_model_response_completed` | Full turn received | Observability |
| `prepare_system_prompt` | Before each model call | Control flow (merged) |
| `on_tool_pre_execute` | Before a tool runs | Control flow (first decisive wins) |
| `on_tool_post_execute` | After a tool succeeds | Control flow (first decisive wins) |
| `on_tool_error` | When a tool fails | Control flow (first decisive wins) |
| `on_completion` | Final text produced | Control flow (first decisive wins) |
| `on_api_error` | API call fails | Retry decision |

### PluginContext

Each hook receives a [`PluginContext`] wrapping a [`hecs::World`] with a dedicated entity for the agent session:

```rust
async fn on_tool_pre_execute(
    &mut self,
    ctx: &PluginContext,
    id: &str,
    name: &str,
    args: &Value,
) -> PreToolAction {
    // Read/write shared state on the agent entity
    if let Some(counter) = ctx.get::<ToolCallCounter>() {
        println!("Tool call #{counter:?}");
    }
    PreToolAction::Continue(None)
}
```

### Built-in plugins

- **`MemoryHistoryPlugin`** — in-memory conversation history (no persistence)
- **`FileHistoryPlugin`** — JSON-file-backed persistence; loads on `init`, saves on `shutdown`

### AgentRunOutput

`agent.run().await` returns an `AgentRunOutput` containing the full `hecs::World`. Use it to inspect plugin state after execution:

```rust
let output = agent.run().await?;
let history: Messages = output.world.get::<History>(output.entity)
    .map(|h| h.0.clone())
    .unwrap_or_default();
```

## Configuration

### Agent options

```rust
AgentOptions::builder()
    .model("gpt-4o")
    .temperature(0.7)
    .max_tokens(4096)
    .max_iterations(10)      // limit agent loop iterations (default: 25)
    .with_tool(&my_tool())
    .build()?
```

### Plugins

Plugins are registered on the builder and receive lifecycle events in registration order:

```rust
Agent::builder()
    .client(client)
    .options(options)
    .plugin(FileHistoryPlugin::new(".session.json")?)
    .plugin(MetricsPlugin::new())
    .build()?
```

State is shared between plugins through a [`hecs::World`] — each plugin reads/writes typed components on the agent entity.

### OpenAI client

```rust
let config = ModelConfig {
    api_key: "sk-...".into(),
    base_url: "https://api.openai.com/v1".into(),
    model: "gpt-4o".into(),
};
let client = OpenAI::new(config);

// OR from environment variables
let config = ModelConfig::from_env()?;
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT — see [LICENSE](./LICENSE).
