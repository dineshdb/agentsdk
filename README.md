# AGENTSDK

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/agentsdk/latest)
[![Build Status](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lean Rust SDK for building AI agents with OpenAI-compatible APIs.
Type-safe tool definitions, streaming, and agentic loops out of the box.

## Features

- **Callback-driven agent loop** — automatically handles multi-turn tool-calling conversations with lifecycle hooks
- **Type-safe tools** — derive tools from plain Rust functions with the `#[tool]` macro
- **OpenAI-compatible** — works with OpenAI, OpenRouter, and any compatible endpoint
- **JSON Schema generation** — automatic input/output schemas via `schemars`

## Installation

```bash
cargo add agentsdk
```

## Quick Start

```rust
use agentsdk::{Agent, AgentListener, AgentOptions, OpenAI, messages, tool, Tool};
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

struct MyHandler;

#[async_trait]
impl AgentListener for MyHandler {
    async fn prepare_system_prompt(&mut self, _history: &Messages) -> Option<std::borrow::Cow<'static, str>> {
        Some("You are a helpful weather assistant.".into())
    }

    async fn on_text_delta(&mut self, text: &str) {
        print!("{text}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ModelConfig::from_env()?;
    let client = OpenAI::new(config);

    let agent = Agent::builder()
        .client(client)
        .options(
            AgentOptions::builder()
                .messages(std::sync::Arc::new(vec![messages::user("What's the weather in Tokyo?")]))
                .with_tool(&get_weather())
                .build()?
        )
        .build()?;

    let mut handler = MyHandler;
    let _history = agent.run(&mut handler).await?;

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
