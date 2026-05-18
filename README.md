# AGENTSDK

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/agentsdk/latest)
[![Build Status](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshdb/agentsdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lean Rust SDK for building AI agents with OpenAI-compatible APIs.
Type-safe tool definitions, streaming, and agentic loops out of the box.

## Features

- **Streaming agent loop** — automatically handles multi-turn tool-calling conversations
- **Type-safe tools** — derive tools from plain Rust functions with the `#[tool]` macro
- **OpenAI-compatible** — works with OpenAI, OpenRouter, and any compatible endpoint
- **JSON Schema generation** — automatic input/output schemas via `schemars`

## Installation

```bash
cargo add agentsdk
```

## Quick Start

```rust
use agentsdk::{Agent, AgentEvent, AgentOptions, OpenAI, messages, tool, Tool};

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAI::builder()
        .api_key("sk-...")
        .base_url("https://api.openai.com/v1")
        .model("gpt-4o")
        .build()?;

    let agent = Agent::builder()
        .client(client)
        .options(
            AgentOptions::builder()
                .system("You are a helpful weather assistant.")
                .messages(std::sync::Arc::new(vec![messages::user("What's the weather in Tokyo?")]))
                .with_tool(&get_weather())
                .build()?
        )
        .build()?;

    let mut stream = agent.stream();
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::ToolCallChunk { name: Some(name), .. } => {
                println!("\n[Calling {name}]");
            }
            AgentEvent::Finished(_) => println!(),
            _ => {}
        }
    }

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
    let model = &ctx.options.model;
    Ok(format!("Model {model} asked for files matching '{pattern}'"))
}
```

## Configuration

### Agent options

```rust
AgentOptions::builder()
    .model("gpt-4o")
    .system("You are helpful.")
    .temperature(0.7)
    .max_tokens(4096)
    .max_steps(10)          // limit agent loop iterations (default: 25)
    .with_tool(&my_tool())
    .build()?
```

### OpenAI client

```rust
OpenAI::builder()
    .api_key("sk-...")
    .base_url("https://api.openai.com/v1")  // optional, defaults to OpenAI
    .model_name("gpt-4o")
    .build()?
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT — see [LICENSE](./LICENSE).
