# agentsdk-plugin-fs

A FileSystem plugin for [agentsdk](https://github.com/dineshdb/agentsdk) that provides tools for common file operations.

---

## 🥧 Part of the Pie Ecosystem

`agentsdk-plugin-fs` is a core component of the **[Pie](https://github.com/dineshdb/pie)** ecosystem.

**Pie** is a fast, minimal AI coding agent written in Rust. While Pie provides the interactive CLI experience and persistent sessions, **AgentSDK** (and this plugin) provides the modular orchestration layer that makes features like filesystem operations possible.

[**Explore Pie →**](https://github.com/dineshdb/pie)

---

## Features

- **read_file**: Read the content of a file (with optional line range).
- **write_file**: Write content to a file (creates parent directories).
- **replace**: Surgical search and replace in a file.
- **list_directory**: List directory entries.
- **glob**: Find files matching a pattern.

## Security Disclaimer

**IMPORTANT**: This plugin does **NOT** perform any path validation or restriction. It allows reading and writing to any path accessible by the process. 

It is the responsibility of the application developer to implement path validation and safety checks. In `agentsdk`, this is best handled using **pre-tool hooks** (`AgentPlugin::on_tool_pre_execute`) to inspect and potentially abort tool calls with unsafe paths.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agentsdk = "0.10.0"
agentsdk-plugin-fs = "0.1.0"
```

## License

MIT
