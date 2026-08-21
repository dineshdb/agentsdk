# agentsdk-plugin-fs

A FileSystem plugin for [agentsdk](https://github.com/dineshdb/agentsdk) that provides tools for common file operations.

---

## 🥧 Part of the Pie Ecosystem

`agentsdk-plugin-fs` is a core component of the **[Pie](https://github.com/dineshdb/pie)** ecosystem.

**Pie** is a fast, minimal AI coding agent written in Rust. While Pie provides the interactive CLI experience and persistent sessions, **AgentSDK** (and this plugin) provides the modular orchestration layer that makes features like filesystem operations possible.

[**Explore Pie →**](https://github.com/dineshdb/pie)

---

## Features

Two plugin implementations share the same handlers but expose different capabilities:

- **`FileSystemPlugin`** (plugin name `fs`) — the full set:
  - **Read**: Read a file (UTF-8, partial content by line range).
  - **Write**: Write a file (overwrites if exists, creates parent directories).
  - **Edit**: Surgical search and replace in a file.
  - **Ls**: List directory entries.
  - **Glob**: Find files matching a pattern.
- **`ReadOnlyFileSystemPlugin`** (plugin name `fs-readonly`) — read-only subset:
  - **Read**, **Ls**, **Glob** only.

The readonly variant never advertises `Write`/`Edit` and rejects them at
dispatch (`Unknown tool`), so a model running against it cannot attempt file
mutations through the filesystem tools at all. Pick the variant when building
your agent:

```rust
use agentsdk_plugin_fs::{FileSystemPlugin, ReadOnlyFileSystemPlugin};

// full read/write access
builder.plugin(FileSystemPlugin::new());
// or read-only
builder.plugin(ReadOnlyFileSystemPlugin::new());
```

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
