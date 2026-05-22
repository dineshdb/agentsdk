# agentsdk-plugin-skills

A Skills plugin for [agentsdk](https://github.com/dineshdb/agentsdk) that manages on-demand knowledge and script execution.

---

## 🥧 Part of the Pie Ecosystem

`agentsdk-plugin-skills` is a core component of the **[Pie](https://github.com/dineshdb/pie)** ecosystem.

**Pie** is a fast, minimal AI coding agent written in Rust. While Pie provides the interactive CLI experience and persistent sessions, **AgentSDK** (and this plugin) provides the modular orchestration layer that makes features like skills possible.

[**Explore Pie →**](https://github.com/dineshdb/pie)

---

## Features

- **Cached Scanning**: All skills in specified search paths are scanned and cached at build time for high performance.
- **On-Demand Loading**: Agents can use the `skills__load` tool to pull in extra knowledge as needed.
- **Dependency Management**: Skills can specify other skills they depend on (via `needs` in frontmatter), which are automatically resolved and loaded.
- **Binary Execution**: Execute any binary or script contained within a skill directory via the `skills__execute` tool.
- **System Prompt Integration**: Automatically informs the agent about all available skills.

## Skill Format

Skills are stored as directories containing a `SKILL.md` file with YAML frontmatter.

```markdown
---
name: my-skill
description: Does something awesome
needs: [other-skill]
---
# My Skill
This is extra knowledge for the LLM.
```

Any other files in the skill directory (or a `bin/` subfolder) can be executed as binaries.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agentsdk = "0.10.0"
agentsdk-plugin-skills = "0.1.0"
```

## Usage

```rust
use agentsdk::Agent;
use agentsdk_plugin_skills::SkillsPlugin;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let skills_plugin = SkillsPlugin::builder()
        .search_paths(vec!["./my-skills".into()])
        .build()?;

    let agent = Agent::builder()
        .plugin(skills_plugin)
        .build()?;
    
    Ok(())
}
```

## License

MIT
