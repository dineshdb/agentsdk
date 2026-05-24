use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::sandbox::Sandbox;
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct FileSystemPlugin;

impl FileSystemPlugin {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentPlugin for FileSystemPlugin {
    fn name(&self) -> &'static str {
        "fs"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "read".into(),
                description: "UTF-8. Supports line ranges.".into(),
                input_schema: schema_for!(ReadInput),
            },
            ToolDefinition {
                name: "write".into(),
                description: "Overwrites if exists. Creates directories.".into(),
                input_schema: schema_for!(WriteInput),
            },
            ToolDefinition {
                name: "replace".into(),
                description:
                    "Surgical search and replace. Fails if old_string is not found or is ambiguous."
                        .into(),
                input_schema: schema_for!(ReplaceInput),
            },
            ToolDefinition {
                name: "list".into(),
                description: "directory entries.".into(),
                input_schema: schema_for!(ListInput),
            },
            ToolDefinition {
                name: "glob".into(),
                description: "paths matching a pattern.".into(),
                input_schema: schema_for!(GlobInput),
            },
        ]
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match call.name.as_str() {
            "read" => {
                let input: ReadInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_read(ctx, &input)
            }
            "write" => {
                let input: WriteInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_write(ctx, &input)
            }
            "replace" => {
                let input: ReplaceInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_replace(ctx, &input)
            }
            "list" => {
                let input: ListInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_list(ctx, &input)
            }
            "glob" => {
                let input: GlobInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_glob(ctx, &input)
            }
            _ => Err(format!("Unknown tool: {}", call.name)),
        }
    }
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ReadInput {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

fn do_read(ctx: &mut PluginContext, input: &ReadInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let content = sandbox
        .0
        .read(Path::new(&input.path))
        .map_err(|e| format!("Failed to read {}: {e}", input.path))?;
    let lines: Vec<&str> = content.lines().collect();
    let start = input.start_line.unwrap_or(1).max(1);
    let end = input.end_line.unwrap_or(lines.len()).min(lines.len());

    if start > end {
        return Err("reached beyond the end of file".to_string());
    }

    let slice = lines
        .get(start.saturating_sub(1)..end)
        .ok_or("Invalid line range")?;
    let result_content = slice.join("\n");

    Ok(json!({
        "path": input.path,
        "content": result_content,
        "start_line": start,
        "end_line": end,
        "total_lines": lines.len()
    }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct WriteInput {
    path: String,
    content: String,
}

fn do_write(ctx: &mut PluginContext, input: &WriteInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    sandbox
        .0
        .write(Path::new(&input.path), &input.content)
        .map_err(|e| format!("Failed to write: {e}"))?;
    Ok(json!({ "status": "success", "path": input.path, "bytes": input.content.len() }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ReplaceInput {
    path: String,
    old_string: String,
    new_string: String,
}

fn do_replace(ctx: &mut PluginContext, input: &ReplaceInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let content = sandbox
        .0
        .read(Path::new(&input.path))
        .map_err(|e| format!("Failed to read: {e}"))?;
    let occurrences = content.matches(&input.old_string).count();
    if occurrences == 0 {
        return Err(format!("String not found in {}", input.path));
    }
    if occurrences > 1 {
        return Err(format!(
            "String found {occurrences} times in {}. Please provide more context to make it unique.",
            input.path
        ));
    }

    let new_content = content.replace(&input.old_string, &input.new_string);
    sandbox
        .0
        .write(Path::new(&input.path), &new_content)
        .map_err(|e| format!("Failed to write: {e}"))?;

    Ok(json!({ "status": "success", "path": input.path }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ListInput {
    path: String,
}

fn do_list(ctx: &mut PluginContext, input: &ListInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let entries = sandbox
        .0
        .list(Path::new(&input.path))
        .map_err(|e| format!("Failed to list directory: {e}"))?;
    let mut result = Vec::new();
    for (name, is_dir) in entries {
        result.push(json!({
            "name": name,
            "is_directory": is_dir,
        }));
    }

    Ok(json!({ "path": input.path, "entries": result }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct GlobInput {
    pattern: String,
}

fn do_glob(ctx: &mut PluginContext, input: &GlobInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let matches = sandbox
        .0
        .glob(&input.pattern)
        .map_err(|e| format!("Glob error: {e}"))?;

    Ok(json!({ "pattern": input.pattern, "matches": matches }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsdk::core::sandbox::Unsandboxed;
    use serial_test::serial;
    use tempfile::tempdir;

    #[tokio::test]
    #[serial]
    async fn test_read_write() {
        let dir = tempdir().unwrap();
        let content = "hello world";

        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Write
        let write_call = PluginToolCall {
            id: "1".into(),
            name: "write".into(),
            arguments: json!({
                "path": "test.txt",
                "content": content
            }),
        };
        plugin.run_tool(&mut ctx, &write_call).await.unwrap();

        // Read
        let read_call = PluginToolCall {
            id: "2".into(),
            name: "read".into(),
            arguments: json!({
                "path": "test.txt"
            }),
        };
        let read_result = plugin.run_tool(&mut ctx, &read_call).await.unwrap();
        assert_eq!(read_result["content"], content);

        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_replace() {
        let dir = tempdir().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Write
        plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "write".into(),
                    arguments: json!({
                        "path": "test.txt",
                        "content": "hello world"
                    }),
                },
            )
            .await
            .unwrap();

        // Replace
        plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "2".into(),
                    name: "replace".into(),
                    arguments: json!({
                        "path": "test.txt",
                        "old_string": "world",
                        "new_string": "rust"
                    }),
                },
            )
            .await
            .unwrap();

        // Read
        let read_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "3".into(),
                    name: "read".into(),
                    arguments: json!({
                        "path": "test.txt"
                    }),
                },
            )
            .await
            .unwrap();
        assert_eq!(read_result["content"], "hello rust");

        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_list() {
        let dir = tempdir().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        let list_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "list".into(),
                    arguments: json!({
                        "path": "."
                    }),
                },
            )
            .await
            .unwrap();

        let entries = list_result["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .any(|e| e["name"] == "a.txt" && e["is_directory"] == false)
        );
        assert!(
            entries
                .iter()
                .any(|e| e["name"] == "subdir" && e["is_directory"] == true)
        );

        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_glob() {
        let dir = tempdir().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        std::fs::write(dir.path().join("c.rs"), "c").unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        let glob_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "glob".into(),
                    arguments: json!({
                        "pattern": "*.txt"
                    }),
                },
            )
            .await
            .unwrap();

        let matches = glob_result["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().any(|m| m.as_str().unwrap() == "a.txt"));
        assert!(matches.iter().any(|m| m.as_str().unwrap() == "b.txt"));

        std::env::set_current_dir(original_dir).unwrap();
    }
}
