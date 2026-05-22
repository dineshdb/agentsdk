use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
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
        _ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match call.name.as_str() {
            "read" => {
                let input: ReadInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_read(&input)
            }
            "write" => {
                let input: WriteInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_write(&input)
            }
            "replace" => {
                let input: ReplaceInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_replace(&input)
            }
            "list" => {
                let input: ListInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_list(&input)
            }
            "glob" => {
                let input: GlobInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                do_glob(&input)
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

fn do_read(input: &ReadInput) -> Result<Value, String> {
    let content = fs::read_to_string(&input.path).map_err(|e| format!("Failed to read: {e}"))?;
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

fn do_write(input: &WriteInput) -> Result<Value, String> {
    let path = Path::new(&input.path);

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directories: {e}"))?;
    }

    fs::write(path, &input.content).map_err(|e| format!("Failed to write: {e}"))?;
    Ok(json!({ "status": "success", "path": input.path, "bytes": input.content.len() }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ReplaceInput {
    path: String,
    old_string: String,
    new_string: String,
}

fn do_replace(input: &ReplaceInput) -> Result<Value, String> {
    let content = fs::read_to_string(&input.path).map_err(|e| format!("Failed to read: {e}"))?;
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
    fs::write(&input.path, new_content).map_err(|e| format!("Failed to write: {e}"))?;

    Ok(json!({ "status": "success", "path": input.path }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ListInput {
    path: String,
}

fn do_list(input: &ListInput) -> Result<Value, String> {
    let entries =
        fs::read_dir(&input.path).map_err(|e| format!("Failed to read directory: {e}"))?;
    let mut result = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Error reading directory entry: {e}"))?;
        let file_name = entry.file_name().to_string_lossy().to_string();
        let file_type = entry
            .file_type()
            .map_err(|e| format!("Error reading file type: {e}"))?;
        let is_dir = file_type.is_dir();
        result.push(json!({
            "name": file_name,
            "is_directory": is_dir,
        }));
    }

    // Sort entries alphabetically by name for deterministic prompt generation
    result.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["name"].as_str().unwrap_or_default())
    });

    Ok(json!({ "path": input.path, "entries": result }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct GlobInput {
    pattern: String,
}

fn do_glob(input: &GlobInput) -> Result<Value, String> {
    let mut matches = Vec::new();
    for entry in glob::glob(&input.pattern).map_err(|e| format!("Invalid glob pattern: {e}"))? {
        let path = entry.map_err(|e| format!("Glob error: {e}"))?;
        matches.push(path.to_string_lossy().to_string());
    }

    // Sort matches alphabetically for deterministic prompt generation
    matches.sort();

    Ok(json!({ "pattern": input.pattern, "matches": matches }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    #[tokio::test]
    #[serial]
    async fn test_read_write() {
        let dir = tempdir().unwrap();
        let _file_path = dir.path().join("test.txt");
        let content = "hello world";

        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
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

        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
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

        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::write(dir.path().join("b.txt"), "b").unwrap();
        fs::write(dir.path().join("c.rs"), "c").unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
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
