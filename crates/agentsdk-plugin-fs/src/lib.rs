use agentsdk::PluginTools;
use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::sandbox::Sandbox;
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use schemars::JsonSchema;
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

#[derive(PluginTools, Serialize, Deserialize)]
enum FsTools {
    /// Read a file. UTF-8. Supports line ranges.
    Read(ReadInput),
    /// Write a file. Overwrites if exists. Creates directories.
    Write(WriteInput),
    /// Surgical search and replace. Fails if old_string is not found or is ambiguous.
    Replace(ReplaceInput),
    /// List directory entries.
    List(ListInput),
    /// List paths matching a pattern.
    Glob(GlobInput),
}

#[async_trait]
impl AgentPlugin for FileSystemPlugin {
    fn name(&self) -> &'static str {
        "fs"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        FsTools::definitions()
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match FsTools::from_call(call)? {
            FsTools::Read(input) => do_read(ctx, &input),
            FsTools::Write(input) => do_write(ctx, &input),
            FsTools::Replace(input) => do_replace(ctx, &input),
            FsTools::List(input) => do_list(ctx, &input),
            FsTools::Glob(input) => do_glob(ctx, &input),
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
    /// Show tree up to this depth. Defaults to 2.
    depth: Option<usize>,
    path: String,
}

fn do_list(ctx: &mut PluginContext, input: &ListInput) -> Result<Value, String> {
    let max_depth = input.depth.unwrap_or(2);
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;

    fn build_tree(
        sandbox: &Sandbox,
        path: &Path,
        prefix: &str,
        is_root: bool,
        depth: usize,
        max_depth: usize,
    ) -> Result<String, String> {
        let entries = sandbox
            .0
            .list(path)
            .map_err(|e| format!("Failed to list directory: {e}"))?;

        if is_root && entries.is_empty() {
            return Ok("(empty)\n".to_string());
        }

        let mut out = String::new();

        if is_root {
            let dirname = path.to_string_lossy();
            out.push_str(&format!("{dirname}\n"));
        }

        if depth >= max_depth {
            return Ok(out);
        }

        let mut dirs: Vec<&str> = Vec::new();
        let mut files: Vec<&str> = Vec::new();
        for (name, is_dir) in entries.iter() {
            if *is_dir {
                dirs.push(name);
            } else {
                files.push(name);
            }
        }
        dirs.sort();
        files.sort();

        let mut by_ext: std::collections::BTreeMap<&str, Vec<&str>> =
            std::collections::BTreeMap::new();
        for name in &files {
            let dot = name.rfind('.').filter(|&i| i > 0);
            let ext = dot.map(|i| &name[i..]).unwrap_or("");
            by_ext.entry(ext).or_default().push(name);
        }

        for ext_files in by_ext.values() {
            let entry = if ext_files.len() == 1 {
                ext_files[0].to_string()
            } else {
                let stems: Vec<&str> = ext_files
                    .iter()
                    .map(|n| {
                        let dot = n.rfind('.').filter(|&i| i > 0);
                        dot.map(|i| &n[..i]).unwrap_or(n)
                    })
                    .collect();
                let base = ext_files[0];
                let dot = base.rfind('.').filter(|&i| i > 0);
                if let Some(dot_idx) = dot {
                    format!("{{{}}}{}", stems.join(","), &base[dot_idx..])
                } else {
                    format!("{{{}}}", stems.join(","))
                }
            };
            out.push_str(&format!("{prefix}{entry}\n"));
        }

        for name in dirs {
            let child_path = path.join(name);
            let child_prefix = format!("{prefix}  ");
            out.push_str(&format!("{prefix}{name}/\n"));
            out.push_str(&build_tree(
                sandbox,
                &child_path,
                &child_prefix,
                false,
                depth + 1,
                max_depth,
            )?);
        }

        Ok(out)
    }

    let tree = build_tree(&sandbox, Path::new(&input.path), "", true, 0, max_depth)?;
    Ok(json!(tree.trim_end()))
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

        let tree = list_result.as_str().unwrap();
        assert!(tree.contains("a.txt"));
        assert!(tree.contains("subdir/"));

        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_list_merge() {
        let dir = tempdir().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        std::fs::write(dir.path().join("main.rs"), "").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("mod.rs"), "").unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();
        std::fs::write(dir.path().join("LICENSE"), "").unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::create_dir(dir.path().join("tests")).unwrap();

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

        let tree = list_result.as_str().unwrap();
        assert!(
            tree.contains("{lib,main,mod}.rs"),
            "merge .rs files: {tree}"
        );
        assert!(tree.contains("README.md"), "single .md: {tree}");
        assert!(tree.contains("src/"), "src/ dir: {tree}");
        assert!(tree.contains("tests/"), "tests/ dir: {tree}");

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
