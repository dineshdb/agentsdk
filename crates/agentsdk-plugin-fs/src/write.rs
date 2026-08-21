//! Full filesystem plugin: read, write, edit, list, glob.

use crate::{
    GlobInput, ListInput, ReadInput, ReplaceInput, WriteInput, do_glob, do_list, do_read,
    do_replace, do_write,
};
use agentsdk::PluginTools;
use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Default, Clone)]
pub struct FileSystemPlugin;

impl FileSystemPlugin {
    pub fn new() -> Self {
        Self
    }
}

#[derive(PluginTools, Serialize, Deserialize)]
enum FsTools {
    /// Read a file. UTF-8. Prefer reading partial content instead of whole file
    Read(ReadInput),
    /// Write a file. Overwrites if exists. Creates directories.
    Write(WriteInput),
    /// Surgical search and replace. Fails if old_string is not found or is ambiguous.
    #[tool(name = "Edit")]
    Replace(ReplaceInput),
    /// List directory entries.
    Ls(ListInput),
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
            FsTools::Ls(input) => do_list(ctx, &input),
            FsTools::Glob(input) => do_glob(ctx, &input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsdk::core::sandbox::{Sandbox, Unsandboxed};
    use serde_json::json;
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
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Write
        let write_call = PluginToolCall {
            id: "1".into(),
            name: "Write".into(),
            arguments: json!({
                "path": "test.txt",
                "content": content
            }),
        };
        plugin.run_tool(&mut ctx, &write_call).await.unwrap();

        // Read
        let read_call = PluginToolCall {
            id: "2".into(),
            name: "Read".into(),
            arguments: json!({
                "path": "test.txt",
                "start_line": 1,
                "lines": 1
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
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Write
        plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "Write".into(),
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
                    name: "Edit".into(),
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
                    name: "Read".into(),
                    arguments: json!({
                        "path": "test.txt",
                        "start_line": 1,
                        "lines": 1
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
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut plugin = FileSystemPlugin::new();
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        let list_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "Ls".into(),
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
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        let list_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "Ls".into(),
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
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        let glob_result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "Glob".into(),
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
