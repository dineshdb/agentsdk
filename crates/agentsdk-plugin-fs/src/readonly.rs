//! Read-only filesystem plugin: read, list, glob. Write and edit are not
//! advertised and cannot be dispatched — the tool enum has no such variants.

use crate::{GlobInput, ListInput, ReadInput, do_glob, do_list, do_read};
use agentsdk::PluginTools;
use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Filesystem plugin restricted to read-only operations.
#[derive(Debug, Default, Clone)]
pub struct ReadOnlyFileSystemPlugin;

impl ReadOnlyFileSystemPlugin {
    pub fn new() -> Self {
        Self
    }
}

#[derive(PluginTools, Serialize, Deserialize)]
enum ReadOnlyFsTools {
    /// Read a file. UTF-8. Prefer reading partial content instead of whole file
    Read(ReadInput),
    /// List directory entries.
    Ls(ListInput),
    /// List paths matching a pattern.
    Glob(GlobInput),
}

#[async_trait]
impl AgentPlugin for ReadOnlyFileSystemPlugin {
    fn name(&self) -> &'static str {
        "fs-readonly"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        ReadOnlyFsTools::definitions()
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match ReadOnlyFsTools::from_call(call)? {
            ReadOnlyFsTools::Read(input) => do_read(ctx, &input),
            ReadOnlyFsTools::Ls(input) => do_list(ctx, &input),
            ReadOnlyFsTools::Glob(input) => do_glob(ctx, &input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsdk::core::sandbox::{Sandbox, Unsandboxed};
    use serde_json::json;

    fn ctx_with_sandbox() -> (hecs::World, hecs::Entity) {
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        (world, entity)
    }

    #[test]
    fn readonly_advertises_exactly_read_ls_glob() {
        let names: Vec<String> = ReadOnlyFileSystemPlugin::new()
            .tools()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(names, vec!["Read", "Ls", "Glob"]);
    }

    #[tokio::test]
    async fn readonly_rejects_write_and_edit() {
        let dir = tempfile::tempdir().unwrap();
        let (world, entity) = ctx_with_sandbox();
        let mut ctx = PluginContext::new(world, entity);
        let mut plugin = ReadOnlyFileSystemPlugin::new();

        for tool in ["Write", "Edit"] {
            let call = PluginToolCall {
                id: "1".into(),
                name: tool.into(),
                arguments: json!({
                    "path": dir.path().join("x.txt"),
                    "content": "no",
                    "old_string": "a",
                    "new_string": "b",
                }),
            };
            let err = plugin.run_tool(&mut ctx, &call).await.unwrap_err();
            assert!(err.contains("Unknown tool"), "{tool}: {err}");
        }

        // Nothing was created on disk either.
        assert!(!dir.path().join("x.txt").exists());
    }

    #[tokio::test]
    async fn readonly_read_ls_glob_work() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "line1\nline2").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let (world, entity) = ctx_with_sandbox();
        let mut ctx = PluginContext::new(world, entity);
        let mut plugin = ReadOnlyFileSystemPlugin::new();
        let base = dir.path().to_string_lossy().to_string();

        let read = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "Read".into(),
                    arguments: json!({ "path": format!("{base}/a.txt"), "start_line": 1, "lines": 2 }),
                },
            )
            .await
            .unwrap();
        assert_eq!(read["content"], "line1\nline2");

        let ls = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "2".into(),
                    name: "Ls".into(),
                    arguments: json!({ "path": base.as_str() }),
                },
            )
            .await
            .unwrap();
        let tree = ls.as_str().unwrap();
        assert!(tree.contains("a.txt"), "ls: {tree}");
        assert!(tree.contains("subdir/"), "ls: {tree}");

        let glob = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "3".into(),
                    name: "Glob".into(),
                    arguments: json!({ "pattern": format!("{base}/*.txt") }),
                },
            )
            .await
            .unwrap();
        let matches = glob["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 1, "glob: {matches:?}");
    }
}
