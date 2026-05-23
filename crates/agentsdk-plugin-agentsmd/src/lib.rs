use agentsdk::core::messages::Messages;
use agentsdk::core::plugin::{AgentPlugin, PluginContext};
use async_trait::async_trait;
use derive_builder::Builder;
use std::borrow::Cow;
use std::env;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during system prompt plugin initialization or execution.
#[derive(Debug, Error)]
pub enum SystemPromptError {
    #[error("Failed to determine current working directory: {0}")]
    CurrentDirError(#[from] std::io::Error),
}

/// A plugin that manages system prompts by loading and merging files from a search path.
///
/// Supports placeholders:
/// - `${HOME}`: The user's home directory.
/// - `${PROJECT_ROOT}`: The root directory of the project (optional).
/// - `${PWD}`: The current working directory.
#[derive(Debug, Clone)]
pub struct AgentsMdPlugin {
    merged_prompt: Option<String>,
}

impl AgentsMdPlugin {
    /// Returns a builder to configure and create a `AgentsMdPlugin`.
    pub fn builder() -> AgentsMdPluginBuilder {
        AgentsMdPluginBuilder::default()
    }
}

/// Configuration for [`AgentsMdPlugin`].
/// This struct is used by `derive_builder` to generate the builder.
#[derive(Builder, Default)]
#[builder(
    pattern = "owned",
    build_fn(skip),
    setter(into, strip_option),
    name = "AgentsMdPluginBuilder"
)]
pub struct AgentsMdConfig {
    #[builder(default)]
    pub search_paths: Vec<String>,
    #[builder(default)]
    pub project_root: Option<PathBuf>,
    #[builder(default)]
    pub home: Option<PathBuf>,
    #[builder(default)]
    pub pwd: Option<PathBuf>,
}

impl AgentsMdPluginBuilder {
    /// Build the `AgentsMdPlugin` by resolving paths and caching the prompt.
    pub fn build(self) -> Result<AgentsMdPlugin, SystemPromptError> {
        let home = self
            .home
            .flatten()
            .or_else(|| env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"));
        let pwd = if let Some(Some(p)) = self.pwd {
            p
        } else {
            env::current_dir()?
        };
        let project_root = self.project_root.flatten();
        let search_paths = self.search_paths.unwrap_or_default();

        let mut prompts = Vec::new();

        for path_str in &search_paths {
            let mut resolved = path_str.to_string();
            resolved = resolved.replace("${HOME}", &home.to_string_lossy());
            resolved = resolved.replace("${PWD}", &pwd.to_string_lossy());

            if let Some(root) = &project_root {
                resolved = resolved.replace("${PROJECT_ROOT}", &root.to_string_lossy());
            } else if resolved.contains("${PROJECT_ROOT}") {
                continue;
            }

            let path = PathBuf::from(resolved);
            if path.exists()
                && path.is_file()
                && let Ok(content) = fs::read_to_string(&path)
                && !content.trim().is_empty()
            {
                prompts.push(content);
            }
        }

        let merged_prompt = if prompts.is_empty() {
            None
        } else {
            Some(prompts.join("\n\n---\n\n"))
        };

        Ok(AgentsMdPlugin { merged_prompt })
    }
}

#[async_trait]
impl AgentPlugin for AgentsMdPlugin {
    fn name(&self) -> &'static str {
        "agentsmd"
    }

    async fn prepare_system_prompt(
        &mut self,
        _ctx: &mut PluginContext,
        _history: &Messages,
    ) -> Option<Cow<'static, str>> {
        self.merged_prompt.as_ref().map(|p| Cow::Owned(p.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsdk::core::plugin::PluginContext;
    use serial_test::serial;
    use tempfile::tempdir;

    #[tokio::test]
    #[serial]
    async fn test_agentsmd_builder() {
        let dir = tempdir().unwrap();
        let home_dir = dir.path().join("home");
        let project_dir = dir.path().join("project");
        fs::create_dir_all(&home_dir).unwrap();
        fs::create_dir_all(&project_dir).unwrap();

        let home_prompt = home_dir.join("SYSTEM.md");
        let project_prompt = project_dir.join("SYSTEM.md");

        fs::write(&home_prompt, "Global prompt").unwrap();
        fs::write(&project_prompt, "Local prompt").unwrap();

        let mut plugin = AgentsMdPlugin::builder()
            .search_paths(vec![
                "${HOME}/SYSTEM.md".into(),
                "${PROJECT_ROOT}/SYSTEM.md".into(),
            ])
            .home(home_dir)
            .project_root(project_dir)
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext { world, entity };
        let history = Messages::default();

        let result = plugin
            .prepare_system_prompt(&mut ctx, &history)
            .await
            .unwrap();
        assert!(result.contains("Global prompt"));
        assert!(result.contains("Local prompt"));
        assert!(result.contains("---"));
    }

    #[tokio::test]
    #[serial]
    async fn test_builder_env_defaults() {
        let dir = tempdir().unwrap();
        let pwd_prompt = dir.path().join("SYSTEM.md");
        fs::write(&pwd_prompt, "PWD prompt").unwrap();

        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(dir.path()).unwrap();

        let mut plugin = AgentsMdPlugin::builder()
            .search_paths(vec!["${PWD}/SYSTEM.md".into()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext { world, entity };
        let history = Messages::default();

        let result = plugin
            .prepare_system_prompt(&mut ctx, &history)
            .await
            .unwrap();
        assert_eq!(result.as_ref(), "PWD prompt");

        env::set_current_dir(original_dir).unwrap();
    }
}
