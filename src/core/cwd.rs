use std::path::{Path, PathBuf};

use super::plugin::PluginContext;

/// The run's working directory, as an ECS component on the agent entity.
///
/// Plugins resolve relative paths (tool arguments, globs) against it instead
/// of the process cwd, so one process can run agents in several directories
/// concurrently. Hosts insert it next to [`crate::core::sandbox::Sandbox`]
/// via `AgentBuilder::component`.
#[derive(Debug, Clone)]
pub struct Cwd(pub PathBuf);

impl Cwd {
    /// The run's cwd, or the process cwd when the host never registered the
    /// component. Preserves pre-`Cwd` behavior for hosts that opt out.
    pub fn from_ctx(ctx: &PluginContext) -> Self {
        ctx.get::<Self>()
            .map_or_else(Self::fallback, |cwd| Self(cwd.0.clone()))
    }

    /// Resolve a possibly-relative path against this directory.
    /// `~` is expanded first; absolute paths pass through unchanged.
    #[must_use]
    pub fn resolve(&self, path: &str) -> PathBuf {
        Self::expand_tilde(path, &self.0)
    }

    /// The directory plugins should assume when none was registered:
    /// the process working directory.
    #[must_use]
    pub fn fallback() -> Self {
        Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    fn expand_tilde(path: &str, base: &Path) -> PathBuf {
        let expanded = match path.strip_prefix("~/") {
            Some(rest) => dirs::home_dir().map_or_else(
                || path.to_string(),
                |h| h.join(rest).to_string_lossy().to_string(),
            ),
            None if path == "~" => dirs::home_dir()
                .map_or_else(|| path.to_string(), |h| h.to_string_lossy().to_string()),
            None => path.to_string(),
        };
        let p = Path::new(&expanded);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base.join(p)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn resolve_joins_base_for_relative_paths() {
        let cwd = Cwd(PathBuf::from("/workspaces/proj"));
        assert_eq!(
            cwd.resolve("src/lib.rs"),
            PathBuf::from("/workspaces/proj/src/lib.rs")
        );
        assert_eq!(cwd.resolve("."), PathBuf::from("/workspaces/proj"));
    }

    #[test]
    fn resolve_passes_absolute_through() {
        let cwd = Cwd(PathBuf::from("/workspaces/proj"));
        assert_eq!(cwd.resolve("/etc/hosts"), PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn resolve_expands_tilde_outside_base() {
        let home = dirs::home_dir().expect("home exists");
        let cwd = Cwd(PathBuf::from("/workspaces/proj"));
        assert_eq!(cwd.resolve("~/notes.md"), home.join("notes.md"));
        assert_eq!(cwd.resolve("~"), home);
    }
}
