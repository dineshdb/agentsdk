//! Sandbox provider that restricts all filesystem operations to a root
//! directory. Symlink escapes are caught by canonicalizing both root and
//! target before the prefix check.

use agentsdk::core::sandbox::{FSProvider, SandboxError, raw_list, raw_read, raw_write};
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// A [`FSProvider`] that confines all reads/writes/lists/globs to
/// `root`. Paths supplied by plugins are resolved relative to `root`.
#[derive(Debug, Clone)]
pub struct ChrootSandbox {
    root: PathBuf,
}

impl ChrootSandbox {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a (possibly relative) path against the root and verify it
    /// doesn't escape via `..` or symlinks.
    fn safe_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let target = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        let canon_root = std::fs::canonicalize(&self.root).map_err(SandboxError::Io)?;
        // Walk up to the nearest existing ancestor so canonicalize succeeds.
        let canon_target = if target.exists() {
            std::fs::canonicalize(&target).map_err(SandboxError::Io)?
        } else {
            let mut cursor = target.as_path();
            let mut tail: Vec<std::ffi::OsString> = Vec::new();
            loop {
                if cursor.exists() {
                    break;
                }
                match cursor.file_name() {
                    Some(name) => {
                        tail.push(name.to_os_string());
                        cursor = cursor.parent().unwrap_or(cursor);
                    }
                    None => break,
                }
            }
            let canon_base = std::fs::canonicalize(cursor).map_err(SandboxError::Io)?;
            tail.iter()
                .rev()
                .fold(canon_base, |acc, comp| acc.join(comp))
        };
        if !canon_target.starts_with(&canon_root) {
            return Err(SandboxError::ReadDenied(format!(
                "path escapes sandbox root: {}",
                path.display()
            )));
        }
        Ok(target)
    }
}

#[async_trait]
impl FSProvider for ChrootSandbox {
    fn read(&self, path: &Path) -> Result<String, SandboxError> {
        raw_read(&self.safe_path(path)?)
    }

    fn write(&self, path: &Path, content: &str) -> Result<(), SandboxError> {
        raw_write(&self.safe_path(path)?, content)
    }

    fn list(&self, path: &Path) -> Result<Vec<(String, bool)>, SandboxError> {
        raw_list(&self.safe_path(path)?)
    }

    fn glob(&self, pattern: &str) -> Result<Vec<String>, SandboxError> {
        let full_pattern = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            self.root.join(pattern).to_string_lossy().into_owned()
        };
        let matches = glob::glob(&full_pattern).map_err(std::io::Error::other)?;
        let mut paths = Vec::new();
        for entry in matches {
            let path = entry.map_err(std::io::Error::other)?;
            self.safe_path(&path)?;
            let rel = path.strip_prefix(&self.root).unwrap_or(&path);
            paths.push(rel.to_string_lossy().into_owned());
        }
        Ok(paths)
    }

    async fn exec(
        &self,
        _cmd: &str,
    ) -> Result<agentsdk::core::sandbox::SandboxOutput, SandboxError> {
        Err(SandboxError::CommandDenied(
            "command execution is not allowed in sandboxed mode".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn read_write_list_within_root() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("sandbox");
        std::fs::create_dir_all(&root).unwrap();
        let sb = ChrootSandbox::new(&root);

        sb.write(Path::new("a/b.txt"), "hello").unwrap();
        assert_eq!(sb.read(Path::new("a/b.txt")).unwrap(), "hello");

        let entries = sb.list(Path::new("a")).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "b.txt");
    }

    #[test]
    fn traversal_rejected() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("sandbox");
        std::fs::create_dir_all(&root).unwrap();
        let sb = ChrootSandbox::new(&root);

        assert!(sb.read(Path::new("../escape")).is_err());
        assert!(sb.write(Path::new("../../etc/passwd"), "x").is_err());
        assert!(sb.list(Path::new("..")).is_err());
    }

    #[test]
    fn glob_returns_relative_paths() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("sandbox");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), "a").unwrap();
        std::fs::write(root.join("b.txt"), "b").unwrap();
        let sb = ChrootSandbox::new(&root);

        let matches = sb.glob("*.txt").unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().any(|m| m == "a.txt"));
        assert!(matches.iter().any(|m| m == "b.txt"));
    }

    #[test]
    fn exec_denied() {
        let tmp = tempdir().unwrap();
        let sb = ChrootSandbox::new(tmp.path());
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(sb.exec("echo hi")).is_err());
    }
}
