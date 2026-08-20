use async_trait::async_trait;
use std::path::Path;

#[derive(Debug)]
pub struct SandboxOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[async_trait]
pub trait FSProvider: Send + Sync {
    // ── Filesystem ───────────────────────────────────────────────

    /// Read file contents. Returns raw string.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    fn read(&self, path: &Path) -> Result<String, SandboxError>;

    /// Write content to file, creating parent dirs as needed.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    fn write(&self, path: &Path, content: &str) -> Result<(), SandboxError>;

    /// List directory entries. Returns (name, `is_dir`) pairs sorted by name.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    fn list(&self, path: &Path) -> Result<Vec<(String, bool)>, SandboxError>;

    /// Find paths matching a glob pattern.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if any path is not allowed or an I/O error occurs.
    fn glob(&self, pattern: &str) -> Result<Vec<String>, SandboxError>;

    // ── Execution ───────────────────────────────────────────────

    /// Execute a command and capture output.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the command is not allowed or execution fails.
    async fn exec(&self, cmd: &str) -> Result<SandboxOutput, SandboxError>;
}

// ── Raw filesystem operations ────────────────────────────────────────
// Shared by Unsandboxed and ChrootSandbox — single source of truth for
// the actual I/O.  Sandbox implementations compose these with their own
// path validation (or lack thereof).

/// Read file contents as a UTF-8 string.
///
/// # Errors
/// Returns [`SandboxError::Io`] on I/O failure.
pub fn raw_read(path: &Path) -> Result<String, SandboxError> {
    Ok(std::fs::read_to_string(path)?)
}

/// Write content to a file, creating parent directories as needed.
///
/// # Errors
/// Returns [`SandboxError::Io`] on I/O failure.
pub fn raw_write(path: &Path, content: &str) -> Result<(), SandboxError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(std::fs::write(path, content)?)
}

/// List directory entries as `(name, is_dir)` pairs sorted by name.
///
/// # Errors
/// Returns [`SandboxError::Io`] on I/O failure.
pub fn raw_list(path: &Path) -> Result<Vec<(String, bool)>, SandboxError> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type()?.is_dir();
        entries.push((name, is_dir));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// Find paths matching a glob pattern.
///
/// # Errors
/// Returns [`SandboxError::Io`] on I/O failure.
pub fn raw_glob(pattern: &str) -> Result<Vec<String>, SandboxError> {
    let matches = glob::glob(pattern).map_err(std::io::Error::other)?;
    let mut paths = Vec::new();
    for entry in matches {
        paths.push(
            entry
                .map_err(std::io::Error::other)?
                .to_string_lossy()
                .into_owned(),
        );
    }
    Ok(paths)
}

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("path not allowed for reading: {0}")]
    ReadDenied(String),
    #[error("path not allowed for writing: {0}")]
    WriteDenied(String),
    #[error("command not allowed: {0}")]
    CommandDenied(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

use std::sync::Arc;

pub struct Sandbox(Arc<dyn FSProvider>);

impl Sandbox {
    pub fn new(provider: impl FSProvider + 'static) -> Self {
        Self(Arc::new(provider))
    }

    /// Read file contents.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    pub fn read(&self, path: &Path) -> Result<String, SandboxError> {
        self.0.read(path)
    }

    /// Write content to file.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    pub fn write(&self, path: &Path, content: &str) -> Result<(), SandboxError> {
        self.0.write(path, content)
    }

    /// List directory entries.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the path is not allowed or an I/O error occurs.
    pub fn list(&self, path: &Path) -> Result<Vec<(String, bool)>, SandboxError> {
        self.0.list(path)
    }

    /// Find paths matching a glob pattern.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if any path is not allowed or an I/O error occurs.
    pub fn glob(&self, pattern: &str) -> Result<Vec<String>, SandboxError> {
        self.0.glob(pattern)
    }

    /// Execute a command.
    ///
    /// # Errors
    /// Returns a [`SandboxError`] if the command is not allowed or execution fails.
    pub async fn exec(&self, cmd: &str) -> Result<SandboxOutput, SandboxError> {
        self.0.exec(cmd).await
    }
}

impl Clone for Sandbox {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Sandbox").finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Unsandboxed;

#[async_trait]
impl FSProvider for Unsandboxed {
    fn read(&self, path: &Path) -> Result<String, SandboxError> {
        raw_read(path)
    }

    fn write(&self, path: &Path, content: &str) -> Result<(), SandboxError> {
        raw_write(path, content)
    }

    fn list(&self, path: &Path) -> Result<Vec<(String, bool)>, SandboxError> {
        raw_list(path)
    }

    fn glob(&self, pattern: &str) -> Result<Vec<String>, SandboxError> {
        raw_glob(pattern)
    }

    async fn exec(&self, cmd: &str) -> Result<SandboxOutput, SandboxError> {
        let output = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(cmd)
            .output()
            .await?;

        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}
