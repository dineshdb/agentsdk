use crate::core::messages::Messages;
use crate::error::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Trait for persisting conversation history.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    /// Saves the message history for a given ID.
    async fn save(&self, id: &str, messages: &Messages) -> Result<()>;
    /// Loads the message history for a given ID. Returns `None` if not found.
    async fn load(&self, id: &str) -> Result<Option<Messages>>;
}

/// A simple file-based implementation of [`HistoryStore`].
#[derive(Debug, Clone)]
pub struct FileHistory {
    dir: PathBuf,
}

impl FileHistory {
    /// Creates a new [`FileHistory`] that stores files in the given directory.
    ///
    /// # Errors
    /// Returns an error if the directory cannot be created.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        Ok(Self { dir: path })
    }

    fn file_path(&self, id: &str) -> PathBuf {
        let mut path = self.dir.clone();
        path.push(format!("{id}.json"));
        path
    }
}

#[async_trait]
impl HistoryStore for FileHistory {
    async fn save(&self, id: &str, messages: &Messages) -> Result<()> {
        let path = self.file_path(id);
        let data = serde_json::to_string_pretty(messages)?;
        fs::write(path, data).await?;
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Messages>> {
        let path = self.file_path(id);
        match fs::read_to_string(path).await {
            Ok(data) => {
                let messages: Messages = serde_json::from_str(&data)?;
                Ok(Some(messages))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::messages;
    use crate::error::AgentSdkError;

    #[tokio::test]
    async fn test_file_history_save_and_load() -> Result<()> {
        let mut dir = std::env::temp_dir();
        dir.push(format!("agentsdk-test-history-{}", std::process::id()));
        let store = FileHistory::new(&dir)?;

        let id = "test-session";
        let messages = vec![
            messages::system("You are a helpful assistant."),
            messages::user("Hello"),
            messages::assistant("Hi there!"),
        ];

        // Save
        store.save(id, &messages).await?;

        // Load
        let loaded = store
            .load(id)
            .await?
            .ok_or_else(|| AgentSdkError::ConfigError("Should have loaded messages".into()))?;
        assert_eq!(messages.len(), loaded.len());

        // Test missing file
        let missing = store.load("non-existent").await?;
        assert!(missing.is_none());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
