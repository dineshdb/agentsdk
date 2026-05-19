use crate::core::messages::{Message, Messages};
use crate::error::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;

/// Trait for managing conversation history.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    /// Loads the message history for a given ID.
    async fn load(&self, id: &str) -> Result<Messages>;
    /// Appends a message to the history for a given ID.
    async fn push(&self, id: &str, message: Message) -> Result<()>;
}

/// A simple in-memory implementation of [`HistoryStore`].
#[derive(Debug, Default, Clone)]
pub struct MemoryHistory {
    storage: Arc<RwLock<std::collections::HashMap<String, Messages>>>,
}

impl MemoryHistory {
    /// Creates a new empty [`MemoryHistory`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl HistoryStore for MemoryHistory {
    async fn load(&self, id: &str) -> Result<Messages> {
        let storage = self.storage.read().await;
        Ok(storage.get(id).cloned().unwrap_or_default())
    }

    async fn push(&self, id: &str, message: Message) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.entry(id.to_string()).or_default().push(message);
        Ok(())
    }
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
    async fn load(&self, id: &str) -> Result<Messages> {
        let path = self.file_path(id);
        match fs::read_to_string(path).await {
            Ok(data) => {
                let messages: Messages = serde_json::from_str(&data)?;
                Ok(messages)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }

    async fn push(&self, id: &str, message: Message) -> Result<()> {
        let mut messages = self.load(id).await?;
        messages.push(message);
        let data = serde_json::to_string_pretty(&messages)?;
        fs::write(self.file_path(id), data).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::messages;

    #[tokio::test]
    async fn test_file_history_save_and_load() -> Result<()> {
        let mut dir = std::env::temp_dir();
        dir.push(format!("agentsdk-test-history-{}", std::process::id()));
        let store = FileHistory::new(&dir)?;

        let id = "test-session";
        let message = messages::user("Hello");

        // Push
        store.push(id, message.clone()).await?;

        // Load
        let loaded = store.load(id).await?;
        assert_eq!(1, loaded.len());

        // Test missing file
        let missing = store.load("non-existent").await?;
        assert!(missing.is_empty());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
