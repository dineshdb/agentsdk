use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    pub path: String,
    #[serde(default)]
    pub rename: HashMap<String, String>,
    #[serde(default)]
    pub derive_extra: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub api_name: Option<String>,
    #[serde(default = "default_deny_unknown_fields")]
    pub deny_unknown_fields: bool,
}

fn default_deny_unknown_fields() -> bool {
    true
}
