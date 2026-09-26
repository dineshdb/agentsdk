use agentsdk::PluginTools;
use agentsdk::core::history::History;
use agentsdk::core::messages::extract_user_text;
use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::sandbox::Sandbox;
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use bm25::{Document, LanguageMode, SearchEngine, SearchEngineBuilder, SearchResult};
use derive_builder::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub enum LoadStatus {
    #[default]
    Unloaded,
    Loaded,
}

#[derive(Debug, Error)]
pub enum SkillsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// A specialized reference file for a skill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct Reference {
    pub title: String,
    pub path: String,
    #[serde(default)]
    pub status: LoadStatus,
}

/// A skill is a piece of extra knowledge or a script that can be loaded on-demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub needs: Vec<String>,
    #[serde(skip)]
    pub references: Vec<Reference>,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(skip)]
    pub status: LoadStatus,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Fetches the content of one remotely served skill — its SKILL.md body
/// (`file: None`) or one of its supporting files, resolved relative to the
/// skill's root. Implemented by the plugin that owns the connection; the
/// skills plugin only sees this trait.
#[async_trait]
pub trait SkillLoader: Send + Sync {
    async fn load(&self, file: Option<&str>) -> Result<String, String>;
}

/// A skill served by a remote source — an MCP server that declares the
/// skills extension, cataloged as `skill://` resources.
#[derive(Clone)]
pub struct RemoteSkill {
    /// The origin the skill came from, shown to the agent so remote
    /// instructions are never indistinguishable from local ones.
    pub server: String,
    pub name: String,
    pub description: String,
    pub loader: Arc<dyn SkillLoader>,
}

impl std::fmt::Debug for RemoteSkill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteSkill")
            .field("server", &self.server)
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// The shared handle a connection plugin fills and the skills plugin reads.
/// Create one with [`remote_skill_sink`], clone it into both plugins, and
/// remote skills become findable and loadable through the same tools as
/// local ones.
pub type RemoteSkillSink = Arc<Mutex<Vec<RemoteSkill>>>;

pub fn remote_skill_sink() -> RemoteSkillSink {
    Arc::new(Mutex::new(Vec::new()))
}

/// A plugin that manages skills.
#[derive(Debug)]
pub struct SkillsPlugin {
    /// Cached skills found during scanning.
    available_skills: HashMap<String, Skill>,
    /// Reference files that have been loaded. Key format: "skill_name/file_name"
    loaded_references: HashMap<String, LoadStatus>,
    /// BM25 search index over skill names and descriptions.
    search_index: SearchEngine<usize>,
    /// Maps skill name to search index document ID.
    skill_ids: HashMap<String, usize>,
    /// Skills served by remote sources, shared with the plugins that
    /// catalog them. Local skills shadow remote ones on name collision.
    remote_skills: RemoteSkillSink,
    /// Remote skills and references already loaded. Key format:
    /// "server/name" and "server/name/file_name".
    loaded_remote: HashSet<String>,
}

impl Clone for SkillsPlugin {
    fn clone(&self) -> Self {
        let (search_index, _) = build_search_index(&self.available_skills);
        Self {
            available_skills: self.available_skills.clone(),
            loaded_references: self.loaded_references.clone(),
            search_index,
            skill_ids: self.skill_ids.clone(),
            remote_skills: self.remote_skills.clone(),
            loaded_remote: self.loaded_remote.clone(),
        }
    }
}

impl SkillsPlugin {
    pub fn builder() -> SkillsPluginBuilder {
        SkillsPluginBuilder::default()
    }

    /// Returns a list of all available skills found during scanning.
    /// Each entry is a tuple of (name, description, references).
    pub fn available_skills(&self) -> Vec<(String, String, Vec<Reference>)> {
        let mut result: Vec<_> = self
            .available_skills
            .values()
            .map(|s| (s.name.clone(), s.description.clone(), s.references.clone()))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    fn resolve_dependencies(&self, names: &[String]) -> Vec<String> {
        let mut resolved = Vec::new();
        let mut visited = HashSet::new();
        let mut stack: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        while let Some(name) = stack.pop() {
            if !visited.insert(name) {
                continue;
            }
            if let Some(skill) = self.available_skills.get(name) {
                for need in &skill.needs {
                    stack.push(need.as_str());
                }
                resolved.push(name.to_string());
            }
        }
        resolved.reverse();
        resolved
    }
}

/// Configuration for [`SkillsPlugin`].
/// This struct is used by `derive_builder` to generate the builder.
#[derive(Builder, Default)]
#[builder(
    pattern = "owned",
    build_fn(skip),
    setter(into, strip_option),
    name = "SkillsPluginBuilder"
)]
pub struct SkillsConfig {
    #[builder(default)]
    #[allow(dead_code)]
    pub search_paths: Vec<PathBuf>,
    /// Skills from remote sources (an MCP plugin fills this handle as its
    /// servers connect). Defaults to an empty sink.
    #[builder(default = "remote_skill_sink()")]
    pub remote_skills: RemoteSkillSink,
}

impl SkillsPluginBuilder {
    pub fn build(self) -> Result<SkillsPlugin, SkillsError> {
        let search_paths = self.search_paths.unwrap_or_default();
        let remote_skills = self.remote_skills.unwrap_or_else(remote_skill_sink);
        let mut available_skills = HashMap::new();

        for path in search_paths {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let md_path = entry.path().join("SKILL.md");
                        if let Ok(raw) = fs::read_to_string(&md_path)
                            && let Some(mut skill) = parse_skill(&raw)
                        {
                            skill.path = entry.path();
                            available_skills.insert(skill.name.clone(), skill);
                        }
                    }
                }
            }
        }

        let (search_index, skill_ids) = build_search_index(&available_skills);

        Ok(SkillsPlugin {
            available_skills,
            loaded_references: HashMap::new(),
            search_index,
            skill_ids,
            remote_skills,
            loaded_remote: HashSet::new(),
        })
    }
}

fn build_search_index(
    available_skills: &HashMap<String, Skill>,
) -> (SearchEngine<usize>, HashMap<String, usize>) {
    let mut skill_ids = HashMap::new();
    let mut sorted: Vec<_> = available_skills.keys().cloned().collect();
    sorted.sort();
    let documents: Vec<Document<usize>> = sorted
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            available_skills.get(name).map(|skill| {
                skill_ids.insert(skill.name.clone(), i);
                let contents = format!("{}: {}", skill.name, skill.description);
                Document { id: i, contents }
            })
        })
        .collect();
    let search_index: SearchEngine<usize> =
        SearchEngineBuilder::with_documents(LanguageMode::default(), documents).build();
    (search_index, skill_ids)
}

const PROMPT: &str = r#"
# Skills

You have access to locally available skills — specialized instructions and tools for various topics.
Skills are automatically searched for each query and matching results are injected into the conversation.

```
.../skills/skill-name/
├── SKILL.md     # Required: metadata + instructions, loaded via 'LoadSkills(skills=["skill-name"])'
├── scripts/     # Optional: executable code, executed via 'RunSkillScript('skill-name/scripts/script.py')
├── references/  # Optional: documentation, loaded-on-demand via 'LoadSkillReference(skill="skill-name", reference="references/file.md")'
├── assets/      # Optional: templates, resources
└── ...          # Any additional files or directories
```

When skill search results appear in the conversation, review them and load relevant skills with `LoadSkills` before answering. Use `LoadSkillReference` for reference files and `RunSkillScript` to execute scripts as instructed.
Skills may also come from connected MCP servers — those carry a `server` field in search results and are loaded the same way, but their scripts are not executable here.

**CRITICAL: DO NOT invent skill names, reference paths, or script paths.**
Only use names and paths from the skill search results. Invented names will be rejected with an error.

Skills are only useful when you follow the newly loaded instructions to refine your query, improvise your plan and reach your goal.
"#;

#[derive(PluginTools, Serialize, Deserialize)]
enum SkillsTools {
    /// Search and then immediately load available skills.
    /// load listed skills and their references as needed.
    #[tool(name = "FindSkills")]
    Find(FindSkillsInput),
    /// Load instructions from skills and their references(optional).
    #[tool(name = "LoadSkills")]
    Load(LoadSkillsInput),
    /// Load a specific reference file from a skill.
    #[tool(name = "LoadSkillReference")]
    Reference(LoadReferenceInput),
    /// Execute a script within a skill, as instructed by the loaded script
    #[tool(name = "RunSkillScript")]
    Run(RunScriptInput),
}

#[async_trait]
impl AgentPlugin for SkillsPlugin {
    fn name(&self) -> &'static str {
        "skills"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        SkillsTools::definitions()
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match SkillsTools::from_call(call)? {
            SkillsTools::Find(input) => {
                serde_json::to_value(self.do_find_skills(&input)?).map_err(|e| e.to_string())
            }
            SkillsTools::Load(input) => self.do_load_skills(ctx, &input).await,
            SkillsTools::Reference(input) => self.do_load_reference(ctx, &input).await,
            SkillsTools::Run(input) => self.do_execute_skill_script(ctx, &input).await,
        }
    }

    async fn prepare_system_prompt(
        &mut self,
        _ctx: &mut PluginContext,
    ) -> Option<Cow<'static, str>> {
        Some(Cow::Borrowed(PROMPT))
    }

    async fn on_iteration_start(&mut self, ctx: &mut PluginContext, iteration: usize) {
        // Only auto-inject on the first iteration of a new user query.
        if iteration > 0 {
            return;
        }

        // Extract the last user message text from history.
        let query = {
            let Some(history) = ctx.get::<History>() else {
                return;
            };
            let Some(last) = history.0.last() else {
                return;
            };
            let Some(text) = extract_user_text(last) else {
                return;
            };
            text
        };

        let find_input = FindSkillsInput {
            query,
            max_results: default_max_results(),
        };

        let Ok(output) = self.do_find_skills(&find_input) else {
            return;
        };

        let Some(mut h) = ctx.get_mut::<History>() else {
            return;
        };

        if output.results.is_empty() {
            let guidance = json!({"message": "No specialized skills match your query. Use your general knowledge, web search, or filesystem tools to find the information you need."});
            h.inject_tool_call(
                "FindSkills",
                &serde_json::to_value(&find_input).unwrap_or_default(),
                &guidance,
            );
        } else {
            h.inject_tool_call(
                "FindSkills",
                &serde_json::to_value(&find_input).unwrap_or_default(),
                &serde_json::to_value(&output).unwrap_or_default(),
            );
        }
    }
}

impl SkillsPlugin {
    /// Find a remote skill by bare name or by the qualified `server/name`.
    /// Local skills shadow remote ones, so callers check the local map first.
    fn find_remote(&self, reference: &str) -> Option<RemoteSkill> {
        let remotes = self.remote_skills.lock().ok()?;
        if let Some((server, name)) = reference.split_once('/') {
            return remotes
                .iter()
                .find(|r| r.server == server && r.name == name)
                .cloned();
        }
        remotes
            .iter()
            .find(|r| r.name == reference && !self.available_skills.contains_key(reference))
            .cloned()
    }

    fn is_remote_skill(&self, name: &str) -> bool {
        let qualified = name.contains('/');
        self.remote_skills.lock().ok().is_some_and(|remotes| {
            remotes.iter().any(|r| {
                if qualified {
                    format!("{}/{}", r.server, r.name) == name
                } else {
                    r.name == name && !self.available_skills.contains_key(name)
                }
            })
        })
    }

    fn warn_unknown_skills(&self, names: &[String], output: &mut String) -> bool {
        let unknown: Vec<&str> = names
            .iter()
            .filter(|n| !self.available_skills.contains_key(n.as_str()) && !self.is_remote_skill(n))
            .map(|n| n.as_str())
            .collect();
        if unknown.is_empty() {
            return false;
        }
        let listed = unknown.join(", ");
        output.push_str(&format!(
            "Warning: The following skills do not exist: {listed}.\n"
        ));
        true
    }

    fn already_loaded_skills(&self, names: &[String]) -> Vec<String> {
        let mut names: Vec<String> = names
            .iter()
            .filter(|n| {
                self.available_skills
                    .get(n.as_str())
                    .is_some_and(|s| s.status == LoadStatus::Loaded)
            })
            .cloned()
            .collect();
        names.sort();
        names.dedup();
        names
    }

    async fn do_load_skills(
        &mut self,
        ctx: &mut PluginContext,
        input: &LoadSkillsInput,
    ) -> Result<Value, String> {
        let mut output = String::new();

        // Partition the request. A name that matches a remote skill — bare,
        // or qualified as server/name, with skill/file shorthand routing to
        // a remote file — loads through its server's loader; everything else
        // flows through the local machinery unchanged.
        let mut all_skill_names: Vec<String> = Vec::new();
        let mut skill_names_to_resolve = Vec::new();
        let mut shorthand_refs = Vec::new();
        let mut remote_requests: Vec<(String, Option<String>)> = Vec::new();

        for s in &input.skills {
            let s = s.strip_prefix('/').unwrap_or(s);
            if self.is_remote_skill(s) {
                all_skill_names.push(s.to_owned());
                remote_requests.push((s.to_owned(), None));
                continue;
            }
            // Support both 'skill/file' and 'skill:file' for backward compatibility
            let parts = s.split_once('/').or_else(|| s.split_once(':'));
            match parts {
                Some((skill_part, file_part)) => {
                    all_skill_names.push(skill_part.to_owned());
                    if self.is_remote_skill(skill_part) {
                        remote_requests.push((skill_part.to_owned(), Some(file_part.to_owned())));
                    } else if self.available_skills.contains_key(skill_part) {
                        shorthand_refs.push((skill_part.to_owned(), file_part.to_owned()));
                        skill_names_to_resolve.push(skill_part.to_owned());
                    }
                }
                None => {
                    all_skill_names.push(s.to_owned());
                    if self.available_skills.contains_key(s) {
                        skill_names_to_resolve.push(s.to_owned());
                    }
                }
            }
        }
        self.warn_unknown_skills(&all_skill_names, &mut output);

        // Local skills: resolve dependencies and load, as before.
        let already_loaded = self.already_loaded_skills(&skill_names_to_resolve);
        let mut skill_names = self.resolve_dependencies(&skill_names_to_resolve);
        skill_names.sort();
        for name in &skill_names {
            self.load_one_skill(name, &mut output);
        }
        for name in &already_loaded {
            output.push_str(&format!(
                "Note: Skill '{name}' is already loaded in the context."
            ));
        }

        // Local shorthand references
        for (skill_name, file_name) in shorthand_refs {
            let mut file_name = file_name;
            if !file_name.ends_with(".md") {
                file_name.push_str(".md");
            }
            self.load_one_reference(ctx, &skill_name, &file_name, &mut output)?;
        }

        // Explicit references: remote ones route to their loader.
        if let Some(refs) = &input.references {
            for sr in refs {
                let skill_name = sr.skill.strip_prefix('/').unwrap_or(&sr.skill);
                if self.is_remote_skill(skill_name) {
                    for file_name in &sr.files {
                        self.load_remote_file(skill_name, file_name, &mut output)
                            .await;
                    }
                } else if self.available_skills.contains_key(skill_name) {
                    for file_name in &sr.files {
                        self.load_one_reference(ctx, skill_name, file_name, &mut output)?;
                    }
                }
            }
        }

        // Remote skills named without a file part.
        for (reference, file) in &remote_requests {
            match file {
                None => {
                    self.load_remote_skill(reference, &mut output).await;
                }
                Some(file_name) => {
                    let mut file_name = file_name.clone();
                    if !file_name.ends_with(".md") {
                        file_name.push_str(".md");
                    }
                    self.load_remote_file(reference, &file_name, &mut output)
                        .await;
                }
            }
        }

        if output.is_empty() {
            if input.skills.is_empty() {
                output.push_str("Warning: No skills requested.");
            } else {
                output.push_str("Warning: Skills not found or none matched.");
            }
        }

        Ok(json!(output))
    }

    async fn do_load_reference(
        &mut self,
        ctx: &mut PluginContext,
        input: &LoadReferenceInput,
    ) -> Result<Value, String> {
        let mut output = String::new();
        let skill_name = input.skill.strip_prefix('/').unwrap_or(&input.skill);

        if !self.available_skills.contains_key(skill_name) && self.is_remote_skill(skill_name) {
            let mut file_name = input.reference.clone();
            if !file_name.ends_with(".md") {
                file_name.push_str(".md");
            }
            self.load_remote_file(skill_name, &file_name, &mut output)
                .await;
            if output.is_empty() {
                return Ok(json!(format!(
                    "Reference '{skill_name}/{file_name}' was already loaded."
                )));
            }
            return Ok(json!(output));
        }

        if self.warn_unknown_skills(&[skill_name.to_string()], &mut output) {
            return Ok(json!(output));
        }

        let mut file_name = input.reference.clone();
        if !file_name.ends_with(".md") {
            file_name.push_str(".md");
        }

        let loaded = self.load_one_reference(ctx, skill_name, &file_name, &mut output)?;

        let path = format!("{skill_name}/{file_name}");

        if !loaded {
            output.push_str(&format!(
                "Warning: Reference '{path}' is already loaded in the context",
            ));
        }

        if output.is_empty() {
            return Ok(json!(format!("Reference '{path}' was already loaded.")));
        }

        Ok(json!(output))
    }

    /// Load a remote skill's SKILL.md through its loader. Idempotent per
    /// session: a second request notes the skill is already loaded rather
    /// than re-fetching.
    async fn load_remote_skill(&mut self, reference: &str, output: &mut String) -> bool {
        let Some(remote) = self.find_remote(reference) else {
            return false;
        };
        let key = format!("{}/{}", remote.server, remote.name);
        if self.loaded_remote.contains(&key) {
            output.push_str(&format!(
                "Note: Skill '{}' is already loaded in the context.",
                remote.name
            ));
            return false;
        }
        match remote.loader.load(None).await {
            Ok(content) => {
                output.push_str(&format!(
                    "## Skill: {} (served by the MCP server {:?})\n{}\n---\n",
                    remote.name, remote.server, content
                ));
                self.loaded_remote.insert(key);
                true
            }
            Err(e) => {
                output.push_str(&format!(
                    "Warning: skill '{}' could not be loaded from server {:?}: {e}\n",
                    remote.name, remote.server
                ));
                false
            }
        }
    }

    /// Load one supporting file of a remote skill.
    async fn load_remote_file(&mut self, reference: &str, file_name: &str, output: &mut String) {
        let Some(remote) = self.find_remote(reference) else {
            return;
        };
        // The main skill content comes first, as with local references.
        self.load_remote_skill(reference, output).await;
        let key = format!("{}/{}/{}", remote.server, remote.name, file_name);
        if self.loaded_remote.contains(&key) {
            output.push_str(&format!(
                "Note: Reference '{file_name}' of skill '{}' is already loaded in the context.",
                remote.name
            ));
            return;
        }
        match remote.loader.load(Some(file_name)).await {
            Ok(content) => {
                output.push_str(&format!(
                    "## {}/{}\n{}\n---\n",
                    remote.name, file_name, content
                ));
                self.loaded_remote.insert(key);
            }
            Err(e) => output.push_str(&format!(
                "Warning: Reference '{file_name}' of skill '{}' could not be loaded: {e}\n",
                remote.name
            )),
        }
    }

    fn load_one_skill(&mut self, name: &str, output: &mut String) -> bool {
        let is_loaded = self
            .available_skills
            .get(name)
            .is_some_and(|s| s.status == LoadStatus::Loaded);
        if is_loaded {
            return false;
        }

        for dep in self.resolve_dependencies(&[name.to_string()]) {
            if self
                .available_skills
                .get(&dep)
                .is_some_and(|s| s.status == LoadStatus::Loaded)
            {
                continue;
            }

            if let Some(skill) = self.available_skills.get_mut(&dep) {
                output.push_str(&format!(
                    "## Skill: {}\n{}\n---\n",
                    skill.name, skill.content
                ));
                skill.status = LoadStatus::Loaded;
            }
        }
        true
    }

    fn load_one_reference(
        &mut self,
        ctx: &mut PluginContext,
        skill_name: &str,
        file_name: &str,
        output: &mut String,
    ) -> Result<bool, String> {
        // Always load the main skill content first
        self.load_one_skill(skill_name, output);

        let key = format!("{skill_name}/{file_name}");
        if self
            .loaded_references
            .get(&key)
            .is_some_and(|s| *s == LoadStatus::Loaded)
        {
            return Ok(false);
        }

        let skill = self
            .available_skills
            .get(skill_name)
            .ok_or_else(|| format!("Skill '{skill_name}' not found for reference loading"))?;

        let file_path = skill.path.join("references").join(file_name);
        let file_path = if file_path.exists() {
            file_path
        } else {
            skill.path.join(file_name)
        };

        if !skill
            .references
            .iter()
            .any(|r| r.path == file_name || r.path.ends_with(&format!("/{file_name}")))
        {
            output.push_str(&format!(
                "Warning: Reference '{file_name}' is not declared in skill '{skill_name}'.\n"
            ));
            return Ok(false);
        }

        let skill_name_owned = skill.name.clone();

        let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
        let content = sandbox.read(&file_path).map_err(|e| {
            format!("Failed to read reference file '{skill_name_owned}/{file_name}': {e}")
        })?;

        output.push_str(&format!(
            "### Reference: {skill_name_owned}/{file_name}\n{content}\n---\n",
        ));
        self.loaded_references.insert(key, LoadStatus::Loaded);

        // Update the reference's status field if it's declared in the skill metadata
        if let Some(skill) = self.available_skills.get_mut(skill_name)
            && let Some(ref_) = skill
                .references
                .iter_mut()
                .find(|r| r.path == file_name || r.path.ends_with(&format!("/{file_name}")))
        {
            ref_.status = LoadStatus::Loaded;
        }

        Ok(true)
    }

    fn discover_scripts(&self, skill: &Skill) -> Vec<String> {
        let scripts_dir = skill.path.join("scripts");
        if !scripts_dir.is_dir() {
            return Vec::new();
        }

        let mut scripts: Vec<String> = match fs::read_dir(&scripts_dir) {
            Ok(entries) => entries
                .flatten()
                .filter(|e| e.path().is_file())
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .collect(),
            Err(_) => Vec::new(),
        };

        scripts.sort();
        scripts
    }

    fn do_find_skills(&self, input: &FindSkillsInput) -> Result<FindSkillsOutput, String> {
        let query = input.query.trim();
        if query.is_empty() {
            return Ok(FindSkillsOutput {
                results: Vec::new(),
                references: Vec::new(),
            });
        }

        // Retrieve more candidates for reranking
        let candidate_count = input.max_results * 3;
        let search_results: Vec<SearchResult<usize>> =
            self.search_index.search(query, candidate_count);

        let query_lower = query.to_lowercase();
        let english_stops = stop_words::get(stop_words::LANGUAGE::English);
        let query_tokens: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|&t| !english_stops.contains(&t))
            .collect();
        let token_count = query_tokens.len().max(1) as f32;

        let mut seen = HashSet::new();
        let mut scored: Vec<(f32, SkillMatch)> = search_results
            .into_iter()
            .filter_map(|r| {
                let name = self
                    .skill_ids
                    .iter()
                    .find(|&(_, id)| id == &r.document.id)
                    .map(|(name, _)| name.clone())?;
                if !seen.insert(name.clone()) {
                    return None;
                }
                let skill = self.available_skills.get(&name)?;

                // Reranking signal: query token overlap with skill name (3x weight)
                let name_lower = skill.name.to_lowercase();
                let name_matches = query_tokens
                    .iter()
                    .filter(|t| name_lower.contains(*t))
                    .count() as f32
                    / token_count;

                // Reranking signal: query token overlap with skill content (0.5x weight)
                let content_lower = skill.content.to_lowercase();
                let content_matches = query_tokens
                    .iter()
                    .filter(|t| content_lower.contains(*t))
                    .count() as f32
                    / token_count;

                let combined = r.score + name_matches * 3.0 + content_matches * 0.5;

                Some((
                    combined,
                    SkillMatch {
                        name: skill.name.clone(),
                        description: skill.description.clone(),
                        score: combined as f64,
                        status: match skill.status {
                            LoadStatus::Loaded => "loaded",
                            LoadStatus::Unloaded => "unloaded",
                        }
                        .to_string(),
                        references: skill.references.clone(),
                        scripts: self.discover_scripts(skill),
                        server: None,
                    },
                ))
            })
            .collect();

        // Remote skills join the same ranking by description/name token
        // overlap — they carry no BM25 document, and local skills shadow a
        // remote one of the same name.
        if let Ok(remotes) = self.remote_skills.lock() {
            for remote in remotes.iter() {
                if self.available_skills.contains_key(&remote.name)
                    || !seen.insert(remote.name.clone())
                {
                    continue;
                }
                let name_lower = remote.name.to_lowercase();
                let name_matches = query_tokens
                    .iter()
                    .filter(|t| name_lower.contains(*t))
                    .count() as f32
                    / token_count;
                let desc_lower = remote.description.to_lowercase();
                let desc_matches = query_tokens
                    .iter()
                    .filter(|t| desc_lower.contains(*t))
                    .count() as f32
                    / token_count;
                let combined = name_matches * 3.0 + desc_matches * 0.5;
                if combined <= 0.0 {
                    continue;
                }
                let status = if self
                    .loaded_remote
                    .contains(&format!("{}/{}", remote.server, remote.name))
                {
                    "loaded"
                } else {
                    "unloaded"
                };
                scored.push((
                    combined,
                    SkillMatch {
                        name: remote.name.clone(),
                        description: remote.description.clone(),
                        score: combined as f64,
                        status: status.to_string(),
                        references: Vec::new(),
                        scripts: Vec::new(),
                        server: Some(remote.server.clone()),
                    },
                ));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(input.max_results);

        let results: Vec<SkillMatch> = scored.into_iter().map(|(_, sm)| sm).collect();

        let mut references: Vec<SkillReferenceInfo> = Vec::new();
        let mut seen = HashSet::new();
        for sm in &results {
            for r in &sm.references {
                let key = format!("{}/{}", sm.name, r.path);
                if seen.insert(key) {
                    references.push(SkillReferenceInfo {
                        skill: sm.name.clone(),
                        title: r.title.clone(),
                        path: r.path.clone(),
                        status: r.status,
                    });
                }
            }
        }

        Ok(FindSkillsOutput {
            results,
            references,
        })
    }

    async fn do_execute_skill_script(
        &self,
        ctx: &mut PluginContext,
        input: &RunScriptInput,
    ) -> Result<Value, String> {
        let skill = self.available_skills.get(&input.skill).ok_or_else(|| {
            if self.is_remote_skill(&input.skill) {
                format!(
                    "Skill '{}' is served by an MCP server. Its scripts are not executable \
                     here — skills from remote servers are instructions to follow, not code \
                     to run.",
                    input.skill
                )
            } else {
                format!(
                    "Skill '{}' not found. DO NOT invent skill or script names. \
                     Use `FindSkills` to search for available skills.",
                    input.skill
                )
            }
        })?;

        let binary_path = skill.path.join("scripts").join(&input.script);

        let binary_path = if binary_path.exists() {
            binary_path
        } else {
            return Err(format!(
                "Script '{}/scripts/{}' not found",
                input.skill, input.script
            ));
        };

        let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
        let mut full_cmd = binary_path.to_string_lossy().to_string();
        if let Some(args) = &input.args {
            full_cmd.push(' ');
            full_cmd.push_str(args);
        }

        let output = sandbox
            .exec(&full_cmd)
            .await
            .map_err(|e| format!("Failed to execute: {e}"))?;

        Ok(json!({
            "exit_code": output.exit_code,
            "stdout": output.stdout,
            "stderr": output.stderr,
        }))
    }
}

pub fn parse_skill(raw: &str) -> Option<Skill> {
    let (yaml, content) = split_frontmatter(raw);
    let mut skill: Skill = serde_yaml::from_str(&yaml)
        .map_err(|e| {
            tracing::error!("Failed to parse skill YAML: {}", e);
            e
        })
        .ok()?;
    skill.content = content.clone();
    skill.references = extract_references(&content);
    Some(skill)
}

fn extract_references(content: &str) -> Vec<Reference> {
    let mut refs = Vec::new();
    for line in content.lines() {
        // Simple markdown link extraction: [text](path)
        // Only capture links pointing to references/ files
        if let (Some(t_start), Some(t_end), Some(p_start), Some(p_end)) = (
            line.find('['),
            line.find(']'),
            line.find('('),
            line.find(')'),
        ) && t_start < t_end
            && t_end < p_start
            && p_start < p_end
        {
            let path = &line[p_start + 1..p_end];
            if path.starts_with("./references/") || path.starts_with("references/") {
                let title = &line[t_start + 1..t_end];
                let clean_path = path.strip_prefix("./").unwrap_or(path);
                refs.push(Reference {
                    title: title.to_string(),
                    path: clean_path.to_string(),
                    status: LoadStatus::Unloaded,
                });
            }
        }
    }
    refs
}

pub fn split_frontmatter(raw: &str) -> (String, String) {
    let lines: Vec<&str> = raw.lines().collect();
    if lines.first().is_some_and(|l| l.trim() == "---") {
        let mut i = 1;
        while i < lines.len() && lines[i].trim() != "---" {
            i += 1;
        }
        if i < lines.len() {
            let yaml = lines[1..i].join("\n");
            let body = lines[i + 1..].join("\n");
            return (yaml, body.trim().to_string());
        }
    }
    (String::new(), raw.trim().to_string())
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct LoadSkillsInput {
    /// List of skill names to load.
    #[serde(default)]
    pub skills: Vec<String>,
    /// Optional: Reference files to load from specific skills.
    pub references: Option<Vec<SkillReference>>,
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct SkillReference {
    /// The name of the skill.
    pub skill: String,
    /// List of filenames to load (e.g., ["extra.md"]).
    pub files: Vec<String>,
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct LoadReferenceInput {
    /// Name of the skill containing the reference.
    pub skill: String,
    /// Reference file path relative to the skill (e.g. "references/CPP.md").
    pub reference: String,
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct RunScriptInput {
    /// The name of the skill containing the binary.
    pub skill: String,
    /// The path of script to execute
    pub script: String,
    /// Optional: Command-line arguments.
    pub args: Option<String>,
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct FindSkillsInput {
    /// Rewrite the query with additional context about the query in hand, project, etc. to make it more relevant.
    /// Specific queries give better results.
    pub query: String,
    /// Maximum number of results to return (default 20).
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_max_results() -> usize {
    20
}

#[derive(JsonSchema, Serialize)]
struct SkillMatch {
    name: String,
    description: String,
    #[serde(skip)]
    #[allow(dead_code)]
    score: f64,
    status: String,
    references: Vec<Reference>,
    scripts: Vec<String>,
    /// Present only when the skill is served by an MCP server; the agent
    /// must be able to tell remote instructions from local ones.
    #[serde(skip_serializing_if = "Option::is_none")]
    server: Option<String>,
}

#[derive(JsonSchema, Serialize)]
struct SkillReferenceInfo {
    skill: String,
    title: String,
    path: String,
    status: LoadStatus,
}

#[derive(JsonSchema, Serialize)]
struct FindSkillsOutput {
    results: Vec<SkillMatch>,
    references: Vec<SkillReferenceInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsdk::core::plugin::PluginContext;
    use agentsdk::core::sandbox::Unsandboxed;
    use serial_test::serial;
    use tempfile::tempdir;

    #[tokio::test]
    #[serial]
    async fn test_load_skills() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();

        let skill_md = "---\nname: test-skill\ndescription: A test skill\n---\nBody content";
        fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({
                        "skills": ["test-skill"]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("Body content"));
        assert!(
            plugin
                .available_skills
                .get("test-skill")
                .is_some_and(|s| s.status == LoadStatus::Loaded)
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_skill_dependencies() {
        let dir = tempdir().unwrap();

        let s1_dir = dir.path().join("s1");
        fs::create_dir_all(&s1_dir).unwrap();
        fs::write(
            s1_dir.join("SKILL.md"),
            "---\nname: s1\ndescription: s1\nneeds: [s2]\n---\ns1 content",
        )
        .unwrap();

        let s2_dir = dir.path().join("s2");
        fs::create_dir_all(&s2_dir).unwrap();
        fs::write(
            s2_dir.join("SKILL.md"),
            "---\nname: s2\ndescription: s2\n---\ns2 content",
        )
        .unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({
                        "skills": ["s1"]
                    }),
                },
            )
            .await
            .unwrap();

        let output = result.as_str().unwrap();
        assert!(output.contains("s1 content"));
        assert!(output.contains("s2 content"));
        assert!(
            plugin
                .available_skills
                .get("s1")
                .is_some_and(|s| s.status == LoadStatus::Loaded)
        );
        assert!(
            plugin
                .available_skills
                .get("s2")
                .is_some_and(|s| s.status == LoadStatus::Loaded)
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_binary() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();

        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\nBody",
        )
        .unwrap();

        #[cfg(unix)]
        {
            let scripts_dir = skill_dir.join("scripts");
            fs::create_dir_all(&scripts_dir).unwrap();
            let script_path = scripts_dir.join("hello.sh");
            fs::write(&script_path, "#!/bin/sh\necho 'hello from script'").unwrap();

            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).unwrap();

            let mut plugin = SkillsPlugin::builder()
                .search_paths(vec![dir.path().to_path_buf()])
                .build()
                .unwrap();

            let mut world = hecs::World::new();
            let entity = world.spawn(());
            world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
            let mut ctx = PluginContext::new(world, entity);

            let result = plugin
                .run_tool(
                    &mut ctx,
                    &PluginToolCall {
                        id: "1".into(),
                        name: "RunSkillScript".into(),
                        arguments: json!({
                            "skill": "test-skill",
                            "script": "hello.sh"
                        }),
                    },
                )
                .await
                .unwrap();

            assert!(
                result["stdout"]
                    .as_str()
                    .unwrap()
                    .contains("hello from script")
            );
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_deterministic_loading_order() {
        let dir = tempdir().unwrap();

        fs::create_dir_all(dir.path().join("a")).unwrap();
        fs::write(
            dir.path().join("a").join("SKILL.md"),
            "---\nname: a\ndescription: a\n---\na content",
        )
        .unwrap();

        fs::create_dir_all(dir.path().join("b")).unwrap();
        fs::write(
            dir.path().join("b").join("SKILL.md"),
            "---\nname: b\ndescription: b\n---\nb content",
        )
        .unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Load A then B
        let result1 = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({ "skills": ["a", "b"] }),
                },
            )
            .await
            .unwrap();

        // Reset loaded skills for comparison
        for skill in plugin.available_skills.values_mut() {
            skill.status = LoadStatus::Unloaded;
        }
        plugin.loaded_references.clear();

        // Load B then A
        let result2 = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "2".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({ "skills": ["b", "a"] }),
                },
            )
            .await
            .unwrap();

        assert_eq!(result1, result2);
        assert!(
            result1.as_str().unwrap().find("Skill: a").unwrap()
                < result1.as_str().unwrap().find("Skill: b").unwrap()
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_load_skills_with_slash_prefix() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\ncontent",
        )
        .unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Load using "/test-skill"
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({
                        "skills": ["/test-skill"]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("content"));
        assert!(
            plugin
                .available_skills
                .get("test-skill")
                .is_some_and(|s| s.status == LoadStatus::Loaded)
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_load_skill_references() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::create_dir_all(skill_dir.join("references")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\nmain content\n\nFor more details see [Extra context](./references/extra.md)",
        )
        .unwrap();
        fs::write(skill_dir.join("references/extra.md"), "reference content").unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        // Verify references were extracted during build
        let skills = plugin.available_skills();
        assert_eq!(
            skills[0].2,
            vec![Reference {
                title: "Extra context".into(),
                path: "references/extra.md".into(),
                status: LoadStatus::Unloaded,
            }]
        );

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Load reference
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({
                        "references": [{
                            "skill": "test-skill",
                            "files": ["references/extra.md"]
                        }]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("reference content"));
        assert!(
            plugin
                .loaded_references
                .contains_key("test-skill/references/extra.md")
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_load_reference_auto_loads_skill() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::create_dir_all(skill_dir.join("references")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\nmain skill content\n\n[Extra context](./references/extra.md)",
        )
        .unwrap();
        fs::write(skill_dir.join("references/extra.md"), "reference content").unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world.insert_one(entity, Sandbox::new(Unsandboxed)).unwrap();
        let mut ctx = PluginContext::new(world, entity);

        // Load reference only
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "LoadSkillReference".into(),
                    arguments: json!({
                        "skill": "test-skill",
                        "reference": "extra"
                    }),
                },
            )
            .await
            .unwrap();

        let output = result.as_str().unwrap();
        assert!(output.contains("main skill content"));
        assert!(output.contains("reference content"));
        assert!(
            plugin
                .available_skills
                .get("test-skill")
                .is_some_and(|s| s.status == LoadStatus::Loaded)
        );
        assert!(plugin.loaded_references.contains_key("test-skill/extra.md"));
    }
}

/// Remote-sourced skills flow through the same Find/Load surface as local
/// ones: the sink is what an MCP plugin fills, the tests stub the loader.
#[cfg(test)]
mod remote_tests {
    #![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use agentsdk::core::plugin::PluginContext;

    struct StubLoader;

    #[async_trait]
    impl SkillLoader for StubLoader {
        async fn load(&self, file: Option<&str>) -> Result<String, String> {
            Ok(match file {
                None => "SKILL BODY".to_owned(),
                Some(f) => format!("BODY OF {f}"),
            })
        }
    }

    fn sink_with_deploy() -> RemoteSkillSink {
        let sink = remote_skill_sink();
        sink.lock().unwrap().push(RemoteSkill {
            server: "mem".to_owned(),
            name: "deploy".to_owned(),
            description: "Ship the worker to the fleet".to_owned(),
            loader: Arc::new(StubLoader),
        });
        sink
    }

    #[tokio::test]
    async fn remote_skills_are_found_and_carry_their_origin() {
        let plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let output = plugin
            .do_find_skills(&FindSkillsInput {
                query: "deploy fleet".to_owned(),
                max_results: 20,
            })
            .unwrap();
        assert_eq!(output.results.len(), 1);
        assert_eq!(output.results[0].name, "deploy");
        assert_eq!(output.results[0].server.as_deref(), Some("mem"));
        assert_eq!(output.results[0].status, "unloaded");
    }

    #[test]
    fn probe() {
        let sink = remote_skill_sink();
        sink.lock().unwrap().push(RemoteSkill {
            server: "mem".to_owned(),
            name: "deploy".to_owned(),
            description: "d".to_owned(),
            loader: Arc::new(StubLoader),
        });
        let plugin = SkillsPlugin::builder()
            .remote_skills(sink.clone())
            .build()
            .unwrap();
        assert_eq!(sink.lock().unwrap().len(), 1);
        assert!(plugin.is_remote_skill("deploy"), "is_remote_skill failed");
    }

    #[tokio::test]
    async fn load_fetches_the_skill_once() {
        let mut plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);

        let first = plugin
            .do_load_skills(
                &mut ctx,
                &LoadSkillsInput {
                    skills: vec!["deploy".to_owned()],
                    references: None,
                },
            )
            .await
            .unwrap();
        let text = first.as_str().unwrap();
        assert!(text.contains("SKILL BODY"), "{text}");
        assert!(text.contains("served by the MCP server \"mem\""), "{text}");

        // A second load must not re-fetch; it notes the skill is loaded.
        let second = plugin
            .do_load_skills(
                &mut ctx,
                &LoadSkillsInput {
                    skills: vec!["deploy".to_owned()],
                    references: None,
                },
            )
            .await
            .unwrap();
        assert!(
            second.as_str().unwrap().contains("already loaded"),
            "{second}"
        );
    }

    #[tokio::test]
    async fn qualified_server_name_loads_a_remote_skill() {
        let mut plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);

        let out = plugin
            .do_load_skills(
                &mut ctx,
                &LoadSkillsInput {
                    skills: vec!["mem/deploy".to_owned()],
                    references: None,
                },
            )
            .await
            .unwrap();
        let text = out.as_str().unwrap();
        assert!(text.contains("SKILL BODY"), "{text}");
    }

    #[tokio::test]
    async fn run_tool_path_loads_qualified_remote_skill() {
        let mut plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);
        let out = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "t".into(),
                    name: "LoadSkills".into(),
                    arguments: json!({"skills": ["mem/deploy"]}),
                },
            )
            .await
            .unwrap();
        let text = out.as_str().unwrap();
        assert!(text.contains("SKILL BODY"), "{text}");
    }

    #[tokio::test]
    async fn load_skill_reference_fetches_through_the_loader() {
        let mut plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);

        let out = plugin
            .do_load_reference(
                &mut ctx,
                &LoadReferenceInput {
                    skill: "deploy".to_owned(),
                    reference: "references/rollout.md".to_owned(),
                },
            )
            .await
            .unwrap();
        let text = out.as_str().unwrap();
        assert!(text.contains("BODY OF references/rollout.md"), "{text}");
        // The main skill content loads first, as with local references.
        assert!(text.contains("SKILL BODY"), "{text}");
    }

    #[tokio::test]
    async fn remote_skill_scripts_are_refused() {
        let plugin = SkillsPlugin::builder()
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = PluginContext::new(world, entity);

        let err = plugin
            .do_execute_skill_script(
                &mut ctx,
                &RunScriptInput {
                    skill: "deploy".to_owned(),
                    script: "run.sh".to_owned(),
                    args: None,
                },
            )
            .await
            .unwrap_err();
        assert!(err.contains("served by an MCP server"), "{err}");
    }

    #[tokio::test]
    async fn local_skills_shadow_a_remote_one_of_the_same_name() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("deploy");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: LOCAL deploy skill\n---\nLOCAL BODY",
        )
        .unwrap();

        let plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .remote_skills(sink_with_deploy())
            .build()
            .unwrap();
        let output = plugin
            .do_find_skills(&FindSkillsInput {
                query: "deploy".to_owned(),
                max_results: 20,
            })
            .unwrap();
        assert_eq!(output.results.len(), 1);
        assert_eq!(output.results[0].server, None, "local must win");
        assert!(output.results[0].description.contains("LOCAL"));
    }
}
