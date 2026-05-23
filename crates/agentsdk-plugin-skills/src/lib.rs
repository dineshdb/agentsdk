use agentsdk::core::messages::Messages;
use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::sandbox::Sandbox;
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use derive_builder::Builder;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SkillsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// A specialized reference file for a skill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reference {
    pub title: String,
    pub path: String,
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
    #[serde(default)]
    pub references: Vec<Reference>,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// A plugin that manages skills.
#[derive(Debug, Clone)]
pub struct SkillsPlugin {
    /// Cached skills found during scanning.
    available_skills: HashMap<String, Skill>,
    /// Skills that have been loaded into the agent's context.
    loaded_skills: HashSet<String>,
    /// Reference files that have been loaded. Key format: "skill_name/file_name"
    loaded_references: HashSet<String>,
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
}

impl SkillsPluginBuilder {
    pub fn build(self) -> Result<SkillsPlugin, SkillsError> {
        let search_paths = self.search_paths.unwrap_or_default();
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

        Ok(SkillsPlugin {
            available_skills,
            loaded_skills: HashSet::new(),
            loaded_references: HashSet::new(),
        })
    }
}

const PROMPT: &str = r#"
## Skills
Skills are extra instruction that gives more knowledge about specific topic.

### Loading Skills & References
- Use the `load` tool to load broad topic instructions (e.g., `skills=['joke']`).
- Use the `reference` tool to load deep-dive files for specific sub-topics.

**CRITICAL:** Before responding to any user query, you MUST check the 'Available skills' and their 'References' below.
If a user's request matches a sub-topic listed under 'References', you MUST load that reference using `load_reference` BEFORE providing an answer.
This applies even if you have already loaded the main skill.

Example:
`reference(path="joke/CPP.md")`

### Available skills:
"#;

#[async_trait]
impl AgentPlugin for SkillsPlugin {
    fn name(&self) -> &'static str {
        "skills"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "load".into(),
                description: "Load instructions from skills and their references(optional).".into(),
                input_schema: schema_for!(LoadSkillsInput),
            },
            ToolDefinition {
                name: "reference".into(),
                description:
                    "Load a specific reference file from a skill using 'skill/file' format.".into(),
                input_schema: schema_for!(LoadReferenceInput),
            },
            ToolDefinition {
                name: "execute".into(),
                description: "execute a script from a skill, as instructed by the loaded script"
                    .into(),
                input_schema: schema_for!(ExecuteSkillScriptInput),
            },
        ]
    }

    async fn run_tool(
        &mut self,
        ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        match call.name.as_str() {
            "load" => {
                let input: LoadSkillsInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                self.do_load_skills(ctx, &input)
            }
            "reference" => {
                let input: LoadReferenceInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                self.do_load_reference(ctx, &input)
            }
            "execute" => {
                let input: ExecuteSkillScriptInput =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                self.do_execute_skill_script(ctx, &input).await
            }
            _ => Err(format!("Unknown tool: {}", call.name)),
        }
    }

    async fn prepare_system_prompt(
        &mut self,
        _ctx: &mut PluginContext,
        _history: &Messages,
    ) -> Option<Cow<'static, str>> {
        let available = self.available_skills();
        if available.is_empty() {
            return None;
        }

        let mut prompt = String::from(PROMPT);
        for (name, desc, refs) in available {
            let status = if self.loaded_skills.contains(&name) {
                "[ALREADY LOADED]"
            } else {
                "[NOT LOADED]"
            };

            prompt.push_str(&format!("- {}: {} {}\n", name, desc, status));
            for r in refs {
                let r_status = if self
                    .loaded_references
                    .contains(&format!("{}/{}", name, r.path))
                {
                    "[ALREADY LOADED]"
                } else {
                    "[NOT LOADED]"
                };
                prompt.push_str(&format!(
                    "    - Reference: {} ({}) {}\n",
                    r.title, r.path, r_status
                ));
            }
        }
        Some(Cow::Owned(prompt))
    }
}

impl SkillsPlugin {
    fn do_load_skills(
        &mut self,
        ctx: &mut PluginContext,
        input: &LoadSkillsInput,
    ) -> Result<Value, String> {
        let mut output = String::new();

        // 1. Handle Skills & Shorthand References
        let mut skill_names_to_resolve = Vec::new();
        let mut shorthand_refs = Vec::new();

        for s in &input.skills {
            let s = s.strip_prefix('/').unwrap_or(s);
            // Support both 'skill/file' and 'skill:file' for backward compatibility
            let (skill_part, file_part) = if let Some(parts) = s.split_once('/') {
                parts
            } else if let Some(parts) = s.split_once(':') {
                parts
            } else {
                skill_names_to_resolve.push(s.to_string());
                continue;
            };

            shorthand_refs.push((skill_part.to_string(), file_part.to_string()));
            skill_names_to_resolve.push(skill_part.to_string());
        }

        // Resolve and sort all skill names for deterministic prompt generation
        let mut skill_names = self.resolve_dependencies(&skill_names_to_resolve);
        skill_names.sort();

        for name in skill_names {
            self.load_one_skill(&name, &mut output);
        }

        // Process Shorthand References
        for (skill_name, file_name) in shorthand_refs {
            let mut file_name = file_name;
            if !file_name.ends_with(".md") {
                file_name.push_str(".md");
            }
            self.load_one_reference(ctx, &skill_name, &file_name, &mut output)?;
        }

        // 2. Handle Explicit References
        if let Some(refs) = &input.references {
            for sr in refs {
                let skill_name = sr.skill.strip_prefix('/').unwrap_or(&sr.skill);
                for file_name in &sr.files {
                    self.load_one_reference(ctx, skill_name, file_name, &mut output)?;
                }
            }
        }

        if output.is_empty() {
            if input.skills.is_empty() {
                return Err("No skills requested".into());
            }
            return Ok(json!("All requested items were already loaded."));
        }

        Ok(json!(output))
    }

    fn do_load_reference(
        &mut self,
        ctx: &mut PluginContext,
        input: &LoadReferenceInput,
    ) -> Result<Value, String> {
        let mut output = String::new();
        let path = input.path.strip_prefix('/').unwrap_or(&input.path);
        let (skill_name, file_name) = path.split_once('/').ok_or_else(|| {
            format!("Invalid reference path format '{path}'. Expected 'skill/file'.")
        })?;

        let mut file_name = file_name.to_string();
        if !file_name.ends_with(".md") {
            file_name.push_str(".md");
        }

        self.load_one_reference(ctx, skill_name, &file_name, &mut output)?;

        if output.is_empty() {
            return Ok(json!(format!("Reference '{path}' was already loaded.")));
        }

        Ok(json!(output))
    }

    fn load_one_skill(&mut self, name: &str, output: &mut String) {
        if self.loaded_skills.contains(name) {
            return;
        }

        for dep in self.resolve_dependencies(&[name.to_string()]) {
            if self.loaded_skills.contains(&dep) {
                continue;
            }

            if let Some(skill) = self.available_skills.get(&dep) {
                output.push_str(&format!(
                    "## Skill: {}\n{}\n---\n",
                    skill.name, skill.content
                ));
            }

            self.loaded_skills.insert(dep);
        }
    }

    fn load_one_reference(
        &mut self,
        ctx: &mut PluginContext,
        skill_name: &str,
        file_name: &str,
        output: &mut String,
    ) -> Result<(), String> {
        // Always load the main skill content first
        self.load_one_skill(skill_name, output);

        let skill = self
            .available_skills
            .get(skill_name)
            .ok_or_else(|| format!("Skill '{skill_name}' not found for reference loading"))?;

        let key = format!("{}/{}", skill.name, file_name);
        if self.loaded_references.contains(&key) {
            return Ok(());
        }

        let file_path = skill.path.join(file_name);

        let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
        let content = sandbox
            .0
            .read(&file_path)
            .map_err(|e| format!("Failed to read reference file '{file_name}': {e}"))?;

        output.push_str(&format!(
            "### Reference: {}/{file_name}\n{content}\n---\n",
            skill.name
        ));
        self.loaded_references.insert(key);
        Ok(())
    }

    async fn do_execute_skill_script(
        &self,
        ctx: &mut PluginContext,
        input: &ExecuteSkillScriptInput,
    ) -> Result<Value, String> {
        let skill = self
            .available_skills
            .get(&input.from_skill)
            .ok_or_else(|| {
                format!(
                    "Skill '{}' not found. Ensure it exists in search paths.",
                    input.from_skill
                )
            })?;

        // Find the binary in the skill's directory or its bin/ subdirectory
        let mut binary_path = None;
        let p = skill.path.join(&input.binary_name);
        if p.exists() {
            binary_path = Some(p);
        } else {
            let p = skill.path.join("bin").join(&input.binary_name);
            if p.exists() {
                binary_path = Some(p);
            }
        }

        let binary_path = binary_path.ok_or_else(|| {
            format!(
                "Binary '{}' not found for skill '{}'",
                input.binary_name, input.from_skill
            )
        })?;

        let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
        let mut full_cmd = binary_path.to_string_lossy().to_string();
        if let Some(args) = &input.args {
            full_cmd.push(' ');
            full_cmd.push_str(args);
        }

        let output = sandbox
            .0
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
    let mut in_references = false;
    for line in content.lines() {
        if line.starts_with("## References") {
            in_references = true;
            continue;
        }
        if in_references && line.starts_with("##") {
            in_references = false;
        }
        if in_references {
            // Very simple markdown link extraction: [text](path)
            if let (Some(t_start), Some(t_end), Some(p_start), Some(p_end)) = (
                line.find('['),
                line.find(']'),
                line.find('('),
                line.find(')'),
            ) && t_start < t_end
                && t_end < p_start
                && p_start < p_end
            {
                let title = &line[t_start + 1..t_end];
                let path = &line[p_start + 1..p_end];
                // Clean up relative paths
                let clean_path = path.strip_prefix("./").unwrap_or(path);
                refs.push(Reference {
                    title: title.to_string(),
                    path: clean_path.to_string(),
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
    /// The reference path in 'skill_name/file_name' format.
    pub path: String,
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ExecuteSkillScriptInput {
    /// The name of the skill containing the binary.
    #[serde(rename = "from_skill")]
    pub from_skill: String,
    /// The filename of the binary to execute.
    #[serde(rename = "binary_name")]
    pub binary_name: String,
    /// Optional: Command-line arguments.
    pub args: Option<String>,
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
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "load".into(),
                    arguments: json!({
                        "skills": ["test-skill"]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("Body content"));
        assert!(plugin.loaded_skills.contains("test-skill"));
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
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "load".into(),
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
        assert!(plugin.loaded_skills.contains("s1"));
        assert!(plugin.loaded_skills.contains("s2"));
    }

    #[tokio::test]
    #[serial]
    async fn test_prepare_system_prompt() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: Test description\n---\nBody",
        )
        .unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };
        let history = Messages::default();

        let prompt = plugin
            .prepare_system_prompt(&mut ctx, &history)
            .await
            .unwrap();
        assert!(prompt.contains("test-skill"));
        assert!(prompt.contains("Test description"));
        assert!(prompt.contains("[NOT LOADED"));

        plugin.loaded_skills.insert("test-skill".into());
        let prompt = plugin
            .prepare_system_prompt(&mut ctx, &history)
            .await
            .unwrap();
        assert!(prompt.contains("[ALREADY LOADED]"));
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
            let script_path = skill_dir.join("hello.sh");
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
            world
                .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
                .unwrap();
            let mut ctx = PluginContext { world, entity };

            let result = plugin
                .run_tool(
                    &mut ctx,
                    &PluginToolCall {
                        id: "1".into(),
                        name: "execute".into(),
                        arguments: json!({
                            "from_skill": "test-skill",
                            "binary_name": "hello.sh"
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
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Load A then B
        let result1 = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "load".into(),
                    arguments: json!({ "skills": ["a", "b"] }),
                },
            )
            .await
            .unwrap();

        // Reset loaded skills for comparison
        plugin.loaded_skills.clear();

        // Load B then A
        let result2 = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "2".into(),
                    name: "load".into(),
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
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Load using "/test-skill"
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "load".into(),
                    arguments: json!({
                        "skills": ["/test-skill"]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("content"));
        assert!(plugin.loaded_skills.contains("test-skill"));
    }

    #[tokio::test]
    #[serial]
    async fn test_load_skill_references() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\nmain content\n\n## References\n- [Extra context](./extra.md)",
        )
        .unwrap();
        fs::write(skill_dir.join("extra.md"), "reference content").unwrap();

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
                path: "extra.md".into()
            }]
        );

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Load reference
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "load".into(),
                    arguments: json!({
                        "references": [{
                            "skill": "test-skill",
                            "files": ["extra.md"]
                        }]
                    }),
                },
            )
            .await
            .unwrap();

        assert!(result.as_str().unwrap().contains("reference content"));
        assert!(plugin.loaded_references.contains("test-skill/extra.md"));
    }

    #[tokio::test]
    #[serial]
    async fn test_load_reference_auto_loads_skill() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: desc\n---\nmain skill content",
        )
        .unwrap();
        fs::write(skill_dir.join("extra.md"), "reference content").unwrap();

        let mut plugin = SkillsPlugin::builder()
            .search_paths(vec![dir.path().to_path_buf()])
            .build()
            .unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn(());
        world
            .insert_one(entity, Sandbox(Box::new(Unsandboxed)))
            .unwrap();
        let mut ctx = PluginContext { world, entity };

        // Load reference only
        let result = plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "1".into(),
                    name: "reference".into(),
                    arguments: json!({
                        "path": "test-skill/extra"
                    }),
                },
            )
            .await
            .unwrap();

        let output = result.as_str().unwrap();
        assert!(output.contains("main skill content"));
        assert!(output.contains("reference content"));
        assert!(plugin.loaded_skills.contains("test-skill"));
        assert!(plugin.loaded_references.contains("test-skill/extra.md"));
    }
}
