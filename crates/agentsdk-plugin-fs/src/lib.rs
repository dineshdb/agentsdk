mod chroot;
mod readonly;
mod write;

pub use chroot::ChrootSandbox;
pub use readonly::ReadOnlyFileSystemPlugin;
pub use write::FileSystemPlugin;

use agentsdk::core::plugin::PluginContext;
use agentsdk::core::sandbox::Sandbox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().to_string();
    }
    path.to_string()
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ReadInput {
    path: String,
    start_line: usize,
    /// Number of lines to read
    lines: usize,
}

fn do_read(ctx: &mut PluginContext, input: &ReadInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let content = sandbox
        .read(Path::new(&expand_tilde(&input.path)))
        .map_err(|e| format!("Failed to read {}: {e}", input.path))?;
    let all_lines: Vec<&str> = content.lines().collect();

    let start = input.start_line.max(1);
    let requested_lines = if input.lines == 0 { 10 } else { input.lines };
    let end = (start + requested_lines - 1).min(all_lines.len());

    if start > all_lines.len() {
        return Ok(json!("You have already read the full lines"));
    }

    let slice = all_lines
        .get(start.saturating_sub(1)..end)
        .ok_or("Invalid line range")?;
    let result_content = slice.join("\n");

    Ok(json!({
        "path": input.path,
        "content": result_content,
        "start_line": start,
        "end_line": end,
        "total_lines": all_lines.len()
    }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct WriteInput {
    path: String,
    content: String,
}

fn do_write(ctx: &mut PluginContext, input: &WriteInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    sandbox
        .write(Path::new(&expand_tilde(&input.path)), &input.content)
        .map_err(|e| format!("Failed to write: {e}"))?;
    Ok(json!({ "status": "success", "path": input.path, "bytes": input.content.len() }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ReplaceInput {
    path: String,
    old_string: String,
    new_string: String,
}

fn do_replace(ctx: &mut PluginContext, input: &ReplaceInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let content = sandbox
        .read(Path::new(&expand_tilde(&input.path)))
        .map_err(|e| format!("Failed to read: {e}"))?;
    let occurrences = content.matches(&input.old_string).count();
    if occurrences == 0 {
        return Err(format!("String not found in {}", input.path));
    }
    if occurrences > 1 {
        return Err(format!(
            "String found {occurrences} times in {}. Please provide more context to make it unique.",
            input.path
        ));
    }

    let new_content = content.replace(&input.old_string, &input.new_string);
    sandbox
        .write(Path::new(&expand_tilde(&input.path)), &new_content)
        .map_err(|e| format!("Failed to write: {e}"))?;

    Ok(json!({ "status": "success", "path": input.path }))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct ListInput {
    /// Show tree up to this depth. Defaults to 2.
    depth: Option<usize>,
    path: String,
}

fn do_list(ctx: &mut PluginContext, input: &ListInput) -> Result<Value, String> {
    let max_depth = input.depth.unwrap_or(2);
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;

    fn build_tree(
        sandbox: &Sandbox,
        path: &Path,
        prefix: &str,
        is_root: bool,
        depth: usize,
        max_depth: usize,
    ) -> Result<String, String> {
        let entries = sandbox
            .list(path)
            .map_err(|e| format!("Failed to list directory: {e}"))?;

        if is_root && entries.is_empty() {
            return Ok("(empty)\n".to_string());
        }

        let mut out = String::new();

        if is_root {
            let dirname = path.to_string_lossy();
            out.push_str(&format!("{dirname}\n"));
        }

        if depth >= max_depth {
            return Ok(out);
        }

        let mut dirs: Vec<&str> = Vec::new();
        let mut files: Vec<&str> = Vec::new();
        for (name, is_dir) in entries.iter() {
            if *is_dir {
                dirs.push(name);
            } else {
                files.push(name);
            }
        }
        dirs.sort();
        files.sort();

        let mut by_ext: std::collections::BTreeMap<&str, Vec<&str>> =
            std::collections::BTreeMap::new();
        for name in &files {
            let dot = name.rfind('.').filter(|&i| i > 0);
            let ext = dot.map(|i| &name[i..]).unwrap_or("");
            by_ext.entry(ext).or_default().push(name);
        }

        for ext_files in by_ext.values() {
            let entry = if ext_files.len() == 1 {
                ext_files[0].to_string()
            } else {
                let stems: Vec<&str> = ext_files
                    .iter()
                    .map(|n| {
                        let dot = n.rfind('.').filter(|&i| i > 0);
                        dot.map(|i| &n[..i]).unwrap_or(n)
                    })
                    .collect();
                let base = ext_files[0];
                let dot = base.rfind('.').filter(|&i| i > 0);
                if let Some(dot_idx) = dot {
                    format!("{{{}}}{}", stems.join(","), &base[dot_idx..])
                } else {
                    format!("{{{}}}", stems.join(","))
                }
            };
            out.push_str(&format!("{prefix}{entry}\n"));
        }

        for name in dirs {
            let child_path = path.join(name);
            let child_prefix = format!("{prefix}  ");
            out.push_str(&format!("{prefix}{name}/\n"));
            out.push_str(&build_tree(
                sandbox,
                &child_path,
                &child_prefix,
                false,
                depth + 1,
                max_depth,
            )?);
        }

        Ok(out)
    }

    let tree = build_tree(
        &sandbox,
        Path::new(&expand_tilde(&input.path)),
        "",
        true,
        0,
        max_depth,
    )?;
    Ok(json!(tree.trim_end()))
}

#[derive(JsonSchema, Deserialize, Serialize)]
struct GlobInput {
    pattern: String,
}

fn do_glob(ctx: &mut PluginContext, input: &GlobInput) -> Result<Value, String> {
    let sandbox = ctx.get::<Sandbox>().ok_or("No sandbox registered")?;
    let matches = sandbox
        .glob(&expand_tilde(&input.pattern))
        .map_err(|e| format!("Glob error: {e}"))?;

    Ok(json!({ "pattern": input.pattern, "matches": matches }))
}
