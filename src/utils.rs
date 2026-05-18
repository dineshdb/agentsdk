pub fn to_terse_json(schema: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = schema.as_object()
        && obj.get("type").and_then(|t| t.as_str()) == Some("object")
        && let Some(properties) = obj.get("properties").and_then(|p| p.as_object())
    {
        let required = obj
            .get("required")
            .and_then(|r| r.as_array())
            .map(|r| {
                r.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();

        let mut terse = serde_json::Map::new();
        for (name, prop) in properties {
            let mut key = name.clone();
            if !required.contains(name.as_str()) {
                key.push('?');
            }

            terse.insert(key, format_property(prop));
        }
        return serde_json::Value::Object(terse);
    }
    schema.clone()
}

fn format_property(prop: &serde_json::Value) -> serde_json::Value {
    let desc = prop
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("");
    let comment = if desc.is_empty() {
        String::new()
    } else {
        format!(" // {desc}")
    };

    if let Some(obj) = prop.as_object() {
        if obj.get("type").and_then(|t| t.as_str()) == Some("object") {
            // For nested objects, we don't append comments here as they might be complex
            return to_terse_json(prop);
        }

        if let Some(enm) = obj.get("enum").and_then(|e| e.as_array()) {
            let vals: Vec<String> = enm.iter().map(|v| v.to_string().replace('"', "")).collect();
            return serde_json::Value::String(format!("<enum: {}>{comment}", vals.join(" | ")));
        }

        if obj.get("type").and_then(|t| t.as_str()) == Some("array")
            && let Some(items) = obj.get("items")
        {
            // For arrays, the comment applies to the array itself
            return serde_json::Value::Array(vec![format_property(items)]);
        }

        let ty = obj
            .get("format")
            .and_then(|f| f.as_str())
            .unwrap_or_else(|| obj.get("type").and_then(|t| t.as_str()).unwrap_or("any"));

        return serde_json::Value::String(format!("<{ty}>{comment}"));
    }
    serde_json::Value::String(format!("<any>{comment}"))
}
