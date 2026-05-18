use agentsdk::core::tools::Tool;
use agentsdk::tool;

#[tool]
/// This is a test tool.
/// It has multiple lines.
fn test_tool() -> Tool {
    let res: Result<String, String> = Ok("ok".to_string());
    res
}

#[tool(desc = "overridden description")]
fn test_tool_override() -> Tool {
    let res: Result<String, String> = Ok("ok".to_string());
    res
}

#[test]
fn test_macro_description() {
    let tool = test_tool();
    assert_eq!(tool.name(), "test_tool");
    assert_eq!(
        tool.description(),
        "This is a test tool.\nIt has multiple lines."
    );

    let tool_overridden = test_tool_override();
    assert_eq!(tool_overridden.description(), "overridden description");
}

#[tool]
/// A tool with parameters.
fn complex_tool(name: String, age: i32, context: agentsdk::core::tools::ToolContext) -> Tool {
    let res: Result<String, String> = Ok(format!(
        "Hello, {}! You are {} years old. Model: {}",
        name, age, context.options.model
    ));
    res
}

#[derive(agentsdk::__private::schemars::JsonSchema, serde::Deserialize, Default, Debug)]
#[schemars(crate = "::agentsdk::__private::schemars")]
/// A search query.
struct SearchQuery {
    /// The search term.
    query: String,
    /// The number of results to return.
    limit: Option<i32>,
}

#[tool]
/// Search for something.
fn search_tool(req: SearchQuery) -> Tool {
    let res: Result<String, String> = Ok(format!(
        "Searching for '{}' (limit: {:?})",
        req.query, req.limit
    ));
    res
}

#[test]
fn test_macro_struct_input() {
    let tool = search_tool();

    // Verify schema
    let schema = serde_json::to_value(tool.input_schema()).unwrap();
    let defs = schema.get("$defs").unwrap();
    let query_def = defs.get("SearchQuery").unwrap();
    let properties = query_def.get("properties").unwrap();

    // Check if descriptions from the struct fields are preserved
    assert_eq!(
        properties.get("query").unwrap().get("description").unwrap(),
        "The search term."
    );
    assert_eq!(
        properties.get("limit").unwrap().get("description").unwrap(),
        "The number of results to return."
    );
}

#[derive(serde::Serialize, serde::Deserialize, agentsdk::__private::schemars::JsonSchema)]
#[schemars(crate = "::agentsdk::__private::schemars")]
struct SearchResponse {
    results: Vec<String>,
}

#[tool]
fn tool_with_struct_output() -> Tool {
    let response = SearchResponse {
        results: vec!["result1".to_string(), "result2".to_string()],
    };
    let res: Result<SearchResponse, String> = Ok(response);
    res
}

#[test]
fn test_macro_struct_output() {
    let tool = tool_with_struct_output();

    let options = agentsdk::AgentOptions::builder().build().unwrap();
    let ctx = agentsdk::core::tools::ToolContext {
        options: std::sync::Arc::new(options),
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(tool.call(ctx, serde_json::json!({}))).unwrap();

    let response: SearchResponse = serde_json::from_value(result).unwrap();
    assert_eq!(response.results.len(), 2);
    assert_eq!(response.results[0], "result1");
}

#[test]
fn test_macro_parameters() {
    let tool = complex_tool();
    assert_eq!(tool.name(), "complex_tool");

    // Verify schema
    let schema = serde_json::to_value(tool.input_schema()).unwrap();
    let properties = schema.get("properties").unwrap();

    assert!(properties.get("name").is_some());
    assert!(properties.get("age").is_some());
    // ToolContext should be excluded
    assert!(properties.get("context").is_none());

    // Verify execution
    let options = agentsdk::AgentOptions::builder()
        .model("test-model")
        .build()
        .unwrap();
    let ctx = agentsdk::core::tools::ToolContext {
        options: std::sync::Arc::new(options),
    };
    let input = serde_json::json!({
        "name": "Alice",
        "age": 30
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(tool.call(ctx, input)).unwrap();
    assert_eq!(
        result,
        serde_json::json!("Hello, Alice! You are 30 years old. Model: test-model")
    );
}
