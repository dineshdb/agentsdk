use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;
use std::collections::HashMap;

pub struct McpPlugin {
    clients: Vec<(String, RunningService<RoleClient, ()>)>,
    tools: Vec<(usize, ToolDefinition)>,
}

impl Default for McpPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl McpPlugin {
    pub fn new() -> Self {
        Self {
            clients: Vec::new(),
            tools: Vec::new(),
        }
    }

    pub async fn add_server(
        &mut self,
        name: impl Into<String>,
        command: tokio::process::Command,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = name.into();
        let client = ().serve(TokioChildProcess::new(command)?).await?;
        self.register_client(name, client).await
    }

    pub async fn add_remote_server(
        &mut self,
        name: impl Into<String>,
        url: &str,
        headers: HashMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = name.into();

        let mut config = StreamableHttpClientTransportConfig::with_uri(url);
        for (k, v) in headers {
            if let (Ok(hname), Ok(hval)) = (
                http::HeaderName::from_bytes(k.as_bytes()),
                http::HeaderValue::from_str(&v),
            ) {
                config.custom_headers.insert(hname, hval);
            }
        }

        let transport = StreamableHttpClientTransport::from_config(config);
        let client = ().serve(transport).await?;
        self.register_client(name, client).await
    }

    async fn register_client(
        &mut self,
        name: String,
        client: rmcp::service::RunningService<rmcp::service::RoleClient, ()>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mcp_tools = client.list_tools(None).await?;

        let client_idx = self.clients.len();
        self.clients.push((name.clone(), client));

        for tool in mcp_tools.tools {
            self.tools.push((
                client_idx,
                ToolDefinition {
                    name: format!("{name}__{}", tool.name),
                    description: tool.description.unwrap_or_default().to_string(),
                    input_schema: serde_json::from_value(Value::Object(
                        std::sync::Arc::try_unwrap(tool.input_schema)
                            .unwrap_or_else(|arc| (*arc).clone()),
                    ))
                    .unwrap_or_default(),
                },
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl AgentPlugin for McpPlugin {
    fn name(&self) -> &'static str {
        "mcp"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|(_, t)| t.clone()).collect()
    }

    async fn run_tool(
        &mut self,
        _ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        let (client_idx, tool_def) = self
            .tools
            .iter()
            .find(|(_, t)| t.name == call.name)
            .ok_or_else(|| format!("Tool {} not found in McpPlugin", call.name))?;

        let (server_name, client) = self
            .clients
            .get_mut(*client_idx)
            .ok_or_else(|| format!("Internal error: Client index {} not found", client_idx))?;

        // Strip the server prefix to get the original tool name
        let prefix = format!("{server_name}__");
        let original_name = tool_def.name.strip_prefix(&prefix).ok_or_else(|| {
            format!(
                "Internal error: Tool name {} does not start with prefix {}",
                tool_def.name, prefix
            )
        })?;

        let mut req = CallToolRequestParams::new(original_name.to_string());
        if let Some(args) = call.arguments.as_object() {
            req = req.with_arguments(args.clone());
        }

        let result = client.call_tool(req).await.map_err(|e| e.to_string())?;

        Ok(serde_json::to_value(result).map_err(|e| e.to_string())?)
    }
}
