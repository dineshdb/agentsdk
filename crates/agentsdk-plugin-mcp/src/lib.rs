use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use rmcp::model::ProtocolVersion;
use rmcp::service::ClientInitializeError;
use rmcp::service::ClientLifecycleMode;
use rmcp::service::ClientServiceExt;
use rmcp::service::RunningService;
use rmcp::transport::IntoTransport;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::TokioChildProcess;
use rmcp::transport::auth::{AuthClient, AuthorizationManager};
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
        let config = transport_config(url, headers);
        let client = Self::negotiate(&name, || {
            StreamableHttpClientTransport::with_client(reqwest::Client::default(), config.clone())
        })
        .await?;
        self.register_client(name, client).await
    }

    /// Connect to a remote server behind OAuth 2.1 authorization.
    ///
    /// The manager injects the bearer token on every request, silently
    /// refreshes an expired token and retries once on a 401 before
    /// surfacing the challenge to the caller. Build it with rmcp's
    /// `transport::auth` helpers (discovery, PKCE, refresh live there).
    pub async fn add_remote_server_authorized(
        &mut self,
        name: impl Into<String>,
        url: &str,
        headers: HashMap<String, String>,
        auth_manager: AuthorizationManager,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = name.into();
        let config = transport_config(url, headers);
        // AuthClient is Clone and shares the manager behind an Arc, so the
        // probe and the legacy fallback speak with the same authorization
        // state.
        let auth_client = AuthClient::new(reqwest::Client::default(), auth_manager);
        let client = Self::negotiate(&name, || {
            StreamableHttpClientTransport::with_client(auth_client.clone(), config.clone())
        })
        .await?;
        self.register_client(name, client).await
    }

    /// Probe the modern `server/discover` lifecycle (SEP-2243 era, protocol
    /// version 2026-07-28) and fall back to the legacy `initialize`
    /// handshake for servers that report themselves legacy. `.serve()`
    /// alone would lock every server to the legacy 2025-11-25 revision,
    /// which modern-only servers reject with -32022.
    ///
    /// Both revisions are named outright rather than through rmcp's
    /// `STANDARD_HEADERS` / `LATEST` aliases. Those aliases track the SDK,
    /// not this negotiation: the next rmcp release that moves `LATEST` to
    /// 2026-07-28 would make the legacy fallback identical to the modern
    /// probe and strand the very servers it exists for.
    ///
    /// Some legacy gateways (observed: mcp.deepwiki.com) answer the
    /// modern-only probe with a JSON-RPC error whose id is a bogus
    /// literal (`"server-error"`) instead of echoing ours. rmcp classifies
    /// any uncorrelated error response as fatal, so the built-in fallback
    /// never fires. Retry once speaking pure legacy.
    ///
    /// `make_transport` is called once per attempt: the legacy fallback
    /// needs a second transport wrapping the same authorization state.
    async fn negotiate<T, E, A>(
        name: &str,
        make_transport: impl Fn() -> T,
    ) -> Result<RunningService<RoleClient, ()>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        match ()
            .serve_with_lifecycle(
                make_transport(),
                ClientLifecycleMode::Auto {
                    preferred_versions: vec![ProtocolVersion::V_2026_07_28],
                    legacy_version: Some(ProtocolVersion::V_2025_11_25),
                },
            )
            .await
        {
            Ok(client) => Ok(client),
            Err(ClientInitializeError::UncorrelatedErrorResponse { .. }) => {
                tracing::debug!(server = %name, "discover probe got an uncorrelated error; retrying with the legacy initialize handshake");
                ().serve_with_lifecycle(make_transport(), ClientLifecycleMode::Initialize)
                    .await
                    .map_err(Into::into)
            }
            Err(e) => Err(e.into()),
        }
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

        // Sort tools alphabetically for deterministic tool listing
        self.tools.sort_by(|(_, a), (_, b)| a.name.cmp(&b.name));

        Ok(())
    }
}

fn transport_config(
    url: &str,
    headers: HashMap<String, String>,
) -> StreamableHttpClientTransportConfig {
    let mut config = StreamableHttpClientTransportConfig::with_uri(url);
    for (k, v) in headers {
        if let (Ok(hname), Ok(hval)) = (
            http::HeaderName::from_bytes(k.as_bytes()),
            http::HeaderValue::from_str(&v),
        ) {
            config.custom_headers.insert(hname, hval);
        }
    }
    config
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

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use agentsdk::core::plugin::PluginToolCall;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tokio::io::AsyncBufReadExt as _;
    use tokio::io::AsyncReadExt as _;
    use tokio::io::AsyncWriteExt as _;
    use tokio::io::BufReader;
    use tokio::net::TcpListener;

    /// Observed requests: "method <body>" lines, in arrival order.
    type Log = Arc<Mutex<Vec<String>>>;

    const ACCEPTED: &str =
        "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";

    /// Serve one JSON-RPC-over-HTTP request per connection, forever.
    ///
    /// `respond` gets the request head lines and JSON body and returns the raw
    /// HTTP response bytes. Requests are appended to `log` before responding.
    async fn serve_requests(
        log: Log,
        respond: impl Fn(Vec<String>, serde_json::Value) -> String + Send + Sync + 'static,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr").to_string();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let log = log.clone();
                let respond = &respond;
                let (rd, mut wr) = sock.into_split();
                let mut rd = BufReader::new(rd);
                let mut head = Vec::new();
                let mut line = String::new();
                loop {
                    line.clear();
                    if rd.read_line(&mut line).await.unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    head.push(line.clone());
                }
                let content_length = head.iter().find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .and_then(|v| v.trim().parse::<usize>().ok())
                });
                let mut body = vec![0u8; content_length.unwrap_or(0)];
                if !body.is_empty() {
                    rd.read_exact(&mut body).await.expect("read body");
                }
                let request: serde_json::Value =
                    serde_json::from_slice(&body).expect("JSON-RPC body");
                log.lock().expect("log lock").push(format!(
                    "{} {request}",
                    request["method"].as_str().unwrap_or_default()
                ));
                // Notifications carry no id and expect no JSON-RPC response.
                let response = if request.get("id").is_some_and(|id| !id.is_null()) {
                    respond(head, request)
                } else {
                    ACCEPTED.to_string()
                };
                wr.write_all(response.as_bytes()).await.expect("respond");
                wr.flush().await.expect("flush");
            }
        });
        format!("http://{addr}/mcp")
    }

    fn json_ok(id: &serde_json::Value, result: serde_json::Value) -> String {
        http_json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    fn json_err(id: &serde_json::Value, code: i64, message: &str) -> String {
        http_json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message}
        }))
    }

    fn http_json(payload: serde_json::Value) -> String {
        let body = payload.to_string();
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn lowercase_headers(head: &[String]) -> String {
        head.join("\n").to_ascii_lowercase()
    }

    fn tool_named(name: &str) -> serde_json::Value {
        json!({
            "name": name,
            "description": "search things",
            "inputSchema": {"type": "object", "properties": {}}
        })
    }

    fn assert_listed(plugin: &McpPlugin, expected: &str) {
        let names: Vec<String> = plugin.tools().iter().map(|t| t.name.clone()).collect();
        assert!(
            names.iter().any(|n| n == expected),
            "expected {expected} in {names:?}"
        );
    }

    async fn call_search(plugin: &mut McpPlugin, prefixed: &str) -> serde_json::Value {
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = agentsdk::core::plugin::PluginContext::new(world, entity);
        plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "call-1".into(),
                    name: prefixed.into(),
                    arguments: json!({"query": "pixel"}),
                },
            )
            .await
            .expect("run_tool succeeds")
    }

    #[tokio::test]
    async fn authorized_transport_connects_without_stored_credentials() {
        // rmcp's AuthClient sends requests unauthenticated while the manager
        // holds no credentials, so a plain modern server must behave exactly
        // as it does for `add_remote_server`.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = serve_requests(log.clone(), move |_head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                "server/discover" => json_ok(
                    &id,
                    json!({
                        "resultType": "complete",
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}},
                        "ttlMs": 60000,
                        "cacheScope": "public"
                    }),
                ),
                "tools/list" => json_ok(
                    &id,
                    json!({
                        "resultType": "complete",
                        "tools": [tool_named("search")]
                    }),
                ),
                other => panic!("authorized server got unexpected method: {other}"),
            }
        })
        .await;

        let manager = AuthorizationManager::new(&url).await.expect("auth manager");
        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server_authorized("linear", &url, HashMap::new(), manager)
            .await
            .expect("authorized transport connects without credentials");
        assert_listed(&plugin, "linear__search");
    }

    #[tokio::test]
    async fn authorized_transport_refuses_server_that_challenges_every_request() {
        // A 401 on every request must fail the connection — never silently
        // proceed. The challenge's metadata URL is a dead port, so reactive
        // discovery cannot rescue it either.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = serve_requests(log, move |_head, _request| {
            "HTTP/1.1 401 Unauthorized\r\nwww-authenticate: Bearer resource_metadata=\"http://127.0.0.1:9/.well-known/oauth-protected-resource\"\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
        })
        .await;

        let manager = AuthorizationManager::new(&url).await.expect("auth manager");
        let mut plugin = McpPlugin::new();
        let err = plugin
            .add_remote_server_authorized("gated", &url, HashMap::new(), manager)
            .await
            .expect_err("server challenging every request must fail the connection");
        tracing::debug!(error = %err, "expected challenge failure");
    }

    #[tokio::test]
    async fn modern_server_negotiates_2026_07_28_and_calls_tools() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = serve_requests(log.clone(), move |head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                // SEP-2243: every request names its method in a header.
                "server/discover" => {
                    assert!(
                        lowercase_headers(&head).contains("mcp-method: server/discover"),
                        "discover must carry the Mcp-Method header, got: {head:?}"
                    );
                    json_ok(
                        &id,
                        json!({
                            "resultType": "complete",
                            "supportedVersions": ["2026-07-28"],
                            "capabilities": {"tools": {}},
                            "ttlMs": 60000,
                            "cacheScope": "public"
                        }),
                    )
                }
                "tools/list" => {
                    let headers = lowercase_headers(&head);
                    assert!(
                        headers.contains("mcp-protocol-version: 2026-07-28"),
                        "post-discover requests must carry the negotiated version, got: {head:?}"
                    );
                    assert!(headers.contains("mcp-method: tools/list"));
                    json_ok(
                        &id,
                        json!({
                            "resultType": "complete",
                            "tools": [tool_named("search")]
                        }),
                    )
                }
                "tools/call" => {
                    let headers = lowercase_headers(&head);
                    assert!(
                        headers.contains("mcp-protocol-version: 2026-07-28"),
                        "tool calls must carry the negotiated version, got: {head:?}"
                    );
                    assert_eq!(
                        request["params"]["name"], "search",
                        "plugin must strip the server prefix before calling"
                    );
                    json_ok(
                        &id,
                        json!({
                            "resultType": "complete",
                            "content": [{"type": "text", "text": "done"}]
                        }),
                    )
                }
                other => panic!("modern server got unexpected method: {other}"),
            }
        })
        .await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("daraz", &url, HashMap::new())
            .await
            .expect("modern-era server connects via server/discover");
        assert_listed(&plugin, "daraz__search");

        let out = call_search(&mut plugin, "daraz__search").await;
        assert_eq!(out["content"][0]["text"], "done", "{out}");

        let methods: Vec<String> = log
            .lock()
            .expect("log lock")
            .iter()
            .map(|entry| entry.split(' ').next().unwrap_or_default().to_string())
            .collect();
        assert_eq!(methods, ["server/discover", "tools/list", "tools/call"]);
    }

    #[tokio::test]
    async fn legacy_server_falls_back_to_initialize_handshake() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = serve_requests(log.clone(), move |_head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                // Pre-discover servers reject the probe as an unknown
                // method; that rejection is what triggers the fallback.
                "server/discover" => json_err(&id, -32601, "method not found"),
                "initialize" => {
                    assert_eq!(
                        request["params"]["protocolVersion"], "2025-11-25",
                        "fallback must propose the legacy revision"
                    );
                    json_ok(
                        &id,
                        json!({
                            "protocolVersion": request["params"]["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "legacy", "version": "1.0"}
                        }),
                    )
                }
                // The handshake's closing notification; answered 202 by the harness.
                "notifications/initialized" => ACCEPTED.to_string(),
                "tools/list" => json_ok(&id, json!({"tools": [tool_named("search")]})),
                other => panic!("legacy server got unexpected method: {other}"),
            }
        })
        .await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("hamrobazaar", &url, HashMap::new())
            .await
            .expect("legacy server connects via initialize fallback");
        assert_listed(&plugin, "hamrobazaar__search");

        let methods: Vec<String> = log
            .lock()
            .expect("log lock")
            .iter()
            .map(|entry| entry.split(' ').next().unwrap_or_default().to_string())
            .collect();
        assert_eq!(
            methods,
            [
                "server/discover",
                "initialize",
                "notifications/initialized",
                "tools/list"
            ]
        );
    }
}
