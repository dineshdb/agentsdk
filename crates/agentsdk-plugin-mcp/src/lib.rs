use agentsdk::core::plugin::{AgentPlugin, PluginContext, PluginToolCall};
use agentsdk::core::tools::ToolDefinition;
use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use rmcp::model::ClientCapabilities;
use rmcp::model::ClientConfig;
use rmcp::model::ExtensionCapabilities;
use rmcp::model::Implementation;
use rmcp::model::JsonObject;
use rmcp::model::ProtocolVersion;
use rmcp::model::ReadResourceRequestParams;
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
use serde_json::json;
use std::collections::HashMap;

/// SEP-2640: skills served as resources under `skill://`, cataloged by
/// `skills/list` and addressed by entry through `skills/get`. A server that
/// declares this extension is negotiated for skills support on connect.
const SKILLS_EXT: &str = "io.modelcontextprotocol/skills";

/// The client declares the extension in its capabilities so a server may
/// rely on it (Tachyon-style servers gate the face on client advertisement).
fn client_config() -> ClientConfig {
    let mut extensions = ExtensionCapabilities::new();
    extensions.insert(SKILLS_EXT.to_string(), JsonObject::default());
    let mut capabilities = ClientCapabilities::default();
    capabilities.extensions = Some(extensions);
    let mut implementation = Implementation::default();
    implementation.name = "pie-mcp".to_owned();
    ClientConfig::new(capabilities, implementation)
}

/// One skill from a server's catalog.
///
/// v1 catalogs through the plain Resources primitive (`resources/list` +
/// `resources/read`) — every skill is an ordinary `skill://` resource, which
/// is all the transport needs. When rmcp learns to decode the extension's
/// `skills/list` entries, the digest manifest they carry slots in here and
/// loads become verifiable.
#[derive(Debug, Clone)]
pub struct McpSkill {
    /// The server this skill came from — half of the (origin, uri) identity
    /// the spec requires hosts to preserve.
    pub server: String,
    pub uri: String,
    pub name: String,
    pub description: String,
}

pub struct McpPlugin {
    clients: Vec<(String, RunningService<RoleClient, ClientConfig>)>,
    tools: Vec<(usize, ToolDefinition)>,
    /// Skills catalogs from every server that declared the extension.
    skills: Vec<McpSkill>,
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
            skills: Vec::new(),
        }
    }

    pub async fn add_server(
        &mut self,
        name: impl Into<String>,
        command: tokio::process::Command,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = name.into();
        let client = client_config()
            .serve(TokioChildProcess::new(command)?)
            .await?;
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
    ) -> Result<RunningService<RoleClient, ClientConfig>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        match client_config()
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
                client_config()
                    .serve_with_lifecycle(make_transport(), ClientLifecycleMode::Initialize)
                    .await
                    .map_err(Into::into)
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn register_client(
        &mut self,
        name: String,
        client: rmcp::service::RunningService<rmcp::service::RoleClient, ClientConfig>,
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

        // A server that declared the skills extension gets asked for its
        // catalog. A failed fetch costs the catalog, never the connection:
        // skills are additive, and tools already work.
        if let Some(peer_info) = self.clients[client_idx].1.peer_info()
            && peer_info
                .capabilities
                .extensions
                .as_ref()
                .is_some_and(|ext| ext.contains_key(SKILLS_EXT))
            {
                match self.fetch_skills(client_idx).await {
                    Ok(count) if count > 0 => {
                        tracing::debug!(server = %name, count, "skills catalog loaded");
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(server = %name, error = %e, "skill catalog fetch failed; no catalog")
                    }
                }
            }

        Ok(())
    }

    /// Pull one server's skill catalog from `resources/list`: every
    /// `skill://<path>/SKILL.md` resource is a skill, named by its final
    /// path segment. Pagination is followed, bounded by a page cap — a
    /// catalog that outlives the cap is truncated rather than unbounded
    /// work on a wedged server.
    async fn fetch_skills(&mut self, client_idx: usize) -> Result<usize, String> {
        const MAX_PAGES: usize = 32;
        let (server, client) = self
            .clients
            .get(client_idx)
            .ok_or("internal error: client index vanished")?;
        let server = server.clone();
        let mut cursor: Option<String> = None;
        let mut count = 0usize;
        for _ in 0..MAX_PAGES {
            let page = client
                .peer()
                .list_resources(Some(
                    rmcp::model::PaginatedRequestParams::default().with_cursor(cursor.clone()),
                ))
                .await
                .map_err(|e| e.to_string())?;
            for resource in page.resources {
                if let Some(skill) = skill_from_resource(&server, &resource) {
                    self.skills.push(skill);
                    count += 1;
                }
            }
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(count)
    }

    /// Resolve a skill by name. `server/name` disambiguates a collision;
    /// a bare name matching exactly one catalog entry is fine.
    fn find_skill(&self, reference: &str) -> Result<&McpSkill, String> {
        let (want_server, want_name) = match reference.split_once('/') {
            Some((s, n)) => (Some(s), n),
            None => (None, reference),
        };
        let matches: Vec<&McpSkill> = self
            .skills
            .iter()
            .filter(|skill| {
                skill.name == want_name && want_server.is_none_or(|srv| srv == skill.server)
            })
            .collect();
        match matches.as_slice() {
            [skill] => Ok(skill),
            [] => Err(format!(
                "no MCP skill named {reference:?}; use find_mcp_skills to list the catalog"
            )),
            many => Err(format!(
                "{} servers serve a skill named {reference:?}: {}; qualify it as server/name",
                many.len(),
                many.iter().map(|s| s.server.as_str()).collect::<Vec<_>>().join(", ")
            )),
        }
    }

    /// Read one file of a skill through `resources/read`. A supporting file
    /// is resolved relative to the skill's root; `..` segments are refused
    /// before anything reaches the wire.
    async fn read_verified(
        &self,
        skill: &McpSkill,
        file: Option<&str>,
    ) -> Result<(String, String), String> {
        let uri = match file {
            None => skill.uri.clone(),
            Some(rel) => {
                let root = skill.uri.strip_suffix("/SKILL.md").ok_or_else(|| {
                    "internal error: skill uri does not end in /SKILL.md".to_owned()
                })?;
                let clean = rel.trim_start_matches('/');
                if clean.is_empty() || clean.split('/').any(|seg| seg == "..") {
                    return Err(format!("invalid skill file reference: {rel:?}"));
                }
                format!("{root}/{clean}")
            }
        };

        let (_, client) = self
            .clients
            .iter()
            .find(|(name, _)| *name == skill.server)
            .ok_or_else(|| format!("internal error: server {:?} not connected", skill.server))?;
        let resource = client
            .peer()
            .read_resource(ReadResourceRequestParams::new(uri.clone()))
            .await
            .map_err(|e| e.to_string())?;
        let content = resource
            .contents
            .first()
            .ok_or_else(|| format!("resource {uri:?} came back empty"))?;
        let text = match content {
            rmcp::model::ResourceContents::TextResourceContents { text, .. } => text.clone(),
            _ => return Err(format!("resource {uri:?} is not text")),
        };
        Ok((uri, text))
    }
}

/// One skill resource from a `resources/list` page: a `skill://` URI naming
/// a SKILL.md, with a description worth surfacing. The skill's name is its
/// final path segment, per the extension's URI constraint.
fn skill_from_resource(server: &str, resource: &rmcp::model::Resource) -> Option<McpSkill> {
    let uri = &resource.uri;
    if !uri.starts_with("skill://") || !uri.ends_with("/SKILL.md") {
        return None;
    }
    let name = resource
        .name
        .rsplit('/')
        .next()
        .filter(|n| !n.is_empty())?
        .to_owned();
    let description = resource.description.as_deref()?.to_owned();
    Some(McpSkill {
        server: server.to_owned(),
        uri: uri.clone(),
        name,
        description,
    })
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
        let mut tools: Vec<ToolDefinition> = self.tools.iter().map(|(_, t)| t.clone()).collect();
        if !self.skills.is_empty() {
            tools.push(ToolDefinition {
                name: "find_mcp_skills".to_owned(),
                description: "List the skills served by connected MCP servers. Returns each \
                              skill's name, description and origin server; load one with \
                              load_mcp_skill before relying on its instructions."
                    .to_owned(),
                input_schema: serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .unwrap_or_default(),
            });
            tools.push(ToolDefinition {
                name: "load_mcp_skill".to_owned(),
                description: "Load one skill (or one of its files) from an MCP server. The \
                              content is verified against the catalog's digest before it is \
                              returned. Reference it as server/name when two servers serve \
                              the same name."
                    .to_owned(),
                input_schema: serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name, or server/name"},
                        "file": {"type": "string", "description": "Optional file inside the skill, relative to its root"}
                    },
                    "required": ["name"]
                }))
                .unwrap_or_default(),
            });
        }
        tools
    }

    async fn run_tool(
        &mut self,
        _ctx: &mut PluginContext,
        call: &PluginToolCall,
    ) -> Result<Value, String> {
        if call.name == "find_mcp_skills" {
            let skills: Vec<&McpSkill> = self.skills.iter().collect();
            return serde_json::to_value(skills.iter().map(|s| json!({
                "server": s.server,
                "name": s.name,
                "description": s.description,
            })).collect::<Vec<_>>())
            .map_err(|e| e.to_string());
        }
        if call.name == "load_mcp_skill" {
            let reference = call
                .arguments
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("load_mcp_skill needs a skill name")?;
            let skill = {
                let found = self.find_skill(reference)?;
                found.clone()
            };
            let (uri, text) = self.read_verified(&skill, call.arguments.get("file").and_then(|v| v.as_str())).await?;
            return serde_json::to_value(json!({
                "server": skill.server,
                "name": skill.name,
                "uri": uri,
                "content": text,
            }))
            .map_err(|e| e.to_string());
        }

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
    use rmcp::transport::CredentialStore;
    use rmcp::transport::auth::InMemoryCredentialStore;
    use rmcp::transport::auth::StoredCredentials;
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

    pub(crate) const ACCEPTED: &str =
        "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";

    /// Serve one JSON-RPC-over-HTTP request per connection, forever.
    ///
    /// `respond` gets the request head lines and JSON body and returns the raw
    /// HTTP response bytes. Requests are appended to `log` before responding.
    pub(crate) async fn serve_requests(
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

    pub(crate) fn json_ok(id: &serde_json::Value, result: serde_json::Value) -> String {
        http_json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    pub(crate) fn json_err(id: &serde_json::Value, code: i64, message: &str) -> String {
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
    async fn authorized_transport_sends_the_stored_bearer_token() {
        // Credentials a completed authorization leaves in the manager's
        // store must ride every request as `Authorization: Bearer <token>`.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = serve_requests(log.clone(), move |head, request| {
            let headers = lowercase_headers(&head);
            assert!(
                headers.contains("authorization: bearer pie-test-token"),
                "authorized requests must carry the stored bearer token, got: {head:?}"
            );
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

        let mut manager = AuthorizationManager::new(&url).await.expect("auth manager");
        // Seed what a finished authorization would have stored. Building
        // the credentials through serde keeps the test free of a direct
        // oauth2 dependency; with no expiry info the manager returns the
        // token as-is instead of refreshing.
        let credentials: StoredCredentials = serde_json::from_value(json!({
            "client_id": "pie-test-client",
            "token_response": {"access_token": "pie-test-token", "token_type": "bearer"}
        }))
        .expect("stored credentials");
        let store = InMemoryCredentialStore::new();
        store.save(credentials).await.expect("seed credentials");
        manager.set_credential_store(store);

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server_authorized("linear", &url, HashMap::new(), manager)
            .await
            .expect("authorized transport connects with stored credentials");
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

/// Test skills support: a server that declares `io.modelcontextprotocol/skills`
/// catalogs its skills as `skill://` resources, which the client lists through
/// `resources/list` and reads through `resources/read`.
#[cfg(test)]
mod skills_tests {
    #![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use agentsdk::core::plugin::PluginToolCall;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;

    type Log = Arc<Mutex<Vec<String>>>;

    const SKILL_MD: &str = "---\nname: deploy\ndescription: Ship the worker\n---\n\n# Deploy\n\ncell deploy . --fleet remote\n";
    const ROLLOUT_MD: &str = "# Rollout\n\n1. healthz\n2. smoke\n";

    /// A skills-capable server: its capabilities declare the extension and
    /// its resources/list enumerates the SKILL.md resources.
    async fn skills_server(log: Log, era: &'static str) -> String {
        crate::tests::serve_requests(log, move |_head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                "server/discover" if era == "modern" => crate::tests::json_ok(
                    &id,
                    json!({
                        "resultType": "complete",
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}, "extensions": {"io.modelcontextprotocol/skills": {}}},
                        "ttlMs": 60000,
                        "cacheScope": "public"
                    }),
                ),
                "server/discover" => crate::tests::json_err(&id, -32601, "method not found"),
                "initialize" => {
                    let ext = &request["params"]["capabilities"]["extensions"];
                    assert!(
                        ext.get("io.modelcontextprotocol/skills").is_some(),
                        "client must declare the skills extension, got: {ext}"
                    );
                    crate::tests::json_ok(
                        &id,
                        json!({
                            "protocolVersion": "2025-11-25",
                            "capabilities": {"tools": {}, "extensions": {"io.modelcontextprotocol/skills": {}}},
                            "serverInfo": {"name": "mem", "version": "0.2.0"}
                        }),
                    )
                }
                "notifications/initialized" => crate::tests::ACCEPTED.to_string(),
                "tools/list" => crate::tests::json_ok(&id, json!({"tools": []})),
                "resources/list" => {
                    if era == "modern" {
                        let caps = &request["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"];
                        assert!(
                            caps.get("extensions").is_some_and(|e| e.get("io.modelcontextprotocol/skills").is_some()),
                            "modern-era requests must carry the client's declared extensions, got: {caps}"
                        );
                    }
                    let mut result = json!({
                        "resources": [
                            {
                                "uri": "skill://deploy/SKILL.md",
                                "name": "deploy",
                                "description": "Ship the worker",
                                "mimeType": "text/markdown"
                            }
                        ]
                    });
                    if era == "modern" {
                        result["resultType"] = json!("complete");
                    }
                    crate::tests::json_ok(&id, result)
                }
                "resources/read" => {
                    let uri = request["params"]["uri"].as_str().unwrap_or_default();
                    let text = match uri {
                        "skill://deploy/SKILL.md" => SKILL_MD,
                        "skill://deploy/references/rollout.md" => ROLLOUT_MD,
                        other => panic!("resources/read for a uri no skill serves: {other}"),
                    };
                    let mut result = json!({"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]});
                    if era == "modern" {
                        result["resultType"] = json!("complete");
                    }
                    crate::tests::json_ok(&id, result)
                }
                other => panic!("skills server got unexpected method: {other}"),
            }
        })
        .await
    }

    async fn run_plugin_tool(
        plugin: &mut McpPlugin,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let mut world = agentsdk::hecs::World::new();
        let entity = world.spawn(());
        let mut ctx = agentsdk::core::plugin::PluginContext::new(world, entity);
        plugin
            .run_tool(
                &mut ctx,
                &PluginToolCall {
                    id: "call-1".into(),
                    name: name.into(),
                    arguments: args,
                },
            )
            .await
    }

    #[tokio::test]
    async fn modern_skills_server_is_cataloged_and_loads() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "modern").await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("skills server connects");

        let catalog = run_plugin_tool(&mut plugin, "find_mcp_skills", json!({}))
            .await
            .expect("find_mcp_skills");
        assert_eq!(
            catalog,
            json!([{"server": "mem", "name": "deploy", "description": "Ship the worker"}])
        );

        let loaded = run_plugin_tool(
            &mut plugin,
            "load_mcp_skill",
            json!({"name": "deploy"}),
        )
        .await
        .expect("load_mcp_skill");
        assert_eq!(loaded["content"], SKILL_MD);
        assert_eq!(loaded["uri"], "skill://deploy/SKILL.md");

        // A supporting file resolves relative to the skill's root.
        let reference = run_plugin_tool(
            &mut plugin,
            "load_mcp_skill",
            json!({"name": "deploy", "file": "references/rollout.md"}),
        )
        .await
        .expect("load reference file");
        assert_eq!(reference["content"], ROLLOUT_MD);

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
                "tools/list",
                "resources/list",
                "resources/read",
                "resources/read"
            ]
        );
    }

    #[tokio::test]
    async fn legacy_skills_server_declares_and_serves() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "legacy").await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("legacy skills server connects via initialize");

        let loaded = run_plugin_tool(&mut plugin, "load_mcp_skill", json!({"name": "deploy"}))
            .await
            .expect("load after legacy handshake");
        assert_eq!(loaded["content"], SKILL_MD);
    }

    #[tokio::test]
    async fn path_traversal_in_a_file_reference_is_refused_without_a_read() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "modern").await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("connect");

        let err = run_plugin_tool(
            &mut plugin,
            "load_mcp_skill",
            json!({"name": "deploy", "file": "../../etc/passwd"}),
        )
        .await
        .expect_err("path traversal in a file reference must be refused");
        assert!(err.contains("invalid skill file reference"), "{err}");

        let methods: Vec<String> = log
            .lock()
            .expect("log lock")
            .iter()
            .map(|entry| entry.split(' ').next().unwrap_or_default().to_string())
            .collect();
        assert_eq!(
            methods,
            ["server/discover", "tools/list", "resources/list"],
            "refused loads must not have reached the wire"
        );
    }

    #[tokio::test]
    async fn a_server_without_the_extension_is_never_asked_for_resources() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = crate::tests::serve_requests(log.clone(), move |_head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                "server/discover" => crate::tests::json_ok(
                    &id,
                    json!({
                        "resultType": "complete",
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}},
                        "ttlMs": 60000,
                        "cacheScope": "public"
                    }),
                ),
                "tools/list" => crate::tests::json_ok(&id, json!({"tools": []})),
                other => panic!("plain server got unexpected method: {other}"),
            }
        })
        .await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("plain", &url, HashMap::new())
            .await
            .expect("plain server connects");

        let catalog = run_plugin_tool(&mut plugin, "find_mcp_skills", json!({}))
            .await
            .expect("find tool exists even with an empty catalog");
        assert_eq!(catalog, json!([]));

        let methods: Vec<String> = log
            .lock()
            .expect("log lock")
            .iter()
            .map(|entry| entry.split(' ').next().unwrap_or_default().to_string())
            .collect();
        assert_eq!(
            methods,
            ["server/discover", "tools/list"],
            "a server that did not declare the extension must not be asked for resources"
        );
    }
}
