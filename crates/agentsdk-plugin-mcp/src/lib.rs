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
use std::collections::HashMap;
use std::sync::Arc;

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

/// The connection handle behind a remote skill's loader: a clone of the
/// client peer, which multiplexes requests over the same transport.
#[derive(Clone)]
struct McpPeerHandle {
    peer: rmcp::service::Peer<RoleClient>,
    root_uri: String,
}

#[async_trait]
impl agentsdk_plugin_skills::SkillLoader for McpPeerHandle {
    async fn load(&self, file: Option<&str>) -> Result<String, String> {
        let uri = match file {
            None => self.root_uri.clone(),
            Some(rel) => {
                // Validate the decoded form too: a percent-encoded `%2e%2e`
                // must not smuggle a traversal segment past the literal
                // check. The wire URI keeps the caller's original encoding.
                let decoded = percent_encoding::percent_decode_str(rel).decode_utf8_lossy();
                let clean = rel.trim_start_matches('/');
                let decoded_clean = decoded.trim_start_matches('/');
                if clean.is_empty()
                    || decoded_clean.is_empty()
                    || decoded_clean.split('/').any(|seg| seg == "..")
                {
                    return Err(format!("invalid skill file reference: {rel:?}"));
                }
                let root = self.root_uri.strip_suffix("/SKILL.md").ok_or_else(|| {
                    "internal error: skill uri does not end in /SKILL.md".to_owned()
                })?;
                format!("{root}/{clean}")
            }
        };
        let resource = self
            .peer
            .read_resource(ReadResourceRequestParams::new(uri.clone()))
            .await
            .map_err(|e| e.to_string())?;
        let content = resource
            .contents
            .first()
            .ok_or_else(|| format!("resource {uri:?} came back empty"))?;
        match content {
            rmcp::model::ResourceContents::TextResourceContents { text, .. } => Ok(text.clone()),
            _ => Err(format!("resource {uri:?} is not text")),
        }
    }
}

pub struct McpPlugin {
    clients: Vec<(String, RunningService<RoleClient, ClientConfig>)>,
    tools: Vec<(usize, ToolDefinition)>,
    /// Where cataloged remote skills are published. Set with
    /// [`McpPlugin::with_remote_skills`]; without it, catalogs are fetched
    /// but dropped.
    remote_skills: Option<agentsdk_plugin_skills::RemoteSkillSink>,
}

impl Default for McpPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl McpPlugin {
    /// Publish cataloged skills into this sink, where a skills plugin reads
    /// them and serves them through its Find/Load tools.
    pub fn with_remote_skills(mut self, sink: agentsdk_plugin_skills::RemoteSkillSink) -> Self {
        self.remote_skills = Some(sink);
        self
    }
}

impl McpPlugin {
    pub fn new() -> Self {
        Self {
            clients: Vec::new(),
            tools: Vec::new(),
            remote_skills: None,
        }
    }

    /// Tool names are prefixed `{server}__`, so a repeated server name would
    /// make two servers' tools collide under one prefix.
    fn ensure_name_is_new(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.clients.iter().any(|(existing, _)| existing == name) {
            return Err(format!(
                "MCP server {name:?} is already registered; server names must be unique because they prefix tool names"
            )
            .into());
        }
        Ok(())
    }

    pub async fn add_server(
        &mut self,
        name: impl Into<String>,
        command: tokio::process::Command,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = name.into();
        self.ensure_name_is_new(&name)?;
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
        self.ensure_name_is_new(&name)?;
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
        self.ensure_name_is_new(&name)?;
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
        // Follow the server's pagination — a catalog bigger than one page
        // would silently lose tools. Bounded like fetch_skills: a wedged
        // server truncates rather than spins forever.
        const MAX_TOOL_PAGES: usize = 32;
        let mut page = client.list_tools(None).await?;
        let mut tools = page.tools;
        for _ in 0..MAX_TOOL_PAGES {
            let Some(cursor) = page.next_cursor else {
                break;
            };
            page = client
                .list_tools(Some(
                    rmcp::model::PaginatedRequestParams::default().with_cursor(Some(cursor)),
                ))
                .await?;
            tools.extend(page.tools);
        }

        let client_idx = self.clients.len();
        self.clients.push((name.clone(), client));

        for tool in tools {
            let input_schema = serde_json::from_value(Value::Object(
                std::sync::Arc::try_unwrap(tool.input_schema)
                    .unwrap_or_else(|arc| (*arc).clone()),
            ))
            .unwrap_or_else(|e| {
                tracing::warn!(server = %name, tool = %tool.name, error = %e, "tool input schema failed to parse; advertising an empty schema");
                Default::default()
            });
            self.tools.push((
                client_idx,
                ToolDefinition {
                    name: format!("{name}__{}", tool.name),
                    description: tool.description.unwrap_or_default().to_string(),
                    input_schema,
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
    /// `skill://<path>/SKILL.md` resource is a skill; the skill's name is the
    /// final `/`-delimited segment of the resource's `name` field. Pagination
    /// is followed, bounded by a page cap — a catalog that outlives the cap is
    /// truncated rather than unbounded work on a wedged server.
    async fn fetch_skills(&mut self, client_idx: usize) -> Result<usize, String> {
        const MAX_PAGES: usize = 32;
        const MAX_SKILLS: usize = 1_000;
        let Some(sink) = self.remote_skills.clone() else {
            return Ok(0);
        };
        let (server, client) = self
            .clients
            .get(client_idx)
            .ok_or("internal error: client index vanished")?;
        let server = server.clone();
        let peer = client.peer().clone();
        let mut cursor: Option<String> = None;
        let mut count = 0usize;
        for _ in 0..MAX_PAGES {
            let page = peer
                .list_resources(Some(
                    rmcp::model::PaginatedRequestParams::default().with_cursor(cursor.clone()),
                ))
                .await
                .map_err(|e| e.to_string())?;
            for resource in page.resources {
                if count >= MAX_SKILLS {
                    tracing::warn!(server = %server, cap = MAX_SKILLS, "skill catalog exceeds the cap; truncating");
                    return Ok(count);
                }
                let Some((name, description)) = skill_name_and_description(&resource) else {
                    continue;
                };
                let loader = McpPeerHandle {
                    peer: peer.clone(),
                    root_uri: resource.uri.clone(),
                };
                sink.lock()
                    .map_err(|_| "skill sink poisoned".to_owned())?
                    .push(agentsdk_plugin_skills::RemoteSkill {
                        server: server.clone(),
                        name,
                        description,
                        loader: Arc::new(loader),
                    });
                count += 1;
            }
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(count)
    }
}

/// A skills resource is a `skill://` URI naming a SKILL.md, with a
/// description worth surfacing. The skill's name is the final
/// `/`-delimited segment of the resource's `name` field.
fn skill_name_and_description(resource: &rmcp::model::Resource) -> Option<(String, String)> {
    if !resource.uri.starts_with("skill://") || !resource.uri.ends_with("/SKILL.md") {
        return None;
    }
    let name = resource
        .name
        .rsplit('/')
        .next()
        .filter(|n| !n.is_empty())?
        .to_owned();
    let description = resource.description.as_deref()?.to_owned();
    Some((name, description))
}

fn transport_config(
    url: &str,
    headers: HashMap<String, String>,
) -> StreamableHttpClientTransportConfig {
    let mut config = StreamableHttpClientTransportConfig::with_uri(url);
    for (k, v) in headers {
        let (Ok(hname), Ok(hval)) = (
            http::HeaderName::from_bytes(k.as_bytes()),
            http::HeaderValue::from_str(&v),
        ) else {
            tracing::warn!(url, header = %k, "dropping invalid custom header (bad name or value)");
            continue;
        };
        config.custom_headers.insert(hname, hval);
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

    #[tokio::test]
    async fn duplicate_server_names_are_rejected() {
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
                other => panic!("duplicate-name server got unexpected method: {other}"),
            }
        })
        .await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("dup", &url, HashMap::new())
            .await
            .expect("first registration with a fresh name succeeds");
        let err = plugin
            .add_remote_server("dup", &url, HashMap::new())
            .await
            .expect_err("a second server with the same name must be rejected");
        tracing::debug!(error = %err, "expected duplicate rejection");

        let count = plugin
            .tools()
            .iter()
            .filter(|t| t.name == "dup__search")
            .count();
        assert_eq!(count, 1, "only the first server's tools remain listed");
    }

    #[tokio::test]
    async fn tools_list_pagination_is_followed() {
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
                "tools/list" => {
                    let cursor = request["params"]["cursor"].as_str().unwrap_or_default();
                    if cursor.is_empty() {
                        json_ok(
                            &id,
                            json!({
                                "resultType": "complete",
                                "tools": [tool_named("search")],
                                "nextCursor": "page-2"
                            }),
                        )
                    } else {
                        json_ok(
                            &id,
                            json!({
                                "resultType": "complete",
                                "tools": [tool_named("extra")]
                            }),
                        )
                    }
                }
                other => panic!("paginating server got unexpected method: {other}"),
            }
        })
        .await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("paged", &url, HashMap::new())
            .await
            .expect("connect");
        assert_listed(&plugin, "paged__search");
        assert_listed(&plugin, "paged__extra");
    }
}

/// Test skills support: a server that declares `io.modelcontextprotocol/skills`
/// catalogs its skills as `skill://` resources, which the client lists through
/// `resources/list` and reads through `resources/read`.
#[cfg(test)]
mod skills_tests {
    #![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

    use super::*;
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

    #[tokio::test]
    async fn modern_skills_server_is_cataloged_into_the_sink() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "modern").await;
        let sink = agentsdk_plugin_skills::remote_skill_sink();

        let mut plugin = McpPlugin::new().with_remote_skills(sink.clone());
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("skills server connects");

        // The loader fetches the SKILL.md and supporting files through the
        // same connection, resolving relative paths under the skill root.
        let loader = {
            let catalog = sink.lock().expect("sink");
            assert_eq!(catalog.len(), 1);
            let skill = &catalog[0];
            assert_eq!(skill.server, "mem");
            assert_eq!(skill.name, "deploy");
            assert_eq!(skill.description, "Ship the worker");
            skill.loader.clone()
        };
        let content = loader.load(None).await.expect("load SKILL.md");
        assert_eq!(content, SKILL_MD);
        let reference = loader
            .load(Some("references/rollout.md"))
            .await
            .expect("load reference file");
        assert_eq!(reference, ROLLOUT_MD);
        assert!(
            loader.load(Some("../escape.md")).await.is_err(),
            "traversal must be refused"
        );
        assert!(
            loader
                .load(Some("references/%2e%2e/escape.md"))
                .await
                .is_err(),
            "percent-encoded traversal must be refused"
        );

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
    async fn skill_catalog_truncates_at_the_cap() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = crate::tests::serve_requests(log.clone(), move |_head, request| {
            let id = request["id"].clone();
            match request["method"].as_str().unwrap_or_default() {
                "server/discover" => crate::tests::json_ok(
                    &id,
                    json!({
                        "resultType": "complete",
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}, "extensions": {"io.modelcontextprotocol/skills": {}}},
                        "ttlMs": 60000,
                        "cacheScope": "public"
                    }),
                ),
                "tools/list" => {
                    crate::tests::json_ok(&id, json!({"resultType": "complete", "tools": []}))
                }
                "resources/list" => {
                    let cursor = request["params"]["cursor"].as_str().unwrap_or_default();
                    let start = if cursor.is_empty() { 0 } else { 600 };
                    let resources: Vec<serde_json::Value> = (start..start + 600)
                        .map(|i| {
                            json!({
                                "uri": format!("skill://bulk{i}/SKILL.md"),
                                "name": format!("bulk{i}"),
                                "description": "bulk"
                            })
                        })
                        .collect();
                    let mut result = json!({ "resources": resources });
                    if cursor.is_empty() {
                        result["nextCursor"] = json!("page-2");
                    }
                    result["resultType"] = json!("complete");
                    crate::tests::json_ok(&id, result)
                }
                other => panic!("bulk catalog server got unexpected method: {other}"),
            }
        })
        .await;

        let sink = agentsdk_plugin_skills::remote_skill_sink();
        let mut plugin = McpPlugin::new().with_remote_skills(sink.clone());
        plugin
            .add_remote_server("bulk", &url, HashMap::new())
            .await
            .expect("connect");

        let catalog = sink.lock().expect("sink");
        assert_eq!(catalog.len(), 1_000, "catalog must truncate at the cap");
        assert!(
            catalog.iter().any(|s| s.name == "bulk999"),
            "the first thousand skills, in page order, are kept"
        );
    }

    #[tokio::test]
    async fn legacy_skills_server_declares_and_serves() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "legacy").await;
        let sink = agentsdk_plugin_skills::remote_skill_sink();

        let mut plugin = McpPlugin::new().with_remote_skills(sink.clone());
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("legacy skills server connects via initialize");

        let loader = sink.lock().expect("sink")[0].loader.clone();
        let content = loader.load(None).await.expect("load");
        assert_eq!(content, SKILL_MD);
    }

    #[tokio::test]
    async fn without_a_sink_the_catalog_is_fetched_and_dropped() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let url = skills_server(log.clone(), "modern").await;

        let mut plugin = McpPlugin::new();
        plugin
            .add_remote_server("mem", &url, HashMap::new())
            .await
            .expect("connect");

        // No tools beyond the server's own were fabricated.
        assert!(plugin.tools().iter().all(|t| t.name.starts_with("mem__")));
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
        let sink = agentsdk_plugin_skills::remote_skill_sink();

        let mut plugin = McpPlugin::new().with_remote_skills(sink.clone());
        plugin
            .add_remote_server("plain", &url, HashMap::new())
            .await
            .expect("plain server connects");

        assert!(sink.lock().expect("sink").is_empty());

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
