# RMCP legacy wire-era pin (board task QS-102, conservative half)

rmcp 3.5.0 (2026-09-28, SEP-2567) moved `ProtocolVersion::LATEST` from
`V_2025_11_25` to `V_2026_07_28` — the discover-only era with **no initialize
handshake**. Any consumer that referenced `LATEST` (or took rmcp's default)
flips its wire era on a routine `cargo update`. This task makes the legacy era
explicit and update-proof in every live consumer. It deliberately does **not**
decide the stay-legacy-vs-modern question — that is the owner's (see the end).

Background: memory concepts `project/citadel/mcp-worker-rmcp-legacy-ceiling`
and `inbox/mem/rmcp-moved-3-1-to-3-5-touching-initialize-and-discover`
(verified 2026-10-02). Ladder in rmcp ≥ 3.5: `NO_INITIALIZE = V_2026_07_28`,
`LATEST_WITH_INITIALIZE = V_2025_11_25` (the explicit legacy-era pin),
`LATEST = V_2026_07_28`.

## Per-consumer state and changes

All work done on fresh worktrees, branch `rmcp-legacy-pin` in each repo; main
checkouts untouched.

| Consumer | Location | rmcp locked | State | Change |
|---|---|---|---|---|
| agentsdk MCP plugin | `~/src/pie/agentsdk` (worktree `agentsdk-rmcp-pin`) | 3.4.0 | live, dual-era client | manifest pin `>=3.4, <3.5` + rationale comment |
| mem MCP server | `~/src/pie/mem` (worktree `mem-rmcp-pin`) | 3.2.0 | live, legacy server | manifest pin `>=3.2, <3.3` + rationale comment; e2e regression test |
| Citadel mcp-worker | `~/src/qretaio/Citadel` | — | **deleted at HEAD** | none possible |
| daraz MCP server | memory says `~/src/mcp/daraz` | — | **not on this machine** | skipped |
| pie-core (adjacent) | `~/src/pie/pie/crates/pie-core` | 3.4.0 | auth-only rmcp use | none needed (see below) |
| `repo` CLI (adjacent) | `~/src/repo` | 3.2.0 | stdio MCP server | none (outside QS-102's consumer list; no version references) |

### agentsdk (this repo) — `agentsdk-plugin-mcp`

The era logic was already deliberate before QS-102: `negotiate()` names both
revisions outright (`preferred_versions: [V_2026_07_28]`,
`legacy_version: Some(V_2025_11_25)`), and its doc comment explains why
`LATEST`/`STANDARD_HEADERS` must never be used there — they track the SDK, not
the negotiation. Literal constants are immune to the LATEST move, so no code
change was needed. Regression coverage already exists:
`legacy_server_falls_back_to_initialize_handshake` (asserts the fallback
speaks `2025-11-25`) and `modern_server_negotiates_2026_07_28_and_calls_tools`.

Change made: `crates/agentsdk-plugin-mcp/Cargo.toml` pins
`rmcp = ">=3.4, <3.5"` with a comment naming QS-102 and the wire-era
rationale. Without the pin, a `cargo update` to 3.5 leaves the literals
correct but changes what `ClientLifecycleMode::Initialize` (the legacy
fallback path) resolves to — rmcp 3.5's `Initialize` mode can no longer mean
"LATEST", and pinning keeps that ambiguity out of the graph until the owner
bumps deliberately.

### mem — `~/src/pie/mem`

Pure rmcp server (stateless streamable HTTP, `rmcp 3.2`, no explicit protocol
references — the server answers whatever revision the client initializes at).
No `LATEST` to replace server-side.

Changes made (worktree `~/src/pie/mem-rmcp-pin`):
- `Cargo.toml`: `rmcp = ">=3.2, <3.3"` with the QS-102 rationale comment.
- `tests/e2e.rs`: `mcp_http_initializes_at_legacy_revision_2025_11_25` —
  spawns the real `mem mcp` HTTP server on an ephemeral port, initializes at
  `2025-11-25`, and asserts the server answers at that same revision. If a
  future bump flips negotiation away from the handshake era, this fails before
  any client does.

### Citadel mcp-worker — nothing left to pin

The Rust sidecar no longer exists in the repo. `~/src/qretaio/Citadel` at HEAD
(`main`, no Cargo.toml) is the celld Worker app only; commit `32433fa`
(2026-09-07, "feat(chat): the chat runs in the isolate again — sidecar
deleted") removed the `agent-worker/` successor of `mcp-worker/`. The memory
gotcha describing the 3.2 sidecar running as a LaunchAgent is stale on this
point — if a stale LaunchAgent binary still runs on 127.0.0.1:8292 from
outside any checkout, it is orphaned runtime state, not repo state. Nothing to
pin; the Citadel row of the legacy-vs-modern decision is moot until/unless a
Rust sidecar returns.

### daraz — repo absent from this machine

`~/src/mcp` does not exist, and a `find` over `~/src` and `~/Projects` finds
nothing daraz-shaped. Memory records it as its own git repo with **no remote**
(`project/daraz/mcp-server`), so if it is not here it is not reachable from
here. Skipped; the same one-line pin (`rmcp = ">=3.1, <3.2"` or its current
locked minor, with the QS-102 comment) applies whenever the checkout
reappears.

### Adjacent, examined and left alone

- **pie-core** (`~/src/pie/pie/crates/pie-core`): uses rmcp only for
  `transport::auth` (OAuth login + credential store). Its MCP sessions come
  from `agentsdk-plugin-mcp` (git dep on this repo), so its era follows this
  repo's, not its own manifest.
- **`repo` CLI** (`~/src/repo`): rmcp 3.2 stdio MCP server, no
  `ProtocolVersion`/`LATEST`/`negotiate` references, and not on QS-102's
  consumer list. Noted here for completeness only.

## The remaining decision (owner's call)

**Stay legacy (current state, zero further work).** Wire speaks
`2025-11-25`; initialize handshake on every connect. Cost: vigilance — every
rmcp bump now requires consciously re-pinning (the manifests say why, and
mem's e2e test fires if negotiation ever drifts). Legacy clients fail loudly
against modern-only servers (`400`, `-32022`); dual-era servers (mem included)
keep working.

**Modern-era client work (the alternative).** What "modern" actually means
mechanically: no initialize handshake (SEP-2567 — per-request `_meta` carries
identity/capabilities instead), an `MCP-Protocol-Version: 2026-07-28` header
on every request, and the discover lifecycle replacing connect-time
negotiation. Cost sketch:
- *agentsdk*: effectively done — `ClientLifecycleMode::Auto` already probes
  discover and falls back; lifting `preferred_versions` to stay on
  `2026-07-28` post-bump is a one-line change plus the existing tests. Un-pin
  `>=3.4, <3.5` deliberately.
- *mem server*: bump rmcp, adopt `ServerHandler::negotiate_initialize`
  (#1247, added 3.3.0) to serve both eras, keep the legacy e2e test and add a
  modern-era one (header validation now rejects duplicate SEP-2243 headers,
  #1274; empty `tool_router` is a new compile-time failure, #1233). Roughly a
  focused day including wire tests.
- *Citadel / daraz*: moot until the code exists again.

The decision is not "legacy vs modern" so much as "which servers must stay
reachable by legacy-only clients" — modern client work is additive; dropping
legacy support is the irreversible part.
