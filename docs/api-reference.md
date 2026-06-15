# API Reference

Everything you can call in lkhu, on one page. lkhu has one process that owns memory — the **daemon** — and three surfaces: the **CLI** (`lkhu ...`), the **MCP tools** Claude uses, and the daemon's **HTTP API**. The MCP server, hooks, and dashboard are thin clients of the daemon; the CLI's memory commands open the engine directly against the same SQLite/FAISS store. Whichever surface you pick, it is the same store and the same zero-LLM pipeline.

This is a reference, not a tutorial. If you're setting up for the first time, start with the [installation guide](installation/macos.md) ([Linux](installation/linux.md), [Windows](installation/windows.md)) or the [docs index](index.md).

```bash
pipx install git+https://github.com/iddk0321/lkhu
# once published to PyPI: pipx install lkhu
lkhu install
```

## The three surfaces at a glance

| Surface | Who calls it | Transport |
|---|---|---|
| [CLI](#cli) | You, in a terminal | Memory commands open the engine in-process; service commands (`daemon`, `dashboard`, `serve`, `hook`) use HTTP to the daemon |
| [MCP tools](#mcp-tools) | Claude Code / Claude Desktop | FastMCP over stdio (`lkhu serve`) → HTTP to the daemon |
| [Daemon HTTP API](#daemon-http-api) | Hooks, MCP, dashboard, your scripts | JSON over `http://127.0.0.1:<port>` |

The daemon auto-starts when any client needs it, so you rarely run `lkhu daemon` yourself.

## CLI

The `lkhu` command is a Typer app. Running `lkhu` with no arguments prints help.

The memory commands (`remember`, `recall`, `forget`, `status`, `export`, `import`) open the SQLite/FAISS store directly rather than calling the daemon, so avoid running them while a daemon is actively writing (see the single-owner note under [Python (embedded)](#python-embedded)).

**Global options**

| Option | Description |
|---|---|
| `--version`, `-V` | Print the version and exit. |

### Setup and health

#### `lkhu install`

```bash
lkhu install
```

The recommended entry point. Creates data directories, generates the codebook (with triple backup) if missing, and registers the **Claude Desktop** MCP server. Does not touch Claude Code — it prints the plugin commands instead (`claude plugin marketplace add iddk0321/lkhu`, then `claude plugin install lkhu@lkhu`). Also checks that Ollama and `snowflake-arctic-embed2` are available. Idempotent: an existing codebook is never regenerated.

#### `lkhu init`

```bash
lkhu init
```

Older setup path, superseded by `install`. Creates directories, generates the codebook (triple backup) if missing, registers the lkhu MCP server in the Claude **Desktop** config file (`claude_desktop_config.json` — despite the success message naming Claude Code), and checks Ollama. Prints "Please restart Claude Code."

#### `lkhu doctor`

```bash
lkhu doctor
```

Diagnoses your setup: Ollama server, codebook integrity, Claude Desktop MCP registration, Claude Code MCP registration, auto-memory hooks, daemon status, and the data directory path.

#### `lkhu uninstall`

```bash
lkhu uninstall
```

Removes the Claude Desktop MCP registration and cleans up legacy Claude Code integrations (direct MCP entry, legacy hooks). Your data and codebook are preserved. To remove the plugin side: `claude plugin uninstall lkhu@lkhu`.

#### `lkhu reset`

```bash
lkhu reset --confirm
```

Deletes **all** data: codebook, codebook backup, database, FAISS index, short-term scent, audit logs, and sidecar `*.meta.json` files. Without `--confirm` it does nothing and exits with code 1.

| Option | Default | Description |
|---|---|---|
| `--confirm` | off | Confirm deletion of all data. |

### Memory operations

#### `lkhu remember`

```bash
lkhu remember "content" [--kind KIND]
```

Store a memory explicitly from the CLI. Explicit memories start at strength 1.3 and are stored with kind `explicit`.

| Option | Default | Description |
|---|---|---|
| `--kind` | `fact` | Accepted, but currently ignored — the engine always stores explicit memories with kind `explicit`. |

#### `lkhu recall`

```bash
lkhu recall "query" [--k N]
```

Search memories from the CLI (handy for debugging). Prints the recalled text, the decoder tier used, and the source memories.

| Option | Default | Description |
|---|---|---|
| `--k` | `5` | Number of memories to retrieve. |

#### `lkhu forget`

```bash
lkhu forget "query" --confirm
```

Archive memories matching the query. Archived memories leave the search index but their audit text is preserved. Without `--confirm` nothing is archived; it only prints a hint ("Add --confirm to actually archive.").

| Option | Default | Description |
|---|---|---|
| `--confirm` | off | Confirm archiving. |

#### `lkhu status`

```bash
lkhu status
```

Prints a statistics table: total memories, archived count, kind distribution, average/max strength, codebook key count, vector dimension, and decoder tier stats.

### Data management

#### `lkhu export`

```bash
lkhu export [--out FILE]
```

Export the audit data (the natural-language shadow log) as JSONL.

| Option | Default | Description |
|---|---|---|
| `--out` | `lkhu_export.jsonl` | Output file path. |

#### `lkhu import`

```bash
lkhu import FILE
```

Import an audit JSONL exported from another machine. Each record's `audit_text` is re-observed under its original `session_id` — vectors are re-encoded locally with your codebook.

#### `lkhu backup`

```bash
lkhu backup
```

Copies `codebook.npy` and `memories.db` to `<data_dir>/backups/daily/<UTC timestamp>/`. See [storage.md](storage.md) for what lives where.

#### `lkhu reembed`

```bash
lkhu reembed --yes
```

Re-encode **all** memory vectors with the current embedding model. Vectors saved with a different model live in a different space, so recall breaks until they are re-encoded — run this after switching `encoder.model`. The text, strength, and metadata are preserved; only the scent vectors change. The codebook is not regenerated (it depends only on the 1024 dimension, not the model). Stop any running daemon/Claude Code first, since they hold the same store. Without `--yes` it only prints how many memories *would* be re-encoded and exits without changing anything.

| Option | Default | Description |
|---|---|---|
| `--yes` | off | Proceed without the confirmation prompt. |

### Services

#### `lkhu daemon`

```bash
lkhu daemon
```

Runs the resident memory daemon — the single process that owns the engine (SQLite + FAISS) and serves the HTTP API. Exits quietly if a daemon is already healthy. Normally auto-launched by clients; you only run it manually for debugging.

#### `lkhu dashboard`

```bash
lkhu dashboard [--no-open]
```

Ensures the daemon is running, prints the dashboard URL, and opens it in your browser. Exits with code 1 if the daemon fails to start.

| Option | Default | Description |
|---|---|---|
| `--no-open` | off | Do not auto-open the browser. |

#### `lkhu serve`

```bash
lkhu serve
```

Runs the MCP server (FastMCP, stdio). Invoked automatically by Claude Code and Claude Desktop — you never run it by hand. It auto-starts the daemon and proxies tool calls to it.

### Evaluation

#### `lkhu eval`

```bash
lkhu eval [--k N] [--offline] [--model NAME] [--out FILE]
```

Score recall quality, noise, multilingual reach, and the save filter against a fixed gold corpus, then print a scorecard. Runs in a throwaway data directory with a fixed codebook seed, so it is deterministic and **never touches your production memories**. Recall is reported as COLD (first session) vs WARM (after simulated real use), alongside save-filter and Hebbian-saturation metrics. On the default embedder (`snowflake-arctic-embed2`): hit@k 1.00, cross-lingual 1.00, noise_rate 0.23, Hebbian noise-saturated (gated) 0, save filter 100/100.

| Option | Default | Description |
|---|---|---|
| `--k` | `5` | Top-k used for the recall metrics. |
| `--offline` | off | Skip the embedding-dependent metrics and run only the embedding-free save-filter checks (no Ollama needed). |
| `--model` | (default embedder) | Ollama embedding model to benchmark instead of the default; its dimension is auto-detected and the eval builds its own codebook at that dimension. |
| `--out` | (none) | Write the scorecard JSON to this path. |

See [evaluation.md](evaluation.md) for the gold corpus and the full metric definitions.

### Hooks

#### `lkhu hook`

```bash
lkhu hook EVENT
```

Internal: handles a Claude Code hook event. Reads JSON from stdin, writes JSON to stdout. Events: `session-start`, `user-prompt`, `stop`. Always exits safely — any error produces a pass-through response, never blocking your work. The plugin wires these up for you; see [auto-memory.md](auto-memory.md).

#### `lkhu install-hooks`

```bash
lkhu install-hooks
```

Legacy: registers the auto recall/save hooks directly in Claude Code. The plugin supersedes this.

#### `lkhu uninstall-hooks`

```bash
lkhu uninstall-hooks
```

Removes lkhu's Claude Code hooks. Other hooks and all data are preserved.

## MCP tools

`lkhu serve` exposes six tools (server name `lkhu`). Each is a thin proxy to a daemon route.

| Tool | Signature | Returns |
|---|---|---|
| `recall` | `recall(query: str, k: int = 5) -> dict` | `{text, tier, llm_used, sources}` |
| `remember` | `remember(content: str, kind: str = "fact") -> dict` | `{"id": "<id>", "stored": true}` |
| `forget` | `forget(query: str, confirm: bool = False) -> dict` | `{archived, confirmed, ids?}` |
| `recall_session` | `recall_session(session_id: str) -> str` | Joined audit text for the session |
| `status` | `status() -> dict` | System statistics (see below) |
| `export` | `export(out_path: str) -> dict` | `{"exported": <count>, "path": <out_path>}` |

**`recall`** searches the top-K relevant memories and decodes the synthesized scent into language. Each entry in `sources` is `{id, audit_text, strength}`; `tier` tells you which [decoder tier](architecture.md) produced the text and `llm_used` is almost always `false`.

**`remember`** stores the content at strength 1.3. The `kind` argument is accepted but currently ignored — explicit memories are always stored with kind `explicit`.

**`forget`** archives matches; the audit text is preserved. With `confirm=false` it returns `{"archived": 0, "confirmed": false}` and changes nothing.

**`status`** returns `{total_memories, archived, kinds, strength_avg, strength_max, codebook_keys, dim, decoder, short_term_norm}`.

## Daemon HTTP API

The daemon is a stdlib `ThreadingHTTPServer` speaking JSON. The hooks, the MCP server, and the dashboard all use these routes — and so can your scripts.

**Base URL**

- Host: `LKHU_DAEMON_HOST` env var, default `127.0.0.1` (loopback only).
- Port: `LKHU_DAEMON_PORT` env var if set; otherwise `37700 + (uid % 100)` — e.g. `37701` for the first macOS user (uid 501); on Windows (no POSIX uid) it falls back to the base `37700`. The canonical example port used throughout the docs is `37701`.

All responses are `application/json; charset=utf-8`, except the dashboard HTML. Unknown paths return `404 {"error": "not found"}`; handler exceptions return `500 {"error": "<message>"}`.

### GET routes

| Route | Returns |
|---|---|
| `GET /health` | `{"ok": true, "memories": <count>}` |
| `GET /status` | Status dict (same shape as the MCP `status` tool) |
| `GET /` and `GET /dashboard` | The web dashboard (HTML) |
| `GET /api/stats` | Dashboard stats dict, plus `"data_dir"` |
| `GET /api/memories?archived=0\|1` | `{"memories": [...]}` — `archived=1` includes archived rows |

### POST routes

All POST routes take a JSON body.

| Route | Body (defaults) | Returns |
|---|---|---|
| `POST /recall` | `{query, k: 5}` | `{text, tier, llm_used, sources}` |
| `POST /remember` | `{content, kind: "fact", session_id: ""}` | `{"id": "<id>"}` |
| `POST /observe` | `{content, session_id: "", strength: null}` | `{"id": "<id>"}` |
| `POST /recent` | `{n: 10}` | `{"memories": [{id, audit_text, strength, kind, created_at}, ...]}` |
| `POST /forget` | `{query, confirm: false}` | `{archived, confirmed, ids?}` |
| `POST /recall_session` | `{session_id}` | `{"text": "<joined audit text>"}` |
| `POST /export` | `{out_path}` | `{"exported": <count>, "path": <out_path>}` |

`remember` stores an explicit memory (kind `explicit`, strength 1.3 — the `kind` field is accepted but currently ignored); `observe` stores a conversation turn at the default turn strength (1.0) — it's what the hooks use. Fields without a listed default are required.

Example:

```bash
curl -s -X POST http://127.0.0.1:37701/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "which database did we pick?", "k": 3}'
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LKHU_DATA` | platformdirs default | Override the data root directory. |
| `LKHU_CLAUDE_CONFIG` | OS-specific | Override the Claude Desktop config file path. |
| `LKHU_CC_SETTINGS` | `~/.claude/settings.json` | Override the Claude Code settings file edited by `install-hooks`/`uninstall-hooks`. |
| `LKHU_DAEMON_HOST` | `127.0.0.1` | Daemon bind host. |
| `LKHU_DAEMON_PORT` | `37700 + (uid % 100)` | Daemon port. |
| `LKHU_DAEMON_LOG` | discarded | File to append daemon stdout/stderr when auto-launched. |

Default data locations are listed in [storage.md](storage.md).

## Python (embedded)

You can use the engine directly as a library — useful for scripts and tests. Note that the daemon expects to be the **sole owner** of the SQLite/FAISS store, so don't open an engine while the daemon is running against the same data directory.

```python
from lkhu.core.engine import LkhuEngine, initialize
from lkhu.platform.ollama import OllamaEmbedder

initialize(register_mcp=False)                 # once, the first time (skips Claude config edits)
engine = LkhuEngine.open(embedder=OllamaEmbedder())
engine.remember("Primary language is Python", kind="fact")
print(engine.recall("which language?", k=3)["text"])
engine.close()
```

For offline use and tests, swap in `from lkhu.core.encoder import HashingEmbedder` as the embedder.

## See also

- [Architecture](architecture.md) — how encode/recall/decode and the daemon fit together
- [Auto-memory](auto-memory.md) — what the hooks save and inject, and the noise filters
- [Storage](storage.md) — file layout, backups, and the audit log
- [FAQ](faq.md)
