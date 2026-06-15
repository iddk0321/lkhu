# Installing lkhu on macOS

Claude Code forgets everything the moment a session ends. **lkhu** fixes that with long-term memory that runs entirely on your Mac — memories are stored as 1024-dimensional latent vectors ("scents"), embeddings are computed locally by Ollama, and nothing ever leaves your machine. No API key, no API cost.

This guide takes you from a bare Mac to working memory. If you're on another OS, see [Linux](linux.md) or [Windows](windows.md).

## Quickstart

If you already have [Homebrew](https://brew.sh) and the Claude Code CLI, this is the whole install:

```bash
# 1. Tools
brew install pipx && pipx ensurepath
brew install --cask ollama-app

# 2. Embedding model (one-time download)
ollama pull snowflake-arctic-embed2

# 3. lkhu itself
pipx install lkhu

# 4. Initialize (codebook + data dirs + Claude Desktop MCP)
lkhu install

# 5. Claude Code plugin (hooks + MCP)
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu

# 6. Verify
lkhu doctor
```

Restart Claude Code and you're done. The rest of this page walks through each step in detail.

## Prerequisites

| Requirement | Why | How to get it |
|---|---|---|
| macOS (Apple Silicon or Intel) | Supported platform | — |
| Homebrew | Easiest way to install the tools below | [brew.sh](https://brew.sh) |
| Python 3.11+ | lkhu requires `>=3.11` | `brew install python@3.11` (or newer) |
| pipx | Installs lkhu in an isolated environment | `brew install pipx && pipx ensurepath` |
| Ollama | Runs the `snowflake-arctic-embed2` embedding model locally | Step 1 below |
| Claude Code CLI | The thing getting the memory | [claude.com/claude-code](https://claude.com/claude-code) |

After `pipx ensurepath`, **open a new terminal** so the PATH change takes effect.

## Step 1 — Install Ollama

lkhu uses [Ollama](https://ollama.com) to run the `snowflake-arctic-embed2` embedding model on your machine. This is a mandatory dependency — it's how text becomes vectors without any cloud API.

```bash
brew install --cask ollama-app
```

Or download the official app from [ollama.com/download](https://ollama.com/download). Either way, launch the Ollama app once so its server is running (you'll see the llama icon in your menu bar).

> **Note:** prefer the desktop app (cask) or the official installer over Homebrew's plain `ollama` formula. The CLI-only formula has been observed to lack the inference runner, in which case embeddings won't work.

Then pull the embedding model — this is a one-time download:

```bash
ollama pull snowflake-arctic-embed2
```

`snowflake-arctic-embed2` is multilingual by design, so your memories work across languages with no extra configuration. For example, you can save a memory in Korean (e.g. "이 프로젝트는 FastAPI를 사용한다") and recall it later with an English query.

> **Stick with `snowflake-arctic-embed2`.** It's the default embedder and what lkhu's eval harness is tuned against. You *can* point `encoder.model` in your config at another 1024-dim model (the previous default, `bge-m3`, still works), but the codebook is built for 1024 dimensions and stored vectors are model-specific — a different model encodes into a different vector space. If you switch models after storing memories, run `lkhu reembed` to re-encode every stored vector with the new model; the codebook, text, strengths, and metadata are preserved.

## Step 2 — Install lkhu

```bash
pipx install lkhu
```

If your default Python is older than 3.11, point pipx at a newer one:

```bash
pipx install --python python3.11 lkhu
```

Check that it landed:

```bash
lkhu --version   # should print 0.1.0
```

If you get `command not found`, jump to [Troubleshooting](#troubleshooting).

## Step 3 — Initialize

```bash
lkhu install
```

This does three things:

- **Creates your codebook** — the key-scent dictionary all memories are encoded against — with a triple backup. If a codebook already exists, it's preserved untouched.
- **Creates the data directories** under `~/Library/Application Support/lkhu`.
- **Registers the MCP server with Claude Desktop** (if you use the Desktop app) by writing to `~/Library/Application Support/Claude/claude_desktop_config.json`.

It also checks that Ollama and `snowflake-arctic-embed2` are reachable, and prints the plugin instructions for the next step. Note that `lkhu install` deliberately does **not** touch Claude Code — that integration goes through the plugin, because directly edited Claude Code config files tend to get clobbered.

> **The codebook is sacrosanct.** `codebook.npy` is generated once and never regenerated — losing it invalidates every memory you've stored. That's why lkhu keeps three copies automatically.

## Step 4 — Install the Claude Code plugin

The plugin wires lkhu into Claude Code: lifecycle hooks for automatic recall/save, plus the MCP server for explicit `remember`/`recall` tools.

```bash
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Then **restart Claude Code**. The plugin needs the `lkhu` command on your PATH (which pipx handled in Step 2) — its hooks and MCP server both invoke the bare `lkhu` command.

What the plugin actually does:

| Hook | When | Effect |
|---|---|---|
| `SessionStart` | Session startup / resume / clear / compact | Injects your most relevant recent memories |
| `UserPromptSubmit` | Every prompt | Recalls related memories, saves the prompt |
| `Stop` | Claude finishes responding | Saves the assistant's final answer |

All of this is pure vector math — **zero LLM calls** in the memory pipeline. See [auto-memory](../auto-memory.md) for the details, including the `<private>` tag for keeping things out of memory.

## Step 5 — Verify with `lkhu doctor`

```bash
lkhu doctor
```

The doctor checks, in order: the Ollama server, codebook integrity, Claude Desktop MCP registration, Claude Code MCP registration, auto-memory hooks, whether the daemon is running, and where your data directory is. Everything green means you're set.

Give it a quick end-to-end test from the CLI:

```bash
lkhu remember "lkhu installed on my MacBook" --kind fact
lkhu recall "what did I install" --k 5
lkhu status
```

And take a look at your memory dashboard:

```bash
lkhu dashboard
```

This starts the background daemon if needed, prints the dashboard URL (served on `127.0.0.1`, port derived from your user id), and opens it in your browser. You'll see strength/age distributions, the consolidation lifecycle, and a sortable table of everything lkhu remembers.

## Where your data lives

Everything is local, under `~/Library/Application Support/lkhu/`:

| Path | What it is |
|---|---|
| `codebook.npy` | The key-scent dictionary — never regenerated |
| `codebook.backup.npy` | Second codebook copy |
| `memories.db` | SQLite — memory metadata, strengths, audit text |
| `vectors.faiss` | FAISS vector index |
| `short_term.npy` | Short-term memory accumulator |
| `audit/` | Append-only natural-language shadow log (JSONL, split by month) |
| `backups/` | `daily/`, `weekly/`, `monthly/` snapshots |
| `logs/`, `stats.json` | Daemon logs and stats |
| `config.yaml` | Optional config overrides |

The third codebook backup lives at `~/Documents/lkhu_codebook.backup.npy`, deliberately outside the app folder.

Want the data somewhere else? Set the `LKHU_DATA` environment variable before running any lkhu command. More detail in [storage](../storage.md).

## Troubleshooting

### `lkhu: command not found`

pipx installs commands into `~/.local/bin`, which may not be on your PATH yet.

```bash
pipx ensurepath
```

Then open a **new** terminal window and try `which lkhu`. If you skipped pipx and used plain `pip`, prefer reinstalling with pipx — it keeps lkhu isolated from your other Python packages.

### Plugin installed, but Claude Code can't start the lkhu MCP server

The plugin runs the bare `lkhu` command, so Claude Code's environment must have it on PATH. Fix PATH as above, then quit and relaunch Claude Code from a terminal where `which lkhu` succeeds.

### `lkhu doctor` says Ollama is unreachable

The Ollama server isn't running. Launch the Ollama app (look for the llama in your menu bar), or start the server manually:

```bash
ollama serve
```

Then confirm the model is present:

```bash
ollama list   # snowflake-arctic-embed2 should appear; if not: ollama pull snowflake-arctic-embed2
```

If you installed Ollama via the Homebrew `ollama` formula and embeddings still fail with the server running, switch to the desktop app (`brew install --cask ollama-app`).

### `lkhu doctor` says the codebook is uninitialized

You haven't run the init step yet:

```bash
lkhu install
```

### Starting over

```bash
lkhu uninstall                     # removes Claude Desktop MCP registration; data preserved
claude plugin uninstall lkhu@lkhu  # removes the Claude Code plugin
```

Your memories and codebook survive both commands. To wipe the live data — codebook, database, index, audit log — there's `lkhu reset --confirm`. This is irreversible, so consider `lkhu backup` or `lkhu export` first. (Snapshots under `backups/` and the codebook copy in `~/Documents` are not touched by `reset`.)

## Next steps

- [How auto-memory works](../auto-memory.md) — the hooks, the noise filter, `<private>` tags
- [Architecture](../architecture.md) — daemon, encoder, recall pipeline
- [FAQ](../faq.md) — common questions
- Back to the [docs index](../index.md)
