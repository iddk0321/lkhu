# Installing lkhu on Linux

**lkhu** (Like Human) gives Claude Code a memory that works like a human brain — memories stored as 1024-dim latent vectors ("scents"), recalled with pure vector math, zero LLM calls in the pipeline. Everything runs locally: embeddings come from Ollama on your own machine, so nothing leaves it and there's no API key to manage.

Linux is arguably the easiest platform for lkhu: Ollama's official installer sets itself up as a systemd service, and the Python tooling you need is one package manager command away.

## Quickstart

If you already have Python 3.11+ and pipx, this is the whole install:

```bash
# 1. Ollama + the embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull snowflake-arctic-embed2

# 2. The lkhu CLI (not on PyPI yet — install from git)
pipx install lkhu

# 3. Initialize (codebook + Claude Desktop MCP config)
lkhu install

# 4. Claude Code integration (plugin: MCP server + auto-memory hooks)
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu

# 5. Verify
lkhu doctor
```

Restart Claude Code and you're done. The rest of this page walks through each step in detail, plus where your data lives and what to do when something doesn't work.

## Step 1: Python 3.11+ and pipx

lkhu requires **Python 3.11 or newer**. Check what you have:

```bash
python3 --version
```

If you're on a recent distro (Ubuntu 24.04+, Fedora 39+, Arch, Debian 12+), the system Python is already new enough. Install pipx from your package manager:

| Distro | Command |
|--------|---------|
| Ubuntu / Debian | `sudo apt install pipx` |
| Fedora | `sudo dnf install pipx` |
| Arch | `sudo pacman -S python-pipx` |

Then make sure pipx-installed commands land on your PATH:

```bash
pipx ensurepath
```

Open a new shell (or `source ~/.bashrc` / `source ~/.zshrc`) so `~/.local/bin` is picked up.

**Why pipx?** It installs lkhu into its own isolated virtualenv and exposes the `lkhu` command globally — no dependency conflicts with your system Python, and the Claude Code plugin needs `lkhu` on PATH to work.

If your system Python is older than 3.11, install a newer one alongside it (on Ubuntu LTS that means the deadsnakes PPA: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11`; on Fedora, `sudo dnf install python3.11`) and point pipx at it in Step 3 with `--python python3.11`.

## Step 2: Ollama and the snowflake-arctic-embed2 model

lkhu embeds text locally with [Ollama](https://ollama.com) and the multilingual **snowflake-arctic-embed2** model (1024 dimensions). This is a mandatory dependency — it's where the "scents" come from.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull snowflake-arctic-embed2
```

On Linux, the official installer registers Ollama as a **systemd service**, so it starts on boot and is always available when lkhu needs an embedding. Verify it's running:

```bash
systemctl status ollama
```

If it isn't (e.g. you installed Ollama some other way), enable it:

```bash
sudo systemctl enable --now ollama
```

A quick end-to-end check that the model actually works:

```bash
ollama list          # snowflake-arctic-embed2 should appear
```

## Step 3: Install the lkhu CLI

lkhu is not on PyPI yet, so install straight from git:

```bash
pipx install lkhu
```

If your default `python3` is older than 3.11:

```bash
pipx install --python python3.11 lkhu
```

Confirm it's on PATH:

```bash
lkhu --version   # should print 0.1.0
```

## Step 4: Run `lkhu install`

```bash
lkhu install
```

This does two things:

- **Creates your codebook** (`codebook.npy`) — the key-scent dictionary every memory is encoded against — with **triple backup**. It's generated once and never regenerated; if it already exists, it's preserved.
- **Registers the MCP server with Claude Desktop** by writing to `~/.config/Claude/claude_desktop_config.json`. If you don't use Claude Desktop on Linux, that file is simply ignored — it's harmless.

It also checks that Ollama and snowflake-arctic-embed2 are reachable, and prints the plugin commands for the next step. `lkhu install` is idempotent — running it again won't touch your existing codebook or memories.

> **The codebook is sacrosanct.** Losing `codebook.npy` invalidates every stored memory. lkhu keeps three copies: the original, `codebook.backup.npy` next to it, and a third at `~/Documents/lkhu_codebook.backup.npy`. Don't delete them.

## Step 5: Install the Claude Code plugin

Claude Code integration ships as a **plugin** (it survives Claude Code updates, unlike directly edited config files). The plugin wires up both the MCP server and the auto-memory hooks:

```bash
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

The plugin runs the bare `lkhu` command, which is why Step 3 (pipx on PATH) must come first.

What you get:

- **6 MCP tools**: `remember`, `recall`, `recall_session`, `forget`, `status`, `export`
- **Auto memory via 3 hooks**: context injection on session start and on every prompt, conversation save on stop — all with zero LLM calls. See [Auto memory](../auto-memory.md) for how the noise filter and `<private>` tags work.

Restart Claude Code to activate.

## Step 6: Verify with `lkhu doctor`

```bash
lkhu doctor
```

This checks everything in one shot: the Ollama server, codebook integrity, Claude Desktop MCP registration, Claude Code MCP registration, auto-memory hooks, whether the daemon is running, and your data directory path.

Other useful commands:

```bash
lkhu status      # memory counts, kind distribution, strength stats
lkhu dashboard   # opens the live web dashboard in your browser
```

## Where your data lives (XDG)

lkhu follows the XDG Base Directory spec via `platformdirs`:

| What | Path |
|------|------|
| Data (memories, codebook, index) | `~/.local/share/lkhu/` |
| Config | `~/.config/lkhu/config.yaml` |
| Third codebook backup | `~/Documents/lkhu_codebook.backup.npy` |

Inside the data directory you'll find `codebook.npy`, `codebook.backup.npy`, `memories.db` (SQLite), `vectors.faiss`, `short_term.npy`, `audit/` (the natural-language shadow log), `backups/`, `logs/`, and `stats.json`.

Two overrides, in priority order: the `LKHU_DATA` environment variable wins, otherwise `XDG_DATA_HOME` is respected if you've set it. Full details in [Storage layout](../storage.md).

**A note on scheduling:** lkhu's maintenance jobs (daily decay + consolidation at 03:00, weekly cleanse Sunday 03:30 — both **UTC**) run inside the lkhu daemon via an embedded APScheduler. There's no cron entry or systemd timer to set up for lkhu itself; the daemon auto-starts on first use.

## Troubleshooting

### `lkhu: command not found`

pipx installs to `~/.local/bin`, which isn't always on PATH. Run `pipx ensurepath`, then open a new shell. Confirm with `which lkhu`.

### `lkhu doctor` says Ollama is unreachable

The Ollama systemd service probably isn't running:

```bash
systemctl status ollama
sudo systemctl enable --now ollama
```

If you skipped the curl installer and run `ollama serve` manually, make sure it's actually up before using lkhu.

### snowflake-arctic-embed2 missing or embedding fails

```bash
ollama pull snowflake-arctic-embed2
```

lkhu expects exactly **1024 dimensions** from the embedder, because the codebook is built for 1024-dim space; any embedder of a different dimension is rejected with a dimension-mismatch error. **Stick with the configured model** (`snowflake-arctic-embed2` by default). You *can* point `encoder.model` in your config at another 1024-dim model — the previous default, `bge-m3`, still works — but stored vectors are model-specific, so after switching models run `lkhu reembed` to re-encode them; the codebook, text, strengths, and metadata are preserved.

### `lkhu doctor` says "claude CLI not found"

That check only matters for Claude Code integration. Install Claude Code (which provides the `claude` CLI), then run the plugin commands from Step 5.

### Codebook reported as "uninitialized"

You haven't run `lkhu install` yet — run it. If you *had* a codebook and it's gone, restore from `codebook.backup.npy` or `~/Documents/lkhu_codebook.backup.npy` before doing anything else. Never regenerate.

### Dashboard / daemon port

The daemon binds to `127.0.0.1` on port `37700 + (your uid % 100)` — e.g. **37701** for a uid ending in 01 (the canonical example port used throughout the docs). Override with the `LKHU_DAEMON_PORT` env var. `lkhu dashboard` prints the exact URL; on a headless server use `lkhu dashboard --no-open` and tunnel the port over SSH.

### Old Python on LTS distros

On distros where `python3` is 3.10 or older, install a newer interpreter (deadsnakes PPA on Ubuntu LTS, `dnf install python3.11` on Fedora) and tell pipx to use it:

```bash
pipx install --python python3.11 lkhu
```

## Next steps

- [Auto memory (hooks + daemon)](../auto-memory.md) — how the automatic remember/recall loop works
- [Storage layout](../storage.md) — backup strategy and every file explained
- [Architecture](../architecture.md) — encoder, 3-tier decoder, lifecycle jobs
- [FAQ](../faq.md) — common questions, including multi-machine sync via `lkhu export` / `lkhu import`
- Installing on another machine? [macOS](macos.md) · [Windows](windows.md)
