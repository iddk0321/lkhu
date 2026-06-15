# Installing lkhu on Windows

**lkhu** (Like Human) gives Claude Code a local, persistent memory that stores what you work on as 1024-dimensional latent vectors ("scents") — no LLM calls in the memory pipeline, no API key, no data leaving your machine. This guide walks you through a clean Windows setup: Python, Ollama, pipx, the `lkhu` CLI, and the Claude Code plugin.

Windows has a few quirks the other platforms don't — the `py` launcher, PATH refreshes after pipx, and the Microsoft Store's `python` alias — so this guide calls each one out where it bites. All examples use PowerShell.

## Quickstart

If you already have Python 3.11+, Git, and Ollama installed, this is the whole thing:

```powershell
ollama pull snowflake-arctic-embed2
py -m pip install --user pipx
py -m pipx ensurepath          # then open a NEW terminal
pipx install git+https://github.com/iddk0321/lkhu
# once published to PyPI: pipx install lkhu
lkhu install
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
lkhu doctor
```

Restart Claude Code and you're done. The rest of this page explains each step, where your data lives, and what to do when something doesn't work.

## Prerequisites

| Requirement | Why | Where |
|---|---|---|
| Python 3.11+ | lkhu is a Python package (`requires-python = ">=3.11"`) | [python.org/downloads](https://www.python.org/downloads/) |
| Git for Windows | `pipx install git+...` clones the repo (lkhu isn't on PyPI yet) | [git-scm.com](https://git-scm.com/download/win) |
| Ollama | Runs the `snowflake-arctic-embed2` embedding model locally (1024-dim) | [ollama.com](https://ollama.com) |
| Claude Code CLI | The `claude plugin` commands wire up hooks + MCP | [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) |

## Step 1 — Install Python 3.11+

Download the installer from [python.org/downloads](https://www.python.org/downloads/). On the first screen of the installer, **check "Add python.exe to PATH"** before clicking Install.

Verify in a new PowerShell window:

```powershell
py --version
# Python 3.12.x  (anything 3.11 or newer is fine)
```

The `py` launcher is installed by default and is the most reliable way to invoke Python on Windows — if you type `python` and the Microsoft Store opens instead, that's the Store's app-execution alias hijacking the command (see [Troubleshooting](#-troubleshooting)). Using `py` sidesteps it entirely.

## Step 2 — Install Ollama and pull snowflake-arctic-embed2

lkhu embeds text with **snowflake-arctic-embed2** (multilingual, 1024 dimensions) running locally through Ollama. This is a hard dependency — every memory save and recall goes through it.

1. Download the Windows installer from [ollama.com](https://ollama.com) and run it. Ollama starts automatically and runs in the system tray.
2. Pull the embedding model:

```powershell
ollama pull snowflake-arctic-embed2
```

Verify it's there:

```powershell
ollama list
# NAME       ...
# snowflake-arctic-embed2     ...
```

Because embedding happens on your machine, nothing you say to Claude is ever sent to a third-party API by lkhu.

> **Stick with `snowflake-arctic-embed2`.** It's the default embedder and what lkhu's eval harness is tuned against. You *can* point `encoder.model` in your config at another 1024-dim model (the previous default, `bge-m3`, still works), but the codebook is built for 1024 dimensions and stored vectors are model-specific — a different model encodes into a different vector space. If you switch models after storing memories, run `lkhu reembed` to re-encode every stored vector with the new model; the codebook, text, strengths, and metadata are preserved.

## Step 3 — Install pipx

[pipx](https://pipx.pypa.io) installs Python CLI tools into isolated environments and puts them on your PATH — exactly what you want for a tool like `lkhu`.

```powershell
py -m pip install --user pipx
py -m pipx ensurepath
```

`ensurepath` edits your user PATH, but **already-open terminals won't see the change**. Close PowerShell and open a new window before continuing.

## Step 4 — Install lkhu

lkhu is not on PyPI yet, so install straight from the repository:

```powershell
pipx install git+https://github.com/iddk0321/lkhu
# once published to PyPI: pipx install lkhu
```

Verify the CLI landed on your PATH:

```powershell
lkhu --version
# 0.1.0
```

If `lkhu` isn't recognized, you skipped the new-terminal step after `ensurepath` — see [Troubleshooting](#-troubleshooting).

## Step 5 — Run lkhu install

```powershell
lkhu install
```

This does two things:

- **Creates your data directory and codebook.** The codebook (`codebook.npy`) is the dictionary of key vectors every memory is encoded with. It's generated once, stored as three copies (the original plus two backups), and must never be regenerated — losing it invalidates all memories. If a codebook already exists, `lkhu install` leaves it alone (the command is idempotent).
- **Registers the MCP server with Claude Desktop**, by writing the `lkhu` entry into `%APPDATA%\Claude\claude_desktop_config.json` (other servers in that file are preserved).

Note what it deliberately does **not** do: it does not touch Claude Code. Claude Code integration goes through the plugin in the next step, because directly editing Claude Code's config files gets clobbered by Claude Code itself — the plugin is the stable path.

## Step 6 — Install the Claude Code plugin

The plugin wires up everything on the Claude Code side: the MCP server (`recall`, `remember`, `forget`, `recall_session`, `status`, `export` tools) and three auto-memory hooks (recall on session start and on each prompt; non-trivial prompts are saved as you go, and the session's final response is saved on stop).

```powershell
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

> **The `lkhu` CLI must be on PATH first** (Steps 3–4). The plugin's hooks and MCP server invoke the bare `lkhu` command — if Claude Code can't find it, the plugin silently does nothing.

Restart Claude Code after installing. How the automatic memory actually works is covered in [Auto memory](../auto-memory.md).

## Step 7 — Verify with lkhu doctor

```powershell
lkhu doctor
```

Doctor checks each link in the chain:

| Check | Healthy looks like |
|---|---|
| Ollama server | Reachable, `snowflake-arctic-embed2` available |
| Codebook | Integrity verified (or "uninitialized (run lkhu install)") |
| Claude Desktop MCP | `lkhu` registered in `claude_desktop_config.json` |
| Claude Code MCP | Registered (or "claude CLI not found" if `claude` isn't on PATH) |
| Auto-memory hooks | Installed |
| Daemon | Running |
| Data directory | Path printed so you know where everything lives |

You can also eyeball your memory store anytime:

```powershell
lkhu status      # totals, kinds, strength stats, codebook keys
lkhu dashboard   # opens the web dashboard in your browser
```

The dashboard daemon listens on `http://127.0.0.1:<port>` by default, where the port is `37700 + (your uid % 100)` (loopback only — nothing is exposed to your network); the canonical example port used throughout the docs is `37701`. `lkhu dashboard` prints the exact URL and opens it for you; pass `--no-open` to skip the browser.

## Where your data lives

Everything is stored under `%LOCALAPPDATA%\lkhu` — typically `C:\Users\<you>\AppData\Local\lkhu`. Path handling goes through `platformdirs` and `pathlib`, so backslashes and OS differences are handled for you.

```powershell
explorer $env:LOCALAPPDATA\lkhu
```

| File / folder | What it is |
|---|---|
| `codebook.npy` | The key-vector dictionary. Sacred — never regenerate. |
| `codebook.backup.npy` | Second codebook copy |
| `memories.db` | SQLite — memory metadata + vectors (source of truth) |
| `vectors.faiss` | FAISS index for fast similarity search |
| `short_term.npy` | Short-term accumulated scent |
| `audit\` | Natural-language shadow log (JSONL, split by month) — for your eyes, never the search target |
| `backups\` | `daily\`, `weekly\`, `monthly\` snapshots |
| `logs\`, `stats.json` | Logs and stats |

The third codebook backup lives outside the data directory, at `Documents\lkhu_codebook.backup.npy`, so it survives even if `%LOCALAPPDATA%\lkhu` is deleted.

To relocate the data root (e.g. onto another drive), set the `LKHU_DATA` environment variable before running any lkhu command. Full details in [Storage layout](../storage.md).

## Troubleshooting

### `lkhu : The term 'lkhu' is not recognized`

pipx put `lkhu.exe` in a directory that isn't on your current session's PATH yet.

```powershell
py -m pipx ensurepath
```

Then **open a new terminal**. If it still fails, `py -m pipx list` shows where pipx installed lkhu so you can confirm it's there.

### `pipx : The term 'pipx' is not recognized`

Same PATH issue, one level up. You can always invoke it through the launcher instead:

```powershell
py -m pipx install git+https://github.com/iddk0321/lkhu
```

### Typing `python` opens the Microsoft Store

Windows ships an app-execution alias that redirects `python` to the Store when no Python is found first on PATH. Either use `py` everywhere (recommended), or disable the alias under **Settings → Apps → Advanced app settings → App execution aliases**.

### `pipx install git+...` fails with "git not found"

Installing from a git URL requires Git on PATH. Install [Git for Windows](https://git-scm.com/download/win), open a new terminal, retry.

### Doctor says Ollama is unreachable

Make sure the Ollama app is running (check the system tray), then:

```powershell
ollama list
```

If `snowflake-arctic-embed2` is missing from the list, `ollama pull snowflake-arctic-embed2`. lkhu raises an error on embedding-dimension mismatch, so any embedder you use must be 1024-dim — the codebook is keyed to the dimension, not to a specific model name. (`snowflake-arctic-embed2` is the default; the previous default `bge-m3` is also 1024-dim.)

### Doctor says "claude CLI not found"

The `claude` command isn't on PATH, so lkhu can't inspect Claude Code's MCP registration. The Claude Desktop side still works. Install the Claude Code CLI, then re-run the plugin commands from Step 6.

### Plugin installed, but Claude never remembers anything

The plugin's hooks run the bare `lkhu` command. If Claude Code was started from an environment where `lkhu` isn't on PATH (common right after install, before restarting), the hooks fail silently by design — they never block your work. Fix: confirm `lkhu --version` works in a fresh terminal, then fully restart Claude Code.

### The daemon port is already taken

The daemon's port (default `37700 + (uid % 100)`, e.g. `37701`) is configurable:

```powershell
setx LKHU_DAEMON_PORT 37755    # persists for new terminals
```

Open a new terminal and re-run `lkhu dashboard` — it prints the URL it actually used.

## Uninstalling

```powershell
claude plugin uninstall lkhu@lkhu   # remove the Claude Code plugin
lkhu uninstall                      # remove the Claude Desktop MCP entry
pipx uninstall lkhu                 # remove the CLI
```

Your data and codebook are **preserved** — `lkhu uninstall` never deletes memories. If you truly want everything gone:

```powershell
lkhu reset --confirm   # ⚠️ irreversible: deletes codebook, database, index, audit
```

## Next steps

- [Auto memory (hooks + daemon)](../auto-memory.md) — how lkhu remembers and recalls without you asking
- [Architecture](../architecture.md) — encoder, recall, 3-tier decoder, lifecycle jobs
- [Storage layout](../storage.md) — every file on disk and how to back it up
- [FAQ](../faq.md) — common questions
- Installing on another machine too? [macOS](macos.md) · [Linux](linux.md)
