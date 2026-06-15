# lkhu — Claude Code plugin

This plugin gives Claude Code persistent memory powered by [lkhu](../../README.md) (Like Human) — memories stored as 1024-dim latent vectors instead of natural language, with zero LLM calls in the memory pipeline.

Editing `settings.json` or `~/.claude.json` by hand is fragile — Claude Code can clobber direct edits. Installing this plugin wires up everything in one step:

- **3 lifecycle hooks** that recall and save memories automatically
- **The lkhu MCP server** (`lkhu serve`, stdio), exposing memory tools to Claude

## Quickstart

The plugin is a thin shell — every hook and the MCP declaration invoke the `lkhu` console command on PATH. Install the CLI first:

```bash
pipx install lkhu

lkhu install   # creates the codebook + data directories (also registers Claude Desktop MCP)
```

You also need [Ollama](https://ollama.com) with the `snowflake-arctic-embed2` embedding model (`ollama pull snowflake-arctic-embed2`). Verify everything with `lkhu doctor`.

Then install the plugin:

```bash
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Restart Claude Code (or start a new session) and the hooks and MCP server are active.

## What the hooks do

All three call `lkhu hook <event>` with a 15-second timeout, make zero LLM calls, and fail open — any error passes through silently and never blocks your work.

| Hook | When | What it does |
|------|------|--------------|
| `SessionStart` | startup / resume / clear / compact | Injects your top memories (by strength and recency) as session context |
| `UserPromptSubmit` | every prompt | Recalls memories relevant to your prompt, injects them, and saves the prompt itself (trivially short prompts are skipped) |
| `Stop` | end of each response | Saves the assistant's final reply as a low-strength memory (code fences stripped) |

Text inside `<private>...</private>` tags is never saved. Details in [docs/auto-memory.md](../../docs/auto-memory.md).

## MCP server

`.mcp.json` registers `lkhu serve` (stdio), which exposes six tools: `recall`, `remember`, `forget`, `recall_session`, `status`, `export`. The server is a thin client of the lkhu daemon, which it starts automatically if needed.

## Local development

Test the plugin from a clone without going through a marketplace:

```bash
claude --plugin-dir path/to/lkhu/plugins/lkhu
```

## Files

```text
plugins/lkhu/
├── .claude-plugin/plugin.json   # manifest (v0.1.0)
├── hooks/hooks.json             # SessionStart / UserPromptSubmit / Stop
├── .mcp.json                    # lkhu serve (stdio)
└── README.md
```

## Uninstall

```bash
claude plugin uninstall lkhu@lkhu
```

Your memories and codebook are untouched — only the plugin registration is removed.

## Learn more

- [Main README](../../README.md) — what lkhu is and why vectors instead of summaries
- [Documentation](../../docs/index.md) — architecture, VSA explained, storage, FAQ
- Installation guides — [macOS](../../docs/installation/macos.md) · [Linux](../../docs/installation/linux.md) · [Windows](../../docs/installation/windows.md)

License: Apache-2.0 · Repo: https://github.com/iddk0321/lkhu
