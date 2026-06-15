# lkhu Documentation

Welcome! **lkhu** (Like Human) gives Claude Code a memory that works like a human brain. Instead of storing your conversations as natural-language summaries and paying an LLM to compress and re-read them, lkhu stores memories as **1024-dimensional latent vectors ("scents")** and runs the entire memory pipeline — save, recall, consolidate, decay, cleanse — with pure vector math. Zero LLM calls in the pipeline; an LLM appears only in a rare decoder fallback (under 5% of decodes, capped at 80 tokens). Everything runs locally via Ollama, so your data never leaves your machine.

If you just want it working, start with the quickstart below. If you want to understand *why* it works, the Concepts section is for you.

## Quickstart

You need [Ollama](https://ollama.com) with the `snowflake-arctic-embed2` embedding model, and the `lkhu` CLI on your PATH:

```bash
ollama pull snowflake-arctic-embed2
pipx install git+https://github.com/iddk0321/lkhu
# once published to PyPI: pipx install lkhu
lkhu install
```

Then add the Claude Code plugin (this wires up the MCP server and the auto-memory hooks):

```bash
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Verify everything with `lkhu doctor`, restart Claude Code, and you're done. For platform-specific details (Homebrew, Windows PATH quirks, XDG paths, troubleshooting), use the per-OS guides below.

## Where to go next

### Getting started

| Guide | Read this if... |
|-------|-----------------|
| [Install on macOS](installation/macos.md) | You're on a Mac and want step-by-step setup with Homebrew. |
| [Install on Windows](installation/windows.md) | You're on Windows and want setup including PATH details. |
| [Install on Linux](installation/linux.md) | You're on Linux and want setup with XDG paths. |

### Concepts

| Guide | Read this if... |
|-------|-----------------|
| [Architecture](architecture.md) | You want the big picture: how the encoder, recall engine, 3-tier decoder, and lifecycle jobs fit together. |
| [VSA "scents" explained](vsa-explained.md) | You're new to vector symbolic architectures and want an intuition for bind / bundle / unbind — no math degree required. |
| [Neuroscience-based design](neuroscience.md) | You're curious how working memory, the hippocampus, Hebbian reinforcement, and REM-sleep consolidation map onto lkhu's components. |

### Guides

| Guide | Read this if... |
|-------|-----------------|
| [Auto memory (hooks + daemon)](auto-memory.md) | You want Claude to remember and recall automatically — no "lkhu, remember this" needed — and want to know how the hooks and daemon work. |
| [Storage layout](storage.md) | You want to know exactly where your data lives on disk (SQLite, FAISS, codebook, audit log) and how to back it up. |
| [API reference](api-reference.md) | You need the full list of CLI commands and MCP tools with their options. |
| [Evaluation](evaluation.md) | You want to measure recall quality objectively (`lkhu eval`) or tune the system against a gold corpus. |

### Comparison

| Guide | Read this if... |
|-------|-----------------|
| [lkhu vs. claude-mem](comparison.md) | You're choosing a memory plugin for Claude Code and want an honest look at the trade-offs between latent-vector memory and natural-language memory. |

### FAQ

| Guide | Read this if... |
|-------|-----------------|
| [FAQ](faq.md) | You have a quick question — Is Ollama required? What if I lose the codebook? Does it really save tokens? |

## Core concepts at a glance

| Concept | Meaning |
|---------|---------|
| **Scent** | A 1024-number vector — the basic unit of memory |
| **Codebook** | The dictionary of key scents. The system's DNA; never regenerated, triple-backed-up |
| **Bind / Bundle / Unbind** | Tie a key and value together / add scents into one / open a bound pair with its key |
| **3-tier decoder** | Audit excerpt → key probe → LLM fallback (<5%, ≤80 tokens) |
| **Biomimetic lifecycle** | Daily decay (×0.99), Hebbian boost on recall (×1.05), session consolidation, weekly cleanse |
| **audit_text** | A natural-language shadow copy kept for your visibility and debugging — never the search target |

## Project info

- **Repository:** [github.com/iddk0321/lkhu](https://github.com/iddk0321/lkhu)
- **Version:** 0.1.0
- **License:** Apache-2.0
- **Requires:** Python 3.11+, Ollama with `snowflake-arctic-embed2`
