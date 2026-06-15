# lkhu — Like Human

> Memory for Claude Code that works like a human brain: thoughts stored as latent vectors, not text. Zero LLM calls, zero API cost, fully local.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)

## Why lkhu

Most memory systems for coding agents store memories as **natural language** — summaries, observations, transcripts. That works, but it has a built-in tax: every save calls an LLM to compress text, and every recall pushes paragraphs back into the context window. The more you remember, the more tokens you burn re-reading your own past.

lkhu takes the human-brain route instead. You don't store your memories as essays — you store compressed traces and reconstruct words only when you need to speak. lkhu does the same: every memory becomes a **"scent"**, a 1024-dimensional latent vector built with [VSA/HRR](docs/vsa-explained.md) (Vector Symbolic Architecture, Holographic Reduced Representations). Storage, association, consolidation, and forgetting are all pure vector math:

- **0 LLM calls** in the entire memory pipeline — save, recall, consolidate, decay, and cleanse use only a local embedding model plus numpy/FAISS operations. An LLM appears in exactly one place: the decoder's Tier-3 fallback (under 5% of decodes, capped at 80 tokens).
- **Fully local.** Embeddings come from [Ollama](https://ollama.com) + `snowflake-arctic-embed2` on your machine. No API key, no API cost, no data leaving your computer.
- **Forgetting is a feature.** Memories decay like the Ebbinghaus curve, strengthen when recalled (Hebbian reinforcement), get consolidated nightly, and weak old ones are archived — so recall stays sharp instead of drowning in noise.

## Quick Start

You need Python 3.11+ and [Ollama](https://ollama.com) installed.

```bash
# 1. Pull the embedding model (one time)
ollama pull snowflake-arctic-embed2

# 2. Install the lkhu CLI
pipx install lkhu

# 3. One-time setup: codebook + Claude Desktop MCP registration
lkhu install

# 4. Add lkhu to Claude Code as a plugin
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Restart Claude Code — done. From now on, lkhu injects recent memories at session start, recalls related memories on every prompt, and saves the conversation automatically. You never have to say "remember this" (though you can). Check everything is wired up with:

```bash
lkhu doctor
```

Per-OS notes: [macOS](docs/installation/macos.md) · [Linux](docs/installation/linux.md) · [Windows](docs/installation/windows.md). To remove: `claude plugin uninstall lkhu@lkhu` and `lkhu uninstall` (your data and codebook are preserved).

## How it works

**Encoding.** Each memory's scent is `0.6 × semantic + 0.4 × structural`. The semantic part is the snowflake-arctic-embed2 embedding; the structural part binds extracted key/value pairs (files, programming languages) to fixed unitary key vectors with circular convolution, then bundles them. Keys live in an append-only **codebook** generated once at install and backed up three times — new keys can be auto-discovered, but existing key vectors are never changed or regenerated.

**Recall.** The query scent goes to FAISS (top-K×3 candidates), which get re-ranked by `similarity × (0.6 + 0.2 × strength + 0.2 × recency)`. Similarity is the primary, query-conditional signal; strength and recency are **multiplicative modulators**, not additive bonuses — so a globally-strong but off-topic memory can never outrank a clearly more similar one. Genuine hits (similarity ≥ 0.58) get a Hebbian strength boost (×1.05); every returned memory's access count still ticks up. The winners are bundled into one composite scent. No LLM involved.

**Decoding (3-tier).** Turning a scent back into language: ① short audit excerpts (~70% of cases), ② unbinding key vectors from the composite and matching against a local vocabulary (~25%), ③ LLM fallback capped at 80 tokens (<5% — and if no LLM is wired in, which is the default install, this tier degrades to a plain audit excerpt: zero LLM calls even here). A natural-language shadow copy (`audit_text`) is always kept for your visibility — but it's never the search target.

**Biomimetic lifecycle.** Daily at 03:00 UTC: strength decays ×0.99 and each session's memories consolidate into a summary scent (a weighted vector sum — 0 LLM calls). Weekly on Sunday: near-duplicates (cosine > 0.95) merge, and memories weaker than 0.1 and older than 30 days are archived, not deleted. See [neuroscience mapping](docs/neuroscience.md) and [architecture](docs/architecture.md).

**Multilingual by construction.** snowflake-arctic-embed2 is a multilingual embedder and lkhu has no per-language keyword lists, so memories work across languages. Example (Korean input shown for illustration):

```bash
lkhu remember "우리 팀 백엔드는 FastAPI를 쓴다"   # stored in Korean
lkhu recall "which web framework does the team use"  # recalled in English
```

## How is this different from claude-mem?

[claude-mem](https://github.com/thedotmack/claude-mem) is the pioneering memory plugin for Claude Code, and lkhu openly borrows from its UX: hooks-based automatic memory, plugin distribution, `<private>` tags, and a local web viewer all appeared there first. The difference is the layer underneath — claude-mem remembers in *language*, lkhu remembers in *vectors*.

| | claude-mem | lkhu |
|---|---|---|
| **Memory representation** | Natural-language summaries (AI-compressed observations) | 1024-dim latent vectors ("scents") via VSA/HRR; NL kept only as an audit shadow |
| **LLM calls** | Claude API compresses observations (e.g. at SessionEnd) | 0 in the pipeline; Tier-3 decode fallback only (<5%, ≤80 tokens) |
| **Privacy / locality** | Summarization goes through the Anthropic API | Fully local (Ollama embedding); no API key, no API cost |
| **Runtime stack** | TypeScript: Node 20+, Bun worker, SQLite+FTS5, Chroma (needs uv) | Python 3.11+ only: numpy, faiss-cpu, SQLite, FastMCP — one pipx install |
| **Retrieval cost** | Progressive disclosure: ~50–100 tokens/result, ~500–1000/detail | Composite scent decoded to a few short lines; Tiers 1–2 cost 0 LLM tokens |
| **Lifecycle / forgetting** | Persistent archive; everything kept and searchable | Decay, Hebbian reinforcement, consolidation, cleanse — designed to forget |
| **Web UI** | Viewer at `localhost:37777`, plus Telegram/Discord/Slack feeds | Built-in dashboard (`lkhu dashboard`): lifecycle, strength/age charts, memory table |
| **Multi-language** | Hybrid FTS5 keyword + semantic search | snowflake-arctic-embed2 multilingual embeddings; no language-specific keyword lists |

**Pick claude-mem if** you want a searchable, citable record of exactly what happened: full-text search over past sessions, rich observation detail, citations, chat-platform feeds, and a mature, battle-tested ecosystem. Keeping everything in natural language is precisely what makes that possible.

**Pick lkhu if** you want memory as a cheap, private background sense: no API spend on remembering, nothing leaving your machine, a single-language Python stack, and a system that compresses and forgets on its own like a brain does. The trade-off is honest — lkhu can't grep your past verbatim, because the past isn't stored as text.

They're different answers to the same question, and you can learn a lot by reading both. Longer write-up: [docs/comparison.md](docs/comparison.md).

## What's inside

**Automatic memory (hooks).** Three Claude Code hooks, all zero-LLM: `SessionStart` injects recent memories, `UserPromptSubmit` recalls related memories and saves your prompt, `Stop` saves the assistant's final reply. System noise and `<private>…</private>` blocks are stripped before saving, and code/tool dumps are stripped from assistant replies. Details: [docs/auto-memory.md](docs/auto-memory.md).

**MCP tools.** Claude can call `recall`, `remember`, `forget`, `recall_session`, `status`, and `export` directly. Reference: [docs/api-reference.md](docs/api-reference.md).

**CLI.** The `lkhu` command for inspection and control:

| Command | What it does |
|---|---|
| `lkhu install` / `lkhu uninstall` | Set up / remove (codebook + Claude Desktop MCP; data preserved) |
| `lkhu doctor` | Diagnose Ollama, codebook, MCP, hooks, daemon |
| `lkhu status` | Memory counts, kinds, strength stats, codebook info |
| `lkhu remember "..." --kind fact` | Store a memory explicitly |
| `lkhu recall "query" --k 5` | Search memories (debugging) |
| `lkhu forget "query" --confirm` | Archive matching memories (audit preserved) |
| `lkhu dashboard` | Open the web dashboard in your browser |
| `lkhu export` / `lkhu import <file>` | Move audit data between machines as JSONL |
| `lkhu backup` | Snapshot the codebook and database |
| `lkhu eval` | Score recall / noise / multilingual / save-filter on a fixed gold corpus (isolated; never touches your memories) |
| `lkhu reembed` | Rebuild all memory vectors with the current embedding model (run after switching models) |

**Dashboard.** A self-contained local web page served by the daemon — stat cards, lifecycle flow, strength/age distributions, and a sortable memory table that refreshes every 5 seconds.

**Daemon architecture.** A single resident daemon owns the engine (SQLite + FAISS); hooks, the MCP server, and the dashboard are thin HTTP clients of it, so they always see one consistent index. It starts automatically when needed. Storage details: [docs/storage.md](docs/storage.md).

## Development

```bash
git clone https://github.com/iddk0321/lkhu
cd lkhu
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # Ollama-dependent tests are skipped by default
ruff format && ruff check && mypy src
```

CI runs on ubuntu/macos/windows × Python 3.11/3.12/3.13. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Docs

- [Documentation index](docs/index.md)
- [Architecture](docs/architecture.md) · [Storage](docs/storage.md) · [API reference](docs/api-reference.md)
- [VSA/HRR explained](docs/vsa-explained.md) · [Neuroscience mapping](docs/neuroscience.md)
- [Auto memory](docs/auto-memory.md) · [FAQ](docs/faq.md) · [Changelog](CHANGELOG.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Built on the shoulders of: Tony Plate's _Holographic Reduced Representations_ (2003), [Snowflake Arctic Embed 2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0), [FAISS](https://github.com/facebookresearch/faiss), [FastMCP](https://github.com/jlowin/fastmcp), and the [Model Context Protocol](https://modelcontextprotocol.io). UX inspiration: [claude-mem](https://github.com/thedotmack/claude-mem).
