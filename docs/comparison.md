# lkhu vs claude-mem (and friends)

If you want persistent memory for Claude Code, you have probably already found [claude-mem](https://github.com/thedotmack/claude-mem). It pioneered the pattern that makes auto-memory feel effortless: lifecycle hooks capture what happens in your sessions, a background service stores it, and relevant context gets injected back into future sessions — no manual note-taking.

lkhu deliberately shares that UX. You install a plugin, hooks fire automatically, a local web viewer shows what's remembered, `<private>` tags keep secrets out. Where the two projects part ways is **underneath**: claude-mem stores memory as natural-language summaries written by the Claude API; lkhu stores memory as 1024-dim latent vectors ("scents") produced by pure vector math, with zero API calls. This page lays out that difference honestly so you can pick the right tool.

## TL;DR

| | claude-mem | lkhu |
|---|---|---|
| Memory is... | Natural-language summaries you can read and search verbatim | Latent vectors; natural language kept only as an audit shadow |
| Compression by... | Claude API (semantic summaries) | VSA/HRR vector math (no text generation) |
| API key / cost | Anthropic API used for compression | None — fully local, zero API calls in the pipeline |
| Forgetting | Keeps history for search, timelines, citations | Decays, consolidates, and merges like a brain |

Both are local-first, hooks-based, plugin-installed, and open source. Neither replaces the other for every use case — see [When to choose claude-mem](#-when-to-choose-claude-mem) and [When to choose lkhu](#-when-to-choose-lkhu).

## Both install the same way (almost)

claude-mem:

```bash
npx claude-mem install
# or, inside Claude Code:
#   /plugin marketplace add thedotmack/claude-mem
#   /plugin install claude-mem
```

lkhu:

```bash
pipx install lkhu
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

lkhu additionally needs [Ollama](https://ollama.com) with `ollama pull snowflake-arctic-embed2` for local embeddings — that is its only external runtime (`lkhu install` pulls the model for you). See the [installation guides](installation/macos.md) for details.

## Architecture, side by side

```text
         claude-mem                            lkhu
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ Claude Code session      │      │ Claude Code session      │
  │ 5 hooks: SessionStart,   │      │ 3 hooks: SessionStart,   │
  │ UserPromptSubmit,        │      │ UserPromptSubmit, Stop   │
  │ PostToolUse, Stop,       │      │                          │
  │ SessionEnd               │      │                          │
  └────────────┬─────────────┘      └────────────┬─────────────┘
               │ tool-use observations           │ cleaned prompt/answer text
               ▼                                 ▼
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ Worker service (Bun)     │      │ lkhu daemon (Python)     │
  │ HTTP API :37777 + web UI │      │ 127.0.0.1 + dashboard    │
  │                          │      │ Ollama embeds locally    │
  │ Claude API summarizes    │      │ (snowflake-arctic-embed2)│
  │ observations into        │      │ ; VSA math makes a       │
  │ NL summaries             │      │ 1024-dim scent vector    │
  └────────────┬─────────────┘      └────────────┬─────────────┘
               ▼                                 ▼
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ SQLite + FTS5 (keyword)  │      │ SQLite (meta + audit     │
  │ Chroma (semantic)        │      │ shadow) + FAISS (cosine) │
  │ → hybrid search,         │      │ → re-rank → 3-tier       │
  │   progressive disclosure │      │   decode (LLM < 5%)      │
  └──────────────────────────┘      └──────────────────────────┘
```

The shapes are similar — hooks feed a resident local service, which owns the storage and serves a web UI. The substrate is not. claude-mem's memory **is text**: an AI-written summary you can grep, quote, and cite. lkhu's memory **is a vector**: prompts and answers are embedded by snowflake-arctic-embed2 on your machine, composed with [HRR bind/bundle operations](vsa-explained.md), and only turned back into language at recall time — usually by excerpting the audit shadow or unbinding key-value structure, with an LLM touched in under 5% of decodes (and in the default install, even that tier falls back to a non-LLM extractive excerpt because no LLM is wired in).

## Detailed comparison

| Dimension | claude-mem | lkhu |
|---|---|---|
| **Storage representation** | Natural-language summaries + tool-usage observations | 1024-dim float32 "scent" vectors (VSA/HRR); `audit_text` shadow kept only for visibility and debugging |
| **Compression mechanism** | Claude API generates semantic summaries of session observations (notably at SessionEnd) | Vector math: strength-weighted bundling per session, similarity-based merging — no text is generated |
| **LLM/API calls, and when** | Claude API calls during compression; requires Anthropic API access | Zero across save / recall / consolidate / decay / cleanse. Only the Tier-3 decode fallback may call an LLM (<5% of decodes, capped at 80 tokens) — and only if you wire one in |
| **Privacy / data locality** | Stored locally (SQLite + Chroma); observations are sent to the Claude API for summarization — the same provider already hosting your session | Fully local end to end: embedding via Ollama on your machine, storage on disk, daemon bound to 127.0.0.1. The memory pipeline never touches the network |
| **Embedding** | Via Chroma, which powers the semantic side of hybrid search (the specific embedding model isn't documented in the README) | Ollama + `snowflake-arctic-embed2`, 1024-dim, multilingual, runs locally |
| **Search method** | Hybrid: SQLite FTS5 keyword + Chroma semantic | FAISS cosine over scents (top-K×3 candidates), re-ranked by similarity × (0.6 + 0.2·strength + 0.2·recency) — strength and recency are multiplicative modulators on the primary similarity signal, not additive bonuses — with gated Hebbian reinforcement of what you recall |
| **Retrieval token cost** | Progressive disclosure: search ~50–100 tokens/result, full observations ~500–1000 tokens; claims ~10x savings by filtering before fetching | Injected memories are single lines capped at 200 chars (≤5 per prompt recall, ≤8 at session start); decode output is an excerpt or `KEY=value` pairs; the project's design target is 1/5–1/10 the token cost of natural-language RAG |
| **Lifecycle / forgetting** | Accumulates; full history retained to power exact search, timelines, and citations | Biomimetic: ×0.99 daily decay, ×1.05 boost on recall, nightly consolidation (03:00 UTC), weekly cleanse (merge >0.95 similarity, archive memories below 0.1 strength that are 30+ days old). [Forgetting is a feature](neuroscience.md) |
| **Stack / runtime deps** | TypeScript/JavaScript: Node.js 20+, Bun, SQLite+FTS5, Chroma (Python, needs uv), Claude Agent SDK + Anthropic API client | Python 3.11+ only: numpy, faiss-cpu, SQLite, FastMCP, APScheduler — plus Ollama as the one external runtime |
| **Install** | `npx claude-mem install`, or Claude Code plugin | One `pipx install`, then the Claude Code plugin |
| **Web UI** | Viewer at `localhost:37777`, managed by Bun | Dashboard built into the daemon; `lkhu dashboard` opens it |
| **Multi-language support** | Not a stated focus; the semantic half of hybrid search helps, keyword (FTS5) matching favors exact tokens | snowflake-arctic-embed2 is natively multilingual; the hook noise filters are length-based, with no keyword lists — see example below |
| **Private content** | `<private>` tags excluded from memory | `<private>` tags stripped by regex before anything is saved |
| **Extras** | Citations, observation timelines, NL search skill, Telegram/Discord/Slack feeds, multi-IDE support | 6 MCP tools (`recall`, `remember`, `forget`, `recall_session`, `status`, `export`), JSONL export/import between machines, triple-backed-up codebook |

**Multilingual example.** Because snowflake-arctic-embed2 embeds 100+ languages into one vector space, a query in one language recalls memories saved in another:

```text
# Saved earlier (English): "Decided to use SQLite for metadata and FAISS for vectors"
# Later query (Korean, shown as an illustrative example):
lkhu recall "어제 결정한 저장소 구조가 뭐였지?"   # "what storage layout did we decide yesterday?"
# → recalls the English memory; no translation step, no keyword lists
```

## When to choose claude-mem

Be honest with yourself about what you need. claude-mem is the better fit if you want:

- **Exact-text recall of everything that happened.** Memories are readable summaries backed by full-text search — you can find the literal sentence from three weeks ago.
- **Citations.** claude-mem can point back at where a remembered fact came from. lkhu's scents are lossy by design; its audit shadow is a debugging aid, not a citation system.
- **Observation timelines.** The PostToolUse hook captures tool-level activity, so you get a timeline of what Claude actually did, not just what was said.
- **Multi-IDE support.** claude-mem works beyond Claude Code; lkhu targets Claude Code (plugin) and Claude Desktop (MCP).
- **A mature ecosystem.** Telegram/Discord/Slack feeds, a natural-language search skill, and a large, active user base.

The trade-off you accept: a heavier stack (Node + Bun + Chroma/uv) and Anthropic API usage for compression.

## When to choose lkhu

lkhu is the better fit if you want:

- **Zero API cost and zero API keys.** The entire memory pipeline is local vector math. Nothing to meter, nothing to configure, nothing that breaks when a key rotates.
- **Full locality and privacy.** Embedding runs in Ollama on your machine; the daemon binds to 127.0.0.1; no session content is sent anywhere for summarization.
- **Forgetting instead of unbounded growth.** Memories decay daily, strengthen when recalled, consolidate nightly, and get merged or archived weekly — the store stays small and relevant on its own. See [neuroscience.md](neuroscience.md).
- **A minimal stack.** One `pipx install` of a pure-Python package (numpy, faiss-cpu, SQLite, FastMCP, APScheduler). No Node, no Bun, no Chroma, no uv.
- **Multilingual memory by construction.** snowflake-arctic-embed2's shared embedding space means Korean, English, Japanese, and 100+ other languages recall each other without per-language code.

The trade-off you accept: recall is reconstructive, not verbatim. lkhu remembers *the gist* — like you do — and decodes it into short excerpts or key-value structure, not exact transcripts. If you need word-for-word history, that's claude-mem's home turf.

## Can I run both?

You shouldn't. Both register hooks on the same Claude Code events (SessionStart, UserPromptSubmit, Stop), so running both means **two memory systems injecting context into every session and every prompt** — doubled token overhead, and each system saving the other's injections as new memories. Nothing crashes (lkhu's hooks are designed to fail open and never block your work), but the result is noisy and expensive. Pick one, and `claude plugin uninstall` the other.

## What about CLAUDE.md and plain RAG?

**CLAUDE.md-style static memory** is hand-curated, deterministic, and perfect for rules and conventions ("always use the venv", "never hardcode paths"). It is not automatic and doesn't capture what *happened* — both lkhu and claude-mem complement a CLAUDE.md rather than replace it. **Generic RAG / vector-DB setups** embed chunks of natural language and re-inject them verbatim, so retrieval cost scales with chunk size and the store grows forever; lkhu's bet is that compressing memories into fixed-size composable vectors — and decoding to language only on demand — cuts retrieval to 1/5–1/10 of that token cost while adding a lifecycle that RAG lacks. The full argument is in [architecture.md](architecture.md) and [vsa-explained.md](vsa-explained.md).

## Credits

claude-mem, by [@thedotmack](https://github.com/thedotmack), pioneered hooks-based automatic memory for Claude Code — the plugin distribution, the background worker with a web viewer, `<private>` tags, and the whole "memory that just happens" experience. lkhu borrows that UX shape gratefully and intentionally; it only swaps out the substrate underneath. If lkhu's biomimetic approach isn't what you need, go give claude-mem a star — it has earned it.

---

*Next: [How auto-memory works](auto-memory.md) · [Architecture](architecture.md) · [FAQ](faq.md)*
