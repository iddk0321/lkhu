# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-16

First release. lkhu (Like Human) gives Claude Code a memory that works like a human brain:
memories are stored as 1024-dimensional latent vectors ("scents") rather than natural-language
summaries, and the entire memory pipeline — save, recall, consolidate, decay, cleanse — runs with
**zero LLM calls** (local embeddings + vector math only).

### Memory engine
- **VSA / HRR core**: numpy FFT-based bind / unbind / bundle with unitary keys (no torch).
- **Codebook**: deterministic key derivation, regeneration guard, integrity checksum, triple backup.
- **Encoder**: `0.6 · semantic + 0.4 · structural` composition; an `Embedder` protocol with a local
  Ollama embedder and an offline `HashingEmbedder` for tests. Default embedder
  **`snowflake-arctic-embed2`** (1024-dim, multilingual) — set `encoder.model` to use another model.
- **Storage**: `LongTermVault` (SQLite source of truth + FAISS index rebuilt on open, thread-safe),
  monthly JSONL `AuditLog` (the natural-language shadow, kept for visibility — never the search
  target), `ShortTermBundle`, `WorkingMemory`.
- **Recall + 3-tier decoder**: FAISS top-K×3 → re-rank where strength and recency *multiplicatively*
  modulate similarity (`similarity × (0.6 + 0.2·strength/max + 0.2·recency)`) → gated Hebbian
  reinforcement (only matches above `reinforce_sim_threshold` are strengthened) → decode via audit
  excerpt, key-unbind probe, or an LLM fallback that degrades to a non-LLM excerpt by default.
  Optional `recall.min_similarity` floor trims low-relevance results.
- **Biomimetic lifecycle**: Ebbinghaus decay (×0.99/day), Hebbian boost on recall (×1.05),
  per-session consolidation (strength-weighted vector sum), glymphatic cleanse (merge cosine > 0.95,
  archive weak + old). The daemon runs any overdue lifecycle on startup, so it works even if the
  machine sleeps through the nightly schedule.

### Integration
- **Daemon** (`lkhu daemon`): a resident local HTTP service (127.0.0.1) that solely owns the engine;
  hooks, the MCP server, and the dashboard are clients, so they always share one consistent store.
- **Claude Code plugin**: repo-root marketplace + `plugins/lkhu/` (plugin.json, hooks.json, .mcp.json).
  SessionStart injects recent memories, UserPromptSubmit recalls + saves, Stop saves the reply gist —
  all language-agnostic, with a noise filter and `<private>` support, and a near-duplicate save dedup.
- **MCP server** (`lkhu serve`): recall / remember / forget / recall_session / status / export tools.
- **Web dashboard** (`lkhu dashboard`): live view of memories, strength/age distribution, and lifecycle.

### Tooling
- **CLI**: init, install, serve, status, remember, forget, recall, doctor, export, import, backup,
  uninstall, reset, dashboard, eval, reembed, daemon, hook, install-hooks, uninstall-hooks.
- **`lkhu eval`**: scores recall quality / noise / cross-lingual / save-filter against a fixed gold
  corpus in an isolated data dir (deterministic; never touches your memories). `--model` benchmarks
  any embedder. Default-embedder scorecard: hit@k 1.00, cross-lingual 1.00, noise 0.23, filter 100/100.
- **`lkhu reembed`**: re-encodes stored memories with the current model (the migration path when
  switching embedders).
- **Platform**: `platformdirs`-based paths (no hardcoded OS paths), 3-OS support, CI across
  ubuntu/macos/windows × Python 3.11–3.13.

### Requirements
- Python 3.11+, and [Ollama](https://ollama.com) with `ollama pull snowflake-arctic-embed2`.

[0.1.0]: https://github.com/iddk0321/lkhu/releases/tag/v0.1.0
