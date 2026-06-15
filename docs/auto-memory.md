# Auto Memory (hooks + daemon)

You shouldn't have to say "lkhu, remember this" all day. Auto memory hooks into Claude Code's lifecycle so that Claude **recalls the past on its own and stores new things as you work** — you just have conversations, and memory happens.

This hooks-based auto-memory UX (lifecycle hooks, plugin distribution, `<private>` tags) was popularized by [claude-mem](https://github.com/thedotmack/claude-mem), and lkhu happily borrows the pattern. The difference is what happens underneath: claude-mem calls the Claude API to compress sessions into natural-language summaries; lkhu's entire auto-memory pipeline runs with **zero LLM calls** — local embeddings and vector math only. See [lkhu vs. claude-mem](comparison.md) for the full trade-off discussion.

## Turn it on

You need Ollama with `snowflake-arctic-embed2`, the `lkhu` CLI on your PATH, and the Claude Code plugin:

```bash
ollama pull snowflake-arctic-embed2
pipx install git+https://github.com/iddk0321/lkhu
# once published to PyPI: pipx install lkhu
lkhu install

claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Restart Claude Code, then verify with:

```bash
lkhu doctor    # checks Ollama, codebook, MCP, hooks, and the daemon
```

The plugin registers three hooks (each runs `lkhu hook <event>` — SessionStart with a 60-second timeout, UserPromptSubmit and Stop with 30 seconds) plus the MCP server. That's it — from your next session, Claude starts with memory.

## What happens, hook by hook

| Claude Code event | When it fires | What lkhu does | LLM calls |
|---|---|---|---|
| **SessionStart** | startup, `/clear`, compact | Injects your **top 8 memories** (ranked by strength and recency) as context — Claude knows the past from message one | 0 |
| **UserPromptSubmit** | every prompt you send | Recalls the **5 most relevant memories** for your prompt and injects them; saves the prompt itself if it's non-trivial | 0 |
| **Stop** | Claude finishes responding | Saves the **prose gist** of the last assistant message (up to 280 chars, code stripped) at **strength 0.6** | 0 |

A few details worth knowing:

- **Injected context is compact.** Memories arrive as a markdown bullet list under a header like `## Related memories (lkhu)`, with each line truncated to 200 characters. No walls of text.
- **Recalling is reinforcing.** The UserPromptSubmit recall runs the full recall pipeline, including the gated Hebbian boost (×1.05 strength on candidates that clear the similarity threshold) — so the memories you actually use get stronger just by you working normally.
- **Saving is deduplicating.** Auto-capture fires on every turn, so the same line gets observed repeatedly. Before inserting a new memory, lkhu compares its scent against the nearest stored one: if a turn from the **same session** is near-identical (cosine ≥ 0.95), it lightly reinforces that existing memory (×1.02) and records the repeat in the audit shadow instead of writing a duplicate row. This keeps the store from filling with copies of one sentence, and it's language-agnostic — it compares vectors, not strings.
- **Stop saves are deliberately weak.** Assistant gists go in at strength 0.6 (explicit `remember` calls get 1.3). If a gist is never recalled again, daily decay (×0.99) quietly forgets it. Forgetting is the cleanup mechanism — no summarization pass needed.
- **Failure never blocks you.** `lkhu hook` catches every error and emits a safe pass-through (`{"continue": true, "suppressOutput": true}`). If the daemon is down or Ollama is missing, your session continues without memory rather than breaking.

## The noise filter

Saving everything verbatim would fill memory with junk. Before any text is embedded or stored, lkhu strips noise — in this order:

1. **`<private>` regions** — removed first, before anything else sees them (details below).
2. **System-injected blocks** — `<system-reminder>`, `<task-notification>`, `<command-name>`, `<command-message>`, `<command-args>`, and `<local-command-stdout>` tags. These are Claude Code plumbing, never worth remembering.
3. **lkhu's own injected context** — any `... (lkhu)` heading plus its bullet list. Without this, the memories injected at SessionStart would get re-saved as new memories every session, creating a feedback loop.
4. **Code fences** — stripped from **assistant text only** (the Stop hook). The conclusion of a response is worth remembering; the 80-line diff inside it is not. Your own prompts keep their code blocks.
5. **Save gate** — after cleaning, the text is dropped if it is either **trivial** (under 8 characters) or **structural noise**.

Both halves of that gate are deliberately **language-agnostic** — there is no keyword list of "unimportant phrases" to maintain per language:

- **Trivial-length** is a pure length check, so short acknowledgements fall under the threshold in any language automatically.
- **Structural noise** catches what a length check misses without any keyword list: a **bare URL**, or a line that is **mostly punctuation, emoji, or symbols**. The escape hatch is one rule — anything containing a **real word (a run of ≥3 letters in any script: Latin, Hangul, CJK, …)** is always kept, so symbol-dense but meaningful prose is never dropped.

```text
# Examples skipped by the save gate (multilingual)
"ok"                  → skipped (trivial, 2 chars)
"응"                  → skipped (trivial, Korean "yeah", 1 char)
"はい"                → skipped (trivial, Japanese "yes", 2 chars)
"https://example.com" → skipped (bare URL)
"👍👍👍 !!!"           → skipped (mostly emoji/punctuation, no real word)

# Kept — contains a ≥3-letter word despite the symbols
"deploy 🚀🚀🚀"        → kept
```

Skipped prompts are still **recalled against** — asking "tests?" will surface related memories — they just aren't saved as memories themselves.

## How `<private>` works

Wrap anything in `<private>` tags and lkhu drops it before embedding, saving, or recalling — it never reaches the vector store, the audit log, or the recall query. (This tag convention also comes from claude-mem.)

```text
You: Deploy the staging build. <private>The API key is sk-live-abc123,
don't store this.</private> Use the blue-green strategy we discussed.
```

What lkhu actually saves as a memory:

```text
Deploy the staging build. Use the blue-green strategy we discussed.
```

Two practical notes:

- The tag **must be closed**. The filter matches `<private>...</private>` pairs; an unclosed `<private>` is not stripped.
- `<private>` protects content from *memory*. It does not hide the text from Claude in the live conversation — Claude still sees and acts on it in the current turn.

For secrets that already slipped into memory, archive them after the fact:

```bash
lkhu forget "staging API key" --confirm
```

## Under the hood: the daemon

Hooks are thin clients. A resident daemon (`lkhu daemon`) exclusively owns the memory engine — the single SQLite + FAISS instance — and hooks, the MCP server, and the CLI all talk to it over local HTTP. This is what prevents index drift from multiple processes opening the database at once.

```text
Claude Code ──hook(stdin JSON)──> lkhu hook <event> ──HTTP──> lkhu daemon ──> LkhuEngine
                                                               (sole owner of SQLite + FAISS)
```

- **Auto-start:** the first hook (or MCP) call launches the daemon if it isn't running. No service to manage.
- **Port:** `37700 + (your uid % 100)` by default, so multiple users on one machine don't collide. Override with `LKHU_DAEMON_PORT`.
- **Lifecycle jobs run here too:** daily decay + consolidation (03:00 UTC) and the weekly cleanse (Sunday 03:30 UTC) run on an embedded scheduler inside the daemon — no OS cron or launchd.
- **Dashboard:** `lkhu dashboard` opens a live view of your memories, served by the same daemon.

See [Architecture](architecture.md) for the full picture and [Storage layout](storage.md) for where the data lives on disk.

## Turning it off

| You installed via... | To disable auto memory |
|---|---|
| The Claude Code plugin (recommended path) | `claude plugin uninstall lkhu@lkhu` — removes the hooks and the MCP server together |
| Legacy direct hooks (`lkhu install-hooks`) | `lkhu uninstall-hooks` — removes only lkhu's hooks; your other hooks are untouched |

Either way, **your memories are preserved** — disabling the hooks stops new automatic saves and injections, nothing more. Related commands:

```bash
lkhu uninstall          # also removes the Claude Desktop MCP entry (data and codebook kept)
lkhu reset --confirm    # nuclear option: deletes ALL data including the codebook
```

## FAQ-sized answers

**Does this send my conversations anywhere?** No. Embedding runs on your local Ollama; storage is local SQLite + FAISS. No API key, no cloud, no per-token cost. The auto-memory pipeline makes 0 LLM calls.

**Will hooks slow down my prompts?** Each hook is a local HTTP round-trip with a hard timeout (60s for SessionStart, 30s for UserPromptSubmit and Stop) and fail-open behavior. SessionStart makes no embedding call at all (it ranks stored memories by strength and recency); UserPromptSubmit costs one local `snowflake-arctic-embed2` embedding for the recall query, plus one more if the prompt is non-trivial and gets saved; Stop costs one embedding for the gist.

**What exactly gets stored?** The cleaned text is embedded into a 1024-dim scent vector (the real memory), and the text itself is kept as `audit_text` — a natural-language shadow used only for showing memories back to you, never as the search target. See [Storage layout](storage.md).

**Can I see what it remembered?** `lkhu dashboard` for the live table, `lkhu status` for totals, `lkhu recall "some topic"` to test retrieval from the CLI.
