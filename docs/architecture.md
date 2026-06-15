# Architecture

Every persistent-memory tool has to answer an awkward question: **who owns the data when five processes want it at once?** A hook fires mid-conversation, an MCP tool call arrives over stdio, you run a CLI command, and a scheduler wants to run nightly decay — all touching the same SQLite database and the same FAISS index.

lkhu's answer is deliberately boring: **one resident daemon owns the data, and the concurrent entry points — hooks, MCP, dashboard — are HTTP clients.** This page explains that design, the three-layer code structure behind it, and exactly what happens when a memory is saved or recalled.

## See it running first

Before the theory, poke at the real thing:

```bash
pipx install lkhu

lkhu install        # codebook + Claude Desktop registration
lkhu dashboard      # starts the daemon if needed, opens the web UI
lkhu doctor         # checks Ollama, codebook, MCP registration, hooks, daemon
```

The daemon listens on loopback only. Its port is derived from your user id — `37700 + (uid % 100)` — so each user on a machine gets a stable, private daemon. On a typical macOS machine (uid 501) that's **37701**; set `LKHU_DAEMON_PORT` to pin it explicitly.

```bash
curl http://127.0.0.1:37701/health
# example output: {"ok": true, "memories": 42}
```

## The big picture

```text
   Claude Code                        Claude Desktop            You
 ┌─────┴──────────┐                         │                    │
 hooks        MCP (stdio)              MCP (stdio)              CLI
 │                │                         │                    │
 lkhu hook …   lkhu serve               lkhu serve      lkhu dashboard / doctor
 │                │                         │                    │
 └────────────────┴─────────────┬───────────┴────────────────────┘
                                │  HTTP (loopback only)
                                ▼
                  ┌───────────────────────────┐
                  │        lkhu daemon        │  ThreadingHTTPServer
                  │   owns the LkhuEngine —   │  + APScheduler (UTC crons)
                  │   the only process that   │  + web dashboard at /
                  │   opens SQLite and FAISS  │
                  └─────────────┬─────────────┘
                                │
                                ▼
                  SQLite  ·  FAISS  ·  audit JSONL
```

Three kinds of clients, one owner:

- **Hooks** (`lkhu hook <event>`) — fired by the Claude Code plugin on `SessionStart`, `UserPromptSubmit`, and `Stop`. They inject remembered context and save conversation turns. See [auto-memory.md](auto-memory.md).
- **MCP server** (`lkhu serve`) — a FastMCP stdio process exposing 6 tools (`recall`, `remember`, `forget`, `recall_session`, `status`, `export`). Each tool is a thin HTTP forward to the daemon.
- **CLI** — `lkhu dashboard`, `lkhu doctor`, and `lkhu hook` go through the same `LkhuClient`. One exception worth knowing: the one-off data commands (`lkhu status`, `lkhu recall`, `lkhu remember`, …) currently open the engine directly in-process rather than going through the daemon.

## Three layers

The source tree under `src/lkhu/` enforces a strict knows-about hierarchy:

| Layer | Path | Knows about | Must not know about |
|---|---|---|---|
| **platform** | `src/lkhu/platform/` | OS paths, Ollama, Claude config files, cron scheduling | Memory semantics |
| **core** | `src/lkhu/core/` | Vectors, memories, lifecycle math | The OS, HTTP, Claude |
| **server** | `src/lkhu/server/` | HTTP, hooks, MCP, the dashboard | OS specifics (goes through platform) |

There is also a small `config/` package: `defaults.yaml` holds every tunable number (weights, crons, thresholds), and `loader.py` merges it with your user config.

### platform/ — OS isolation

Hard rule of the project: **no hardcoded paths, anywhere.** Everything OS-specific funnels through this layer, so `core/` runs identically on macOS, Linux, and Windows.

- `paths.py` — all file locations via `platformdirs` (`LKHU_DATA` env overrides the data root). See [storage.md](storage.md) for the full layout.
- `ollama.py` — the `snowflake-arctic-embed2` embedder (1024-dim), availability checks, model pulls.
- `scheduler.py` — APScheduler wrapper, embedded in the daemon. No OS cron or launchd.
- `mcp_config.py` — Claude Desktop config file handling (its path differs per OS).
- `claude_code.py` / `claude_hooks.py` — legacy direct Claude Code registration (the plugin supersedes both).
- `setup.py` — `lkhu install` / `lkhu uninstall` orchestration.

### core/ — the OS-agnostic engine

The actual memory system. Pure Python + numpy + faiss; no HTTP, no Claude awareness, fully testable offline (a `HashingEmbedder` stands in for Ollama in tests).

```text
                      LkhuEngine (facade)
                             │
        ┌───────────┬────────┼────────────┬───────────────┐
        ▼           ▼        ▼            ▼               ▼
     Encoder     Recall   Decoder     Lifecycle        Codebook
   (semantic +  (re-rank  (3-tier)   decay · consoli-  (sealed key
    structure)  + Hebbian)            date · cleanse    scents)
        └───────────┴────┬───┴────────────┘
                         ▼
              VSA engine — numpy FFT HRR
            bind · unbind · bundle · cosine
                         │
            ┌────────────┼─────────────────┐
            ▼            ▼                 ▼
     ShortTermBundle  LongTermVault     AuditLog
     (accumulated     (SQLite + FAISS)  (JSONL shadow,
      session scent)                     user-visible)
```

Modules: `vsa` (HRR ops), `codebook`, `encoder`, `decoder`, `recall`, `memory`, `long_term`, `short_term`, `working_memory`, `audit`, `decay`, `consolidator`, `glymphatic`, `metrics`, and `engine` — the facade everything else talks to. The math behind the VSA engine is covered in [vsa-explained.md](vsa-explained.md); the biology it mimics in [neuroscience.md](neuroscience.md).

### server/ — daemon, hooks, MCP, dashboard

- `daemon.py` — the resident service: stdlib `ThreadingHTTPServer`, JSON routes, scheduler, dashboard.
- `client.py` — `LkhuClient` (the HTTP client all entry points share) plus `ensure_daemon_running()`.
- `hooks.py` — the three Claude Code hook handlers, including noise filtering and `<private>` stripping.
- `mcp.py` / `tools.py` — FastMCP stdio server and its 6 tool definitions.
- `dashboard.py` — a single self-contained HTML page; no build step, vanilla JS polling `/api/stats` and `/api/memories` every 5 seconds.

## Why one daemon owns the engine

You could let every process open the database directly — many tools do. lkhu doesn't, for three concrete reasons:

1. **FAISS lives in RAM.** The vector index is rebuilt from SQLite on open and after every archive/delete. If two processes each held an index while writing to SQLite, their indexes would silently diverge from the rows — recall would return ghosts.
2. **SQLite wants one writer.** Hooks fire mid-conversation, an MCP `remember` call arrives, the scheduler runs at 3 AM — concurrent writers from separate processes invite lock contention and lost updates. One owner serializes everything behind a single `RLock`.
3. **The scheduler must run exactly once.** Decay multiplies strength by 0.99 daily. Three processes each running decay would forget three times faster than designed.

So the daemon (`lkhu daemon`) is the one resident process that opens `LkhuEngine`, and `run_daemon()` exits quietly if a healthy daemon already answers `/health` — there is never more than one daemon. (One-off CLI data commands like `lkhu status` and `lkhu recall` currently open the engine directly for the duration of the command; the long-lived, concurrent entry points all go through the daemon.)

**Auto-launch:** no client makes you start the daemon by hand. `ensure_daemon_running()` checks `/health`, and on failure spawns `python -m lkhu daemon` fully detached (`start_new_session` on POSIX, `DETACHED_PROCESS | NEW_PROCESS_GROUP` on Windows), then polls health every 0.3 s for up to 20 s. The MCP server, the hooks, and `lkhu dashboard` all do this transparently.

### The HTTP API

Everything is JSON over loopback:

| Route | Body | Returns |
|---|---|---|
| `GET /health` | — | `{"ok": true, "memories": N}` |
| `GET /status` | — | Engine status (counts, kinds, strengths, decoder tiers) |
| `GET /` , `GET /dashboard` | — | Dashboard HTML |
| `GET /api/stats` | — | Dashboard stats + `data_dir` |
| `GET /api/memories?archived=0\|1` | — | Memory dump |
| `POST /recall` | `query`, `k` (5) | Decoded text + tier + sources |
| `POST /remember` | `content`, `kind` ("fact"), `session_id` | `{"id": ...}` |
| `POST /observe` | `content`, `session_id`, `strength` | `{"id": ...}` |
| `POST /recent` | `n` (10) | Recent memories |
| `POST /forget` | `query`, `confirm` (false) | Archive result |
| `POST /recall_session` | `session_id` | `{"text": ...}` |
| `POST /export` | `out_path` | `{"exported": N, "path": ...}` |

Full parameter details live in [api-reference.md](api-reference.md).

## Data flow: saving a memory

Say you finish a conversation turn and the `Stop` hook fires. Here is the complete path — note that **no step calls an LLM**:

1. **A client makes the call.** The Stop hook posts the cleaned last assistant reply to `/observe` (kind `turn`). Explicit saves via the `remember` MCP tool go to `/remember` instead (kind `explicit`, strength 1.3); `lkhu remember` reaches the same `engine.remember` in-process.
2. **Dedup check (auto-saves only).** Before inserting a `turn`, `engine.observe` checks for a near-identical memory in the *same session* (cosine ≥ `long_term.dedup_threshold`, default 0.95). If one exists, it reinforces that memory (× `dedup_reinforce`, default 1.02) and records the observation in the audit shadow and short-term bundle instead of writing a duplicate row. Explicit `remember` saves skip this.
4. **Encode.** Ollama's `snowflake-arctic-embed2` embeds the text into a 1024-dim **semantic scent**. A rule-based extractor (regex for file paths, a canonical-name table for programming languages — max 7 pairs, zero LLM) builds a **structural scent**: the bundle of `bind(K_key, embed(value))` for each pair. The final scent is `normalize(0.6·semantic + 0.4·structure)`.
5. **Store.** `LongTermVault.insert` writes the SQLite row — vector as a float32 BLOB, plus strength, timestamps, `session_id`, `kind`, and `audit_text`. SQLite is the single source of truth; FAISS gets the normalized vector keyed by the SQLite rowid.
6. **Accumulate.** `ShortTermBundle.add` sums the vector into the running session scent (an unnormalized accumulator, persisted as `.npy`).
7. **Shadow.** `AuditLog.append` writes the natural-language copy to `audit/YYYY-MM/DD.jsonl`. This is for your eyes and for recovery — it is never the search target.

## Data flow: recalling a memory

You type a prompt; the `UserPromptSubmit` hook posts it to `/recall`:

1. **Encode the query** exactly as in saving (same 0.6/0.4 mix).
2. **Candidate search.** FAISS returns the top `k×3` nearest vectors (inner product on normalized vectors = cosine similarity). An optional `recall.min_similarity` floor (default `0.0` = off; e.g. ~0.30 for arctic2) drops candidates that aren't actually close to the query.
3. **Re-rank — multiplicatively.** Similarity is the primary, query-conditional signal; strength and recency are **multiplicative modulators, not additive bonuses.** Each candidate scores `similarity × (0.6 + 0.2·(strength/max_strength) + 0.2·recency)`, where recency decays with a 7-day half-life (`0.5^(age_days/7)`). Because the modulator multiplies similarity rather than adding to it, a globally-strong but off-topic memory can no longer outrank a clearly more similar one — among comparably-similar items, strength merely breaks ties. Take the top `k`.
4. **Gated Hebbian reinforcement.** Every returned memory gets `access_count + 1` and a fresh `last_accessed_at`. The `strength × 1.05` boost (capped at 1.5) is **gated**: only candidates whose query similarity reaches `recall.reinforce_sim_threshold` (default 0.58) are strengthened, so low-relevance filler that merely survived the top-k cut does not accrue strength and pin itself at the cap. (`recall()` also takes a `reinforce` flag; the eval harness passes `False` for side-effect-free reads.)
5. **Synthesize.** The top-k vectors are bundled, weighted by score, into one composite scent.
6. **Decode** the composite into a few lines of text via the 3-tier decoder: Tier 1 quotes short audit excerpts (~70% of cases), Tier 2 unbinds key scents and matches values at cosine ≥ 0.7 (~25%), Tier 3 falls back to an LLM capped at 80 tokens (<5% — and if no LLM is wired in, a non-LLM extractive excerpt is used instead; the stock engine wires none, so a default install makes zero LLM calls even here).

Steps 1–5 are pure vector math. The only place an LLM can appear in the entire system is Tier 3.

## ⏰ Background lifecycle

The daemon embeds APScheduler (UTC, no OS cron):

- **Daily, 03:00 UTC** — decay (long-term ×0.99, short-term ×0.7) and session consolidation (strength-weighted bundling of the last 2 days, `min_session_size` 3, zero LLM calls; the summary's `audit_text` is just `"[session X: merged N items]"`).
- **Sunday, 03:30 UTC** — glymphatic cleanse: merge near-duplicates (cosine > 0.95), archive memories that are both weak (< 0.1) and old (> 30 days).

Both crons are configurable in YAML. The why behind these numbers — Ebbinghaus, Hebb, sleep consolidation — is in [neuroscience.md](neuroscience.md).

**Catch-up on startup.** A cron only fires while the daemon happens to be alive at the scheduled wall-clock time, so a laptop that sleeps through 03:00 would never decay or consolidate. To fix that, the daemon calls `engine.run_due_lifecycle()` on startup (state persisted in `data_dir/lifecycle_state.json`): it runs the daily job if more than 20 hours have elapsed since the last run, and the weekly cleanse if more than 7 days have. On a brand-new store it seeds the weekly clock instead of cleansing, so the first real cleanse lands a week out. The job is idempotent within its window.

## Thread safety

The daemon's HTTP server handles requests on worker threads, and APScheduler runs jobs on its own background thread, so the engine must tolerate concurrent access within the one owning process. `LongTermVault` serializes everything with `check_same_thread=False` plus an `RLock`, and FAISS is pinned to a single OpenMP thread (`faiss.omp_set_num_threads(1)`).

## Design notes: the OpenMP story

lkhu's HRR operations were originally built on `torchhd`. It worked — until faiss entered the same process. PyTorch and faiss each ship their own OpenMP runtime, and loading both triggered the infamous **OMP Error #179** (duplicate runtime) under multithreading.

The fix exploited a happy fact: HRR's bind and unbind are circular convolution and circular correlation, which are pointwise multiplication (and conjugate multiplication) in the frequency domain. That's a two-liner in numpy:

```python
def bind(a, b):          # circular convolution
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=dim)

def unbind(bound, key):  # circular correlation
    return np.fft.irfft(np.fft.rfft(bound) * np.conj(np.fft.rfft(key)), n=dim)
```

(Simplified — see `src/lkhu/core/vsa.py` for the real thing, including unitary vector generation and normalization.)

The numpy reimplementation produces identical results to torchhd for recovery rate and membership tests, drops the entire torch dependency tree from the install, and the OpenMP clash is gone by construction. Sometimes the best dependency is the one you delete.

## Related reading

- [vsa-explained.md](vsa-explained.md) — bind/unbind/bundle and why holographic vectors work
- [neuroscience.md](neuroscience.md) — the biological model behind decay, reinforcement, consolidation
- [storage.md](storage.md) — SQLite schema, FAISS index, on-disk file layout
- [auto-memory.md](auto-memory.md) — the hook pipeline and noise filtering in detail
- [api-reference.md](api-reference.md) — every CLI command, MCP tool, and HTTP route
- [faq.md](faq.md) — common questions
- Install guides: [macOS](installation/macos.md) · [Linux](installation/linux.md) · [Windows](installation/windows.md)
