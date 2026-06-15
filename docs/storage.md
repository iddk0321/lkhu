# Storage: Where Your Data Lives

Your memories are yours. lkhu keeps every byte on your machine — vectors, metadata, and the natural-language shadow — in a handful of plain files you can inspect, back up, and carry to another computer. This doc walks through each file, what's inside it, and the commands that manage them.

If you've ever wondered "what does lkhu actually know about me, and where is it?" — this page answers that in full.

## Quickstart: find and inspect your data

Paths are never hardcoded. They come from `platformdirs` via `src/lkhu/platform/paths.py`, following each OS's conventions:

| OS | Data root |
|----|-----------|
| macOS | `~/Library/Application Support/lkhu/` |
| Linux | `~/.local/share/lkhu/` |
| Windows | `%LOCALAPPDATA%\lkhu\` |

Override resolution order: explicit `base` argument (tests) → `LKHU_DATA` environment variable → the platformdirs default above. Config lives separately at the OS config dir (e.g. `~/.config/lkhu/config.yaml` on Linux) as `config.yaml`.

Three ways to look inside, no SQL required:

```bash
lkhu status      # totals, kind distribution, strength stats, codebook keys
lkhu dashboard   # live web UI — stats, lifecycle, sortable memory table
lkhu export      # dump every memory's natural-language text to JSONL
```

## The directory tree

```text
~/Library/Application Support/lkhu/        # (macOS example)
├── codebook.npy                  # Key-scent dictionary (15 keys × 1024-dim) — sacred
├── codebook.npy.meta.json        # Seed, key order, SHA-256 checksum
├── codebook.backup.npy           # Second codebook copy (+ its .meta.json)
├── memories.db                   # SQLite — metadata + scent BLOBs (source of truth)
├── short_term.npy                # Short-term accumulated scent (one 1024-dim vector)
├── lifecycle_state.json          # Last-run timestamps for the daily/weekly jobs (catch-up)
├── audit/                        # Natural-language shadow log
│   └── 2026-06/                  # Month directory
│       └── 11.jsonl              # Day file, append-only JSONL
├── backups/
│   ├── daily/  weekly/  monthly/ # `lkhu backup` writes timestamped dirs into daily/
└── logs/                         # Server / consolidation / cleanse logs
```

Two things live **outside** this tree on purpose:

- `~/Documents/lkhu_codebook.backup.npy` — the third codebook backup, kept in your Documents folder so it survives even if the data dir is wiped.
- `config.yaml` — in the OS config dir, per platform convention.

## How one memory is written

Storing a single memory touches **three places at once** (`core/engine.py::_store`):

```text
remember("User prefers pytest over unittest")
        │
        ├─ ① encode() → 1024-dim "scent" vector (semantic 0.6 + structure 0.4)
        │
        ├─ ② LongTermVault.insert()  → memories.db (SQLite BLOB) + in-memory FAISS index
        ├─ ③ ShortTermBundle.add()   → short-term accumulator (persisted to short_term.npy on close)
        └─ ④ AuditLog.append()       → audit/2026-06/11.jsonl (natural-language copy)
```

The split is deliberate: the **vector is for search**, the **natural language is for your eyes**. Recall never reads the audit log to find matches — it works entirely in vector space, then decodes results back to text. See [architecture.md](./architecture.md) for the full pipeline.

## memories.db — SQLite, the single source of truth

`core/long_term.py`. Every memory's metadata and its original scent vector, in one ordinary SQLite file you can open with any tool.

### Schema

```sql
CREATE TABLE memories (
    rowid            INTEGER PRIMARY KEY AUTOINCREMENT,  -- doubles as the FAISS id
    id               TEXT UNIQUE NOT NULL,               -- UUIDv7 (time-ordered)
    vector           BLOB NOT NULL,                      -- float32[1024] = 4096 bytes
    strength         REAL NOT NULL,                      -- 0–1.5, decay/reinforcement target
    created_at       TEXT NOT NULL,                      -- ISO 8601 timestamp
    last_accessed_at TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,         -- times recalled
    session_id       TEXT NOT NULL DEFAULT '',
    kind             TEXT NOT NULL DEFAULT 'turn',       -- turn | explicit | summary | merged
    audit_text       TEXT NOT NULL DEFAULT '',           -- natural-language shadow
    source_ids       TEXT NOT NULL DEFAULT '[]',         -- JSON: origins of summary/merged rows
    archived         INTEGER NOT NULL DEFAULT 0          -- 1 = excluded from search
);
-- Indexes: idx_session(session_id), idx_archived(archived)
```

### What the numbers mean

- **One scent = 4 KB.** `1024 floats × 4 bytes (float32)`. A million memories is roughly 4 GB of vectors — most users will never get near that.
- **`strength` tells a story.** An explicit `lkhu remember` starts at 1.3; one recall multiplies by 1.05 → 1.365. Daily decay multiplies by 0.99. The cap is 1.5.
- **The four kinds:** `turn` (auto-captured conversation), `explicit` (you asked to remember), `summary` (nightly consolidation), `merged` (near-duplicates fused by the weekly cleanse).

### How a vector becomes a BLOB

```python
# store: numpy float32 array → bytes
blob = np.ascontiguousarray(vec, dtype=np.float32).tobytes()   # 4096 bytes
# restore: bytes → numpy array
vec = np.frombuffer(blob, dtype=np.float32)                    # shape (1024,)
```

## FAISS — the search index (rebuilt, not trusted)

There is a reserved path for `vectors.faiss`, but in practice **the index is rebuilt in memory from SQLite every time the engine opens** (`_rebuild_index`), and again after any archive or delete. SQLite is the truth; FAISS is a disposable accelerator. A stale index file can never disagree with the database, because the database always wins.

- **Index type:** `IndexIDMap2(IndexFlatIP(1024))`. Scents are L2-normalized, so inner product equals cosine similarity. `IDMap2` tags each vector with its SQLite `rowid`, so a search hit maps straight back to its row.
- **Search flow:** top-K×3 candidates by the query scent → re-rank by `similarity × (0.6 + 0.2·(strength/max_strength) + 0.2·recency)` → top-K. Similarity is the query-conditional primary signal; strength and recency are *multiplicative modulators*, not additive bonuses (`core/recall.py`).
- **Concurrency:** the daemon owns the only engine instance; SQLite/FAISS access is serialized with an `RLock` (`check_same_thread=False`), and `faiss.omp_set_num_threads(1)` avoids OpenMP runtime clashes.

## codebook.npy — sacred, never regenerate

`core/codebook.py`. The key-scent dictionary: 15 named keys (`K_USER`, `K_TOPIC`, `K_FILE`, `K_LANGUAGE`, ...) each mapped to a unitary 1024-dim vector. Every structural scent in every memory was built by binding values to these keys — **lose the codebook and every stored vector becomes uninterpretable noise**. That's why it gets DNA-level protection:

- **Regeneration guard.** `Codebook.save()` raises `FileExistsError` rather than overwrite an existing codebook. `lkhu init` and `lkhu install` only generate one if none exists.
- **Checksum guard.** The sidecar `codebook.npy.meta.json` stores a SHA-256 checksum of the matrix bytes plus the key list; `load()` verifies both and refuses a corrupted or truncated file.
- **Triple backup.** Saved to `codebook.npy`, replicated to `codebook.backup.npy` in the data dir, and to `~/Documents/lkhu_codebook.backup.npy` — each with its own meta sidecar.

The file itself is tiny: a `(15, 1024)` float32 matrix, 61,568 bytes (15 × 1024 × 4 + numpy header).

```json
{
  "dim": 1024,
  "master_seed": 4004614247728964048,
  "keys": ["K_USER", "K_TIME", "K_TOPIC", "K_VALUE", "K_PRIORITY", "K_DECISION",
           "K_QUESTION", "K_ANSWER", "K_EMOTION", "K_SESSION", "K_FACT",
           "K_PREFERENCE", "K_PROJECT", "K_FILE", "K_LANGUAGE"],
  "checksum": "1ac26228…",
  "format": 1
}
```

Each key's vector is **deterministically derived** from `blake2b(master_seed::key_name)` seeding a unitary-vector generator. Two consequences: the same seed reproduces the entire codebook bit-for-bit, and adding a new key in the future can never disturb existing keys. The `master_seed` itself comes from `secrets.randbits(63)` at first init and is then fixed forever. More on why unitary vectors matter in [vsa-explained.md](./vsa-explained.md).

## audit/ — the natural-language shadow

`core/audit.py`. Every memory's original text, preserved as **append-only JSONL** at `audit/YYYY-MM/DD.jsonl` (one directory per month, one file per day). This is the "shadow" of hard rule 4: it exists for *your* visibility, debugging, and recovery — it is **never the primary search target**. Search happens in vector space; the audit text resurfaces only when results are decoded back to language (Tier-1 decode excerpts) or when you read it yourself.

A real line looks like this:

```json
{"id": "019e9b24-1fd8-7de3-abb2-ae6c9e835c18", "session_id": "real",
 "kind": "explicit", "audit_text": "The user's name is Donguk and they develop in Python on macOS",
 "created_at": "2026-06-06T04:14:56.471697+00:00"}
```

Because `audit_text` is plain UTF-8, it stores any language as-is. Example (multilingual support): a memory saved as `"사용자는 다크 모드를 선호한다"` (Korean for "the user prefers dark mode") lands in the JSONL byte-for-byte and embeds through the multilingual snowflake-arctic-embed2 model like any English text.

The audit log is also lkhu's **interchange format** — `lkhu export` and `lkhu import` move memories between machines through it (details below).

## short_term.npy — the accumulator

`core/short_term.py`. The hippocampal short-term buffer: a single unnormalized 1024-dim vector into which every new scent is summed. It decays fast (×0.7 per day, applied by the nightly job — effectively gone in 3 days), so only themes that keep recurring stay strong. (The nightly consolidation itself builds its session summaries from the SQLite rows, not from this vector.) One vector, one `.npy` file, saved on engine close.

## ⏳ How the data changes over time

A background scheduler (APScheduler, embedded in the `lkhu daemon` process — no OS cron) rewrites these files on a fixed rhythm. All times are **UTC**.

> **Catch-up on startup.** The cron only fires while the daemon is alive at the scheduled wall-clock time, so a laptop that sleeps through 03:00 would otherwise never decay or consolidate. To fix that, the daemon runs `engine.run_due_lifecycle()` on launch and replays any overdue job: it runs the daily job (decay + consolidation) if more than 20 h have elapsed since the last run, and the weekly cleanse if more than 7 days have. The last-run timestamps live in `lifecycle_state.json`. On a brand-new store it seeds the weekly clock without cleansing, so the first real cleanse lands a week out.

| Operation | When | What changes on disk |
|-----------|------|---------------------|
| **Decay** | Daily 03:00 UTC | Every `strength` ×0.99; short-term vector ×0.7 |
| **Consolidation** | Daily 03:00 UTC | Sessions with ≥3 memories (last 2 days) get one `kind='summary'` row (strength 1.2) — strength-weighted vector sum, 0 LLM calls |
| **Glymphatic cleanse** | Sunday 03:30 UTC | Pairs with cosine > 0.95 merge into a `kind='merged'` row; memories with strength < 0.1 *and* age > 30 days get `archived=1` |
| **Hebbian boost** | On every recall | Recalled rows: `strength` ×1.05 (cap 1.5), `access_count` +1, `last_accessed_at` updated |

> **Archive, not delete.** `lkhu forget` and the cleanse set `archived=1` — the row stays in SQLite with its `audit_text` intact, it's just removed from the search index. Nothing in the automatic lifecycle ever hard-deletes a memory. The biology behind these numbers is in [neuroscience.md](./neuroscience.md).

## Backup, export, import, reset

### Backup

```bash
lkhu backup
```

Copies `codebook.npy` and `memories.db` to `backups/daily/<UTC timestamp>/` inside the data dir. Cheap and instant — run it whenever you like. (The codebook additionally has its standing triple backup, independent of this command.)

### Export — your data in plain text

```bash
lkhu export                       # writes lkhu_export.jsonl
lkhu export --out my-memories.jsonl
```

Dumps the audit records — every memory's natural-language text plus id, kind, session, timestamp — into one JSONL file. This is your no-lock-in guarantee: everything lkhu knows, readable by `jq`, a text editor, or you.

### Import — move to another machine

```bash
lkhu import my-memories.jsonl
```

Reads each record and **re-observes** its `audit_text` under its original `session_id`. That word matters: import re-encodes the text with the *destination* machine's codebook and embedder, so memories transfer cleanly even though codebooks are machine-unique. Natural language is the portable layer; vectors are regenerated locally.

### Re-embed — after switching models

```bash
lkhu reembed --yes
```

Vectors are model-specific: a scent encoded by one embedder is meaningless to another, so recall breaks the moment you change `encoder.model`. `reembed` walks every memory (archived included), re-encodes its `audit_text` with the *current* embedder, and **replaces the stored vector in place** — metadata, `strength`, `access_count`, and the audit shadow are all preserved. The codebook stays untouched: it depends on the dimension (1024), not the model, so any other 1024-dim model reuses it. Run this once after the switch, then recall is back to normal.

### Reset — the nuclear option

```bash
lkhu reset --confirm
```

Deletes the codebook, its in-dir backup, `memories.db`, `vectors.faiss`, `short_term.npy`, the entire `audit/` directory, and all `.meta.json` sidecars. Without `--confirm` it refuses and exits with code 1. The Documents-folder codebook backup (`~/Documents/lkhu_codebook.backup.npy`) is intentionally left untouched — a last lifeline if you reset by mistake.

## Inspecting what's stored

**The dashboard** is the richest view. It's served by the daemon (default port `37700 + your-uid % 100`, e.g. `http://127.0.0.1:37701/`; override with `LKHU_DAEMON_PORT`):

```bash
lkhu dashboard            # starts the daemon if needed, opens your browser
lkhu dashboard --no-open  # just print the URL
```

You get stat cards (active / archived / summaries / merged sources), the lifecycle parameters live from config, strength and age distributions, and a sortable table of every memory — strength bar, kind badge, content excerpt, recall count, session. A "Show archived" checkbox reveals what forgetting has set aside.

**From the terminal:**

```bash
lkhu status                      # counts, kinds, avg/max strength, codebook keys, decoder tiers
lkhu recall "faiss locking" --k 5  # run a real recall, see tier + sources
lkhu doctor                      # checks Ollama, codebook integrity, daemon, integrations
```

**Raw access** is always an option — it's just SQLite and JSONL:

```bash
sqlite3 "$HOME/Library/Application Support/lkhu/memories.db" \
  "SELECT kind, strength, substr(audit_text, 1, 60) FROM memories ORDER BY strength DESC LIMIT 10;"
```

## Privacy guarantees

- **Nothing leaves your machine.** Embeddings run on local Ollama (`snowflake-arctic-embed2`); the entire save/recall/consolidate/decay/cleanse pipeline is local vector math with zero external API calls. No API key, no telemetry, no cloud.
- **The daemon binds to localhost.** `127.0.0.1` by default (`LKHU_DAEMON_HOST` to change); the dashboard and HTTP API are not exposed to your network.
- **`<private>` tags are honored.** Anything you wrap in `<private>...</private>` is stripped by the auto-memory hooks before saving — see [auto-memory.md](./auto-memory.md).
- **One honest caveat:** `memories.db` and `audit/` are unencrypted files under your user account. Anyone with access to your disk can read them — the same trust model as your shell history or browser profile. Use full-disk encryption (FileVault, BitLocker, LUKS) as you would for any personal data.

## At a glance

| File | Format | Role | Search target? |
|------|--------|------|----------------|
| `memories.db` | SQLite | Metadata + scent BLOBs — single source of truth | ✅ |
| FAISS index | In-memory | Fast cosine search, rebuilt from SQLite on open | ✅ (derived) |
| `codebook.npy` (+meta) | numpy + JSON | Key-scent dictionary — sacred, checksummed, triple-backed-up | — (interpretation key) |
| `audit/*.jsonl` | JSONL | Natural-language shadow — your eyes, export/import | ❌ (never searched) |
| `short_term.npy` | numpy | Short-term accumulator of recent scents | — |
| `lifecycle_state.json` | JSON | Last-run timestamps for daily/weekly jobs (catch-up on startup) | — |
| `backups/` | Copies | `lkhu backup` snapshots of codebook + db | — |

**The design in one sentence:** search, association, and forgetting all happen in vector space; natural language is kept only as the human-facing port — which is why lkhu barely ever calls an LLM, and why everything fits in a few files you fully own.

---

Next: [architecture.md](./architecture.md) for how these files are wired together, or the [FAQ](./faq.md) for common what-ifs (lost codebook, moving between machines).
