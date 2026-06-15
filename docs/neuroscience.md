# Neuroscience-inspired design

Most memory tools for AI agents are **append-only**: every observation is written down and
kept forever. That sounds safe, but it isn't how memory actually works — and it's why those
systems slowly fill with stale, redundant, low-value text that you have to pay to search
through on every query.

lkhu takes the opposite bet. It borrows the **shape** of human memory — fast working memory,
slow consolidation, reinforcement, and active forgetting — and implements each stage with
local vector math instead of language. The result is a store that gets *sharper* over time
because the noise is allowed to fade.

> **Honest framing:** this is *inspiration, not simulation.* lkhu does not model neurons,
> spikes, or biochemistry. It takes a handful of well-known principles from memory research
> and maps them onto concrete, testable code. The brain is the metaphor; numpy is the
> machinery.

## See it before you read about it

You don't need to understand any of the theory below to use lkhu. Install it, talk to Claude
Code, and the lifecycle runs on its own.

```bash
# Install the CLI (lkhu is not on PyPI yet)
pipx install git+https://github.com/iddk0321/lkhu
# (once published to PyPI: pipx install lkhu)

# Generate the codebook + register the Claude Desktop MCP server
lkhu install

# Add the Claude Code plugin (the lkhu command must be on PATH first)
claude plugin marketplace add iddk0321/lkhu
claude plugin install lkhu@lkhu
```

Then open the live dashboard and watch memories gain and lose strength in real time:

```bash
lkhu dashboard
```

The dashboard (served by the daemon at `http://127.0.0.1:37701/` — the exact port is per-user,
`37700 + uid % 100`, overridable via `LKHU_DAEMON_PORT`) has a
**Lifecycle panel** showing every parameter discussed on this page — daily decay, recall
boost, max strength, the consolidation and cleanse schedules, and the merge threshold — pulled
straight from the running config. The rest of this document explains what those numbers mean.

## The map: brain mechanism to lkhu component

Each row below is a real mechanism from memory research, the lkhu component that echoes it,
and the exact parameter that governs it.

| Brain mechanism | lkhu component | Parameter / behavior |
|---|---|---|
| Working memory (prefrontal cortex) | `WorkingMemory` | RAM deque, max 50 turns, 30-min idle-flush rule |
| Hippocampal buffer (short-term index) | `ShortTermBundle` | single accumulated scent, decays ×0.7/day |
| Long-term cortical storage | `LongTermVault` | SQLite (source of truth) + FAISS index |
| Hebbian plasticity ("fire together, wire together") | `RecallEngine` | strength ×1.05 on recall (gated to genuine hits, sim ≥ 0.58), capped at 1.5 |
| Ebbinghaus forgetting curve | `DecayEngine` | strength ×0.99 every day |
| Sleep / systems consolidation | `Consolidator` | nightly strength-weighted bundle, 0 LLM calls |
| Glymphatic clearance (sleep "rinse") | `GlymphaticCleaner` | weekly merge >0.95 similarity, archive weak+old |
| Language production (Broca's area) | `Decoder` | 3-tier decode, LLM only in Tier-3 (<5%) |

Source: `src/lkhu/core/` — `working_memory.py`, `short_term.py`, `long_term.py`,
`recall.py`, `decay.py`, `consolidator.py`, `glymphatic.py`, `decoder.py`.

## Working memory: the prefrontal buffer

Human working memory holds a handful of items for a short while — enough to follow a
conversation, not enough to be a permanent record. lkhu's `WorkingMemory` is a RAM-resident
deque capped at **50 turns** (oldest evicted first) with a **30-minute** idle-flush rule —
`working_memory.max_turns` and `flush_idle_minutes` in the config. It is volatile by design
and never the long-term store.

The same "not every passing remark deserves to be permanent" instinct governs what enters the
store from a live session. The auto-memory hooks strip system blocks, code/tool dumps, and
`<private>` regions before saving anything; inputs under 8 characters are skipped entirely;
and the assistant's closing prose is saved at a deliberately low strength (**0.6**, capped at
280 chars) — so unless it later proves useful enough to be recalled and reinforced, daily
decay quietly clears it.

## Hippocampal consolidation: from fast buffer to slow store

In the brain, the hippocampus captures experiences quickly, then — largely during sleep —
relevant patterns are gradually transferred into cortex for long-term storage. The
fast-but-fragile buffer and the slow-but-durable store are different systems.

lkhu mirrors this split:

- **Short-term** lives in `ShortTermBundle` — a single accumulated scent vector that decays
  fast (**×0.7 per day**, ~3-day retention). It captures the gist of recent activity.
- **Long-term** lives in `LongTermVault` — SQLite as the single source of truth, with a FAISS
  index rebuilt from it for cosine search.
- **Consolidation** runs on a schedule inside the daemon's APScheduler (`Consolidator`,
  nightly at **03:00 UTC**). For each session over the last 2 days with at least
  `min_session_size = 3` memories, it computes a **strength-weighted bundle** of that session's
  turns into a single `summary` memory (strength `1.2`, with an `audit_text` like
  `[session X: merged N items]`).

The scheduler only fires while the daemon is awake at the scheduled wall-clock time, so on a
laptop that sleeps through 03:00 the nightly job used to be silently skipped. The daemon now
closes that gap with a **startup catch-up**: on boot it calls `engine.run_due_lifecycle()`,
which runs the daily job (decay + consolidation) if more than 20 h have passed since it last
ran, and the weekly job (glymphatic cleanse) if more than 7 days have. The last-run timestamps
live in `lifecycle_state.json`, and a brand-new store seeds the weekly clock without cleansing
(there is nothing to clean yet). So the lifecycle actually happens on real, intermittently-on
machines — not only on a server that stays up at 03:00.

Here's the part that matters: consolidation uses **zero LLM calls.** Where an append-only
system would call out to a language model to "summarize this session," lkhu just adds vectors
together, weighted by strength. Concepts that recurred or were reinforced dominate the sum;
one-off noise washes out. The summary is a vector, not a paragraph — cheap to compute, cheap
to store, and produced entirely on your machine.

## Hebbian plasticity: what you recall, you reinforce

The classic Hebbian slogan is *"cells that fire together wire together."* Synapses that are
used repeatedly strengthen; unused ones weaken. Recall itself is an act of reinforcement —
this is the basis of the **spaced-repetition** effect, where revisiting a fact makes it stick.

lkhu's `RecallEngine` does exactly this. Every time a memory surfaces in the top-K results of
a recall, it gets a Hebbian update:

- `access_count += 1`
- `last_accessed_at = now`
- `strength = min(1.5, strength × 1.05)` — a **5% boost per recall**, capped at **1.5**

**The strength boost is gated, the access count is not.** Surfacing in the top-K is necessary
but not sufficient: only candidates whose query similarity clears `reinforce_sim_threshold`
(default **0.58**) get the ×1.05 strength bump. A filler result that merely ranked top-K by
being the least-bad match still gets `access_count += 1`, but its strength is left alone. This
matters because Hebbian reinforcement is a rich-get-richer loop — without the gate, a piece of
noise that keeps getting dragged into top-K would accrue strength on every unrelated query and
eventually pin itself at the cap, crowding out real memories. The gate enforces the actual
Hebbian rule ("fire *together*"): you only wire two things together when they genuinely
co-activate, not when one happens to be nearby. (Recall can also be run side-effect-free with
`reinforce=False`, which the eval harness uses so that *measuring* recall never mutates
strengths.)

So a memory you keep genuinely matching is continually pushed back up the strength curve,
outrunning the daily decay below. A memory nobody ever recalls — or that only ever shows up as
weak filler — just keeps fading. Usefulness, not recency or verbosity, decides what survives.

Recall ranking itself is **multiplicative, not additive**: `score = similarity × (0.6 +
0.2·(strength/max_strength) + 0.2·recency)`. Similarity is the primary, query-conditional
signal; strength and recency are *modulators* that gently lift or demote it, not bonuses added
on top. The distinction is load-bearing: an additive strength term is query-independent, so a
globally-strong but off-topic memory could outrank a clearly more similar one — which is
exactly the failure that hurt recall before. With multiplication, a strong recent memory still
outranks a marginally-more-similar ignored one, but a strong *off-topic* one cannot beat a
clearly-better match. (Recency here has a 7-day half-life.) See
[VSA "scent" explained](vsa-explained.md) for how the similarity term is computed.

## The Ebbinghaus forgetting curve: decay as a feature

In 1885 Hermann Ebbinghaus measured his own memory and found that retention drops off roughly
exponentially over time unless a memory is refreshed. lkhu's `DecayEngine` applies that curve
directly: **every memory's strength is multiplied by 0.99 once per day** (the daily job, again
at 03:00 UTC).

That's a gentle slope — about a 1% loss per day for an untouched memory — and a single gated
Hebbian ×1.05 on recall more than reverses one day of decay. The two forces together produce
the behavior you want:

- A memory that *genuinely* matches a query even occasionally stays strong indefinitely.
- A memory never recalled — or only ever surfaced as weak, below-threshold filler — drifts
  steadily toward zero strength, because the gate denies it the reinforcement that would save it.

Decay alone doesn't delete anything; it just lowers strength. Deletion (well, archival) is the
glymphatic stage's job, and it only acts on memories that are *both* weak *and* old.

## Glymphatic clearance: the weekly rinse

While you sleep, the brain's glymphatic system flushes metabolic waste that builds up during
waking hours. It's literal cleanup — and memory benefits from the equivalent: pruning
redundancy and clearing what's no longer load-bearing.

lkhu's `GlymphaticCleaner` runs weekly (**Sunday 03:30 UTC**) and does two things:

1. **Merge near-duplicates.** Any pair of memories with cosine similarity **> 0.95** is fused
   into one `merged` memory: the normalized sum of the two vectors, taking the higher of the
   two strengths. The originals are archived. Saying the same thing five different ways
   collapses into one strong memory instead of five weak ones.
2. **Archive the weak and old.** Memories with **strength < 0.1 AND age > 30 days** are
   archived. Both conditions are required — a weak-but-recent memory gets a chance to be
   reinforced, and a strong-but-old one stays.

Crucially, "forget" here means **archive, not delete.** The row stays in SQLite, its
`audit_text` intact; it just leaves the active FAISS index and stops showing up in recall. You
can still see archived rows in the dashboard (toggle "Show archived") and in `lkhu export`.
True deletion only happens via the explicit `lkhu reset --confirm`.

## Why forgetting makes memory *better*

This is the counterintuitive core of the design, so it's worth stating plainly: **a memory
system that never forgets is a worse memory system.**

The job of memory isn't to maximize what's stored — it's to maximize **signal relative to
noise** at the moment of recall. Every memory you keep is something a search has to consider
and rank against. Keep everything, and the useful memories are buried under a growing pile of
trivia, dead ends, and five-different-phrasings-of-the-same-fact. Recall quality degrades as
the store grows.

lkhu's lifecycle is a continuous signal-vs-noise filter:

- **Decay** lets unused memories fade, so they stop competing for ranking slots.
- **Hebbian reinforcement** pushes genuinely useful memories *up* against that decay — and
  because the boost is *gated* by similarity, noise that merely shows up in results is never
  promoted, so it keeps fading instead of being kept alive by accident.
- **Consolidation** distills a noisy session into the concepts that actually recurred.
- **Glymphatic merge + archive** removes redundancy and clears the weak-and-stale.

The memories that survive are the ones that proved useful by being recalled, or important
enough to be reinforced. That's a feature you cannot get by adding more storage — only by
forgetting well.

## Contrast: append-only memory systems

Tools like [claude-mem](https://github.com/thedotmack/claude-mem) — a pioneering, popular
project that directly inspired lkhu's hooks-based UX, plugin distribution, `<private>` tags,
and web viewer — take an append-only, natural-language approach: capture observations during a
session, compress them with an LLM into text summaries, and store them for full-text and
semantic search. It's a strong design with real advantages, and the comparison below is meant
to be honest about both directions.

| | Append-only NL memory (e.g. claude-mem) | lkhu |
|---|---|---|
| What's stored | Natural-language summaries (text) | 1024-dim latent vectors ("scents") |
| LLM calls in pipeline | Yes — summarization on session end | **Zero** (only Tier-3 decode fallback, <5%) |
| Growth over time | Monotonic — append, keep forever | Self-pruning — decay, merge, archive |
| Exact past wording | **Preserved and searchable** (FTS5) | Approximate; `audit_text` shadow only |
| Citations to source | **Yes** | Not a primary feature |
| Cost per stored item | API tokens to summarize | Local embedding only — no API cost |
| Stack | Node/Bun + SQLite/FTS5 + Chroma | Python only — numpy, faiss, SQLite |

**Where append-only wins:** if you need to retrieve the *exact* text of something said three
weeks ago, with citations back to the originating session, a full-text store is the right tool.
claude-mem's FTS5 + Chroma hybrid search, observation detail, citations, and multi-IDE
ecosystem are genuinely strong, and lkhu does not try to replace them.

**Where lkhu wins:** if you want memory that stays small and sharp on its own, costs nothing
per item (no API key, no data leaving your machine), and works in any language without
per-language keyword lists — lkhu's biomimetic, forget-by-default design is built for that.

The full feature-by-feature comparison lives in [lkhu vs claude-mem](comparison.md).

lkhu keeps a natural-language `audit_text` shadow copy for every memory, but only for your
visibility, debugging, and Tier-1 decode excerpts — it is **never the primary search target.**
Search and ranking happen entirely in vector space.

## The 3-tier decoder: language only at the very end

When a memory finally needs to come back *as words*, the `Decoder` tries the cheapest path
first. This is the only place an LLM can appear anywhere in the pipeline, and it's the
exception, not the rule.

| Tier | Method | LLM? | Typical share |
|---|---|---|---|
| 1 | Join short `audit_text` excerpts (each < 150 chars) | No | ~70% |
| 2 | `unbind` the composite scent against codebook keys, match audit vocabulary (≥2 keys, cosine ≥ 0.7) | No | ~25% |
| 3 | LLM fallback, capped at 80 tokens (or non-LLM extractive excerpt if no LLM is wired in) | Sometimes | **<5%** |

So roughly **95% of decodes touch no language model at all.** Even Tier 3 has a hard 80-token
ceiling and degrades gracefully to an extractive excerpt when no LLM callable is provided.

## Core philosophy

> Just as a person remembers well without knowing how their brain works, lkhu processes
> everything internally as scents — fading, reinforcing, consolidating, and clearing on its
> own — and hands Claude Code a single line of natural language only at the moment it's needed.

## Keep reading

- [Architecture](architecture.md) — how the daemon, hooks, MCP, and CLI fit together
- [VSA "scent" explained](vsa-explained.md) — the intuition behind bind / bundle / unbind
- [Storage layout](storage.md) — where memories live and in what form
- [Auto memory (hooks + daemon)](auto-memory.md) — how recall and save happen without typing
- [lkhu vs claude-mem](comparison.md) — the full feature-by-feature comparison
- [API reference](api-reference.md) — CLI commands and MCP tools
- [FAQ](faq.md)
