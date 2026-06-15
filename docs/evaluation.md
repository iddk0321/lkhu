# Evaluation

> "Is the memory actually helpful?" should be a number, not a vibe. `lkhu eval` makes it one.

lkhu ships a small evaluation harness that scores recall quality, noise robustness,
multilingual recall, and the save filter against a fixed, hand-labeled gold corpus — all in a
throwaway data directory, so your real memories are never touched.

## Running it

```bash
lkhu eval                          # full run (needs Ollama + snowflake-arctic-embed2)
lkhu eval --offline                # filter metrics only, no Ollama
lkhu eval --k 5                    # top-k for recall metrics (default 5)
lkhu eval --model bge-m3           # benchmark another embedder (dimension auto-detected)
lkhu eval --out score.json         # also write the raw scorecard as JSON
```

Example output (default embedder, snowflake-arctic-embed2):

```
lkhu eval — ollama (k=5)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric                                ┃       Value ┃ Good when ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ save filter — signal kept             │        100% │    → 100% │
│ save filter — hard noise dropped      │        100% │    → 100% │
│ recall — hit@k (cold → warm)          │ 100% → 100% │    → 100% │
│ recall — noise rate (cold → warm)     │ 0.23 → 0.23 │       → 0 │
│ recall — cross-lingual hit@k (warm)   │        100% │    → 100% │
│ Hebbian — noise saturated (ungated)   │           1 │     lower │
│ Hebbian — noise saturated (gated)     │           0 │       → 0 │
└───────────────────────────────────────┴─────────────┴───────────┘
```

## What the metrics mean

- **hit@k** — fraction of queries where a genuinely relevant memory appears in the top *k*.
- **noise rate** — fraction of returned context that is labeled chatter (lower is better).
- **cross-lingual hit@k** — hit@k restricted to queries whose answer is stored in a *different*
  language (e.g. a Korean memory recalled by an English query). Exercises the multilingual
  embedding with no per-language keyword lists.
- **Hebbian noise saturated** — how many chatter memories drift up to the strength cap. The
  similarity gate (`recall.reinforce_sim_threshold`) should keep this at 0.
- **save filter** — does the hook keep durable signal and drop catchable junk (URLs, emoji,
  acknowledgements)?

### Cold vs. warm

Recall is reported twice. **Cold** is a brand-new store (your first-ever session). **Warm** is
after a simulated stretch of real use: the harness repeatedly recalls real topics (using
*held-out paraphrases*, so the test queries stay unseen) and applies daily decay. Memories you
keep asking about get reinforced; never-asked chatter decays out of the way. The gap between cold
and warm is lkhu's biomimetic lifecycle doing its job — for example, noise rate drops from cold
to warm because un-reinforced chatter loses strength and is demoted in ranking.

## How recall ranking works (and why)

Candidates come from FAISS by cosine similarity, then are re-ranked. **Similarity is the
primary, query-conditional signal.** Strength and recency are *multiplicative* modulators, not
additive bonuses:

```
score = similarity × (sim_weight + strength_weight·(strength/max) + recency_weight·recency)
```

An additive strength term is query-independent, so a globally-strong but off-topic memory could
outrank a clearly more relevant one — which measurably hurt recall. Multiplicative coupling lets
strength gently demote weak/stale memories and break ties among comparably-similar items, while
never flipping a clearly-higher-similarity result.

An optional `recall.min_similarity` cosine floor (default `0.0`, i.e. off) can drop
low-similarity candidates before re-ranking, trimming the noise rate at the cost of occasionally
losing a marginal hit. The right value is embedder-specific (around `0.30` for
snowflake-arctic-embed2), so it ships disabled.

## Determinism

The scorecard is reproducible: measurement uses a side-effect-free recall
(`reinforce=False`, so scoring doesn't mutate strengths), and the eval codebook uses a fixed
seed. Two back-to-back runs produce identical numbers — essential for trusting a before/after
comparison while tuning.

## Tuning loop

The harness is built to be iterated against. The pattern (see the repo's recall-quality work):

1. `lkhu eval` for a baseline.
2. Pick the worst metric; form one hypothesis (e.g. adjust `recall.reinforce_sim_threshold`,
   `recall.min_similarity`, the rerank weights, or `long_term.dedup_threshold` in
   `config/defaults.yaml`).
3. Change one thing; keep `ruff` + `pytest` green.
4. `lkhu eval` again; keep the change only if the target metric improved without regressing
   others.

Two honest caveats:

- **hit@k and cross-lingual hit@k are largely bounded by the embedding model.** Once a relevant
  memory simply isn't near the query in snowflake-arctic-embed2's space, no amount of lkhu-side re-ranking pulls
  it into the top *k*. Don't chase these past the embedder's ceiling by making the corpus easier
  — that's gaming the benchmark, not improving the product.
- **Grow the corpus** in `src/lkhu/eval/corpus.py` with new realistic cases (especially
  multilingual and long chatter) rather than special-casing strings. A bigger, harder test set
  is the honest way to raise confidence.
