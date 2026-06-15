# VSA explained: how lkhu remembers with "scents"

lkhu stores your memories as 1024-dimensional vectors instead of sentences. That raises an obvious question: if a memory is just a list of numbers, how do you put *structure* into it — "the language is **Python**", "the file is **tests/conftest.py**" — and how do you ever get that structure back out?

The answer is **VSA** (Vector Symbolic Architecture), specifically Tony Plate's **HRR** (Holographic Reduced Representation). This page walks you through the whole trick with nothing but high-school trig intuition and a few lines of numpy. You don't need a PhD. You don't even need lkhu installed — just numpy.

## Try it first

Paste this into any Python session. It's the entire mathematical core of lkhu in ~30 lines:

```python
import numpy as np

dim = 1024
rng = np.random.default_rng(42)

def normalize(v):
    return v / np.linalg.norm(v)

def bind(a, b):        # circular convolution, via FFT
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=dim)

def unbind(bound, key):  # circular correlation = conjugate multiply
    return np.fft.irfft(np.fft.rfft(bound) * np.conj(np.fft.rfft(key)), n=dim)

def unitary(rng, dim):   # a "key" vector: magnitude 1 at every frequency
    phase = rng.uniform(-np.pi, np.pi, size=dim // 2 + 1)
    spectrum = np.exp(1j * phase)
    spectrum[0] = 1.0    # DC component must be real
    spectrum[-1] = 1.0   # Nyquist too (even dim)
    return normalize(np.fft.irfft(spectrum, n=dim))

key   = unitary(rng, dim)
value = normalize(rng.standard_normal(dim))

box = bind(key, value)            # lock the value with the key
got = unbind(box, key)            # open it again

print(np.dot(normalize(got), value))   # ≈ 1.0  — perfect recovery
print(np.dot(normalize(box), key))     # ≈ -0.01 — the box reveals nothing
print(np.dot(normalize(box), value))   # ≈ 0.04  — about the key or the value
```

A value goes in, looks like random noise from the outside, and comes back out *exactly* when you hold the key. That's the whole game. The rest of this page explains why it works, and how lkhu builds memories on top of it.

The production versions of these functions live in [`src/lkhu/core/vsa.py`](https://github.com/iddk0321/lkhu/blob/main/src/lkhu/core/vsa.py) — they're the same one-liners with `float32` casting and docstrings.

## What's a "scent"?

A **scent** is lkhu's word for a 1024-dimensional `float32` vector with length (L2 norm) 1. Something like `[0.12, -0.45, 0.88, ..., -0.71]`.

Why 1024 dimensions? Because high-dimensional space is *roomy*. Two randomly chosen unit vectors in 1024 dimensions are almost always nearly perpendicular:

```python
a = normalize(rng.standard_normal(1024))
b = normalize(rng.standard_normal(1024))
np.dot(a, b)   # ≈ -0.06 — essentially orthogonal
```

This near-orthogonality is the foundation everything else rests on. It means every random vector is effectively a **unique signal**: similarity with anything unrelated hovers around zero, so any similarity you *do* measure is meaningful.

Scents come from two places in lkhu:

| Source | What it gives you |
|---|---|
| **snowflake-arctic-embed2 embeddings** (via Ollama, 1024-dim) | "Similar things smell similar" — *dog* and *cat* land close together, *dog* and *car* land far apart. Multilingual by construction. |
| **The codebook** (unitary key vectors) | Fixed, random, mutually near-orthogonal "keys" like `K_LANGUAGE`, `K_FILE` — structural slots, not meanings. |

Comparing two scents is one dot product — **cosine similarity** — and it costs microseconds. No LLM is involved at any point.

## The three operations

HRR gives you exactly three operations, and lkhu's whole memory pipeline (save, recall, consolidate, cleanse) is built from them.

### bind — lock two scents together

```python
bound = np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=dim)
```

This is **circular convolution**, computed the fast way: FFT both vectors, multiply elementwise in the frequency domain, inverse-FFT back. Three properties matter:

- **Same size in, same size out.** Binding two 1024-dim scents gives a 1024-dim scent — structure never inflates the representation.
- **The result resembles neither input.** `bound` is near-orthogonal to both `a` and `b` (we measured ≈ -0.01 and ≈ 0.04 above). The pairing is hidden.
- **It's reversible** — that's what unbind is for.

Think of it as a locked box: `bind(key, value)` puts the value in a box that only this key opens.

### bundle — pour scents into one bottle

```python
bundled = normalize(np.sum(vectors, axis=0))
```

Bundle is just elementwise addition followed by normalization. The magic is that the sum **preserves membership**: each ingredient stays detectable by cosine similarity.

```python
dog, cat, car = (normalize(rng.standard_normal(dim)) for _ in range(3))
zoo = normalize(dog + cat)

np.dot(zoo, dog)   # ≈ 0.70 — dog is in there
np.dot(zoo, car)   # ≈ 0.01 — car is not
```

Each member's similarity drops as you add more (roughly `1/√n` for n members), but against a near-zero noise floor, "present" and "absent" stay easy to tell apart for quite a few members. This is why lkhu caps structural key-value pairs at 7 per memory — beyond that, the bottle gets crowded.

### unbind — open the box with the key

```python
recovered = np.fft.irfft(np.fft.rfft(bound) * np.conj(np.fft.rfft(key)), n=dim)
```

This is **circular correlation**: same FFT trick, but you multiply by the *complex conjugate* of the key's spectrum. `unbind(bind(key, value), key) ≈ value`.

And because bind distributes over addition, you can unbind straight through a bundle:

```text
unbind(bind(K1, v1) + bind(K2, v2), K1)  ≈  v1 + noise
```

The `bind(K2, v2)` term, unbound with the *wrong* key, turns into more near-orthogonal noise — it doesn't corrupt the answer, it just sits quietly at cosine ≈ 0.

## Why unitary keys make recovery ≈ 1.0

Here's the one piece of actual math on this page, and it's three lines.

In the frequency domain, bind is multiplication: `C(f) = A(f) · B(f)`. Unbinding with key A multiplies by the conjugate:

```text
C(f) · conj(A(f)) = A(f) · conj(A(f)) · B(f) = |A(f)|² · B(f)
```

So unbinding recovers `B` scaled by `|A(f)|²` at every frequency. If the key's spectrum has **magnitude exactly 1 at every frequency** — that's what **unitary** means — then `|A(f)|² = 1` everywhere and you get `B` back *exactly*. A unitary key only rotates phases, and the conjugate rotates them all back.

A random (non-unitary) key has lumpy magnitudes, so each frequency of `B` comes back scaled differently — recognizable, but distorted:

```python
# unitary key:  cosine(recovered, value) ≈ 1.0   (exact)
# random key:   cosine(recovered, value) ≈ 0.66  (recognizable but noisy)
```

That's why `unitary_vector()` in `vsa.py` builds keys directly in the frequency domain: magnitude 1, random phase in [-π, π], with the DC and Nyquist components forced to 1.0 so the inverse FFT yields a real-valued vector.

### The codebook: keys that never change

lkhu's keys live in the **codebook** (`codebook.npy`) — 15 unitary vectors named `K_USER`, `K_TIME`, `K_TOPIC`, `K_VALUE`, `K_PRIORITY`, `K_DECISION`, `K_QUESTION`, `K_ANSWER`, `K_EMOTION`, `K_SESSION`, `K_FACT`, `K_PREFERENCE`, `K_PROJECT`, `K_FILE`, `K_LANGUAGE`.

Each key is derived deterministically from a master seed plus its name (via `blake2b`), so adding a new key never disturbs existing ones. But the master seed itself is random and **irreplaceable**: every stored memory was bound with these exact keys, so regenerating the codebook would render every memory unreadable. lkhu refuses to overwrite an existing codebook, verifies a SHA-256 checksum on load, and keeps three backups (data dir, a `.backup` sibling, and `~/Documents/lkhu_codebook.backup.npy`). See [storage.md](storage.md) for the full layout.

## How the encoder composes a memory

When lkhu stores a memory, the encoder (`src/lkhu/core/encoder.py`) builds the scent from two parts:

```text
scent = normalize( 0.6 · semantic  +  0.4 · structure )
```

- **Semantic scent (weight 0.6):** the snowflake-arctic-embed2 embedding of the whole text. This carries the fuzzy "what is this about" signal and dominates what FAISS sees (FAISS indexes the final composite scent, in which this part has the larger weight).
- **Structure scent (weight 0.4):** rule-extracted key-value pairs, each bound to its codebook key and bundled: `bundle([bind(K, embed(value)) for K, value in pairs])`.

The key-value extraction is deliberately boring — pure rules, zero LLM calls, capped at 7 pairs:

| Key | How it's found |
|---|---|
| `K_FILE` | A regex for file paths with common source extensions (`.py`, `.ts`, `.rs`, `.yaml`, ...) |
| `K_LANGUAGE` | A 15-entry canonical-name table (python, rust, go, c++, ...), matched longest-first so "c" doesn't shadow "c++" |

If no pairs are found, the semantic scent is used alone. Either way the result is a single unit-norm 1024-dim vector — one memory, one scent.

## A tiny worked example

Let's encode this sentence and then read structure back out of it. (Numbers below are from a real run using lkhu's offline `HashingEmbedder`; snowflake-arctic-embed2 gives the same shape of result with better semantics.)

```python
from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Encoder, HashingEmbedder, extract_kv
from lkhu.core import vsa

cb  = Codebook.generate(["K_FILE", "K_LANGUAGE"], dim=1024, seed=7)
enc = Encoder(HashingEmbedder(1024), cb, auto_discovery=False)

text = "Set up pytest fixtures in tests/conftest.py for the Python encoder"
extract_kv(text)
# [('K_FILE', 'tests/conftest.py'), ('K_LANGUAGE', 'Python')]

scent = enc.encode(text)   # one unit vector: 0.6·semantic + 0.4·structure
```

Now probe the *composite* scent with a key, and compare the result against a small vocabulary:

```python
emb = HashingEmbedder(1024)
probe = vsa.unbind(scent, cb["K_LANGUAGE"])

vsa.cosine(probe, emb.embed("Python"))       # ≈ 0.38  ← winner
vsa.cosine(probe, emb.embed("Rust"))         # ≈ 0.05
vsa.cosine(probe, emb.embed("JavaScript"))   # ≈ 0.04
vsa.cosine(probe, emb.embed("Go"))           # ≈ 0.01
```

Notice the recovered value isn't at cosine 1.0 anymore — the structure part only carries weight 0.4, it shares a bundle with the `K_FILE` pair, and the semantic part adds its own noise. But that's fine: every wrong answer sits near zero, so **the right answer wins by an order of magnitude**. You don't need exact recovery, just a clear nearest neighbor. Plate called this step **clean-up memory**: snap a noisy vector to the closest known item.

Probing with the other key works the same way:

```python
probe = vsa.unbind(scent, cb["K_FILE"])
vsa.cosine(probe, emb.embed("tests/conftest.py"))  # ≈ 0.39  ← winner
vsa.cosine(probe, emb.embed("Python"))             # ≈ -0.04
```

Because snowflake-arctic-embed2 is multilingual, the semantic side needs no per-language handling. For example (Korean shown purely as a multilingual illustration), embedding "인코더를 파이썬으로 작성했다" lands near "wrote the encoder in Python" in the same vector space — no keyword lists, no translation step.

## How lkhu reads memories back

This is exactly what the 3-tier decoder does at recall time. After FAISS finds the top memories and they're fused into one composite scent, the decoder tries the cheapest path first:

1. **Tier 1 — audit excerpt:** if the matched memories have short `audit_text` shadows (under 150 chars), just quote them. Zero math, zero LLM. (~70% of decodes.)
2. **Tier 2 — unbind probe:** unbind the composite with each of the high-value codebook keys (`K_TOPIC`, `K_DECISION`, `K_LANGUAGE`, `K_FILE`, ...), snap each result to the nearest known vocabulary item, and accept matches at cosine ≥ 0.7. If at least 2 keys match, emit something like `LANGUAGE=Python, FILE=tests/conftest.py`. Still zero LLM. (~25%.)
3. **Tier 3 — LLM fallback:** only when both fail, ask an LLM — capped at 80 tokens, and under 5% of decodes in practice. (If no LLM is wired in at all, lkhu degrades to a plain audit excerpt instead — still zero LLM calls.)

That's the punchline of the whole architecture: because structure is *recoverable from the vectors themselves*, lkhu almost never has to pay an LLM to turn memories back into words. See [architecture.md](architecture.md) for the full pipeline.

## Where this comes from

None of this math is new — lkhu just applies it to agent memory:

- **Tony Plate, *Holographic Reduced Representation: Distributed Representation for Cognitive Structures* (CSLI Publications, 2003)** — the definitive HRR reference: circular convolution binding, superposition bundling, clean-up memory, capacity analysis. Plate's 1995 IEEE paper covers the core ideas in shorter form.
- VSA is a family; HRR is the member that works with plain real-valued vectors and FFTs, which is why lkhu's entire symbolic layer is ~150 lines of numpy.

One implementation note: lkhu originally used the `torchhd` library for these operations, but torch and faiss each link their own OpenMP runtime and crash when combined (OMP Error #179). Since HRR bind/unbind are *literally* the FFT one-liners you saw above, the torch dependency was dropped — same math, ~1 GB smaller install, thread-safe on all three OSes.

## Further reading

- [Architecture](architecture.md) — where these operations sit in the save/recall/consolidate/cleanse pipeline
- [Neuroscience-based design](neuroscience.md) — why decay, reinforcement, and consolidation mimic the brain
- [Storage layout](storage.md) — `codebook.npy`, FAISS index, SQLite, and the audit shadow log
- [FAQ](faq.md) — common questions, including "what happens if I lose the codebook?"
- [Docs index](index.md)
