"""lkhu evaluation harness — measurable recall quality, noise robustness, and filtering.

The whole point of an iterative "is it actually helpful?" loop is that *helpful* has to be a
number, not a vibe. This package builds an isolated engine over a fixed gold corpus and scores:

- recall quality   — do paraphrased queries surface the right durable memory?
- noise robustness  — how much returned context is junk (lower is better)?
- multilingual      — does a Korean memory answer an English query and vice versa?
- save filter       — does the hook filter drop catchable junk and keep signal?
- Hebbian health    — does repeated recall saturate strengths into noise attractors?

Run via ``lkhu eval`` (Ollama) or ``lkhu eval --offline`` (filter/mechanics only).
"""

from lkhu.eval.harness import Scorecard, run_eval

__all__ = ["Scorecard", "run_eval"]
