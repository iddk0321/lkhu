# Contributing to lkhu

Thanks for stopping by! **lkhu** (Like Human) gives Claude Code memory that works like a human brain: memories are stored as 1024-dim latent vectors ("scents"), not natural-language summaries, and the entire memory pipeline runs with **zero LLM calls**. That design constraint shapes every contribution — so this guide gets you productive fast, then tells you the few rules you really can't break.

You don't need to understand VSA math to contribute. Tests, docs, CLI polish, and platform fixes are all just as valuable as core algorithm work.

## Dev setup

You need Python 3.11+ and nothing else to run the test suite (Ollama is optional, see below).

```bash
git clone https://github.com/iddk0321/lkhu.git
cd lkhu
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -m "not ollama"             # full suite, no Ollama required
```

Everything lives in one `pyproject.toml` — there is no `requirements.txt`. Run every Python command with `.venv` activated; `.venv` is git-ignored.

### Optional: real-embedding tests

Tests marked `ollama` need a running [Ollama](https://ollama.com) server with the default embedding model, `snowflake-arctic-embed2` (1024-dim, multilingual). CI skips them, but you can run them locally:

```bash
ollama pull snowflake-arctic-embed2
pytest -m ollama
```

(`bge-m3` was the previous default and still works if you set `encoder.model` in user config, but new work should target `snowflake-arctic-embed2`.)

## Where things live

| Path | What it is |
|------|-----------|
| `src/lkhu/core/` | OS-agnostic business logic: encoder, decoder, VSA ops, recall, consolidation, decay |
| `src/lkhu/platform/` | All OS differences: paths, Ollama, MCP config, scheduler, setup |
| `src/lkhu/server/` | Daemon, MCP server, Claude Code hooks, dashboard |
| `src/lkhu/cli.py` | The `lkhu` Typer CLI |
| `tests/` | `unit/`, `integration/`, `platform/` |
| `docs/` | User and design docs — start with [docs/architecture.md](docs/architecture.md) |

[CLAUDE.md](CLAUDE.md) holds the project's permanent decisions; skim it before touching core code.

## Quality bar

Run these before pushing (CI runs the ruff and pytest steps; `mypy` is enforced at review time):

```bash
ruff format .                      # formatter
ruff check .                       # linter
mypy src/lkhu                      # strict mode is enabled in pyproject.toml
pytest -m "not ollama" --cov --cov-report=term-missing
```

- **Type hints are required.** `mypy` runs in strict mode over `src/lkhu`.
- **Docstrings** follow Google style.
- **Coverage target is 80%+.** New code should come with tests; bug fixes should come with a regression test.

### Recall-quality signal: `lkhu eval`

Beyond the unit suite, `lkhu eval` is the objective signal for *usefulness* — how well recall, noise filtering, and cross-lingual matching actually work. It runs against a fixed gold corpus (18 signal items, soft/hard noise, 16 queries with 8 cross-lingual) in a throwaway data directory with a seeded codebook, so it is deterministic and never touches production memories.

```bash
lkhu eval                          # full scorecard (needs Ollama)
lkhu eval --offline                # save-filter metrics only, no embeddings
lkhu eval --k 5 --out scorecard.json
lkhu eval --model bge-m3           # compare another embedder (dimension auto-detected)
```

It reports recall as **COLD** (first session) vs **WARM** (after simulated use), plus save-filter and Hebbian-saturation metrics. On the default embedder (`snowflake-arctic-embed2`) the current scorecard is hit@k 1.00, cross-lingual 1.00, noise_rate 0.23, Hebbian noise-saturated 0, save filter 100/100. **If your change touches encoder, recall ranking, the save filter, or the lifecycle, run `lkhu eval` and make sure these numbers don't regress.**

## Commits and branches

- Commits follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `chore:`, ...
- `main` is protected. Work on a `feat/...` branch (or `fix/...`, `docs/...`) and open a PR against `main`.

## CI

Every PR runs the matrix in [.github/workflows/test.yml](.github/workflows/test.yml): **ubuntu / macos / windows × Python 3.11 / 3.12 / 3.13** — 9 jobs, all of which must be green. Each job runs `ruff check`, `ruff format --check`, and `pytest -m "not ollama"` with coverage.

If your change is platform-sensitive (paths, process spawning, config file locations), test it through `src/lkhu/platform/` abstractions rather than branching on `sys.platform` in business logic — CI will catch it otherwise, but it's faster to get it right up front.

## Hard Rules

These are non-negotiable. PRs that break them will be asked to change, no matter how nice the feature is.

1. **No hardcoded paths.** Never write `~/Library/...` or `C:\...` in code. All filesystem locations come from `src/lkhu/platform/paths.py` (built on `platformdirs` + `pathlib.Path`). OS differences live in the platform layer only — `core/` must be OS-agnostic.
2. **The codebook is sacrosanct.** `codebook.npy` (the key-scent dictionary) is never modified or regenerated once created — losing it invalidates every stored memory. Adding new keys is fine (it never changes existing ones); changing or reseeding existing keys is not. The code guards this (`Codebook.save()` refuses to overwrite); don't work around the guard.
3. **Minimize LLM calls — that's the whole point.** Save, recall, consolidation, decay, and cleansing run on local embeddings and vector math alone. The only place an LLM may appear is the decoder's Tier-3 fallback (<5% of decodes, capped at 80 tokens). If your feature "just needs one LLM call," it needs a redesign instead.
4. **`audit_text` is a shadow.** The natural-language copy exists for user visibility and debugging (and Tier-1 decode excerpts) — it is never the primary search target. Don't build features that search or rank on `audit_text`; search goes through vectors. But always preserve it: users must be able to read what the system remembers.

## First issue?

Genuinely welcome. Good entry points:

- **Docs and examples** — if something in [docs/](docs/index.md) confused you, fixing it is a real contribution.
- **Tests** — coverage gaps in `tests/unit/` are easy to find and low-risk to fill.
- **Platform quirks** — Windows and Linux path/process edge cases always need more eyes.
- **CLI polish** — `lkhu doctor`, `lkhu status`, and friends can always be friendlier.

Browse [open issues](https://github.com/iddk0321/lkhu/issues), comment that you're taking one, and ask anything you like there — unclear questions usually mean unclear docs, which is our bug, not yours. Small PRs are easier to review than big ones; when in doubt, open an issue first and sketch your approach.

## PR checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy src/lkhu` passes
- [ ] `pytest -m "not ollama"` passes
- [ ] No new hardcoded paths; no new LLM calls outside decoder Tier 3
- [ ] Relevant docs updated
- [ ] Conventional Commit messages

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
