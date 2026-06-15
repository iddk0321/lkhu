# CLAUDE.md — lkhu project rules

> Claude Code reads this file automatically. It holds the project's permanent rules and decisions.

## One-line project definition

**lkhu** (Like Human) — an open-source MCP plugin that gives Claude Code "memory that works like a human brain." It stores and processes memories as latent vectors ("scents") rather than natural language, cutting token cost to 1/5–1/10 of conventional RAG.

## Confirmed core decisions (do not change)

| Item | Decision | Note |
|------|----------|------|
| Name | `lkhu` (Like Human) | Used for PyPI/CLI/MCP alike |
| Architecture | v2 (VSA-based latent representation) | Not natural-language RAG |
| Embedding | Ollama + `snowflake-arctic-embed2` (1024-dim) | Mandatory dependency. Any 1024-dim Ollama model works (set `encoder.model`); switching requires `lkhu reembed`. `bge-m3` is the previous default and still usable. |
| VSA ops | NumPy FFT HRR (circular convolution/correlation) | bind/bundle/unbind. `torchhd` was dropped (torch/faiss OpenMP clash, ~1GB) for an equivalent FFT implementation. |
| Vector search | `faiss-cpu` | |
| Storage | SQLite (meta + audit) + FAISS (vectors) | |
| MCP | `fastmcp` (stdio) | |
| CLI | `typer` + `rich` | |
| Scheduler | `APScheduler` (embedded in server) | No OS-native cron/launchd |
| Path handling | `platformdirs` + `pathlib.Path` | Never hardcode OS-dependent paths |
| Platform | macOS / Windows / Linux | Three OSes supported at once |
| Python | 3.11 or higher | |
| Dependency management | Single `pyproject.toml` | No requirements.txt |
| Virtual environment | `.venv` (project root) | Excluded from git |
| License | Apache License 2.0 | NOTICE file included |

## Hard Rules

1. **No hardcoded paths.** Do not write OS-specific paths like `~/Library/...` or `C:\...` directly in code. Always obtain paths only through `lkhu/platform/paths.py`.
2. **The Codebook is sacrosanct.** The key-scent dictionary (codebook.npy) is never changed or regenerated once created. Losing it invalidates all memories, so keep three backups.
3. **Minimizing natural language is the reason for being.** Minimizing LLM (external model) calls is this project's core value. Consolidation, cleansing, and associative search are performed with vector operations only; the LLM is called only in the decoder Tier 3 fallback (under 5% of all cases).
4. **audit_text is a shadow.** The natural-language copy (audit_text) is for debugging/recovery only and is not the primary search target. However, it must always be preserved for user visibility.
5. **OS differences are isolated in the platform layer.** Business logic (core/) must be OS-agnostic.
6. **Every PR must pass CI on all 3 OSes.** ubuntu/macos/windows × Python 3.11/3.12/3.13.

## Directory structure (target)

```
lkhu/
├── src/lkhu/
│   ├── cli.py
│   ├── core/          # OS-agnostic business logic
│   │   ├── encoder.py decoder.py vsa.py codebook.py
│   │   ├── working_memory.py short_term.py long_term.py
│   │   ├── recall.py consolidator.py glymphatic.py decay.py audit.py
│   ├── platform/      # OS-difference isolation
│   │   ├── paths.py ollama.py mcp_config.py scheduler.py process.py
│   ├── server/        # mcp.py tools.py
│   └── config/        # defaults.yaml loader.py
├── tests/             # unit/ integration/ platform/
├── docs/
├── examples/
├── .github/workflows/ # test.yml publish.yml
├── pyproject.toml README.md LICENSE NOTICE .gitignore
```

## Coding conventions

- Formatter: `ruff format` / Linter: `ruff check`
- Type hints required (aiming for mypy strict)
- Docstrings: Google style
- Tests: `pytest`, target 80%+ coverage
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` ...)
- Branches: `main` protected, features on `feat/...` branches

## Virtual environment usage (consistency)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
pip install -e ".[dev]"
```

Run every Python command with `.venv` activated. `.venv` is included in `.gitignore`.

## Core algorithm summary

- **encode**: semantic scent (0.6) + structural scent (0.4). Structure = Σ bind(key, value)
- **recall**: FAISS top-K×3 → re-rank (similarity × (0.6 + 0.2·strength + 0.2·recency); strength/recency are multiplicative modulators, not additive bonuses) → gated Hebbian strength update (only candidates ≥ reinforce_sim_threshold) → synthesized scent
- **decode (3-tier)**: ① audit excerpt ② key unbind probe ③ LLM fallback (80 tokens)
- **consolidate**: per-session weighted sum of scents (0 LLM calls)
- **decay**: daily ×0.99, reinforced ×1.05 on recall
- **cleanse**: merge similarity > 0.95, prune strength < 0.1 & 30 days+

## References

- VSA/HRR: Tony Plate (2003); torchhd docs (conceptual reference — the library itself is no longer a dependency)
- Embedder: `snowflake-arctic-embed2` (ollama.com/library/snowflake-arctic-embed2); previous default `bge-m3` (huggingface.co/BAAI/bge-m3)
- MCP: modelcontextprotocol.io
