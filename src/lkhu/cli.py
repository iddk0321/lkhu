"""lkhu CLI entry point (Typer-based). See design doc §9.

Commands: init / serve / status / recall / doctor / export / import / backup / uninstall / reset.
Heavy dependencies (faiss, etc.) are lazily imported inside each command.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lkhu import __version__

app = typer.Typer(
    name="lkhu",
    help="lkhu (Like Human) — an AI memory system that remembers like a human.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    """Handle the ``--version`` flag and exit immediately."""
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """lkhu command-line interface."""


def _open_engine():
    """Open the production engine (Ollama embedder). If no codebook, show help and exit."""
    from lkhu.core.engine import LkhuEngine
    from lkhu.platform.ollama import OllamaEmbedder

    try:
        return LkhuEngine.open(embedder=OllamaEmbedder())
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def init() -> None:
    """Create directories, generate codebook (triple backup), register Claude Code config."""
    from lkhu.core.engine import initialize
    from lkhu.platform.ollama import OllamaEmbedder

    console.print("🧠 [bold]Starting lkhu initialization...[/bold]")
    info = initialize()
    emb = OllamaEmbedder()
    ollama_ok = emb.is_available()

    console.print(f"  Data directory: {info['data_dir']}")
    if info["codebook_created"]:
        console.print("  [green]✓[/green] codebook generated + triple backup")
    else:
        console.print("  [yellow]•[/yellow] codebook already exists, preserved (not regenerated)")
    console.print(
        "  [green]✓[/green] Claude Code config registered"
        if info["mcp_registered"]
        else "  [yellow]•[/yellow] Claude config registration skipped"
    )
    if ollama_ok:
        console.print("  [green]✓[/green] Ollama + snowflake-arctic-embed2 available")
    else:
        console.print(
            "  [yellow]![/yellow] Ollama not detected — install from https://ollama.com then "
            "run 'ollama pull snowflake-arctic-embed2'."
        )
    console.print(Panel.fit("✅ Done! Please restart Claude Code.", border_style="green"))


@app.command()
def install() -> None:
    """Generate codebook + register Claude Desktop MCP. (Claude Code installs as a plugin.)"""
    from lkhu.platform import setup
    from lkhu.platform.ollama import OllamaEmbedder

    console.print("🧠 [bold]Starting lkhu installation...[/bold]")
    result = setup.install()
    console.print(f"  Data directory: {result['data_dir']}")
    console.print(
        "  [green]✓[/green] codebook "
        + ("generated (triple backup)" if result["codebook_created"] else "preserved")
    )
    console.print("  [green]✓[/green] Claude Desktop MCP registered")
    if OllamaEmbedder().is_available():
        console.print("  [green]✓[/green] Ollama + snowflake-arctic-embed2 available")
    else:
        console.print(
            "  [yellow]![/yellow] Ollama not detected — install from https://ollama.com then "
            "run 'ollama pull snowflake-arctic-embed2'"
        )
    console.print(
        Panel.fit(
            "✅ Desktop installation complete!\n\n"
            "[bold]Install Claude Code as a plugin:[/bold]\n"
            "  claude plugin marketplace add <lkhu repo>\n"
            "  claude plugin install lkhu@lkhu\n"
            "  → restart Claude Code",
            border_style="green",
        )
    )


@app.command()
def serve() -> None:
    """Run the MCP server (invoked automatically by Claude Code)."""
    from lkhu.server.mcp import run

    run()


@app.command()
def status() -> None:
    """Show system status and statistics."""
    engine = _open_engine()
    try:
        s = engine.status()
    finally:
        engine.close()

    table = Table(title="lkhu status", show_header=False)
    table.add_row("Total memories", str(s["total_memories"]))
    table.add_row("Archived", str(s["archived"]))
    table.add_row("Kind distribution", str(s["kinds"]))
    table.add_row("Avg/max strength", f"{s['strength_avg']} / {s['strength_max']}")
    table.add_row("codebook key count", str(s["codebook_keys"]))
    table.add_row("Dimension", str(s["dim"]))
    table.add_row("Decoder tier", str(s["decoder"]))
    console.print(table)


@app.command()
def remember(
    content: str,
    kind: str = typer.Option("fact", help="Memory kind (fact/decision, etc.)"),
) -> None:
    """Store a memory explicitly (directly from the CLI)."""
    engine = _open_engine()
    try:
        mem = engine.remember(content, kind=kind)
    finally:
        engine.close()
    console.print(f"[green]✓[/green] Stored (id={mem.id[:8]}…): {content}")


@app.command()
def forget(
    query: str,
    confirm: bool = typer.Option(False, "--confirm", help="Confirm archiving."),
) -> None:
    """Archive memories matching the query (audit is preserved)."""
    engine = _open_engine()
    try:
        res = engine.forget(query, confirm=confirm)
    finally:
        engine.close()
    if not confirm:
        console.print("[yellow]•[/yellow] Add --confirm to actually archive.")
    else:
        console.print(f"[green]✓[/green] Archived {res['archived']} memories.")


@app.command()
def recall(query: str, k: int = typer.Option(5, help="Number of memories to retrieve")) -> None:
    """Search memories directly from the CLI (for debugging)."""
    engine = _open_engine()
    try:
        result = engine.recall(query, k=k)
    finally:
        engine.close()
    body = result["text"] or "[dim](no relevant memories)[/dim]"
    console.print(Panel(body, title=f"recall (tier {result['tier']})"))
    for src in result["sources"]:
        console.print(f"  • [dim]{src['strength']}[/dim] {src['audit_text']}")


@app.command()
def doctor() -> None:
    """Diagnose the environment: Ollama / Codebook / MCP (Desktop·Code) / hooks / daemon."""
    from lkhu.core.codebook import Codebook
    from lkhu.platform import claude_code, claude_hooks, mcp_config
    from lkhu.platform.ollama import OllamaEmbedder
    from lkhu.platform.paths import LkhuPaths
    from lkhu.server.client import LkhuClient

    paths = LkhuPaths()
    table = Table(title="lkhu doctor")
    table.add_column("Item")
    table.add_column("Status")

    def mark(ok: bool, yes: str = "OK", no: str = "missing") -> str:
        return f"[green]{yes}[/green]" if ok else f"[red]{no}[/red]"

    emb = OllamaEmbedder()
    table.add_row("Ollama server", mark(emb.is_available()))

    if Codebook.is_initialized(paths.codebook_path):
        try:
            Codebook.load(paths.codebook_path)
            table.add_row("Codebook integrity", mark(True))
        except ValueError:
            table.add_row("Codebook integrity", "[red]corrupted[/red]")
    else:
        table.add_row("Codebook", "[yellow]uninitialized (run lkhu install)[/yellow]")

    table.add_row(
        "Claude Desktop MCP", mark(mcp_config.is_registered(), "registered", "unregistered")
    )
    if claude_code.is_available():
        table.add_row(
            "Claude Code MCP", mark(claude_code.is_registered(), "registered", "unregistered")
        )
    else:
        table.add_row("Claude Code MCP", "[yellow]claude CLI not found[/yellow]")
    table.add_row(
        "Auto-memory hooks", mark(claude_hooks.hooks_installed(), "installed", "not installed")
    )
    table.add_row("Daemon", mark(LkhuClient().health(), "running", "stopped"))
    table.add_row("Data directory", str(paths.data_dir))
    console.print(table)


@app.command()
def export(
    out: str = typer.Option("lkhu_export.jsonl", help="Output file path"),
) -> None:
    """Export audit data as JSONL."""
    engine = _open_engine()
    try:
        count = engine.export(out)
    finally:
        engine.close()
    console.print(f"[green]✓[/green] Exported {count} records to {out}.")


@app.command(name="import")
def import_(file: str) -> None:
    """Import audit JSONL from another machine and reconstruct memories."""
    import json
    from pathlib import Path

    engine = _open_engine()
    count = 0
    try:
        for line in Path(file).read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("audit_text", "")
            if text:
                engine.observe(text, session_id=rec.get("session_id", ""))
                count += 1
    finally:
        engine.close()
    console.print(f"[green]✓[/green] Imported {count} memories.")


@app.command()
def backup() -> None:
    """Copy the codebook and database to the backup directory."""
    import shutil
    from datetime import UTC, datetime

    from lkhu.platform.paths import LkhuPaths

    paths = LkhuPaths()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    dest = paths.backups_dir / "daily" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in (paths.codebook_path, paths.db_path):
        if src.exists():
            shutil.copy2(src, dest / src.name)
            copied.append(src.name)
    console.print(f"[green]✓[/green] Backup complete: {dest} ({', '.join(copied) or 'no targets'})")


@app.command()
def uninstall() -> None:
    """Remove Claude Desktop MCP + legacy integration (data is preserved).

    Remove the Claude Code plugin with 'claude plugin uninstall lkhu@lkhu'.
    """
    from lkhu.platform import setup

    result = setup.uninstall()
    console.print("[green]✓[/green] Removed lkhu integration (data and codebook preserved).")
    console.print(f"  Desktop MCP: {'removed' if result['desktop_mcp'] else 'none'}")
    console.print(
        "  [dim]Remove the Claude Code plugin with 'claude plugin uninstall lkhu@lkhu'[/dim]"
    )


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion of all data."),
) -> None:
    """Delete all data (dangerous, requires --confirm)."""
    from lkhu.platform.paths import LkhuPaths

    if not confirm:
        console.print(
            "[red]Danger:[/red] all memories and the codebook will be deleted. Add --confirm."
        )
        raise typer.Exit(code=1)

    paths = LkhuPaths()
    targets = [
        paths.codebook_path,
        paths.codebook_backup_path,
        paths.db_path,
        paths.faiss_path,
        paths.short_term_path,
        paths.audit_dir,
    ]
    for t in targets:
        if t.is_dir():
            import shutil

            shutil.rmtree(t, ignore_errors=True)
        elif t.exists():
            t.unlink()
        # Also remove codebook meta
        meta = t.with_name(t.name + ".meta.json")
        if meta.exists():
            meta.unlink()
    console.print("[green]✓[/green] All data deleted.")


@app.command()
def dashboard(
    no_open: bool = typer.Option(False, "--no-open", help="Do not auto-open the browser"),
) -> None:
    """Open the web dashboard (visualizes stored memories, strength, and lifecycle)."""
    import webbrowser

    from lkhu.server.client import daemon_url, ensure_daemon_running

    if not ensure_daemon_running():
        console.print("[red]✗[/red] Failed to start the daemon. Check with 'lkhu doctor'.")
        raise typer.Exit(code=1)
    url = daemon_url() + "/"
    console.print(f"[green]✓[/green] Dashboard: [bold]{url}[/bold]  (stays up without Ctrl+C)")
    if not no_open:
        webbrowser.open(url)


@app.command(name="eval")
def eval_(
    k: int = typer.Option(5, help="Top-k for recall metrics"),
    offline: bool = typer.Option(
        False, "--offline", help="Skip embedding-dependent metrics (filter only, no Ollama)"
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="Ollama embedding model to benchmark; dimension auto-detected",
    ),
    out: str = typer.Option("", "--out", help="Write the scorecard JSON to this path"),
) -> None:
    """Score recall quality / noise / multilingual / save-filter on a fixed gold corpus.

    Runs in a throwaway data directory — production memories are never touched. This is the
    objective signal for iterating on usefulness: higher hit@k and lower noise_rate are better.
    Pass --model to compare a different embedder (the eval builds its own codebook at the
    model's dimension, so it never affects your stored memories).
    """
    import json as _json

    from lkhu.eval import run_eval

    if offline:
        from lkhu.core.encoder import HashingEmbedder

        embedder = HashingEmbedder(dim=1024)
    else:
        from lkhu.platform.ollama import OllamaEmbedder

        if model:
            probe = OllamaEmbedder(model=model, dim=1)
            embedder = OllamaEmbedder(model=model, dim=probe.detect_dim())
            console.print(f"[dim]model={model} dim={embedder.dim}[/dim]")
        else:
            embedder = OllamaEmbedder()

    console.print("[dim]Running eval over the gold corpus (isolated data dir)…[/dim]")
    card = run_eval(embedder, k=k, offline=offline)

    table = Table(title=f"lkhu eval — {card.mode} (k={card.k})", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Good when", justify="right")
    table.add_row("save filter — signal kept", f"{card.filter['signal_kept_rate']:.0%}", "→ 100%")
    table.add_row(
        "save filter — hard noise dropped",
        f"{card.filter['hard_noise_dropped_rate']:.0%}",
        "→ 100%",
    )
    if not offline:
        cold, warm = card.recall_cold, card.recall
        table.add_row(
            "recall — hit@k (cold → warm)",
            f"{cold['hit_at_k']:.0%} → {warm['hit_at_k']:.0%}",
            "→ 100%",
        )
        table.add_row(
            "recall — noise rate (cold → warm)",
            f"{cold['noise_rate']:.2f} → {warm['noise_rate']:.2f}",
            "→ 0",
        )
        table.add_row(
            "recall — cross-lingual hit@k (warm)",
            f"{warm['cross_lingual_hit_at_k']:.0%}",
            "→ 100%",
        )
        table.add_row(
            "Hebbian — noise saturated (ungated)",
            f"{card.hebbian_baseline['noise_saturated']:.0f}",
            "lower",
        )
        table.add_row(
            "Hebbian — noise saturated (gated)",
            f"{card.hebbian_gated['noise_saturated']:.0f}",
            "→ 0",
        )
    console.print(table)

    if out:
        from pathlib import Path as _Path

        _Path(out).write_text(_json.dumps(card.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        console.print(f"[green]✓[/green] Scorecard written to {out}")


@app.command()
def reembed(
    yes: bool = typer.Option(False, "--yes", help="Proceed without the confirmation prompt"),
) -> None:
    """Rebuild all memory vectors with the current embedding model (after switching models).

    Vectors saved with a different model live in a different space, so recall breaks until they
    are re-encoded. This preserves your memories' text, strength, and metadata — only the scent
    vectors change. Stop the daemon first (it holds the same store): `lkhu` has no stop command,
    so quit the process running `lkhu daemon`/Claude Code, or it will be re-encoded under it.
    """
    engine = _open_engine()
    try:
        n = engine.vault.count(include_archived=True)
        if not yes:
            console.print(
                f"[yellow]•[/yellow] This will re-encode {n} memories with the current model. "
                "Re-run with --yes to proceed."
            )
            return
        count = engine.reembed()
        console.print(f"[green]✓[/green] Re-embedded {count} memories with the current model.")
    finally:
        engine.close()


@app.command()
def daemon() -> None:
    """Run the memory daemon (a resident service shared by hooks, MCP, and CLI)."""
    from lkhu.server.daemon import run_daemon

    run_daemon()


@app.command()
def hook(event: str) -> None:
    """Handle a Claude Code hook event (stdin JSON → stdout JSON). Internal, always exits safely."""
    import json
    import sys

    from lkhu.server import hooks as hookmod
    from lkhu.server.client import LkhuClient, ensure_daemon_running

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001
        data = {}

    out = hookmod.PASS
    try:
        if ensure_daemon_running():
            out = hookmod.dispatch(event, data, LkhuClient())
    except Exception:  # noqa: BLE001 — so a memory-system error never blocks the user's work
        out = hookmod.PASS

    sys.stdout.write(json.dumps(out, ensure_ascii=False))


@app.command(name="install-hooks")
def install_hooks_cmd() -> None:
    """Register auto recall/save hooks in Claude Code."""
    from lkhu.platform import claude_hooks

    path = claude_hooks.install_hooks()
    console.print(f"[green]✓[/green] Claude Code hooks registered: {path}")
    console.print("  New claude sessions auto-recall at start and auto-save mid-conversation.")


@app.command(name="uninstall-hooks")
def uninstall_hooks_cmd() -> None:
    """Remove the Claude Code auto recall/save hooks."""
    from lkhu.platform import claude_hooks

    claude_hooks.uninstall_hooks()
    console.print("[green]✓[/green] Removed lkhu hooks (other hooks and data preserved).")


if __name__ == "__main__":  # pragma: no cover
    app()
