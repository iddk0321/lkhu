"""lkhu basic usage example.

Before running: `lkhu init` (generate codebook). Requires Ollama + bge-m3.
For an offline demo, use HashingEmbedder instead of OllamaEmbedder.
"""

from __future__ import annotations

from lkhu.core.engine import LkhuEngine, initialize
from lkhu.platform.ollama import OllamaEmbedder


def main() -> None:
    # First run only: create the data directory and codebook (preserved if they exist).
    initialize(register_mcp=False)

    engine = LkhuEngine.open(embedder=OllamaEmbedder())
    try:
        # Store memories
        sid = "demo"
        engine.remember(
            "The user is on macOS and their main language is Python", kind="fact", session_id=sid
        )
        engine.remember(
            "The project is named lkhu and the license is Apache 2.0",
            kind="decision",
            session_id=sid,
        )
        engine.remember("Decided to prioritize accuracy first", kind="decision", session_id=sid)

        # Search + decode
        result = engine.recall("the user's development environment", k=3)
        print("recall tier:", result["tier"])
        print("text:", result["text"])
        print("sources:", [s["audit_text"] for s in result["sources"]])

        # Restore a session
        print("\nFull session audit:")
        print(engine.recall_session("demo"))

        # Statistics
        print("\nstatus:", engine.status()["total_memories"], "memories")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
