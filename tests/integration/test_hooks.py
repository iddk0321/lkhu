"""Phase 8 — hook handler tests (session-start injection / prompt recall+save / stop save)."""

from __future__ import annotations

import json

import pytest

from lkhu.core.encoder import HashingEmbedder
from lkhu.core.engine import LkhuEngine, initialize
from lkhu.server import hooks
from lkhu.server.client import LkhuClient
from lkhu.server.daemon import DaemonServer


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data"))
    initialize(base=None, mcp_config_path=tmp_path / "claude.json")
    engine = LkhuEngine.open(embedder=HashingEmbedder(dim=1024))
    server = DaemonServer(engine, host="127.0.0.1", port=0)
    server.start_background()
    c = LkhuClient(base_url=f"http://127.0.0.1:{server.port}")
    yield c
    server.shutdown()


def test_session_start_empty_is_pass(client) -> None:
    assert hooks.handle_session_start({}, client) == hooks.PASS


def test_session_start_injects_recent(client) -> None:
    client.remember("The user prefers the dark theme", kind="preference")
    out = hooks.handle_session_start({"source": "startup"}, client)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "dark theme" in ctx


def test_user_prompt_injects_and_saves(client) -> None:
    client.remember("The main language is Python and development is on macOS", kind="fact")
    before = client.status()["total_memories"]
    out = hooks.handle_user_prompt(
        {"prompt": "what was my development language again", "session_id": "s"}, client
    )
    # Related memory injected
    assert "Python" in out["hookSpecificOutput"]["additionalContext"]
    # The prompt itself is saved
    assert client.status()["total_memories"] == before + 1


def test_user_prompt_empty_is_pass(client) -> None:
    assert hooks.handle_user_prompt({"prompt": "   "}, client) == hooks.PASS


def test_stop_saves_last_assistant(client, tmp_path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": "question"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Final response: the task is finished"}
                            ],
                        },
                    }
                ),
            ]
        )
    )
    before = client.status()["total_memories"]
    out = hooks.handle_stop({"transcript_path": str(transcript), "session_id": "s"}, client)
    assert out == hooks.PASS
    assert client.status()["total_memories"] == before + 1
    # Verify the saved content
    assert "Final response" in client.recall("task finished", k=1)["text"]


def test_stop_no_transcript_is_pass(client) -> None:
    assert hooks.handle_stop({}, client) == hooks.PASS


def test_dispatch_unknown_event_is_pass(client) -> None:
    assert hooks.dispatch("nope", {}, client) == hooks.PASS


# ── Noise filter (pure functions) ──────────────────────────────────────────


def test_strip_removes_system_blocks() -> None:
    txt = "<task-notification>background task done</task-notification>actually important content"
    out = hooks._strip_noise(txt)
    assert "task-notification" not in out and "background task" not in out
    assert "actually important content" in out


def test_strip_removes_injected_lkhu_context() -> None:
    txt = "## Related memories (lkhu)\n- old memory one\n- old memory two\nMy real question is this"
    out = hooks._strip_noise(txt)
    assert "old memory" not in out  # prevents re-save loop
    assert "real question" in out


def test_clean_assistant_removes_code() -> None:
    txt = "The cause was a race condition, fixed with a mutex ```python\nx = 1\n``` that's all"
    out = hooks._clean_assistant(txt)
    assert "x = 1" not in out
    assert "fixed with a mutex" in out


def test_is_trivial() -> None:
    assert hooks._is_trivial("ㅇㅇ")
    assert hooks._is_trivial("응 그래")
    assert hooks._is_trivial("ㅇㅇ해봐")
    assert not hooks._is_trivial("my name is Donguk and I use macOS")


# ── Noise filter (integration) ─────────────────────────────────────────────


def test_user_prompt_skips_trivial(client) -> None:
    before = client.status()["total_memories"]
    hooks.handle_user_prompt({"prompt": "ㅇㅇ"}, client)
    assert client.status()["total_memories"] == before  # not saved


def test_user_prompt_strips_system_block_before_save(client) -> None:
    hooks.handle_user_prompt(
        {
            "prompt": "<task-notification>x</task-notification> my hobby is hiking",
            "session_id": "s",
        },
        client,
    )
    text = client.recall("hobby", k=1)["text"]
    assert "task-notification" not in text
    assert "hiking" in text


def test_stop_saves_prose_only_at_low_strength(client, tmp_path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Conclusion: added a cache for speed ```js\nlet a=1\n```",
                        }
                    ],
                },
            }
        )
    )
    hooks.handle_stop({"transcript_path": str(transcript), "session_id": "s"}, client)
    recent = client.recent(n=1)
    assert recent[0]["strength"] == pytest.approx(0.6, abs=0.01)  # low strength
    assert "let a=1" not in recent[0]["audit_text"]  # code removed
    assert "added a cache" in recent[0]["audit_text"]  # prose preserved


def test_stop_skips_pure_code(client, tmp_path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "```\ncode only\n```"}],
                },
            }
        )
    )
    before = client.status()["total_memories"]
    hooks.handle_stop({"transcript_path": str(transcript), "session_id": "s"}, client)
    assert client.status()["total_memories"] == before  # code only → not saved
