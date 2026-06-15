"""Claude Code hook handlers — automatic recall/save (0 LLM calls).

What claude-mem does as "compression" via an LLM worker, lkhu replaces with vector
storage + lifecycle (decay/glymphatic). Each handler takes (input dict, daemon client)
and returns a hook output dict (for testability).

Hook output convention (Claude Code):
    - Context injection: {"hookSpecificOutput": {"hookEventName": ..., "additionalContext": "..."}}
    - Otherwise:         {"continue": True, "suppressOutput": True}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lkhu.server.client import LkhuClient

__all__ = [
    "handle_session_start",
    "handle_user_prompt",
    "handle_stop",
    "dispatch",
    "PASS",
]

PASS: dict[str, Any] = {"continue": True, "suppressOutput": True}

_SESSION_RECENT = 8  # number of memories to inject at session start
_PROMPT_RECALL_K = 5  # number of memories to recall per prompt
_STOP_AUDIT_MAX = 280  # max length of response to save on Stop
_INJECT_LINE_MAX = (
    200  # max length per memory line on injection (save tokens — avoid injecting long text whole)
)
_ASSISTANT_STRENGTH = (
    0.6  # save assistant responses at low strength → decay cleans them up if unused
)
_MIN_CHARS = 8  # shorter than this is not worth saving (trivial response)

# ── Noise filter ──────────────────────────────────────────────────────────
# ① System-injected blocks (never worth remembering). Includes context injected by lkhu itself.
_SYSTEM_BLOCK_RE = re.compile(
    r"<(system-reminder|task-notification|command-name|command-message|command-args|"
    r"local-command-stdout)>.*?</\1>",
    re.DOTALL,
)
_LKHU_CONTEXT_RE = re.compile(
    r"#{1,6}[^\n]*\(lkhu\)\n(?:[-*][^\n]*\n?)*"
)  # injected memory block (prevents re-save loop)
_PRIVATE_RE = re.compile(r"<private>.*?</private>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)  # ② code/tool dumps


def _strip_noise(text: str) -> str:
    """Remove system blocks, injected lkhu context, and private regions (zero value)."""
    text = _PRIVATE_RE.sub(" ", text)
    text = _SYSTEM_BLOCK_RE.sub(" ", text)
    text = _LKHU_CONTEXT_RE.sub(" ", text)
    return text


def _is_trivial(text: str) -> bool:
    """Whether the text is too short to be worth saving.

    Length-based, hence language-agnostic: short acknowledgements in any language fall
    under the threshold automatically, with no per-language keyword list.
    """
    return len(text.strip()) < _MIN_CHARS


_URL_ONLY_RE = re.compile(r"^\s*<?https?://\S+>?\s*$")
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)  # a run of ≥3 letters in any script


def _is_noise(text: str) -> bool:
    """Whether the text is structural junk not worth remembering (language-agnostic).

    Catches what a length check misses without any keyword list: a bare URL, or a line that is
    mostly punctuation/emoji/symbols. Anything containing a real word (a run of ≥3 letters in
    any script — Latin, Hangul, CJK, …) is kept, so symbol-dense but meaningful prose is not
    dropped. ``str.isalnum`` is Unicode-aware for the fallback ratio test.
    """
    t = text.strip()
    if not t:
        return True
    if _URL_ONLY_RE.match(t):
        return True
    if _WORD_RE.search(t):
        return False
    meaningful = sum(1 for ch in t if ch.isalnum())
    return meaningful / len(t) < 0.5


def _skip_save(text: str) -> bool:
    """Combined save gate: skip trivial (too short) or structurally noisy content."""
    return _is_trivial(text) or _is_noise(text)


def _clean_prompt(text: str) -> str:
    """Clean a prompt: remove system blocks + normalize whitespace."""
    return " ".join(_strip_noise(text).split())


def _clean_assistant(text: str) -> str:
    """Clean an assistant response: strip system blocks + code/tool dumps, keep prose."""
    cleaned = _CODE_FENCE_RE.sub(" ", _strip_noise(text))
    return " ".join(cleaned.split())


def _format_memories(memories: list[dict[str, Any]], title: str) -> str:
    """Build context-injection markdown from memories (truncate long text)."""
    lines = [f"## {title} (lkhu)"]
    for m in memories:
        text = " ".join((m.get("audit_text") or "").split())  # normalize whitespace/newlines
        if not text:
            continue
        if len(text) > _INJECT_LINE_MAX:
            text = text[:_INJECT_LINE_MAX].rstrip() + "…"
        lines.append(f"- {text}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _inject(event: str, context: str) -> dict[str, Any]:
    """Build an additionalContext injection output (pass through if context is empty)."""
    if not context:
        return PASS
    return {
        "hookSpecificOutput": {"hookEventName": event, "additionalContext": context},
        "suppressOutput": True,
    }


def handle_session_start(data: dict[str, Any], client: LkhuClient) -> dict[str, Any]:
    """At session start, inject top memories by strength/recency as context."""
    memories = client.recent(n=_SESSION_RECENT)
    context = _format_memories(memories, "Remembered from previous sessions")
    return _inject("SessionStart", context)


def handle_user_prompt(data: dict[str, Any], client: LkhuClient) -> dict[str, Any]:
    """Inject prompt-related memories, and save only non-trivial prompts."""
    prompt = _clean_prompt(data.get("prompt") or "")
    if not prompt:
        return PASS
    # 1) Recall related memories → inject (short queries are still recalled)
    result = client.recall(prompt, k=_PROMPT_RECALL_K)
    context = _format_memories(result.get("sources", []), "Related memories")
    # 2) Save only when worth keeping (embedding only, 0 LLM calls)
    if not _skip_save(prompt):
        client.observe(prompt, session_id=data.get("session_id", ""))
    return _inject("UserPromptSubmit", context)


def _extract_text(message: Any) -> str:
    """Extract text from a transcript message (string or list of blocks)."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return " ".join(p for p in parts if p)
    return ""


def handle_stop(data: dict[str, Any], client: LkhuClient) -> dict[str, Any]:
    """On session stop, save the prose gist of the last assistant response at low strength.

    Code/tool dumps/system blocks are stripped, leaving only prose. Conclusions are
    preserved, but at low strength so decay cleans them up if unused (biomimetic cleanup).
    """
    transcript_path = data.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        return PASS
    last_assistant = ""
    for line in Path(transcript_path).read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "assistant":
            last_assistant = _extract_text(entry.get("message", {}))

    prose = _clean_assistant(last_assistant)
    if prose and not _skip_save(prose):
        client.observe(
            prose[:_STOP_AUDIT_MAX],
            session_id=data.get("session_id", ""),
            strength=_ASSISTANT_STRENGTH,
        )
    return PASS


_HANDLERS = {
    "session-start": handle_session_start,
    "user-prompt": handle_user_prompt,
    "stop": handle_stop,
}


def dispatch(event: str, data: dict[str, Any], client: LkhuClient) -> dict[str, Any]:
    """Call the handler by event name (unsupported events pass through)."""
    handler = _HANDLERS.get(event)
    if handler is None:
        return PASS
    return handler(data, client)
