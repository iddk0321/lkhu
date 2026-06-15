"""Fixed gold corpus for the lkhu evaluation harness.

Hand-labeled so the harness can compute objective metrics. Three memory classes:

- SIGNAL     — durable facts/decisions/preferences. Must be stored and retrievable.
- SOFT_NOISE — plausible-length chatter that passes the save filter but must rank LOW on
               recall (this is what the ranking/lifecycle, not the filter, has to suppress).
- HARD_NOISE — acknowledgements / URLs / emoji the save filter SHOULD drop outright.

Bilingual on purpose (English + Korean), with cross-lingual queries, to exercise the
multilingual-embedding claim with no per-language keyword lists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    text: str
    topic: str | None = None  # None = noise (no relevant topic)
    lang: str = "en"


@dataclass(frozen=True)
class Query:
    text: str
    topic: str  # the topic whose SIGNAL items count as relevant
    lang: str = "en"
    cross_lingual: bool = False  # answer lives in a different language than the query


# ── Durable signal (must be kept + retrievable) ──────────────────────────────
SIGNAL: list[Item] = [
    Item("The backend service is built with FastAPI and Python 3.11.", "stack", "en"),
    Item("We chose PostgreSQL over MySQL for the primary database.", "db", "en"),
    Item("Deployments go through GitHub Actions to the staging cluster first.", "deploy", "en"),
    Item("The user prefers tabs over spaces and a 100-character line limit.", "style", "en"),
    Item("The project is licensed under Apache 2.0.", "license", "en"),
    Item("The user's name is Donguk and they develop on macOS.", "identity", "en"),
    Item("프론트엔드는 React와 TypeScript, 빌드 도구는 Vite를 사용한다.", "frontend", "ko"),
    Item("인증은 JWT 액세스 토큰과 리프레시 토큰 방식으로 결정했다.", "auth", "ko"),
    Item("데이터베이스 마이그레이션은 Alembic으로 관리한다.", "migration", "ko"),
    Item("API 응답은 모두 snake_case로 통일하기로 합의했다.", "casing", "ko"),
    Item("The team's primary database of record is PostgreSQL.", "db", "en"),
    Item("코드 스타일: 탭을 쓰고 한 줄은 100자로 제한한다.", "style", "ko"),
    Item("Unit tests run with pytest, targeting 80% coverage.", "testing", "en"),
    Item("The app is containerized with Docker and deployed on Kubernetes.", "container", "en"),
    Item("Secrets live in environment variables and are never committed to git.", "secrets", "en"),
    Item("Redis sits in front of the database as the caching layer.", "cache", "en"),
    Item("로깅은 구조화된 JSON 형식으로 표준화했다.", "logging", "ko"),
    Item("에러는 전역 예외 핸들러에서 일관된 형식으로 처리한다.", "errors", "ko"),
]

# ── Soft noise (passes the filter; must NOT dominate recall) ──────────────────
SOFT_NOISE: list[Item] = [
    Item("ok that looks good, let's ship the current build to TestFlight today.", None, "en"),
    Item("hmm wait, can you move that button a little further to the right please.", None, "en"),
    Item("아 그거 말고 다른 방법으로 한번 해줘 지금 화면이 너무 큼.", None, "ko"),
    Item("좋아요 동욱님 그러면 그 작업부터 먼저 시작하면 될 것 같아요.", None, "ko"),
    Item("let me know when the app logo is done so I can submit it.", None, "en"),
    Item("그래 일단 그렇게 진행하고 나중에 다시 확인해보자.", None, "ko"),
    Item("wait actually never mind, let's revisit this part tomorrow morning.", None, "en"),
    Item("음 그거 한번 다시 해보고 안 되면 말해줘 고마워.", None, "ko"),
    Item("cool cool, sounds good to me, go ahead with that approach.", None, "en"),
]

# ── Hard noise (the save filter SHOULD drop these) ───────────────────────────
HARD_NOISE: list[Item] = [
    Item("ㅇㅇ", None, "ko"),
    Item("응 그래", None, "ko"),
    Item("ok", None, "en"),
    Item("👍👍🔥", None, "en"),
    Item("https://github.com/DDDangkong/lkhu", None, "en"),
    Item("...", None, "en"),
    Item("ㅋㅋㅋ", None, "ko"),
    Item(
        "<task-notification><tool-use-id>toolu_01ABC</tool-use-id></task-notification>", None, "en"
    ),
]

# ── Warm-up queries (held out from QUERIES; used to simulate real use over time) ──
# These reinforce the genuinely-relevant memories per topic the way a user repeatedly asking
# about real topics would. They are paraphrases — DIFFERENT strings from the test QUERIES — so
# measuring on QUERIES afterwards is a proper train/test split, not memorization. Noise has no
# warm-up query, so over time it is never reinforced and falls behind (the lifecycle thesis).
WARMUP_QUERIES: list[Query] = [
    Query("which framework runs the server side?", "stack", "en"),
    Query("what is our main datastore?", "db", "en"),
    Query("what's the CI release pipeline?", "deploy", "en"),
    Query("tabs or spaces, and what line length?", "style", "en"),
    Query("what open-source license do we use?", "license", "en"),
    Query("what's the developer's name and machine?", "identity", "en"),
    Query("프론트엔드 기술 스택이 뭐였지?", "frontend", "ko"),
    Query("로그인 인증 방식은 어떻게 정했지?", "auth", "ko"),
    Query("DB 스키마 변경은 무슨 도구로 하나?", "migration", "ko"),
    Query("API 필드 이름 표기 규칙은?", "casing", "ko"),
    Query("how do we run the test suite?", "testing", "en"),
    Query("how is the app packaged and orchestrated?", "container", "en"),
    Query("어떻게 비밀 값을 관리하나?", "secrets", "ko", cross_lingual=True),
    Query("캐시는 무엇을 쓰나?", "cache", "ko", cross_lingual=True),
    Query("how is logging standardized?", "logging", "en", cross_lingual=True),
    Query("에러 처리는 어떻게 하나?", "errors", "ko"),
]

# ── Queries (relevant = SIGNAL items sharing the topic) ───────────────────────
QUERIES: list[Query] = [
    Query("what web framework does the backend use?", "stack", "en"),
    Query("which database did we pick?", "db", "en"),
    Query("how do we deploy the app?", "deploy", "en"),
    Query("what are the code style preferences?", "style", "en"),
    Query("what license is the project under?", "license", "en"),
    Query("who is the user and what OS do they use?", "identity", "en"),
    # cross-lingual: Korean memory, English query
    Query("what frontend framework and build tool are used?", "frontend", "en", cross_lingual=True),
    Query("how is authentication handled?", "auth", "en", cross_lingual=True),
    # cross-lingual: English memory, Korean query
    Query("데이터베이스는 무엇을 쓰기로 했나?", "db", "ko", cross_lingual=True),
    Query("배포는 어떻게 하나?", "deploy", "ko", cross_lingual=True),
    # more same-language topics
    Query("what testing framework and coverage target do we use?", "testing", "en"),
    Query("how is the application containerized and run?", "container", "en"),
    Query("what format is logging standardized to?", "logging", "en", cross_lingual=True),
    # more cross-lingual pairs (English memory ↔ Korean query, and vice versa)
    Query("비밀 값(시크릿)은 어디에 저장하나?", "secrets", "ko", cross_lingual=True),
    Query("what caching layer is in front of the database?", "cache", "en", cross_lingual=True),
    Query("how are errors handled across the app?", "errors", "en", cross_lingual=True),
]
