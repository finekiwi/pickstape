"""UI tests for the Pickstape Streamlit chatbot app.

Uses Streamlit's native AppTest framework to run app.py headlessly with
all external dependencies mocked — no real CSV loading, no LLM calls.

Pattern mirrors the catchup project: ExitStack-based patching fixture +
PickstapeHarness wrapper for readable test code.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).parent.parent / "app.py")

# ── Test data builders ────────────────────────────────────────


def _sample_track(**overrides: Any) -> dict:
    """Build a minimal recommendation dict matching engine output format."""
    base: dict[str, Any] = {
        "track_id": "test_001",
        "track_name": "Test Song",
        "track_artist": "Test Artist",
        "track_album_name": "Test Album",
        "track_popularity": 80,
        "playlist_genres": ["pop", "r&b"],
        "energy": 0.75,
        "valence": 0.60,
        "danceability": 0.68,
        "tempo": 120.0,
    }
    base.update(overrides)
    return base


def _invoke_result(
    response_text: str = "테스트 추천 결과입니다.",
    recommendations: list[dict] | None = None,
    intent: str = "emotion",
    params: dict | None = None,
) -> dict:
    """Build a mock graph.invoke() return value."""
    return {
        "user_input": "",
        "intent": intent,
        "params": params or {},
        "recommendations": recommendations if recommendations is not None else [_sample_track()],
        "response_text": response_text,
        "error": None,
    }


def _empty_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Minimal (empty) DataFrame + feature matrix for engine mock."""
    return pd.DataFrame(columns=["track_id", "track_name"]), np.zeros((0, 8))


# ── App harness ───────────────────────────────────────────────


class PickstapeHarness:
    """Convenience wrapper around AppTest for Pickstape UI tests."""

    def __init__(self, app: AppTest, mock_graph: MagicMock) -> None:
        self.app = app
        self.mock_graph = mock_graph

    # ── interaction ──

    def run(self, user_input: str | None = None) -> "PickstapeHarness":
        """Rerun the app, optionally submitting a chat message."""
        if user_input is not None:
            self.app.chat_input[0].set_value(user_input)
        self.app.run()
        return self

    def set_invoke_result(self, **kwargs: Any) -> "PickstapeHarness":
        """Override the mock graph's next invoke() return value."""
        self.mock_graph.invoke.return_value = _invoke_result(**kwargs)
        return self

    # ── inspection ──

    @property
    def messages(self) -> list[dict]:
        """Current st.session_state.messages."""
        # SafeSessionState does not implement .get(); use getattr with default.
        return getattr(self.app.session_state, "messages", [])

    @property
    def markdown_values(self) -> list[str]:
        """Text content of all rendered st.markdown elements."""
        return [m.value for m in self.app.markdown]

    # ── assertion helpers ──

    def assert_no_exception(self) -> None:
        """Fail the test if the app raised an unhandled exception.

        AppTest returns ElementList() (not None) when there are no exceptions,
        so use a truthiness check rather than identity comparison.
        """
        assert not list(self.app.exception), f"Unexpected exception: {self.app.exception}"


# ── Fixture ───────────────────────────────────────────────────


@pytest.fixture
def make_app():
    """Factory fixture: patches all external deps for the lifetime of the test.

    Usage::

        def test_something(make_app):
            h = make_app()            # default mock result
            h = make_app(_invoke_result(response_text="..."))  # custom result
            h.run(user_input="...")   # simulate user typing
    """
    df, matrix = _empty_dataset()

    with ExitStack() as stack:
        mock_bg = stack.enter_context(patch("src.agent.build_graph"))
        stack.enter_context(
            patch("src.recommender.load_and_preprocess", return_value=(df, matrix))
        )
        mock_re = stack.enter_context(patch("src.recommender.RecommendationEngine"))
        # Default: is_ambiguous_title returns False (non-ambiguous titles)
        mock_re.return_value.is_ambiguous_title.return_value = False

        def _build(
            invoke_result: dict | None = None,
            ambiguous: bool = False,
        ) -> PickstapeHarness:
            import streamlit as st

            # Clear process-level cache so init_resources() re-runs with current mock.
            st.cache_resource.clear()

            mock_re.return_value.is_ambiguous_title.return_value = ambiguous

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = invoke_result or _invoke_result()
            mock_bg.return_value = mock_graph

            at = AppTest.from_file(_APP_PATH, default_timeout=15)
            at.run()
            return PickstapeHarness(at, mock_graph)

        yield _build


# ── Tests: initial state ──────────────────────────────────────


class TestInitialState:
    """App renders correctly on first load before any user interaction."""

    def test_initial_run_has_no_exception(self, make_app):
        h = make_app()
        h.assert_no_exception()

    def test_welcome_message_stored_in_session_state(self, make_app):
        h = make_app()
        msgs = h.messages
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert "환영" in msgs[0]["content"]

    def test_welcome_message_has_no_recommendations(self, make_app):
        h = make_app()
        assert h.messages[0]["recommendations"] is None

    def test_chat_input_is_rendered(self, make_app):
        h = make_app()
        assert len(h.app.chat_input) >= 1

    def test_init_graph_not_called_more_than_once_on_rerun(self, make_app):
        h = make_app()
        # Trigger a rerun without input — cache should prevent re-init.
        h.run()
        h.mock_graph.invoke.assert_not_called()


# ── Tests: chat flow ──────────────────────────────────────────


class TestChatFlow:
    """User submits a query and receives a response with recommendations."""

    def test_user_message_appended_to_history(self, make_app):
        h = make_app()
        h.run(user_input="우울한 기분에 어울리는 노래 추천해줘")

        user_msgs = [m for m in h.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "우울한 기분에 어울리는 노래 추천해줘"

    def test_assistant_response_appended_to_history(self, make_app):
        # app.py uses template — LLM response_text is ignored
        h = make_app(_invoke_result(response_text="LLM이 생성한 텍스트.", intent="emotion"))
        h.run(user_input="슬픈 노래 추천해줘")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2  # welcome + response
        # Template (not LLM text) should appear
        assert "LLM이 생성한 텍스트." not in assistant_msgs[-1]["content"]
        assert "골라봤어요" in assistant_msgs[-1]["content"]

    def test_recommendations_stored_in_assistant_message(self, make_app):
        tracks = [_sample_track(track_name=f"Song {i}") for i in range(5)]
        h = make_app(_invoke_result(recommendations=tracks))
        h.run(user_input="신나는 곡 추천해줘")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        recs = assistant_msgs[-1].get("recommendations")
        assert recs is not None
        assert len(recs) == 5

    def test_graph_invoke_called_with_user_input(self, make_app):
        h = make_app()
        h.run(user_input="카페 음악 추천해줘")
        h.mock_graph.invoke.assert_called_once_with({"user_input": "카페 음악 추천해줘"})

    def test_multi_turn_conversation_accumulates_messages(self, make_app):
        h = make_app()
        h.run(user_input="첫 번째 질문")
        h.run(user_input="두 번째 질문")

        user_msgs = [m for m in h.messages if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_invoke_called_once_per_turn(self, make_app):
        h = make_app()
        h.run(user_input="첫 번째")
        h.run(user_input="두 번째")
        assert h.mock_graph.invoke.call_count == 2


# ── Tests: _is_routeable short-circuit ───────────────────────


class TestIsRouteable:
    """Non-routeable input bypasses graph.invoke() and returns FALLBACK_REASK."""

    def test_jamo_only_does_not_invoke_graph(self, make_app):
        """ㅋㅋㅋ contains no full Korean syllables → short-circuited."""
        h = make_app()
        h.run(user_input="ㅋㅋㅋ")
        h.assert_no_exception()
        h.mock_graph.invoke.assert_not_called()

    def test_jamo_only_returns_fallback_message(self, make_app):
        h = make_app()
        h.run(user_input="ㅠㅠ")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert "죄송해요" in assistant_msgs[-1]["content"]

    def test_jamo_only_has_no_recommendations(self, make_app):
        h = make_app()
        h.run(user_input="ㅋㅋ")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert assistant_msgs[-1].get("recommendations") is None

    def test_punctuation_only_does_not_invoke_graph(self, make_app):
        """??? contains no Korean or Latin letters."""
        h = make_app()
        h.run(user_input="???")
        h.mock_graph.invoke.assert_not_called()

    def test_korean_syllable_routes_to_graph(self, make_app):
        """Input with full Korean syllables must reach graph.invoke()."""
        h = make_app()
        h.run(user_input="슬픈 노래 추천해줘")
        h.mock_graph.invoke.assert_called_once()


# ── Tests: post-processing ────────────────────────────────────


class TestPostProcessing:
    """app.py filters applied after graph.invoke(): blocklist, dedup, CJK cleaning."""

    def test_blocklisted_track_removed_from_recommendations(self, make_app):
        """Tracks whose name is in _TITLE_BLOCKLIST must not appear in stored recs."""
        blocked = _sample_track(track_name="Suicidal", track_artist="YNW Melly")
        safe = _sample_track(track_name="Good Vibes", track_artist="Artist B")
        h = make_app(_invoke_result(recommendations=[blocked, safe]))
        h.run(user_input="신나는 노래 추천해줘")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        recs = assistant_msgs[-1].get("recommendations") or []
        names = [r["track_name"] for r in recs]
        assert "Suicidal" not in names
        assert "Good Vibes" in names

    def test_duplicate_tracks_deduped(self, make_app):
        """Same (track_name, track_artist) pair must appear only once."""
        dup = _sample_track(track_name="Lollipop", track_artist="Lil Wayne")
        h = make_app(_invoke_result(recommendations=[dup, dup]))
        h.run(user_input="힙합 추천해줘")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        recs = assistant_msgs[-1].get("recommendations") or []
        assert len(recs) == 1

    def test_all_tracks_blocked_triggers_fallback_message(self, make_app):
        """When every rec is filtered out, response_text must be FALLBACK_REASK."""
        blocked = _sample_track(track_name="suicidal", track_artist="X")
        h = make_app(
            _invoke_result(
                response_text="골라봤어요!",
                recommendations=[blocked],
            )
        )
        h.run(user_input="슬픈 노래")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        assert last.get("recommendations") is None
        assert "죄송해요" in last["content"]

    def test_llm_response_text_ignored_always(self, make_app):
        """app.py uses template — any LLM response_text (including foreign chars) is ignored."""
        h = make_app(
            _invoke_result(
                response_text="자연스럽게げて좋은 곡 مرحبا이에요.",  # hiragana + Arabic
                recommendations=[_sample_track()],
                intent="emotion",
            )
        )
        h.run(user_input="신나는 노래")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        content = assistant_msgs[-1]["content"]
        assert "げ" not in content
        assert "مرحبا" not in content
        assert "골라봤어요" in content  # template always applied

    def test_emotion_template_contains_mood_label(self, make_app):
        """Emotion template includes Korean mood label when params.mood is provided."""
        h = make_app(
            _invoke_result(
                intent="emotion",
                params={"mood": "excited"},
                recommendations=[_sample_track()],
            )
        )
        h.run(user_input="신나는 노래 추천해줘")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert "신나는" in assistant_msgs[-1]["content"]

    def test_situation_template_contains_situation(self, make_app):
        """Situation template mentions the specific situation."""
        h = make_app(
            _invoke_result(
                intent="situation",
                params={"situation": "카페"},
                recommendations=[_sample_track()],
            )
        )
        h.run(user_input="카페 음악")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert "카페" in assistant_msgs[-1]["content"]

    def test_similar_template_contains_seed_track(self, make_app):
        """Similar template mentions the seed track name."""
        h = make_app(
            _invoke_result(
                intent="similar",
                params={"seed_track": "Yellow", "seed_artist": "Coldplay"},
                recommendations=[_sample_track()],
            )
        )
        h.run(user_input="Yellow 비슷한 곡")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert "Yellow" in assistant_msgs[-1]["content"]

    def test_recommendations_capped_at_five(self, make_app):
        """Engine returning >5 tracks should be capped at 5 in stored recs."""
        tracks = [
            _sample_track(track_name=f"Song {i}", track_artist=f"Artist {i}")
            for i in range(8)
        ]
        h = make_app(_invoke_result(recommendations=tracks))
        h.run(user_input="노래 추천해줘")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        recs = assistant_msgs[-1].get("recommendations") or []
        assert len(recs) == 5


# ── Tests: fallback intent (graph returns empty recs) ─────────


class TestFallback:
    """Graph returns empty recommendations → FALLBACK_REASK shown."""

    def test_fallback_response_has_no_recommendations(self, make_app):
        h = make_app(
            _invoke_result(
                response_text="죄송해요, 요청을 정확히 이해하지 못했어요.",
                recommendations=[],
                intent="fallback",
            )
        )
        h.run(user_input="잘 모르겠어요")
        h.assert_no_exception()

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        # empty list → None (app uses `recommendations or None`)
        assert last.get("recommendations") is None

    def test_empty_recs_replaces_response_with_fallback(self, make_app):
        """Even if LLM produced response text, empty recs must show FALLBACK_REASK."""
        h = make_app(
            _invoke_result(
                response_text="골라봤어요!",
                recommendations=[],
                intent="fallback",
            )
        )
        h.run(user_input="아무거나 추천해줘")
        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert "죄송해요" in assistant_msgs[-1]["content"]


# ── Tests: error handling ─────────────────────────────────────


class TestErrorHandling:
    """App stays stable when graph.invoke() raises an exception."""

    def test_invoke_exception_does_not_crash_app(self, make_app):
        h = make_app()
        h.mock_graph.invoke.side_effect = ConnectionError("서버 응답 없음")
        h.run(user_input="테스트 쿼리")
        h.assert_no_exception()

    def test_invoke_exception_appends_error_message_to_history(self, make_app):
        h = make_app()
        h.mock_graph.invoke.side_effect = ConnectionError("서버 응답 없음")
        h.run(user_input="테스트 쿼리")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2  # welcome + error
        # Error message from app.py
        assert "서버" in assistant_msgs[-1]["content"]

    def test_user_message_persists_after_invoke_error(self, make_app):
        h = make_app()
        h.mock_graph.invoke.side_effect = RuntimeError("예상치 못한 오류")
        h.run(user_input="에러 유발 쿼리")

        user_msgs = [m for m in h.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "에러 유발 쿼리"

    def test_conversation_continues_after_error(self, make_app):
        """After an error turn, the next successful turn should work normally."""
        h = make_app()
        h.mock_graph.invoke.side_effect = ConnectionError("일시적 오류")
        h.run(user_input="에러 유발")

        h.mock_graph.invoke.side_effect = None
        h.set_invoke_result(response_text="LLM 텍스트 — 무시됨.")
        h.run(user_input="재시도")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3  # welcome + error + normal
        assert "골라봤어요" in assistant_msgs[-1]["content"]  # template, not LLM text


# ── Tests: similar disambiguation ────────────────────────────


class TestDisambiguation:
    """Ambiguous similar titles (≥3 artists) without artist → re-ask, no cards."""

    def test_ambiguous_title_without_artist_shows_reask(self, make_app):
        """is_ambiguous_title=True + no seed_artist → disambiguation message, no recs."""
        h = make_app(
            _invoke_result(
                intent="similar",
                params={"seed_track": "Stay", "seed_artist": None},
                recommendations=[_sample_track()],
            ),
            ambiguous=True,
        )
        h.run(user_input="Stay랑 비슷한 곡 추천해줘")
        h.assert_no_exception()

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        assert "Stay" in last["content"]
        assert "아티스트" in last["content"]
        assert last.get("recommendations") is None

    def test_unambiguous_title_without_artist_proceeds_normally(self, make_app):
        """is_ambiguous_title=False → normal similar flow even without seed_artist."""
        h = make_app(
            _invoke_result(
                intent="similar",
                params={"seed_track": "Blinding Lights", "seed_artist": None},
                recommendations=[_sample_track()],
            ),
            ambiguous=False,
        )
        h.run(user_input="Blinding Lights랑 비슷한 곡 추천해줘")
        h.assert_no_exception()

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        assert "찾아봤어요" in last["content"]
        assert last.get("recommendations") is not None

    def test_ambiguous_title_with_artist_proceeds_normally(self, make_app):
        """seed_artist provided → disambiguation skipped, normal flow."""
        h = make_app(
            _invoke_result(
                intent="similar",
                params={"seed_track": "Stay", "seed_artist": "Justin Bieber"},
                recommendations=[_sample_track()],
            ),
            ambiguous=True,  # would be ambiguous, but artist given
        )
        h.run(user_input="Stay Justin Bieber랑 비슷한 곡 추천해줘")
        h.assert_no_exception()

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        # Artist provided → disambiguation not triggered
        assert "아티스트" not in last["content"]
        assert last.get("recommendations") is not None
