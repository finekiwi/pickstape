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
) -> dict:
    """Build a mock graph.invoke() return value."""
    return {
        "user_input": "",
        "intent": intent,
        "params": {},
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
        stack.enter_context(patch("src.recommender.RecommendationEngine"))

        def _build(invoke_result: dict | None = None) -> PickstapeHarness:
            import streamlit as st

            # Clear process-level cache so init_graph() re-runs with current mock.
            st.cache_resource.clear()

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
        h = make_app(_invoke_result(response_text="잘 맞는 곡들이에요."))
        h.run(user_input="슬픈 노래 추천해줘")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2  # welcome + response
        assert "잘 맞는 곡들이에요." in assistant_msgs[-1]["content"]

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


# ── Tests: fallback intent ────────────────────────────────────


class TestFallback:
    """Fallback intent returns a re-ask message with no recommendations."""

    def test_fallback_response_has_no_recommendations(self, make_app):
        h = make_app(
            _invoke_result(
                response_text="죄송해요, 요청을 정확히 이해하지 못했어요.",
                recommendations=[],
                intent="fallback",
            )
        )
        h.run(user_input="ㅋㅋㅋ")
        h.assert_no_exception()

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        last = assistant_msgs[-1]
        # empty list → None (app uses `recommendations or None`)
        assert last.get("recommendations") is None

    def test_fallback_response_text_is_displayed(self, make_app):
        h = make_app(
            _invoke_result(
                response_text="죄송해요, 요청을 정확히 이해하지 못했어요.",
                recommendations=[],
                intent="fallback",
            )
        )
        h.run(user_input="???")
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
        h.set_invoke_result(response_text="정상 응답이에요.")
        h.run(user_input="재시도")

        assistant_msgs = [m for m in h.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3  # welcome + error + normal
        assert "정상 응답이에요." in assistant_msgs[-1]["content"]
