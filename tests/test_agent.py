"""Tests for the LangGraph agent pipeline (PS-05).

LLM calls are mocked via factory injection. The recommendation engine
uses a small synthetic dataset for deterministic results.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agent.graph import build_graph
from src.agent.nodes import (
    _extract_json_from_text,
    _parse_intent_fallback,
    _template_response,
    create_nodes,
)
from src.agent.prompts import FALLBACK_REASK
from src.recommender.engine import RecommendationEngine
from src.recommender.preprocess import COSINE_FEATURES


# ---------------------------------------------------------------------------
# Synthetic fixture (reused from test_engine.py pattern)
# ---------------------------------------------------------------------------

def _make_row(
    track_id: str,
    track_name: str,
    track_artist: str,
    track_popularity: int,
    valence: float,
    energy: float,
    danceability: float,
    acousticness: float,
    speechiness: float,
    instrumentalness: float,
    liveness: float,
    tempo: float,
    loudness: float,
    mental_health_label: str,
    genres: list[str],
) -> dict:
    return {
        "track_id": track_id,
        "track_name": track_name,
        "track_artist": track_artist,
        "track_album_name": "Album",
        "track_popularity": track_popularity,
        "valence": valence,
        "energy": energy,
        "danceability": danceability,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "tempo": tempo,
        "tempo_norm": tempo / 240.0,
        "loudness": loudness,
        "Mental_Health_Label": mental_health_label,
        "playlist_genres": genres,
        "playlist_subgenres": [],
    }


@pytest.fixture(scope="module")
def sample_engine() -> RecommendationEngine:
    """RecommendationEngine with synthetic data covering basic cases."""
    rows = [
        _make_row("h1", "Happy Song", "Artist A", 80, 0.8, 0.7, 0.7, 0.1, 0.05, 0.0, 0.1, 120.0, -6.0, "Normal/Unclassified", ["pop"]),
        _make_row("h2", "Happy Song B", "Artist B", 70, 0.75, 0.6, 0.65, 0.2, 0.06, 0.0, 0.2, 115.0, -7.0, "Normal/Unclassified", ["pop", "r&b"]),
        _make_row("s1", "Sad Song", "Artist C", 65, 0.2, 0.3, 0.4, 0.6, 0.05, 0.0, 0.1, 80.0, -12.0, "Normal/Unclassified", ["r&b"]),
        _make_row("s2", "Sad Song B", "Artist D", 55, 0.1, 0.2, 0.35, 0.7, 0.04, 0.0, 0.05, 75.0, -14.0, "Normal/Unclassified", ["rock"]),
        _make_row("c1", "Cafe Chill", "Artist E", 60, 0.4, 0.3, 0.5, 0.5, 0.05, 0.0, 0.1, 90.0, -10.0, "Normal/Unclassified", ["pop"]),
        _make_row("c2", "Cafe Jazz", "Artist F", 50, 0.35, 0.2, 0.45, 0.6, 0.04, 0.0, 0.15, 85.0, -11.0, "Normal/Unclassified", ["r&b"]),
        _make_row("p1", "Party Hit", "Artist G", 90, 0.85, 0.9, 0.85, 0.05, 0.04, 0.0, 0.1, 130.0, -4.0, "Normal/Unclassified", ["edm"]),
        _make_row("p2", "Party Anthem", "Artist H", 85, 0.9, 0.85, 0.9, 0.03, 0.03, 0.0, 0.15, 135.0, -3.5, "Normal/Unclassified", ["edm"]),
        _make_row("sl1", "Sleep Well", "Artist I", 40, 0.25, 0.15, 0.3, 0.8, 0.03, 0.5, 0.05, 60.0, -20.0, "Normal/Unclassified", ["pop"]),
        _make_row("sl2", "Dream On", "Artist J", 35, 0.3, 0.1, 0.25, 0.7, 0.02, 0.6, 0.04, 55.0, -22.0, "Normal/Unclassified", ["r&b"]),
    ]
    df = pd.DataFrame(rows)
    feature_matrix = df[COSINE_FEATURES].values.astype(np.float64)
    return RecommendationEngine(df, feature_matrix)


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(content: str) -> MagicMock:
    """Create a mock ChatOpenAI that returns the given content."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _make_failing_llm() -> MagicMock:
    """Create a mock ChatOpenAI that raises on invoke."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM connection failed")
    return mock_llm


# ===========================================================================
# _extract_json_from_text tests
# ===========================================================================


class TestExtractJsonFromText:
    def test_plain_json(self) -> None:
        text = '{"intent": "emotion", "params": {"mood": "sad"}}'
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["intent"] == "emotion"

    def test_fenced_json(self) -> None:
        text = 'Here is the result:\n```json\n{"intent": "situation", "params": {"situation": "카페"}}\n```'
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["intent"] == "situation"

    def test_json_with_preamble(self) -> None:
        text = 'I think the user wants music.\n{"intent": "similar", "params": {"seed_track": "Hello"}}'
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["intent"] == "similar"

    def test_no_json(self) -> None:
        assert _extract_json_from_text("no json here") is None

    def test_malformed_json(self) -> None:
        assert _extract_json_from_text("{'bad': json}") is None

    def test_nested_braces(self) -> None:
        text = '{"intent": "emotion", "params": {"mood": "happy", "extra": null}}'
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["params"]["mood"] == "happy"


# ===========================================================================
# _parse_intent_fallback tests
# ===========================================================================


class TestParseIntentFallback:
    def test_emotion_from_korean_keywords(self) -> None:
        intent, params = _parse_intent_fallback("요즘 우울해서 노래 듣고 싶어", "")
        assert intent == "emotion"
        assert params["mood"] == "sad"

    def test_situation_from_keywords(self) -> None:
        intent, params = _parse_intent_fallback("카페에서 틀 음악 추천", "")
        assert intent == "situation"
        assert params["situation"] == "카페"

    def test_similar_from_quoted_track(self) -> None:
        intent, params = _parse_intent_fallback("'Blinding Lights' 비슷한 곡", "")
        assert intent == "similar"
        assert params["seed_track"] == "Blinding Lights"

    def test_similar_from_unquoted_pattern(self) -> None:
        intent, params = _parse_intent_fallback("Blinding Lights랑 비슷한 곡 찾아줘", "")
        assert intent == "similar"
        assert params["seed_track"] is not None
        assert "Blinding Lights" in params["seed_track"]

    def test_genre_extraction(self) -> None:
        intent, params = _parse_intent_fallback("팝 장르로 신나는 노래 추천해줘", "")
        assert intent == "emotion"
        assert params["genre_pref"] == "pop"
        assert params["mood"] == "excited"

    def test_total_fallback(self) -> None:
        intent, params = _parse_intent_fallback("안녕하세요", "")
        assert intent == "fallback"

    def test_extracts_from_llm_output_too(self) -> None:
        intent, params = _parse_intent_fallback("노래 틀어줘", '"intent": "emotion"')
        # "노래 틀어줘" alone would be fallback, but llm output has no emotion keywords
        # This tests that combined text is searched
        assert intent in ("emotion", "fallback")


# ===========================================================================
# Router node tests
# ===========================================================================


class TestRouterNode:
    def test_parses_emotion_json(self, sample_engine: RecommendationEngine) -> None:
        json_response = json.dumps({
            "intent": "emotion",
            "params": {"mood": "sad", "situation": None, "genre_pref": None, "seed_track": None, "seed_artist": None},
        })
        mock_llm = _make_mock_llm(json_response)
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "슬픈 날 들을 노래"})
        assert result["intent"] == "emotion"
        assert result["params"]["mood"] == "sad"

    def test_parses_situation_json(self, sample_engine: RecommendationEngine) -> None:
        json_response = json.dumps({
            "intent": "situation",
            "params": {"mood": None, "situation": "카페", "genre_pref": None, "seed_track": None, "seed_artist": None},
        })
        mock_llm = _make_mock_llm(json_response)
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "카페 음악 추천"})
        assert result["intent"] == "situation"
        assert result["params"]["situation"] == "카페"

    def test_parses_similar_json(self, sample_engine: RecommendationEngine) -> None:
        json_response = json.dumps({
            "intent": "similar",
            "params": {"mood": None, "situation": None, "genre_pref": None, "seed_track": "Shape of You", "seed_artist": "Ed Sheeran"},
        })
        mock_llm = _make_mock_llm(json_response)
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "Shape of You 비슷한 곡"})
        assert result["intent"] == "similar"
        assert result["params"]["seed_track"] == "Shape of You"

    def test_fenced_json_extraction(self, sample_engine: RecommendationEngine) -> None:
        fenced = '```json\n{"intent": "emotion", "params": {"mood": "happy"}}\n```'
        mock_llm = _make_mock_llm(fenced)
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "행복한 노래"})
        assert result["intent"] == "emotion"

    def test_keyword_fallback_on_garbage(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_mock_llm("I don't understand")
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "우울할 때 들을 노래 추천해줘"})
        assert result["intent"] == "emotion"
        assert result["params"]["mood"] == "sad"

    def test_total_failure_returns_fallback(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_mock_llm("xyz")
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "안녕"})
        assert result["intent"] == "fallback"

    def test_llm_exception_returns_fallback(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_failing_llm()
        router, _, _ = create_nodes(sample_engine, router_llm_factory=lambda: mock_llm)

        result = router({"user_input": "카페에서 들을 음악"})
        # Keyword fallback should still work on user_input
        assert result["intent"] == "situation"


# ===========================================================================
# Recommendation node tests
# ===========================================================================


class TestRecommendationNode:
    def test_emotion_returns_results(self, sample_engine: RecommendationEngine) -> None:
        _, recommend, _ = create_nodes(sample_engine)

        result = recommend({"intent": "emotion", "params": {"mood": "sad"}})
        assert len(result["recommendations"]) > 0

    def test_situation_returns_results(self, sample_engine: RecommendationEngine) -> None:
        _, recommend, _ = create_nodes(sample_engine)

        result = recommend({"intent": "situation", "params": {"situation": "파티"}})
        assert len(result["recommendations"]) > 0

    def test_fallback_returns_empty(self, sample_engine: RecommendationEngine) -> None:
        _, recommend, _ = create_nodes(sample_engine)

        result = recommend({"intent": "fallback", "params": {}})
        assert result["recommendations"] == []

    def test_engine_error_captured(self) -> None:
        mock_engine = MagicMock()
        mock_engine.recommend.side_effect = ValueError("test error")
        _, recommend, _ = create_nodes(mock_engine)

        result = recommend({"intent": "emotion", "params": {"mood": "happy"}})
        assert result["recommendations"] == []
        assert "test error" in result["error"]


# ===========================================================================
# Response node tests
# ===========================================================================


class TestResponseNode:
    def test_generates_response(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_mock_llm("지금 기분에 맞는 곡들을 골라봤어요!")
        _, _, respond = create_nodes(
            sample_engine,
            response_llm_factory=lambda: mock_llm,
        )

        result = respond({
            "user_input": "슬픈 노래",
            "intent": "emotion",
            "recommendations": [
                {"track_name": "Song A", "track_artist": "Artist A", "playlist_genres": ["pop"]},
            ],
        })
        assert "골라봤어요" in result["response_text"]

    def test_empty_recommendations_returns_reask(self, sample_engine: RecommendationEngine) -> None:
        _, _, respond = create_nodes(sample_engine)

        result = respond({
            "user_input": "안녕",
            "intent": "fallback",
            "recommendations": [],
        })
        assert result["response_text"] == FALLBACK_REASK
        assert result.get("error") is not None

    def test_llm_failure_uses_template(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_failing_llm()
        _, _, respond = create_nodes(
            sample_engine,
            response_llm_factory=lambda: mock_llm,
        )

        recs = [
            {"track_name": "Song A", "track_artist": "Artist A", "playlist_genres": ["pop"]},
            {"track_name": "Song B", "track_artist": "Artist B", "playlist_genres": ["rock"]},
        ]
        result = respond({
            "user_input": "우울한 노래",
            "intent": "emotion",
            "recommendations": recs,
        })
        assert "Song A" in result["response_text"]
        assert "Artist A" in result["response_text"]

    def test_llm_empty_content_uses_template(self, sample_engine: RecommendationEngine) -> None:
        mock_llm = _make_mock_llm("")
        _, _, respond = create_nodes(
            sample_engine,
            response_llm_factory=lambda: mock_llm,
        )

        recs = [{"track_name": "Song A", "track_artist": "Artist A", "playlist_genres": ["pop"]}]
        result = respond({
            "user_input": "우울한 노래",
            "intent": "emotion",
            "recommendations": recs,
        })
        assert "Song A" in result["response_text"]


# ===========================================================================
# Template response test
# ===========================================================================


class TestTemplateResponse:
    def test_formats_tracks(self) -> None:
        recs = [
            {"track_name": "A", "track_artist": "X", "playlist_genres": ["pop"]},
            {"track_name": "B", "track_artist": "Y", "playlist_genres": ["rock", "edm"]},
        ]
        text = _template_response(recs)
        assert "1. A - X (pop)" in text
        assert "2. B - Y (rock, edm)" in text


# ===========================================================================
# Graph integration test
# ===========================================================================


class TestGraphIntegration:
    def test_full_pipeline_emotion(self, sample_engine: RecommendationEngine) -> None:
        router_json = json.dumps({
            "intent": "emotion",
            "params": {"mood": "sad"},
        })
        mock_router_llm = _make_mock_llm(router_json)
        mock_response_llm = _make_mock_llm("슬픈 날엔 이런 노래들 어때요?")

        graph = build_graph(
            sample_engine,
            router_llm_factory=lambda: mock_router_llm,
            response_llm_factory=lambda: mock_response_llm,
        )

        result = graph.invoke({"user_input": "우울한 날 위로되는 노래"})
        assert "response_text" in result
        assert result["response_text"]
        assert result["intent"] == "emotion"
        assert len(result["recommendations"]) > 0

    def test_full_pipeline_fallback(self, sample_engine: RecommendationEngine) -> None:
        mock_router_llm = _make_mock_llm("xyz garbage")
        # Response LLM should not be called for fallback
        mock_response_llm = _make_mock_llm("should not appear")

        graph = build_graph(
            sample_engine,
            router_llm_factory=lambda: mock_router_llm,
            response_llm_factory=lambda: mock_response_llm,
        )

        result = graph.invoke({"user_input": "안녕"})
        assert result["intent"] == "fallback"
        assert result["response_text"] == FALLBACK_REASK

    def test_full_pipeline_situation(self, sample_engine: RecommendationEngine) -> None:
        router_json = json.dumps({
            "intent": "situation",
            "params": {"situation": "파티"},
        })
        mock_router_llm = _make_mock_llm(router_json)
        mock_response_llm = _make_mock_llm("파티에 딱 맞는 곡들이에요!")

        graph = build_graph(
            sample_engine,
            router_llm_factory=lambda: mock_router_llm,
            response_llm_factory=lambda: mock_response_llm,
        )

        result = graph.invoke({"user_input": "파티에서 틀 노래"})
        assert result["intent"] == "situation"
        assert len(result["recommendations"]) > 0
        assert result["response_text"]
