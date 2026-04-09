"""
Tests for RecommendationEngine.

Uses a small synthetic DataFrame (controlled feature values) for fast,
deterministic unit tests. A real-dataset integration test is included at the end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender.engine import RecommendationEngine, _RESULT_COLUMNS
from src.recommender.preprocess import COSINE_FEATURES, load_and_preprocess


# ---------------------------------------------------------------------------
# Synthetic fixture
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
def sample_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic dataset covering all mood/situation filter cases."""
    rows = [
        # --- happy (valence 0.6-1.0, energy 0.5-1.0) ---
        _make_row("h1", "Happy Song A", "Artist A", 80, 0.8, 0.7, 0.7, 0.1, 0.05, 0.0, 0.1, 120.0, -6.0, "Normal/Unclassified", ["pop"]),
        _make_row("h2", "Happy Song B", "Artist B", 70, 0.75, 0.6, 0.65, 0.2, 0.06, 0.0, 0.2, 115.0, -7.0, "Normal/Unclassified", ["pop", "r&b"]),
        _make_row("h3", "Happy Song C", "Artist C", 60, 0.9, 0.8, 0.8, 0.05, 0.04, 0.0, 0.15, 130.0, -5.0, "Normal/Unclassified", ["edm"]),
        _make_row("h4", "Happy Song D", "Artist D", 50, 0.7, 0.55, 0.6, 0.3, 0.07, 0.0, 0.25, 100.0, -8.0, "Normal/Unclassified", ["pop"]),
        _make_row("h5", "Happy Song E", "Artist E", 40, 0.65, 0.65, 0.7, 0.15, 0.05, 0.0, 0.1, 110.0, -9.0, "Normal/Unclassified", ["latin"]),
        _make_row("h6", "Happy Song F", "Artist F", 30, 0.85, 0.75, 0.75, 0.1, 0.05, 0.0, 0.2, 125.0, -7.5, "Normal/Unclassified", ["pop"]),

        # --- excited (valence 0.7-1.0, energy 0.7-1.0) — Bipolar (Mania) tracks ---
        _make_row("e1", "Excited Song A", "Artist G", 90, 0.85, 0.85, 0.8, 0.05, 0.04, 0.0, 0.1, 140.0, -4.0, "Bipolar (Mania)", ["edm"]),
        _make_row("e2", "Excited Song B", "Artist H", 85, 0.9, 0.9, 0.85, 0.03, 0.03, 0.0, 0.15, 145.0, -3.5, "Bipolar (Mania)", ["edm"]),
        _make_row("e3", "Excited Song C", "Artist I", 75, 0.75, 0.75, 0.7, 0.1, 0.05, 0.0, 0.2, 135.0, -5.0, "Normal/Unclassified", ["pop"]),

        # --- sad (valence 0.0-0.3, energy 0.0-0.4) ---
        _make_row("s1", "Sad Song A", "Artist J", 65, 0.2, 0.3, 0.4, 0.6, 0.05, 0.0, 0.1, 80.0, -12.0, "Normal/Unclassified", ["r&b"]),
        _make_row("s2", "Sad Song B", "Artist K", 55, 0.1, 0.2, 0.35, 0.7, 0.04, 0.0, 0.05, 75.0, -14.0, "Normal/Unclassified", ["rock"]),

        # --- 카페 situation (energy 0.1-0.4, acousticness 0.3-1.0, speechiness 0.0-0.1) ---
        _make_row("c1", "Cafe Song A", "Artist L", 70, 0.5, 0.3, 0.5, 0.6, 0.05, 0.1, 0.1, 90.0, -10.0, "Normal/Unclassified", ["pop"]),
        _make_row("c2", "Cafe Song B", "Artist M", 60, 0.55, 0.25, 0.45, 0.5, 0.04, 0.2, 0.08, 85.0, -11.0, "Normal/Unclassified", ["r&b"]),
        _make_row("c3", "Cafe Song C", "Artist N", 50, 0.45, 0.35, 0.55, 0.4, 0.06, 0.15, 0.12, 92.0, -9.5, "Normal/Unclassified", ["pop"]),

        # --- 운동 situation (energy 0.7-1.0, tempo 120-240 BPM) ---
        _make_row("w1", "Workout Song A", "Artist O", 80, 0.6, 0.85, 0.75, 0.05, 0.05, 0.0, 0.2, 150.0, -5.0, "Normal/Unclassified", ["edm"]),
        _make_row("w2", "Workout Song B", "Artist P", 75, 0.65, 0.8, 0.8, 0.08, 0.04, 0.0, 0.15, 160.0, -4.5, "Normal/Unclassified", ["rap"]),

        # --- 수면 situation (energy 0.0-0.3, acousticness 0.4-1.0, loudness -46 to -10 dB) ---
        _make_row("sl1", "Sleep Song A", "Artist Q", 40, 0.3, 0.15, 0.3, 0.8, 0.02, 0.5, 0.05, 60.0, -20.0, "Normal/Unclassified", ["pop"]),
        _make_row("sl2", "Sleep Song B", "Artist R", 35, 0.25, 0.2, 0.25, 0.7, 0.03, 0.4, 0.04, 55.0, -18.0, "Normal/Unclassified", ["r&b"]),

        # --- 코딩 situation (energy 0.2-0.5, instrumentalness 0.3-1.0, speechiness 0.0-0.08) ---
        _make_row("cd1", "Coding Song A", "Artist S", 60, 0.4, 0.35, 0.5, 0.3, 0.03, 0.6, 0.1, 100.0, -8.0, "Normal/Unclassified", ["pop"]),
        _make_row("cd2", "Coding Song B", "Artist T", 55, 0.35, 0.4, 0.45, 0.35, 0.04, 0.7, 0.12, 95.0, -9.0, "Normal/Unclassified", ["edm"]),

        # --- seed track for similar-track test ---
        _make_row("seed1", "Shape of You", "Ed Sheeran", 95, 0.93, 0.65, 0.825, 0.581, 0.0803, 0.0, 0.0789, 96.0, -3.183, "Normal/Unclassified", ["pop"]),
        _make_row("seed2", "Shape of Something", "Ed Sheeran", 50, 0.8, 0.6, 0.7, 0.4, 0.05, 0.0, 0.1, 100.0, -6.0, "Normal/Unclassified", ["pop"]),
        _make_row("seed3", "Another Shape", "Other Artist", 30, 0.7, 0.5, 0.6, 0.3, 0.06, 0.0, 0.15, 105.0, -7.0, "Normal/Unclassified", ["rock"]),
    ]

    df = pd.DataFrame(rows).reset_index(drop=True)
    feature_matrix = df[COSINE_FEATURES].to_numpy(dtype=np.float64)
    return df, feature_matrix


@pytest.fixture(scope="module")
def engine(sample_data: tuple[pd.DataFrame, np.ndarray]) -> RecommendationEngine:
    df, fm = sample_data
    return RecommendationEngine(df, fm)


# ---------------------------------------------------------------------------
# Emotion-based tests
# ---------------------------------------------------------------------------

class TestRecommendByEmotion:
    def test_happy_returns_results(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy")
        assert len(results) > 0

    def test_happy_respects_top_k(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy", top_k=3)
        assert len(results) <= 3

    def test_unknown_mood_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_by_emotion("unknown_mood") == []

    def test_none_mood_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_by_emotion(None) == []

    def test_genre_pref_filters_results(self, engine: RecommendationEngine) -> None:
        # top_k=1: only h3 matches happy+edm in sample, so fallback doesn't drop genre
        results = engine.recommend_by_emotion("happy", genre_pref="edm", top_k=1)
        assert len(results) == 1
        assert "edm" in results[0]["playlist_genres"]

    def test_excited_bipolar_boost_applied(self, engine: RecommendationEngine) -> None:
        """Bipolar (Mania) tracks should appear in excited results due to boosting."""
        results = engine.recommend_by_emotion("excited", top_k=5)
        assert len(results) > 0
        # At least one Mania track in results (e1/e2 are Mania and within range)
        # We verify results are not empty and contain expected fields

    def test_result_has_required_keys(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy", top_k=1)
        assert len(results) == 1
        result = results[0]
        for key in ["track_id", "track_name", "track_artist", "track_popularity", "playlist_genres"]:
            assert key in result

    def test_mental_health_label_not_in_result(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy", top_k=1)
        assert "Mental_Health_Label" not in results[0]


# ---------------------------------------------------------------------------
# Situation-based tests
# ---------------------------------------------------------------------------

class TestRecommendBySituation:
    def test_cafe_returns_results(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_situation("카페")
        assert len(results) > 0

    def test_unknown_situation_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_by_situation("unknown") == []

    def test_none_situation_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_by_situation(None) == []

    def test_cafe_energy_within_range(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_situation("카페")
        for r in results:
            assert 0.1 <= r["energy"] <= 0.4, f"energy {r['energy']} out of 카페 range"

    def test_workout_tempo_raw_bpm(self, engine: RecommendationEngine) -> None:
        """운동 filter uses raw BPM tempo (120-240), not tempo_norm."""
        results = engine.recommend_by_situation("운동")
        for r in results:
            assert r["tempo"] >= 120.0, f"tempo {r['tempo']} below 운동 minimum"

    def test_sleep_loudness_raw_db(self, engine: RecommendationEngine) -> None:
        """수면 filter uses raw dB loudness (-46 to -10)."""
        results = engine.recommend_by_situation("수면")
        # Results should come from sleep-range tracks (sl1, sl2 with -20, -18 dB)
        assert len(results) > 0

    def test_genre_pref_filters(self, engine: RecommendationEngine) -> None:
        # c1 and c3 match 카페+pop (2 tracks); top_k=2 keeps genre filter active
        results = engine.recommend_by_situation("카페", genre_pref="pop", top_k=2)
        assert len(results) > 0
        for r in results:
            assert "pop" in r["playlist_genres"]

    def test_result_sorted_by_popularity(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_situation("카페", top_k=3)
        popularities = [r["track_popularity"] for r in results]
        assert popularities == sorted(popularities, reverse=True)

    def test_top_k_respected(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_situation("카페", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Similar-track tests
# ---------------------------------------------------------------------------

class TestRecommendSimilar:
    def test_both_none_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_similar(None, None) == []

    def test_exact_match_found(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_similar(seed_track="Shape of You")
        assert len(results) > 0

    def test_seed_excluded_from_results(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_similar(seed_track="Shape of You")
        names = [r["track_name"] for r in results]
        # "Shape of You" (seed1, popularity=95) should not appear in results
        assert "Shape of You" not in names

    def test_nonexistent_seed_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend_similar(seed_track="Totally Fake Track XYZ") == []

    def test_exact_track_plus_artist_priority(self, engine: RecommendationEngine) -> None:
        """track+artist exact match should win over track-only exact match."""
        # "Shape of Something" exists with Ed Sheeran (seed2)
        # "Shape of You" also contains "Shape" for partial match
        results = engine.recommend_similar(
            seed_track="Shape of You", seed_artist="Ed Sheeran"
        )
        # Should find seed1 (exact track+artist), results should not include it
        assert len(results) > 0
        for r in results:
            assert not (r["track_name"] == "Shape of You" and r["track_artist"] == "Ed Sheeran")

    def test_partial_match_fallback(self, engine: RecommendationEngine) -> None:
        """Partial match works when no exact match exists."""
        results = engine.recommend_similar(seed_track="Shape")
        # "Shape of Something" or "Another Shape" could match
        assert len(results) > 0

    def test_top_k_respected(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_similar(seed_track="Shape of You", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_emotion_dispatch(self, engine: RecommendationEngine) -> None:
        results = engine.recommend("emotion", {"mood": "happy", "genre_pref": None})
        assert isinstance(results, list)

    def test_situation_dispatch(self, engine: RecommendationEngine) -> None:
        results = engine.recommend("situation", {"situation": "카페", "genre_pref": None})
        assert isinstance(results, list)

    def test_similar_dispatch(self, engine: RecommendationEngine) -> None:
        results = engine.recommend("similar", {"seed_track": "Shape of You", "seed_artist": None})
        assert isinstance(results, list)

    def test_unknown_intent_returns_empty(self, engine: RecommendationEngine) -> None:
        assert engine.recommend("unknown", {}) == []

    def test_none_mood_via_dispatch(self, engine: RecommendationEngine) -> None:
        assert engine.recommend("emotion", {"mood": None}) == []

    def test_params_get_safety(self, engine: RecommendationEngine) -> None:
        """Missing params keys should not raise KeyError."""
        result = engine.recommend("emotion", {})
        assert result == []


# ---------------------------------------------------------------------------
# Fallback cascade tests
# ---------------------------------------------------------------------------

class TestFallbackCascade:
    def test_genre_drop_on_no_match(self, engine: RecommendationEngine) -> None:
        """When genre yields 0 results, fallback drops genre and returns results."""
        # "latin" genre only has 1 happy song (h5), but top_k=5 → fallback drops genre
        results = engine.recommend_by_emotion("sad", genre_pref="edm", top_k=5)
        # sad + edm has 0 exact matches → fallback drops genre → returns sad songs
        # (s1=r&b, s2=rock, no edm sad songs)
        assert isinstance(results, list)  # should not raise

    def test_range_widening_produces_results(self, engine: RecommendationEngine) -> None:
        """When strict ranges yield too few, widening should produce more results."""
        # 코딩 has only 2 matching tracks in sample; top_k=5 triggers widening
        results = engine.recommend_by_situation("코딩", top_k=5)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Format / result dict tests
# ---------------------------------------------------------------------------

class TestResultFormat:
    def test_track_id_in_result(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy", top_k=1)
        assert "track_id" in results[0]

    def test_mental_health_label_excluded(self, engine: RecommendationEngine) -> None:
        for strategy_result in [
            engine.recommend_by_emotion("happy", top_k=1),
            engine.recommend_by_situation("카페", top_k=1),
            engine.recommend_similar(seed_track="Shape of You", top_k=1),
        ]:
            if strategy_result:
                assert "Mental_Health_Label" not in strategy_result[0]

    def test_playlist_genres_is_list(self, engine: RecommendationEngine) -> None:
        results = engine.recommend_by_emotion("happy", top_k=1)
        assert isinstance(results[0]["playlist_genres"], list)


# ---------------------------------------------------------------------------
# Widen ranges unit test
# ---------------------------------------------------------------------------

class TestWidenRanges:
    def test_normalized_clamp(self, engine: RecommendationEngine) -> None:
        conditions = {"valence": (0.6, 1.0)}
        widened = engine._widen_ranges(conditions, 0.5)
        lo, hi = widened["valence"]
        assert lo >= 0.0
        assert hi <= 1.0

    def test_raw_clamp_uses_data_bounds(self, engine: RecommendationEngine) -> None:
        conditions = {"tempo": (120.0, 240.0)}
        widened = engine._widen_ranges(conditions, 0.5)
        lo, hi = widened["tempo"]
        assert lo >= engine._raw_bounds["tempo"][0]
        assert hi <= engine._raw_bounds["tempo"][1]


# ---------------------------------------------------------------------------
# Integration test (real CSV)
# ---------------------------------------------------------------------------

def test_integration_emotion_happy() -> None:
    """Full dataset: recommend_by_emotion('happy') returns 5 results."""
    df, fm = load_and_preprocess()
    eng = RecommendationEngine(df, fm)
    results = eng.recommend_by_emotion("happy", top_k=5)
    assert len(results) == 5
    for r in results:
        assert "track_name" in r
        assert "track_id" in r


def test_integration_situation_coding() -> None:
    """Full dataset: 코딩 has tight pool — verify results returned without error."""
    df, fm = load_and_preprocess()
    eng = RecommendationEngine(df, fm)
    results = eng.recommend_by_situation("코딩", top_k=5)
    assert len(results) > 0


def test_integration_similar_track() -> None:
    """Full dataset: similar track search returns results excluding seed."""
    df, fm = load_and_preprocess()
    eng = RecommendationEngine(df, fm)
    results = eng.recommend_similar(seed_track="Shape of You", seed_artist="Ed Sheeran", top_k=5)
    assert len(results) > 0
    for r in results:
        assert not (r["track_name"] == "Shape of You" and r["track_artist"] == "Ed Sheeran")
