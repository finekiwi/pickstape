"""
Recommendation engine: filtering + cosine similarity.

Three strategies:
- recommend_by_emotion: MOOD_MAPPING filter → cosine similarity → Mental_Health_Label boost
- recommend_by_situation: SITUATION_MAPPING hard filter → popularity sort
- recommend_similar: seed track cosine similarity

All filtering operates on self.df columns (not feature_matrix).
feature_matrix is used only for cosine similarity computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.mappings import (
    MENTAL_HEALTH_BOOST,
    MOOD_MAPPING,
    SITUATION_MAPPING,
)
from src.recommender.preprocess import COSINE_FEATURES

# Features that use raw scale (not [0, 1]) — only affects _widen_ranges clamp logic
_RAW_SCALE_FEATURES: frozenset[str] = frozenset({"tempo", "loudness"})

# Columns included in each result dict
_RESULT_COLUMNS: list[str] = [
    "track_id",
    "track_name",
    "track_artist",
    "track_album_name",
    "track_popularity",
    "playlist_genres",
    "danceability",
    "energy",
    "valence",
    "tempo",
]


class RecommendationEngine:
    """Hybrid recommendation engine combining feature filtering and cosine similarity."""

    def __init__(self, df: pd.DataFrame, feature_matrix: np.ndarray) -> None:
        """Initialise with preprocessed data from load_and_preprocess().

        Parameters
        ----------
        df:
            Cleaned DataFrame (28 352 rows after dedup). Must contain all
            columns in COSINE_FEATURES plus tempo, loudness, playlist_genres,
            Mental_Health_Label, track_popularity.
        feature_matrix:
            Shape (N, 8) float64 array aligned with df. Columns follow
            COSINE_FEATURES order.
        """
        self.df = df.reset_index(drop=True)
        self.feature_matrix = feature_matrix

        # Precompute column medians for ideal-vector construction.
        # Keys are COSINE_FEATURES column names (including tempo_norm).
        self._medians: dict[str, float] = {
            feat: float(self.df[feat].median()) for feat in COSINE_FEATURES
        }

        # Precompute raw-scale column bounds for fallback clamp.
        self._raw_bounds: dict[str, tuple[float, float]] = {
            feat: (float(self.df[feat].min()), float(self.df[feat].max()))
            for feat in _RAW_SCALE_FEATURES
            if feat in self.df.columns
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_by_emotion(
        self,
        mood: str | None,
        genre_pref: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Recommend tracks matching an emotional state.

        Strategy: valence/energy range filter → cosine similarity against an
        ideal mood vector → optional Mental_Health_Label additive boost → Top-k.

        Parameters
        ----------
        mood:
            One of the keys in MOOD_MAPPING. Returns [] for unknown/None.
        genre_pref:
            Optional single genre string (e.g. "pop"). Matched against
            playlist_genres list column.
        top_k:
            Number of results to return (default 5).
        """
        if not mood or mood not in MOOD_MAPPING:
            return []

        conditions = dict(MOOD_MAPPING[mood])
        mask = self._filter_by_ranges(conditions)
        if genre_pref:
            mask = self._apply_genre_filter(mask, genre_pref)

        # Fallback cascade when candidates are insufficient
        mask, conditions = self._fallback_cascade(mask, conditions, genre_pref, top_k)

        indices = np.where(mask.values)[0]
        if len(indices) == 0:
            return []

        # Cosine similarity against ideal mood vector
        ideal = self._build_ideal_vector(mood)
        sub_matrix = self.feature_matrix[indices]
        scores = cosine_similarity(ideal.reshape(1, -1), sub_matrix)[0]

        # Mental_Health_Label additive boost (ADR-007: soft signal, not hard filter)
        if mood in MENTAL_HEALTH_BOOST:
            label, multiplier = MENTAL_HEALTH_BOOST[mood]
            boost_offset = (multiplier - 1.0) * float(np.median(scores))
            label_match = self.df.iloc[indices]["Mental_Health_Label"].values == label
            scores[label_match] += boost_offset

        top_local = np.argsort(scores)[::-1][:top_k]
        top_global = indices[top_local]
        return self._format_results(top_global)

    def recommend_by_situation(
        self,
        situation: str | None,
        genre_pref: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Recommend tracks suitable for a situational context.

        Strategy: hard feature-range filter → optional genre filter →
        popularity sort → Top-k.

        Parameters
        ----------
        situation:
            One of the keys in SITUATION_MAPPING. Returns [] for unknown/None.
        genre_pref:
            Optional genre string.
        top_k:
            Number of results to return.
        """
        if not situation or situation not in SITUATION_MAPPING:
            return []

        conditions = dict(SITUATION_MAPPING[situation])
        mask = self._filter_by_ranges(conditions)
        if genre_pref:
            mask = self._apply_genre_filter(mask, genre_pref)

        mask, conditions = self._fallback_cascade(mask, conditions, genre_pref, top_k)

        candidates = self.df[mask].nlargest(top_k, "track_popularity")
        return self._format_results(candidates.index.tolist())

    def recommend_similar(
        self,
        seed_track: str | None = None,
        seed_artist: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Recommend tracks similar to a seed track via cosine similarity.

        Parameters
        ----------
        seed_track:
            Track name to search for. Returns [] if both seed_track and
            seed_artist are None.
        seed_artist:
            Optional artist name for disambiguation.
        top_k:
            Number of results to return (seed itself excluded).
        """
        if seed_track is None and seed_artist is None:
            return []

        seed_idx = self._find_seed_index(seed_track, seed_artist)
        if seed_idx is None:
            return []

        seed_vec = self.feature_matrix[seed_idx].reshape(1, -1)
        scores = cosine_similarity(seed_vec, self.feature_matrix)[0]

        # Exclude all rows matching the seed's track_name + track_artist.
        # The dataset can have multiple track_ids for the same song (different
        # versions / releases), so excluding only by index is insufficient.
        seed_row = self.df.iloc[seed_idx]
        same_track_mask = (
            (self.df["track_name"] == seed_row["track_name"])
            & (self.df["track_artist"] == seed_row["track_artist"])
        )
        scores[same_track_mask.values] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._format_results(top_indices)

    def recommend(self, intent: str, params: dict) -> list[dict]:
        """Dispatch router output to the appropriate recommendation strategy.

        Parameters
        ----------
        intent:
            One of "emotion", "situation", "similar". Returns [] for unknown.
        params:
            Dict with optional keys: mood, situation, genre_pref, seed_track,
            seed_artist. None values are safe.
        """
        if intent == "emotion":
            return self.recommend_by_emotion(
                mood=params.get("mood"),
                genre_pref=params.get("genre_pref"),
            )
        elif intent == "situation":
            return self.recommend_by_situation(
                situation=params.get("situation"),
                genre_pref=params.get("genre_pref"),
            )
        elif intent == "similar":
            return self.recommend_similar(
                seed_track=params.get("seed_track"),
                seed_artist=params.get("seed_artist"),
            )
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_ranges(self, conditions: dict[str, tuple[float, float]]) -> pd.Series:
        """Build a boolean mask by applying feature range conditions to self.df.

        All conditions reference self.df columns directly — both raw-scale
        (tempo BPM, loudness dB) and normalized [0, 1] features.
        """
        mask = pd.Series(True, index=self.df.index)
        for feat, (lo, hi) in conditions.items():
            mask &= (self.df[feat] >= lo) & (self.df[feat] <= hi)
        return mask

    def _apply_genre_filter(self, mask: pd.Series, genre_pref: str) -> pd.Series:
        """Narrow mask to tracks whose playlist_genres list contains genre_pref.

        playlist_genres is list[str]; genre_pref is a single string.
        Uses `genre_pref in genres_list` — not character-level iteration.
        """
        genre_match = self.df["playlist_genres"].apply(lambda gs: genre_pref in gs)
        return mask & genre_match

    def _build_ideal_vector(self, mood: str) -> np.ndarray:
        """Construct an 8-dim ideal feature vector for cosine similarity.

        Features specified in MOOD_MAPPING (valence, energy) are set to the
        midpoint of their range. All other features use the dataset column
        median as a neutral value that avoids penalising tracks for features
        the mood does not constrain.

        Note: MOOD_MAPPING uses raw column names (valence, energy), but the
        cosine vector uses tempo_norm (not tempo). The mapping has no tempo
        entry, so tempo_norm falls back to its dataset median automatically.
        """
        vector = np.array([self._medians[feat] for feat in COSINE_FEATURES])
        for feat, (lo, hi) in MOOD_MAPPING[mood].items():
            if feat in COSINE_FEATURES:
                idx = COSINE_FEATURES.index(feat)
                vector[idx] = (lo + hi) / 2.0
        return vector

    def _find_seed_index(
        self,
        seed_track: str | None,
        seed_artist: str | None,
    ) -> int | None:
        """Find df row index for a seed track using a 3-stage priority search.

        Priority (highest to lowest):
        1. track exact + artist exact (only when seed_artist is provided)
        2. track exact only
        3. partial match (case-insensitive substring)

        Within each stage, the track with the highest track_popularity wins.
        Returns None if no match is found at any stage.
        """
        if seed_track is None:
            return None

        track_lower = seed_track.lower()
        name_col = self.df["track_name"].str.lower()

        # Stage 1: exact track + exact artist (requires seed_artist)
        if seed_artist:
            artist_lower = seed_artist.lower()
            artist_col = self.df["track_artist"].str.lower()
            stage1 = self.df[(name_col == track_lower) & (artist_col == artist_lower)]
            if not stage1.empty:
                return int(stage1["track_popularity"].idxmax())

        # Stage 2: exact track name only
        stage2 = self.df[name_col == track_lower]
        if not stage2.empty:
            return int(stage2["track_popularity"].idxmax())

        # Stage 3: partial match (substring, literal, case-insensitive)
        stage3 = self.df[
            self.df["track_name"].str.contains(seed_track, case=False, regex=False)
        ]
        if not stage3.empty:
            return int(stage3["track_popularity"].idxmax())

        return None

    def _format_results(self, indices) -> list[dict]:
        """Convert df row indices to a list of result dicts.

        Returns only the columns in _RESULT_COLUMNS. Mental_Health_Label is
        intentionally excluded (ADR-004: never expose diagnosis labels to UI).
        """
        available = [c for c in _RESULT_COLUMNS if c in self.df.columns]
        return self.df.iloc[indices][available].to_dict("records")

    def _widen_ranges(
        self,
        conditions: dict[str, tuple[float, float]],
        factor: float,
    ) -> dict[str, tuple[float, float]]:
        """Expand each feature range by factor × range_width on both sides.

        Clamp bounds differ by feature type:
        - Normalized [0, 1] features: clamp to [0.0, 1.0]
        - Raw-scale features (tempo, loudness): clamp to observed min/max in df

        Example: valence (0.6, 1.0), factor=0.2
            width=0.4, delta=0.08 → (0.52, 1.08) → clamped to (0.52, 1.0)
        """
        widened: dict[str, tuple[float, float]] = {}
        for feat, (lo, hi) in conditions.items():
            delta = (hi - lo) * factor
            new_lo = lo - delta
            new_hi = hi + delta
            if feat in _RAW_SCALE_FEATURES and feat in self._raw_bounds:
                col_min, col_max = self._raw_bounds[feat]
                new_lo = max(new_lo, col_min)
                new_hi = min(new_hi, col_max)
            else:
                new_lo = max(new_lo, 0.0)
                new_hi = min(new_hi, 1.0)
            widened[feat] = (new_lo, new_hi)
        return widened

    def _fallback_cascade(
        self,
        mask: pd.Series,
        conditions: dict[str, tuple[float, float]],
        genre_pref: str | None,
        top_k: int,
    ) -> tuple[pd.Series, dict[str, tuple[float, float]]]:
        """Progressively relax filter conditions when candidates < top_k.

        Stage 1: drop genre_pref
        Stage 2: widen feature ranges by 20%
        Stage 3: widen feature ranges by 50% (return whatever remains)

        Returns the relaxed (mask, conditions) pair.
        """
        if mask.sum() >= top_k:
            return mask, conditions

        # Stage 1: drop genre filter
        if genre_pref:
            mask = self._filter_by_ranges(conditions)
            if mask.sum() >= top_k:
                return mask, conditions

        # Stage 2: widen by 20%
        conditions_20 = self._widen_ranges(conditions, 0.2)
        mask = self._filter_by_ranges(conditions_20)
        if mask.sum() >= top_k:
            return mask, conditions_20

        # Stage 3: widen by 50%
        conditions_50 = self._widen_ranges(conditions, 0.5)
        mask = self._filter_by_ranges(conditions_50)
        return mask, conditions_50
