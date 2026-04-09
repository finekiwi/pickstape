"""
Data preprocessing pipeline for Music_recommendation.csv.

Provides load_and_preprocess() which returns a cleaned DataFrame and a
NumPy feature matrix ready for cosine similarity computation.
"""

from __future__ import annotations

from pathlib import Path

from collections import Counter

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "Music_recommendation.csv"

# Columns used for cosine similarity. tempo_norm replaces raw tempo so that
# all values are on the [0, 1] scale.
COSINE_FEATURES: list[str] = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo_norm",
]

# Columns that carry no predictive or display value after preprocessing.
_DROP_COLS: list[str] = [
    "instrument",
    "track_album_id",
    "track_album_release_date",
    "playlist_id",
    "playlist_name",
    "key",
    "mode",
]


def load_and_preprocess(
    data_path: Path | str = DATA_PATH,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and preprocess the music recommendation dataset.

    Steps
    -----
    1. Load CSV (utf-8-sig encoding).
    2. Drop rows with missing track_name / track_artist / track_album_name.
    3. Aggregate playlist_genre and playlist_subgenre per track_id into lists
       so that multi-genre tracks are not silently truncated during dedup.
    4. Deduplicate by track_id, keeping the row with the highest
       track_popularity. Tie-breaking order: playlist_genre, playlist_subgenre,
       playlist_name (alphabetical) for reproducibility. playlist_name is
       dropped in step 6 and does not appear in the returned DataFrame.
    5. Merge aggregated genre lists back onto the deduplicated frame.
    6. Drop low-value columns.
    7. Add tempo_norm — tempo divided by a fixed 240 BPM ceiling — for the
       cosine feature matrix. Using a fixed divisor keeps the transform
       data-independent and safe for subset fixtures. Raw tempo (BPM) is
       preserved for situation-based hard filters. Raw loudness (dB) is also
       preserved for the same reason.
    8. Reset index for alignment with feature_matrix.
    9. Build the (N, 8) cosine feature matrix from COSINE_FEATURES.

    Parameters
    ----------
    data_path:
        Path to Music_recommendation.csv. Defaults to the project-level data
        directory relative to this file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame indexed by integer position. Contains tempo (BPM),
        loudness (dB), tempo_norm, playlist_genres (list), and
        playlist_subgenres (list).
    feature_matrix : np.ndarray
        Shape (N, 8) float64 array aligned with df. Each row corresponds to
        the same row in df. Columns follow COSINE_FEATURES order.
    """
    # --- 1. Load CSV ---
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    # --- 2. Drop rows with missing core text fields ---
    df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

    # --- 3. Aggregate genres/subgenres before dedup ---
    # 1,686 tracks appear in multiple playlist_genre categories. Collapsing to
    # a single row would silently drop genre information for those tracks.
    genre_agg = (
        df.groupby("track_id")
        .agg(
            playlist_genres=("playlist_genre", lambda x: [g for g, _ in Counter(x).most_common()]),
            playlist_subgenres=("playlist_subgenre", lambda x: sorted(set(x))),
        )
        .reset_index()
    )

    # --- 4. Deduplicate by track_id (keep max popularity row) ---
    # playlist_name is used here only as a stable tiebreaker; it is dropped in
    # step 6 via _DROP_COLS so it does not appear in the returned DataFrame.
    df = df.sort_values(
        [
            "track_id",
            "track_popularity",
            "playlist_genre",
            "playlist_subgenre",
            "playlist_name",
        ],
        ascending=[True, False, True, True, True],
    ).drop_duplicates(subset="track_id", keep="first")

    # --- 5. Merge aggregated genre lists ---
    df = df.merge(genre_agg, on="track_id", how="left")

    # Remove original single-value genre columns (replaced by list columns)
    df = df.drop(columns=["playlist_genre", "playlist_subgenre"])

    # --- 6. Drop low-value columns ---
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # --- 7. Add tempo_norm ---
    # Fixed denominator (240 BPM) makes the transform data-independent and
    # safe for subset fixtures and future test data that may not span the full
    # BPM range observed in the training dataset.
    _TEMPO_MAX = 240.0
    df["tempo_norm"] = df["tempo"] / _TEMPO_MAX

    # --- 8. Reset index for alignment with feature_matrix ---
    df = df.reset_index(drop=True)

    # --- 9. Build cosine feature matrix ---
    feature_matrix = df[COSINE_FEATURES].to_numpy(dtype=np.float64)

    return df, feature_matrix
