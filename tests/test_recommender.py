"""Structural smoke tests for the Music_recommendation dataset."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.recommender.preprocess import COSINE_FEATURES, load_and_preprocess

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "Music_recommendation.csv"

EXPECTED_ROW_COUNT = 32833  # header excluded
EXPECTED_COLUMNS = {
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "playlist_genre",
    "danceability",
    "energy",
    "tempo",
    "Mental_Health_Label",
}


def test_csv_loads() -> None:
    """Dataset file exists and is readable by pandas."""
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset loaded but is empty"


def test_csv_row_count() -> None:
    """Dataset contains the expected number of rows (32,832 data rows)."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} rows, got {len(df)}"
    )


def test_csv_expected_columns_present() -> None:
    """Dataset contains the minimum required columns."""
    df = pd.read_csv(DATA_PATH)
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_csv_no_bom_in_first_column() -> None:
    """First column header must not contain a UTF-8 BOM character."""
    df = pd.read_csv(DATA_PATH)
    first_col = df.columns[0]
    assert "\ufeff" not in first_col, (
        f"UTF-8 BOM detected in first column header: {first_col!r}"
    )


# ---------------------------------------------------------------------------
# load_and_preprocess() unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preprocessed() -> tuple[pd.DataFrame, np.ndarray]:
    """Run load_and_preprocess() once per module and share the result."""
    return load_and_preprocess()


def test_preprocess_return_types(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """load_and_preprocess() must return (DataFrame, ndarray)."""
    df, feature_matrix = preprocessed
    assert isinstance(df, pd.DataFrame), "First return value must be a DataFrame"
    assert isinstance(feature_matrix, np.ndarray), (
        "Second return value must be a NumPy ndarray"
    )


def test_preprocess_track_id_unique(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """track_id must be unique after deduplication."""
    df, _ = preprocessed
    assert df["track_id"].is_unique, "Duplicate track_id values found after dedup"


def test_preprocess_no_nulls_in_core_columns(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Core columns must contain no null values."""
    df, _ = preprocessed
    core_cols = ["track_id", "track_name", "track_artist", "track_album_name"]
    for col in core_cols:
        null_count = df[col].isna().sum()
        assert null_count == 0, f"Null values found in column '{col}': {null_count}"


def test_preprocess_tempo_norm_range(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """tempo_norm values must fall within [0, 1]."""
    df, _ = preprocessed
    assert df["tempo_norm"].between(0.0, 1.0).all(), (
        "tempo_norm contains values outside the expected [0, 1] range"
    )


def test_preprocess_feature_matrix_shape(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Feature matrix must have shape (N, 8) aligned with the DataFrame."""
    df, feature_matrix = preprocessed
    n_features = len(COSINE_FEATURES)
    assert feature_matrix.shape == (len(df), n_features), (
        f"Expected feature_matrix shape ({len(df)}, {n_features}), "
        f"got {feature_matrix.shape}"
    )


def test_preprocess_playlist_genres_are_lists(
    preprocessed: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """playlist_genres and playlist_subgenres columns must contain lists."""
    df, _ = preprocessed
    assert df["playlist_genres"].apply(isinstance, args=(list,)).all(), (
        "playlist_genres column contains non-list values"
    )
    assert df["playlist_subgenres"].apply(isinstance, args=(list,)).all(), (
        "playlist_subgenres column contains non-list values"
    )
