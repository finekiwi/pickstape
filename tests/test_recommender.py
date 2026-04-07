"""Structural smoke tests for the Music_recommendation dataset."""

import pathlib

import pandas as pd

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
