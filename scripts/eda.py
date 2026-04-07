"""
EDA script for Music_recommendation.csv.

Usage:
    python scripts/eda.py              # stdout
    python scripts/eda.py > out.md     # redirect to file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "Music_recommendation.csv"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

MOOD_MAPPING: dict[str, dict[str, tuple[float, float]]] = {
    "happy":   {"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
    "sad":     {"valence": (0.0, 0.3), "energy": (0.0, 0.4)},
    "angry":   {"valence": (0.0, 0.4), "energy": (0.7, 1.0)},
    "calm":    {"valence": (0.3, 0.6), "energy": (0.0, 0.4)},
    "excited": {"valence": (0.7, 1.0), "energy": (0.7, 1.0)},
    "anxious": {"valence": (0.1, 0.4), "energy": (0.4, 0.7)},
    "empty":   {"valence": (0.2, 0.5), "energy": (0.2, 0.5)},
}

SITUATION_MAPPING: dict[str, dict[str, tuple[float, float]]] = {
    "카페":    {"energy": (0.1, 0.4), "acousticness": (0.3, 1.0), "speechiness": (0.0, 0.1)},
    "파티":    {"energy": (0.7, 1.0), "danceability": (0.7, 1.0), "valence": (0.5, 1.0)},
    "코딩":    {"energy": (0.2, 0.5), "instrumentalness": (0.3, 1.0), "speechiness": (0.0, 0.08)},
    "운동":    {"energy": (0.7, 1.0), "tempo": (120.0, 240.0)},
    "수면":    {"energy": (0.0, 0.3), "acousticness": (0.4, 1.0), "loudness": (-46.0, -15.0)},
    "드라이브": {"energy": (0.5, 0.8), "valence": (0.4, 0.8), "tempo": (90.0, 140.0)},
}

# Overall genre distribution (computed once for bias comparison)
_GENRE_DIST: pd.Series | None = None


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def section(title: str) -> None:
    print(f"\n## {title}\n")


def md_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------

def section_overview(df: pd.DataFrame) -> None:
    section("1. Dataset Overview")

    total_rows, total_cols = df.shape
    unique_tracks = df["track_id"].nunique()
    unique_artists = df["track_artist"].nunique()
    unique_albums = df["track_album_id"].nunique()
    unique_playlists = df["playlist_id"].nunique()

    print(f"- Total rows: {total_rows:,}")
    print(f"- Total columns: {total_cols}")
    print(f"- Unique track_id: {unique_tracks:,}")
    print(f"- Unique artists: {unique_artists:,}")
    print(f"- Unique albums: {unique_albums:,}")
    print(f"- Unique playlists: {unique_playlists:,}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["column", "missing_count"]
    missing["pct"] = (missing["missing_count"] / total_rows * 100).round(2)
    print(f"\n### Missing Values\n")
    if missing.empty:
        print("No missing values.")
    else:
        print(md_table(missing))

    # Duplicate track analysis
    section_duplicates(df)


def section_duplicates(df: pd.DataFrame) -> None:
    print("\n### Duplicate Track Analysis\n")

    dup_rows = df.duplicated(subset="track_id").sum()
    unique_tracks = df["track_id"].nunique()
    print(f"- Rows with duplicate track_id: {dup_rows:,}")
    print(f"- Unique tracks: {unique_tracks:,}")
    print(f"- After dedup (by track_id, keep max popularity): {unique_tracks:,} rows")

    # Distribution: how many playlists does a track appear in?
    playlist_counts = (
        df.groupby("track_id")["playlist_id"]
        .nunique()
        .value_counts()
        .sort_index()
        .reset_index()
    )
    playlist_counts.columns = ["n_playlists", "n_tracks"]
    playlist_counts["pct"] = (playlist_counts["n_tracks"] / unique_tracks * 100).round(1)
    print("\n**Track × Playlist appearance distribution:**\n")
    print(md_table(playlist_counts))
    print(
        "\n> Dedup strategy: keep row with max `track_popularity` per `track_id`. "
        "Ensures diverse playlist coverage is collapsed to the best-known version."
    )


# ---------------------------------------------------------------------------
# 2. Audio feature distributions
# ---------------------------------------------------------------------------

def section_audio_features(df: pd.DataFrame) -> None:
    section("2. Audio Feature Distributions")

    stats = df[AUDIO_FEATURES].describe().T.round(3)
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].reset_index()
    stats.columns = ["feature", "mean", "std", "min", "p25", "median", "p75", "max"]
    print(md_table(stats))

    print(
        "\n> **Normalization required**: `loudness` (range -46~1 dB) and `tempo` (0~239 BPM) "
        "are on different scales from the 0-1 normalized features. Min-max scaling needed before "
        "cosine similarity calculation."
    )


# ---------------------------------------------------------------------------
# 3. Correlation analysis
# ---------------------------------------------------------------------------

def section_correlation(df: pd.DataFrame) -> None:
    section("3. Correlation Analysis (feature redundancy for cosine similarity)")

    corr = df[AUDIO_FEATURES].corr().round(3)

    # Find pairs with |r| > 0.4
    pairs = []
    features = AUDIO_FEATURES
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.loc[features[i], features[j]]
            if abs(r) >= 0.4:
                pairs.append({"feature_a": features[i], "feature_b": features[j], "r": r})

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
        print("**Pairs with |r| ≥ 0.4:**\n")
        print(md_table(pairs_df))
    else:
        print("No feature pairs with |r| ≥ 0.4.")

    print(
        "\n> **Implication for cosine similarity**: Highly correlated features contribute "
        "redundant signal. e.g. energy–loudness: if both included, 'energy' direction is "
        "over-weighted in distance calculation. Consider excluding `loudness` from similarity "
        "features and using it only as a hard filter (situation mapping)."
    )


# ---------------------------------------------------------------------------
# 4. Category distributions
# ---------------------------------------------------------------------------

def section_categories(df: pd.DataFrame) -> None:
    global _GENRE_DIST
    section("4. Category Distributions")

    # Genre
    print("### playlist_genre\n")
    genre_dist = df["playlist_genre"].value_counts().reset_index()
    genre_dist.columns = ["genre", "count"]
    genre_dist["pct"] = (genre_dist["count"] / len(df) * 100).round(1)
    _GENRE_DIST = genre_dist.set_index("genre")["pct"]
    print(md_table(genre_dist))

    # Subgenre top 10
    print("\n### playlist_subgenre (top 10)\n")
    sg = df["playlist_subgenre"].value_counts().head(10).reset_index()
    sg.columns = ["subgenre", "count"]
    sg["pct"] = (sg["count"] / len(df) * 100).round(1)
    print(md_table(sg))

    # Mental_Health_Label
    print("\n### Mental_Health_Label\n")
    mhl = df["Mental_Health_Label"].value_counts().reset_index()
    mhl.columns = ["label", "count"]
    mhl["pct"] = (mhl["count"] / len(df) * 100).round(1)
    print(md_table(mhl))

    # Mental_Health_Label × genre cross-tab
    print("\n### Mental_Health_Label × genre (row %)\n")
    cross = pd.crosstab(
        df["Mental_Health_Label"], df["playlist_genre"], normalize="index"
    ).round(3) * 100
    cross = cross.round(1).reset_index()
    print(md_table(cross))

    # instrument
    print("\n### instrument\n")
    inst = df["instrument"].value_counts().reset_index()
    inst.columns = ["instrument", "count"]
    inst["pct"] = (inst["count"] / len(df) * 100).round(1)
    print(md_table(inst))

    # popularity
    print("\n### track_popularity stats\n")
    pop_stats = df["track_popularity"].describe().round(1).reset_index()
    pop_stats.columns = ["stat", "value"]
    print(md_table(pop_stats))


# ---------------------------------------------------------------------------
# 5. Mapping validation
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, conditions: dict[str, tuple[float, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for feature, (lo, hi) in conditions.items():
        mask &= (df[feature] >= lo) & (df[feature] <= hi)
    return df[mask]


def _genre_bias(matched: pd.DataFrame, label: str) -> str:
    """Return bias note if any genre deviates > 20pp from overall distribution."""
    if matched.empty or _GENRE_DIST is None:
        return "N/A"
    match_dist = matched["playlist_genre"].value_counts(normalize=True) * 100
    biased = []
    for g, pct in match_dist.items():
        overall = _GENRE_DIST.get(g, 0.0)
        if abs(pct - overall) > 20:
            biased.append(f"{g} ({pct:.0f}% vs {overall:.0f}% overall)")
    return "; ".join(biased) if biased else "none"


TOP_LABELS = {"Normal/Unclassified", "Bipolar (Mania)", "Anxiety"}


def section_mapping_validation(df: pd.DataFrame) -> None:
    section("5. Mapping Validation")
    total = len(df)

    # Pre-compute label-filtered subset for post-filter coverage comparison
    df_label_filtered = df[df["Mental_Health_Label"].isin(TOP_LABELS)]

    print("### MOOD_MAPPING\n")
    mood_rows = []
    for mood, conditions in MOOD_MAPPING.items():
        matched = _apply_filters(df, conditions)
        n = len(matched)
        pct = round(n / total * 100, 2)
        flag = "⚠️ <300" if n < 300 else "✅"
        bias = _genre_bias(matched, mood)

        # Label distribution
        label_dist = matched["Mental_Health_Label"].value_counts(normalize=True).round(3) * 100
        top_label = label_dist.idxmax() if not label_dist.empty else "N/A"
        top_label_pct = round(label_dist.max(), 1) if not label_dist.empty else 0

        # Post-filter count: audio filters applied on top-3-label subset
        n_post = len(_apply_filters(df_label_filtered, conditions))

        mood_rows.append({
            "mood": mood,
            "matched": n,
            "post_label_filter": n_post,
            "coverage_%": pct,
            "status": flag,
            "top_label": f"{top_label} ({top_label_pct}%)",
            "genre_bias": bias,
        })
    mood_df = pd.DataFrame(mood_rows)
    print(md_table(mood_df))

    print(
        "\n> `post_label_filter`: matches remaining after restricting to "
        "`Mental_Health_Label ∈ {Normal/Unclassified, Bipolar (Mania), Anxiety}` "
        "**before** audio-feature filtering. Use this column to evaluate whether "
        "a hard label gate is safe for each mood."
    )

    # Warn with adjustment suggestions for <300
    print()
    for row in mood_rows:
        if row["matched"] < 300:
            print(
                f"> **{row['mood']}** ({row['matched']} matches): "
                f"Consider widening ranges. Current conditions: "
                + "; ".join(
                    f"{f}={v}" for f, v in MOOD_MAPPING[row["mood"]].items()
                )
            )

    print("\n### SITUATION_MAPPING\n")
    sit_rows = []
    for situation, conditions in SITUATION_MAPPING.items():
        matched = _apply_filters(df, conditions)
        n = len(matched)
        pct = round(n / total * 100, 2)
        flag = "⚠️ <300" if n < 300 else "✅"
        bias = _genre_bias(matched, situation)

        # Post-filter count with top-3 label restriction
        n_post = len(_apply_filters(df_label_filtered, conditions))

        sit_rows.append({
            "situation": situation,
            "matched": n,
            "post_label_filter": n_post,
            "coverage_%": pct,
            "status": flag,
            "genre_bias": bias,
        })
    sit_df = pd.DataFrame(sit_rows)
    print(md_table(sit_df))

    print(
        "\n> `post_label_filter`: matches remaining after restricting to "
        "`Mental_Health_Label ∈ {Normal/Unclassified, Bipolar (Mania), Anxiety}` "
        "**before** audio-feature filtering. Situations with post-filter count < 300 "
        "must not use Mental_Health_Label as a hard gate."
    )

    print()
    for row in sit_rows:
        if row["matched"] < 300:
            conds = SITUATION_MAPPING[row["situation"]]
            print(
                f"> **{row['situation']}** ({row['matched']} matches): "
                f"Consider widening ranges. Current: "
                + "; ".join(f"{f}={v}" for f, v in conds.items())
            )


# ---------------------------------------------------------------------------
# 6. Preprocessing recommendations
# ---------------------------------------------------------------------------

def section_preprocessing(df: pd.DataFrame) -> None:
    section("6. Preprocessing Recommendations")

    # Cosine similarity feature set
    sim_features = [f for f in AUDIO_FEATURES if f not in ("loudness",)]
    print("**Cosine similarity feature set (proposed):**")
    print(", ".join(f"`{f}`" for f in sim_features))
    print(
        "\n> `loudness` excluded: high correlation with `energy` (~0.68) causes redundant "
        "signal in cosine distance. Used only as hard filter in situation mapping."
    )

    print("\n**Min-max normalization targets:** `loudness`, `tempo`")
    print("**Dedup:** `track_id`, keep row with max `track_popularity`")
    print("**Drop:** missing rows (5 each in track_name, track_artist, track_album_name)")
    print("**Exclude columns:** `instrument` (84% Unknown), ID columns (track_album_id, playlist_id, playlist_name)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("# EDA Report — Music Recommendation Dataset")
    print("\n> Auto-generated by `scripts/eda.py`. Interpretations written separately in eda_report.md.")

    df = load_data()

    section_overview(df)
    section_audio_features(df)
    section_correlation(df)
    section_categories(df)
    section_mapping_validation(df)
    section_preprocessing(df)


if __name__ == "__main__":
    main()
