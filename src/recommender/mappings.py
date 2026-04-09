"""
Mapping tables for mood/situation → audio feature ranges and Mental_Health_Label boosting.

All feature ranges reference df column names:
- Normalized [0, 1]: valence, energy, danceability, acousticness,
  speechiness, instrumentalness, liveness
- Raw scale: tempo (BPM), loudness (dB)  ← filters use df["tempo"] / df["loudness"]

Range values are derived from EDA (refs/eda_report.md, scripts/eda.py).
Key quantitative decisions are annotated inline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (lo, hi) range for a single audio feature
FeatureRange = tuple[float, float]

# mood/situation → {feature_name: (lo, hi)}
MoodConfig = dict[str, FeatureRange]
SituationConfig = dict[str, FeatureRange]


# ---------------------------------------------------------------------------
# MOOD_MAPPING
# ---------------------------------------------------------------------------
# Ranges chosen from EDA valence/energy percentile distributions per mood cluster.
# EDA pool sizes (post-filter, before genre/label filters):
#   happy: ~5,400  sad: ~928  angry: ~4,600  calm: ~932
#   excited: ~4,400  anxious: ~3,800  empty: ~1,800

MOOD_MAPPING: dict[str, MoodConfig] = {
    "happy":   {"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
    "sad":     {"valence": (0.0, 0.3), "energy": (0.0, 0.4)},
    "angry":   {"valence": (0.0, 0.4), "energy": (0.7, 1.0)},
    "calm":    {"valence": (0.3, 0.6), "energy": (0.0, 0.4)},
    "excited": {"valence": (0.7, 1.0), "energy": (0.7, 1.0)},
    "anxious": {"valence": (0.1, 0.4), "energy": (0.4, 0.7)},
    "empty":   {"valence": (0.2, 0.5), "energy": (0.2, 0.5)},
}


# ---------------------------------------------------------------------------
# SITUATION_MAPPING
# ---------------------------------------------------------------------------
# Keys "tempo" and "loudness" use raw scale (BPM / dB).
# All other keys use the normalized [0, 1] scale.
#
# EDA pool sizes (post-filter):
#   카페: ~1,200  파티: ~3,100  코딩: ~381  운동: ~4,800  수면: ~234  드라이브: ~2,900
#
# 코딩: instrumentalness >= 0.3 is strict (dataset median=0.0, p75=0.005).
#   381 tracks is sufficient for top-5 but fallback cascade covers edge cases.
# 수면: loudness threshold relaxed from -15 → -10 dB after EDA showed
#   -15 dB cut left only 109/234 tracks (54% loss). ADR-007.

SITUATION_MAPPING: dict[str, SituationConfig] = {
    "카페":    {
        "energy":       (0.1, 0.4),
        "acousticness": (0.3, 1.0),
        "speechiness":  (0.0, 0.1),
    },
    "파티":    {
        "energy":       (0.7, 1.0),
        "danceability": (0.7, 1.0),
        "valence":      (0.5, 1.0),
    },
    "코딩":    {
        "energy":           (0.2, 0.5),
        "instrumentalness": (0.3, 1.0),   # EDA: p75=0.005 → tight pool (~381곡), fallback ready
        "speechiness":      (0.0, 0.08),
    },
    "운동":    {
        "energy": (0.7, 1.0),
        "tempo":  (120.0, 240.0),          # raw BPM; EDA: 운동 tracks peak at 128-160 BPM
    },
    "수면":    {
        "energy":       (0.0, 0.3),
        "acousticness": (0.4, 1.0),
        "loudness":     (-46.0, -10.0),    # raw dB; relaxed from -15→-10 (EDA: -15 cut 54% of pool)
    },
    "드라이브": {
        "energy":  (0.5, 0.8),
        "valence": (0.4, 0.8),
        "tempo":   (90.0, 140.0),          # raw BPM
    },
}


# ---------------------------------------------------------------------------
# MENTAL_HEALTH_BOOST
# ---------------------------------------------------------------------------
# Soft ranking signal — additive score offset, never a hard filter (ADR-007).
# Hard filtering by label causes dangerous pool loss in low-coverage paths:
#   수면: 234 → 109 tracks (-54%), 코딩: 381 → 284 tracks (-25%).
#
# Only moods with strong EDA label overlap are boosted:
#   excited → Bipolar (Mania): 98.9% of excited-range tracks carry this label
#   angry   → Anxiety:         73.1% of angry-range tracks carry this label
# Other moods dominated by Normal/Unclassified (46–96%) → no meaningful signal.
#
# EDA: r(energy, loudness) = 0.677 → loudness excluded from cosine features (ADR-006).
# Format: mood → (Mental_Health_Label value, multiplier for boost offset calculation)

MENTAL_HEALTH_BOOST: dict[str, tuple[str, float]] = {
    "excited": ("Bipolar (Mania)", 1.5),   # 98.9% signal strength
    "angry":   ("Anxiety", 1.2),           # 73.1% signal strength
}
