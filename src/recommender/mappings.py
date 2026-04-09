"""
Mapping tables for mood/situation → audio feature ranges and Mental_Health_Label boosting.

All feature ranges reference df column names:
- Normalized [0, 1]: valence, energy, danceability, acousticness,
  speechiness, instrumentalness, liveness
- Raw scale: tempo (BPM), loudness (dB)  ← filters use df["tempo"] / df["loudness"]
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
        "energy":            (0.2, 0.5),
        "instrumentalness":  (0.3, 1.0),
        "speechiness":       (0.0, 0.08),
    },
    "운동":    {
        "energy": (0.7, 1.0),
        "tempo":  (120.0, 240.0),  # raw BPM
    },
    "수면":    {
        "energy":       (0.0, 0.3),
        "acousticness": (0.4, 1.0),
        "loudness":     (-46.0, -10.0),  # raw dB — EDA 검증: -15 → -10
    },
    "드라이브": {
        "energy":  (0.5, 0.8),
        "valence": (0.4, 0.8),
        "tempo":   (90.0, 140.0),  # raw BPM
    },
}


# ---------------------------------------------------------------------------
# MENTAL_HEALTH_BOOST
# ---------------------------------------------------------------------------
# Only moods with strong EDA signal are boosted.
# Format: mood → (Mental_Health_Label value, additive multiplier)
# - "excited" → Bipolar (Mania): 98.9% overlap in EDA
# - "angry"   → Anxiety:         73.1% overlap in EDA
# Other moods are dominated by Normal/Unclassified → no meaningful signal.

MENTAL_HEALTH_BOOST: dict[str, tuple[str, float]] = {
    "excited": ("Bipolar (Mania)", 1.5),
    "angry":   ("Anxiety", 1.2),
}
