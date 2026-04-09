"""Shared test fixtures and helpers."""

from __future__ import annotations


def make_row(
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
    """Build a single synthetic row matching the preprocessed DataFrame schema."""
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
