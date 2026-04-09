"""Reusable Streamlit UI components for Pickstape.

Three public functions:
- render_sidebar()              — sidebar branding and usage hints
- render_recommendation_cards() — cassette-style track cards
- render_chat_message()         — unified chat bubble renderer
"""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> None:
    """Render the Pickstape sidebar with branding and usage examples."""
    st.sidebar.title("Pickstape")
    st.sidebar.caption("AI가 골라 담은 당신만의 테이프")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "기분, 상황, 또는 좋아하는 곡을 말해주세요. "
        "Pickstape가 딱 맞는 음악을 골라드려요."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**사용 예시**")
    st.sidebar.markdown(
        "- 우울한 기분에 어울리는 노래 추천해줘\n"
        "- 카페에서 들을 잔잔한 음악 추천해줘\n"
        "- 운동할 때 들을 신나는 곡 골라줘\n"
        "- 'Blinding Lights'랑 비슷한 곡 찾아줘"
    )


def render_recommendation_cards(recommendations: list[dict]) -> None:
    """Render a list of recommendation dicts as styled HTML cards.

    Parameters
    ----------
    recommendations:
        List of dicts from the recommendation engine. Expected keys:
        track_name, track_artist, track_album_name, playlist_genres (list),
        energy, valence, danceability, tempo (raw BPM float).
    """
    for track in recommendations:
        name = track.get("track_name", "Unknown")
        artist = track.get("track_artist", "Unknown")
        album = track.get("track_album_name", "")
        genres: list[str] = track.get("playlist_genres") or []
        energy = track.get("energy", 0.0)
        valence = track.get("valence", 0.0)
        danceability = track.get("danceability", 0.0)
        tempo = track.get("tempo", 0.0)

        # Genre badges
        badges_html = "".join(
            f'<span class="genre-badge">{g}</span>' for g in genres
        )

        # Audio feature bars (energy, valence, danceability)
        def _bar(label: str, value: float) -> str:
            pct = int(round(value * 100))
            return (
                f'<div class="feature-row">'
                f'  <span class="feature-label">{label}</span>'
                f'  <div class="feature-track">'
                f'    <div class="feature-fill" style="width:{pct}%"></div>'
                f'  </div>'
                f'  <span class="feature-value">{value:.2f}</span>'
                f'</div>'
            )

        bars_html = (
            _bar("Energy", energy)
            + _bar("Valence", valence)
            + _bar("Dance", danceability)
        )

        tempo_html = f'<div class="tempo-row">Tempo: {int(round(tempo))} BPM</div>'

        card_html = (
            f'<div class="recommendation-card">'
            f'  <div class="card-title">{name}</div>'
            f'  <div class="card-meta">{artist} &middot; {album}</div>'
            f'  <div style="margin-bottom:0.5rem">{badges_html}</div>'
            f'  {bars_html}'
            f'  {tempo_html}'
            f'</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)


def render_chat_message(
    role: str,
    content: str,
    recommendations: list[dict] | None = None,
) -> None:
    """Render a single chat message bubble.

    Used for both history replay and real-time response rendering so that
    the output is identical in both cases.

    Parameters
    ----------
    role:
        "user" or "assistant".
    content:
        The message text.
    recommendations:
        Optional list of recommendation dicts. Rendered as cards below
        the message text when present and non-empty.
    """
    with st.chat_message(role):
        st.markdown(content)
        if recommendations:
            render_recommendation_cards(recommendations)
