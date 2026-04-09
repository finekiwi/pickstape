"""Reusable Streamlit UI components for Pickstape.

Three public functions:
- render_sidebar()              — sidebar branding and usage hints
- render_recommendation_cards() — cassette-style track cards
- render_chat_message()         — unified chat bubble renderer
"""

from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st


_EXAMPLES = [
    ("A1", "우울한 기분에 어울리는 노래 추천해줘"),
    ("A2", "카페에서 들을 잔잔한 음악 추천해줘"),
    ("A3", "운동할 때 들을 신나는 곡 골라줘"),
    ("A4", "'Blinding Lights'랑 비슷한 곡 찾아줘"),
]

@lru_cache(maxsize=1)
def _get_logo_b64() -> str | None:
    """Load logo.png as base64, cached via lru_cache."""
    path = Path("assets/logo.png")
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def render_sidebar() -> None:
    """Render the Pickstape sidebar with cassette-label style."""
    b64 = _get_logo_b64()
    logo_img = (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:36px;height:36px;object-fit:contain;vertical-align:middle;margin-right:8px">'
        if b64 else ""
    )

    tracks_html = "".join(
        f'<div class="sidebar-track{" sidebar-track-even" if i % 2 == 1 else ""}">'
        f'  <span class="sidebar-track-num">{html.escape(num)}</span>'
        f'  <span class="sidebar-track-title">{html.escape(title)}</span>'
        f'</div>'
        for i, (num, title) in enumerate(_EXAMPLES)
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-logo-block">
          <div class="sidebar-logo-row">
            {logo_img}<span class="sidebar-logo">PICKSTAPE</span>
          </div>
          <div class="sidebar-subtitle">AI가 골라 담은 당신만의 테이프</div>
          <div class="sidebar-tape-meta">SIDE A · 90 min</div>
        </div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-desc">기분, 상황, 또는 좋아하는 곡을 말해주세요.</div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-tracklist-block">
          <div class="sidebar-section-label">TRACKLIST</div>
          {tracks_html}
        </div>
        <div class="sidebar-rec">REC <span class="sidebar-rec-dot">●</span> PICKSTAPE 2026</div>
        """,
        unsafe_allow_html=True,
    )


_FEATURE_COLORS = [
    (0.8, "rgba(255,255,255,1.0)"),
    (0.6, "rgba(255,255,255,0.8)"),
    (0.4, "rgba(255,255,255,0.6)"),
    (0.2, "rgba(255,255,255,0.45)"),
    (0.0, "rgba(255,255,255,0.3)"),
]

_REEL_HTML = (
    '<div class="reel-hole"><div class="reel-hole-inner"></div></div>'
    '<div class="reel-hole"><div class="reel-hole-inner"></div></div>'
)


def _bar_color(value: float) -> str:
    for threshold, color in _FEATURE_COLORS:
        if value >= threshold:
            return color
    return "#F0D4DE"


def _vbar(label: str, value: float) -> str:
    height = int(round(value * 24))
    color = _bar_color(value)
    return (
        f'<div class="vbar-col">'
        f'  <div class="vbar-fill" style="height:{height}px;background:{color}"></div>'
        f'  <div class="vbar-label">{label}</div>'
        f'</div>'
    )


def _build_card_html(track: dict, show_find_similar: bool = False) -> str:
    """Build the full VHS cassette card as a single HTML string.

    Includes an explicit .vhs-inner-panel (white) so the white panel is
    always rendered as part of the HTML — not dependent on Streamlit DOM
    structure. Spotify link is an HTML anchor inside .card-actions.
    When show_find_similar=True, a visual placeholder div is rendered in
    .card-actions to reserve vertical space; the actual clickable button
    is a Streamlit st.button() rendered separately (see render_recommendation_card).
    """
    name = html.escape(track.get("track_name", "Unknown"))
    artist = html.escape(track.get("track_artist", "Unknown"))
    album = html.escape(track.get("track_album_name", "") or "")
    genres: list[str] = [html.escape(g) for g in (track.get("playlist_genres") or [])[:2]]
    energy = track.get("energy", 0.0)
    valence = track.get("valence", 0.0)
    danceability = track.get("danceability", 0.0)
    acousticness = track.get("acousticness", 0.0)
    instrumentalness = track.get("instrumentalness", 0.0)
    tempo_norm = min(track.get("tempo", 0.0) / 240.0, 1.0)
    track_id = track.get("track_id", "")

    badges_html = "".join(f'<span class="genre-badge">{g}</span>' for g in genres)
    bars_html = (
        _vbar("E", energy)
        + _vbar("D", danceability)
        + _vbar("A", acousticness)
        + _vbar("I", instrumentalness)
        + _vbar("V", valence)
        + _vbar("T", tempo_norm)
    )
    spotify_html = (
        f'<a class="spotify-btn" href="https://open.spotify.com/track/{quote(track_id, safe="")}" '
        f'target="_blank" rel="noopener noreferrer">▶ Spotify에서 열기</a>'
        if track_id else ""
    )
    # Placeholder reserves the same height as the real st.button so the
    # card's inner-panel height stays consistent whether the button is shown or not.
    find_similar_placeholder = (
        '<div class="find-similar-placeholder">비슷한 곡 찾기</div>'
        if show_find_similar else ""
    )
    return (
        f'<div class="vhs-card">'
        f'  <div class="vhs-header">'
        f'    <span class="vhs-label">SIDE A</span>'
        f'    <div class="vhs-reels">{_REEL_HTML}</div>'
        f'  </div>'
        f'  <div class="vhs-inner-panel">'
        f'    <div class="card-main">'
        f'      <div class="card-info">'
        f'        <div class="card-title">{name}</div>'
        f'        <div class="card-meta">{artist} &middot; {album}</div>'
        f'        <div class="card-badges">{badges_html}</div>'
        f'      </div>'
        f'      <div class="card-actions">'
        f'        {spotify_html}'
        f'        {find_similar_placeholder}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="vbar-row">{bars_html}</div>'
        f'</div>'
    )


def render_recommendation_card(
    track: dict,
    show_find_similar: bool = False,
    btn_key: str | None = None,
) -> bool:
    """Render a single VHS cassette card.

    The card HTML (including the white inner panel) is rendered as a single
    st.markdown() block. When show_find_similar=True, the HTML contains a
    visual placeholder in .card-actions and the real clickable button is
    rendered immediately after with CSS aligning it over the placeholder.

    Returns True if '비슷한 곡 찾기' was clicked.
    """
    st.markdown(_build_card_html(track, show_find_similar), unsafe_allow_html=True)
    if show_find_similar and btn_key:
        return st.button("비슷한 곡 찾기", key=btn_key)
    return False


def render_recommendation_cards(
    recommendations: list[dict],
    show_feedback: bool = False,
    msg_id: int | None = None,
) -> dict | None:
    """Render recommendation dicts as VHS cassette cards in a 2-column grid.

    Parameters
    ----------
    recommendations:
        List of dicts from the recommendation engine. Expected keys:
        track_name, track_artist, track_album_name, playlist_genres (list),
        energy, valence, danceability, acousticness, instrumentalness,
        tempo (raw BPM float).
    show_feedback:
        If True, render the feedback UI and "이 곡으로 더 찾기" buttons below
        the cards. Only the latest recommendation turn should pass True.
    msg_id:
        Message ID used to generate unique button keys. Required for
        "이 곡으로 더 찾기" buttons; buttons are hidden when None.

    Returns
    -------
    dict | None
        If a "이 곡으로 더 찾기" button was clicked, returns
        {"track_name": str, "track_artist": str, "track_id": str}.
        Returns None if no button was clicked.
        Only the first click is captured (defensive against multiple True
        returns, though Streamlit only returns True for one button per rerun).
    """
    selected_seed: dict | None = None
    show_find_similar = show_feedback and msg_id is not None

    for i in range(0, len(recommendations), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(recommendations):
                track = recommendations[idx]
                track_id = track.get("track_id", str(idx))
                btn_key = f"find_similar_{msg_id}_{track_id}" if show_find_similar else None
                with col:
                    clicked = render_recommendation_card(
                        track,
                        show_find_similar=show_find_similar,
                        btn_key=btn_key,
                    )
                    if clicked and selected_seed is None:
                        selected_seed = {
                            "track_name": track.get("track_name", ""),
                            "track_artist": track.get("track_artist", ""),
                            "track_id": track_id,
                        }

    if not show_feedback:
        return selected_seed

    # Feedback UI — UI only, no logic
    st.markdown(
        '<div class="feedback-label">이 테이프가 마음에 드셨나요?</div>',
        unsafe_allow_html=True,
    )
    key_seed = (recommendations[0].get("track_id") or "fb") if recommendations else "fb"
    _, col_l, col_r, _ = st.columns([3, 1, 1, 3], gap="small")
    with col_l:
        if st.button("좋아요", key=f"fb_good_{key_seed}", use_container_width=True):
            st.toast("감사해요! 더 좋은 테이프를 만들어볼게요 🎵")
    with col_r:
        if st.button("다른 분위기로", key=f"fb_bad_{key_seed}", use_container_width=True):
            st.toast("다른 기분이나 상황을 말해주세요!")

    return selected_seed


def render_chat_message(
    role: str,
    content: str,
    recommendations: list[dict] | None = None,
    show_feedback: bool = False,
    msg_id: int | None = None,
) -> dict | None:
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
    show_feedback:
        If True, render the feedback UI and "이 곡으로 더 찾기" buttons.
        Should be True only for the active recommendation turn
        (controlled by app.py via active_feedback_id).
    msg_id:
        Passed through to render_recommendation_cards for button key uniqueness.

    Returns
    -------
    dict | None
        Passes through the return value of render_recommendation_cards —
        the selected seed track if a "이 곡으로 더 찾기" button was clicked,
        or None otherwise.
    """
    if role == "user":
        st.markdown(
            f'<div style="background-color:#FFF0F5;padding:12px 16px;'
            f'border-radius:20px 20px 4px 20px;margin:8px 0;color:#2D1B33;'
            f'max-width:60%;margin-left:auto;">'
            f'{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )
        return None

    avatar = "assets/logo.png" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if recommendations:
            return render_recommendation_cards(
                recommendations,
                show_feedback=show_feedback,
                msg_id=msg_id,
            )
    return None
