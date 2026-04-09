"""Pickstape — Streamlit chatbot entry point.

Flow:
    load data → build engine → compile graph (cached)
    → chat UI loop → graph.invoke() → render response + cards
"""

from __future__ import annotations

import re

import streamlit as st

# ── Page config (must be the first Streamlit call) ───────────
st.set_page_config(
    page_title="Pickstape",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Response post-processing (runs every render, not cached) ─
#
# All defensive checks live here so they're immune to @st.cache_resource
# retaining stale graph/engine instances across Streamlit hot reloads.

# Inputs with no full Korean syllables or Latin letters (e.g. "ㅋㅋㅋ", "ㅠㅠ")
# carry no routeable intent — short-circuit to FALLBACK_REASK.
_ROUTEABLE_RE = re.compile(r"[가-힣a-zA-Z]")

# is_blocked_title imported below after set_page_config (Streamlit constraint on imports).

_FALLBACK_REASK: str = (
    "죄송해요, 요청을 정확히 이해하지 못했어요. 😅\n"
    "어떤 기분이신지, 어떤 상황에서 들을 음악인지, "
    "또는 좋아하는 곡 이름을 알려주시면 딱 맞는 곡을 골라드릴게요!"
)

# Korean labels for mood values (used in contextual templates).
_MOOD_LABELS: dict[str, str] = {
    "happy":   "행복한",
    "sad":     "슬픈",
    "angry":   "화난",
    "calm":    "잔잔한",
    "excited": "신나는",
    "anxious": "불안한",
    "empty":   "공허한",
}


def _is_routeable(text: str) -> bool:
    """Return True if the input contains enough content to route.

    Returns False for:
    - Inputs longer than 500 characters (prevents LLM timeout)
    - Inputs with no Korean syllables or Latin letters (e.g. emoji-only, jamo-only)
    """
    if len(text.strip()) > 500:
        return False
    return bool(_ROUTEABLE_RE.search(text))


def _build_template_response(intent: str, params: dict) -> str:
    """Build a short, contextual Korean response using intent and extracted params.

    LLM free-generation was replaced with templates: Qwen3.5-4B produced
    inconsistent Korean, leaked foreign characters, and used awkward phrasing.
    Templates are stable, fast, and presentation-safe.
    """
    if intent == "emotion":
        mood = params.get("mood")
        label = _MOOD_LABELS.get(mood, "지금 기분에 맞는") if mood else "지금 기분에 맞는"
        return f"{label} 분위기의 곡들을 골라봤어요. 마음에 드는 곡이 있길 바라요!"
    if intent == "situation":
        situation = params.get("situation") or "이 상황"
        return f"{situation}에 딱 맞는 곡들을 골라봤어요. 좋은 시간 되세요!"
    if intent == "similar":
        seed = params.get("seed_track") or ""
        artist = params.get("seed_artist") or ""
        if seed:
            ref = f"'{seed}'" + (f" - {artist}" if artist else "")
            return f"{ref}와 비슷한 분위기의 곡들을 찾아봤어요."
        return "비슷한 느낌의 곡들을 찾아봤어요."
    return "추천 곡을 골라봤어요."


def _filter_recommendations(recs: list[dict]) -> list[dict]:
    """Blocklist (word-boundary) → dedup by (track_name, track_artist).

    Uses is_blocked_title() from engine so the check logic is never duplicated.
    Applied after graph.invoke() so it is immune to @st.cache_resource
    retaining stale engine instances across hot reloads.
    """
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for r in recs:
        name = r.get("track_name", "").strip()
        artist = r.get("track_artist", "").strip().lower()
        if is_blocked_title(name):
            continue
        key = (name.lower(), artist)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    return result


# ── Imports (after set_page_config) ─────────────────────────
from src.agent import build_graph
from src.recommender import RecommendationEngine, load_and_preprocess
from src.recommender.engine import is_blocked_title
from src.ui.components import render_chat_message, render_recommendation_cards, render_sidebar
from src.ui.styles import inject_base_css


# ── Resource initialisation (run once, cached across reruns) ─
@st.cache_resource(show_spinner="데이터를 불러오는 중...")
def init_resources():
    """Load dataset, build recommendation engine, compile LangGraph.

    Returns (graph, engine) so app.py can call engine helpers (e.g.
    is_ambiguous_title) without going through the cached graph.
    """
    df, feature_matrix = load_and_preprocess()
    engine = RecommendationEngine(df, feature_matrix)
    return build_graph(engine), engine


# Initialise — show error page if data/model setup fails
try:
    graph, engine = init_resources()
except Exception as e:
    st.error(
        "데이터 초기화에 실패했어요. "
        "앱을 새로고침하거나 서버 상태를 확인해주세요."
    )
    st.exception(e)
    st.stop()

# ── CSS + sidebar ─────────────────────────────────────────────
inject_base_css()
render_sidebar()

# ── Session state ─────────────────────────────────────────────
_WELCOME: str = (
    "안녕하세요! Pickstape에 오신 걸 환영해요.\n\n"
    "기분이 어때요? 어떤 상황에서 들을 음악인지, "
    "또는 좋아하는 곡 이름을 알려주시면 딱 맞는 테이프를 골라드릴게요!"
)

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = [
        {"id": 0, "role": "assistant", "content": _WELCOME, "recommendations": None}
    ]
if "active_feedback_id" not in st.session_state:
    st.session_state.active_feedback_id = None
if "_next_msg_id" not in st.session_state:
    st.session_state._next_msg_id = 1

# ── "비슷한 곡 찾기" — query param handler ────────────────────
# Triggered when user clicks the HTML anchor in the card:
#   <a href="?fs=1&fn=Track+Name&fa=Artist+Name">비슷한 곡 찾기</a>
# Streamlit reruns with those params; we process here and clear immediately.
if st.query_params.get("fs") == "1":
    _fs_name = st.query_params.get("fn", "")
    _fs_artist = st.query_params.get("fa", "")
    st.query_params.clear()
    if _fs_name:
        with st.spinner("비슷한 곡을 찾고 있어요..."):
            _raw_recs = engine.recommend_similar(
                seed_track=_fs_name, seed_artist=_fs_artist, top_k=8,
            )
        _recommendations = _filter_recommendations(_raw_recs)[:4]
        _response_text = (
            f"'{_fs_name}'와(과) 비슷한 분위기의 곡들을 찾아봤어요."
            if _recommendations else _FALLBACK_REASK
        )
        _msg_id = st.session_state._next_msg_id
        st.session_state._next_msg_id += 1
        st.session_state.active_feedback_id = _msg_id if _recommendations else None
        st.session_state.messages.append({
            "id": _msg_id,
            "role": "assistant",
            "content": _response_text,
            "recommendations": _recommendations or None,
        })
        st.rerun()

# ── Replay chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    _is_active = msg.get("id") == st.session_state.active_feedback_id
    render_chat_message(
        msg["role"], msg["content"], msg.get("recommendations"),
        show_feedback=_is_active,
        msg_id=msg.get("id") if _is_active else None,
    )

# ── Chat input ────────────────────────────────────────────────
if user_input := st.chat_input("어떤 음악을 찾고 계세요?"):
    # Display and store user message
    render_chat_message("user", user_input)
    st.session_state.messages.append(
        {"id": st.session_state._next_msg_id, "role": "user", "content": user_input, "recommendations": None}
    )
    st.session_state._next_msg_id += 1

    # Invoke agent (skip graph for non-routeable input)
    if not _is_routeable(user_input):
        response_text = _FALLBACK_REASK
        recommendations = []
    else:
        try:
            with st.spinner("테이프를 고르고 있어요..."):
                result = graph.invoke({"user_input": user_input})
            intent = result.get("intent", "fallback")
            params = result.get("params") or {}

            # Disambiguation: similar + no artist + common title (≥3 distinct artists)
            seed_track_val = params.get("seed_track") or ""
            seed_artist_val = params.get("seed_artist") or ""
            if (
                intent == "similar"
                and seed_track_val
                and not seed_artist_val
                and engine.is_ambiguous_title(seed_track_val)
            ):
                artists = (
                    engine.df[
                        engine.df["track_name"].str.lower() == seed_track_val.lower()
                    ]["track_artist"]
                    .unique()[:3]
                )
                examples = ", ".join(f"'{seed_track_val} - {a}'" for a in artists)
                response_text = (
                    f"'{seed_track_val}'이라는 제목의 곡이 여러 아티스트에게 있어요. "
                    f"어떤 버전을 찾으시나요? (예: {examples})"
                )
                recommendations = []
            else:
                response_text = _build_template_response(intent, params)
                recommendations = _filter_recommendations(result.get("recommendations", []))[:4]
                if not recommendations:
                    response_text = _FALLBACK_REASK
        except Exception:
            response_text = "서버 연결에 실패했어요. 잠시 후 다시 시도해주세요."
            recommendations = []

    # Store assistant message, then rerun so replay handles all rendering
    # (including feedback — avoids double-render path)
    msg_id = st.session_state._next_msg_id
    st.session_state._next_msg_id += 1
    st.session_state.active_feedback_id = msg_id if bool(recommendations) else None
    st.session_state.messages.append(
        {
            "id": msg_id,
            "role": "assistant",
            "content": response_text,
            "recommendations": recommendations or None,
        }
    )
    st.rerun()
