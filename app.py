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
    page_icon="🎵",
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

# Matches any whitespace-delimited token that contains at least one CJK
# ideograph — the entire token is dropped, not just the CJK character.
# "분위기를營造하기" → whole token removed (vs. leaving "분위기를하기").
_CJK_WORD_RE = re.compile(r"\S*[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+\S*")

# Track titles that must never appear in the UI regardless of audio features.
_TITLE_BLOCKLIST: frozenset[str] = frozenset({
    "suicidal", "suicide", "kill yourself", "kys",
    "die", "i want to die", "self harm",
})

_FALLBACK_REASK: str = (
    "죄송해요, 요청을 정확히 이해하지 못했어요. 😅\n"
    "어떤 기분이신지, 어떤 상황에서 들을 음악인지, "
    "또는 좋아하는 곡 이름을 알려주시면 딱 맞는 곡을 골라드릴게요!"
)

# Detects scripts beyond Korean + ASCII + Latin-1 supplement.
# Arabic, Devanagari, Thai, Latin Extended Additional (Vietnamese ổ ọ etc.),
# and any surviving hiragana/katakana/CJK after _CJK_WORD_RE are all caught.
# When triggered, response_text is replaced with a clean per-intent template.
_SUSPICIOUS_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff"    # Hiragana + Katakana
    r"\u4e00-\u9fff"     # CJK Unified Ideographs
    r"\u3400-\u4dbf"     # CJK Extension A
    r"\uf900-\ufaff"     # CJK Compatibility
    r"\u0600-\u06ff"     # Arabic
    r"\u0900-\u097f"     # Devanagari
    r"\u0e00-\u0e7f"     # Thai
    r"\u1e00-\u1eff"     # Latin Extended Additional (Vietnamese etc.)
    r"]"
)

_INTENT_TEMPLATES: dict[str, str] = {
    "emotion": "지금 기분에 잘 어울리는 곡들을 골라봤어요. 마음에 드는 곡이 있길 바라요!",
    "situation": "이 상황에 딱 맞는 곡들이에요. 좋은 시간 되세요!",
    "similar": "비슷한 느낌의 곡들을 찾아봤어요. 새로운 음악도 마음에 드셨으면 해요!",
}
_DEFAULT_TEMPLATE: str = "추천 곡을 골라봤어요."


def _is_routeable(text: str) -> bool:
    """Return True if the input contains enough content to route."""
    return bool(_ROUTEABLE_RE.search(text))


def _clean_text(text: str) -> str:
    """Drop tokens containing CJK characters and collapse leftover whitespace."""
    cleaned = _CJK_WORD_RE.sub("", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def _validate_response(text: str, intent: str) -> str:
    """Return intent template when text is empty or contains suspicious scripts.

    Catches hiragana, katakana, CJK, Arabic, Devanagari, Thai, and Latin
    Extended Additional characters that slip through _clean_text().
    """
    if not text or _SUSPICIOUS_SCRIPT_RE.search(text):
        return _INTENT_TEMPLATES.get(intent, _DEFAULT_TEMPLATE)
    return text


def _filter_recommendations(recs: list[dict]) -> list[dict]:
    """Blocklist → dedup by (track_name, track_artist).

    Applied after graph.invoke() so it is immune to @st.cache_resource
    retaining stale engine instances across hot reloads.
    """
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for r in recs:
        name = r.get("track_name", "").strip().lower()
        artist = r.get("track_artist", "").strip().lower()
        if name in _TITLE_BLOCKLIST:
            continue
        key = (name, artist)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    return result


# ── Imports (after set_page_config) ─────────────────────────
from src.agent import build_graph
from src.recommender import RecommendationEngine, load_and_preprocess
from src.ui.components import render_chat_message, render_recommendation_cards, render_sidebar
from src.ui.styles import inject_base_css


# ── Resource initialisation (run once, cached across reruns) ─
@st.cache_resource(show_spinner="데이터를 불러오는 중...")
def init_graph():
    """Load dataset, build recommendation engine, compile LangGraph."""
    df, feature_matrix = load_and_preprocess()
    engine = RecommendationEngine(df, feature_matrix)
    return build_graph(engine)


# Initialise — show error page if data/model setup fails
try:
    graph = init_graph()
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
        {"role": "assistant", "content": _WELCOME, "recommendations": None}
    ]

# ── Replay chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    render_chat_message(msg["role"], msg["content"], msg.get("recommendations"))

# ── Chat input ────────────────────────────────────────────────
if user_input := st.chat_input("어떤 음악을 찾고 계세요?"):
    # Display and store user message
    render_chat_message("user", user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "recommendations": None}
    )

    # Invoke agent (skip graph for non-routeable input)
    if not _is_routeable(user_input):
        response_text = _FALLBACK_REASK
        recommendations = []
    else:
        try:
            with st.spinner("테이프를 고르는 중..."):
                result = graph.invoke({"user_input": user_input})
            intent = result.get("intent", "fallback")
            response_text = _validate_response(
                _clean_text(result.get("response_text", "")),
                intent,
            )
            # Cap at 5 so blocklist removal doesn't leave fewer cards than expected.
            # Engine is asked for 8 (nodes.py top_k=8) as a buffer.
            recommendations = _filter_recommendations(result.get("recommendations", []))[:5]
            if not recommendations:
                response_text = _FALLBACK_REASK
        except Exception:
            response_text = "서버 연결에 실패했어요. 잠시 후 다시 시도해주세요."
            recommendations = []

    # Display and store assistant message (always, even on error)
    render_chat_message("assistant", response_text, recommendations or None)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "recommendations": recommendations or None,
        }
    )
