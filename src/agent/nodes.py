"""LangGraph node functions for the Pickstape recommendation agent.

Three nodes form a fixed pipeline:
    router_node → recommendation_node → response_node
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

# Matches at least one full Korean syllable (AC00–D7A3) or Latin letter.
# Inputs that contain only jamo (ㅋ, ㅠ, …), punctuation, or digits do not
# carry enough semantic content to route — they fall back to FALLBACK_REASK.
_HAS_CONTENT_RE = re.compile(r"[가-힣a-zA-Z]")

# CJK unified ideographs + extension A + compatibility + hiragana + katakana —
# stripped from all LLM outputs. Qwen leaks Chinese and occasionally Japanese;
# post-processing is more reliable than prompting alone.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import FALLBACK_REASK, RESPONSE_SYSTEM_PROMPT, RESPONSE_TEMPLATES, ROUTER_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.recommender.engine import RecommendationEngine
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# ── Valid values ─────────────────────────────────────────────

_VALID_INTENTS = {"emotion", "situation", "similar"}

# ── Keyword → param mappings for fallback ────────────────────

_MOOD_KEYWORDS: dict[str, str] = {
    "우울": "sad",
    "슬프": "sad",
    "슬픈": "sad",
    "신나": "excited",
    "신난": "excited",
    "화나": "angry",
    "짜증": "angry",
    "불안": "anxious",
    "초조": "anxious",
    "공허": "empty",
    "멍": "empty",
    "행복": "happy",
    "기분 좋": "happy",
    "즐거": "happy",
    "잔잔": "calm",
    "차분": "calm",
    "편안": "calm",
}

_SITUATION_KEYWORDS: list[str] = ["카페", "파티", "코딩", "운동", "수면", "드라이브"]

_GENRE_KEYWORDS: dict[str, str] = {
    "팝": "pop",
    "pop": "pop",
    "락": "rock",
    "록": "rock",
    "rock": "rock",
    "힙합": "rap",
    "랩": "rap",
    "rap": "rap",
    "알앤비": "r&b",
    "r&b": "r&b",
    "라틴": "latin",
    "latin": "latin",
    "일렉": "edm",
    "edm": "edm",
}

_SIMILAR_KEYWORDS: list[str] = ["비슷", "유사", "같은", "similar"]

_EMOTION_KEYWORDS: list[str] = [
    "기분",
    "감정",
    "우울",
    "슬프",
    "슬픈",
    "신나",
    "화나",
    "짜증",
    "불안",
    "초조",
    "공허",
    "멍",
    "행복",
    "즐거",
    "위로",
    "잔잔",
    "차분",
    "편안",
]


# ── JSON extraction helpers ──────────────────────────────────


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract a JSON object from LLM output.

    Strategy:
    1. Fenced JSON block (```json ... ```)
    2. Balanced brace matching (first { to its matching })
    """
    # 1) fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) balanced brace matching
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_intent_fallback(
    user_input: str,
    llm_output: str,
) -> tuple[str, dict[str, Any]]:
    """Keyword-based fallback when JSON parsing fails completely.

    Extracts both intent and params from the raw text.
    Returns (intent, params) — intent may be "fallback" if nothing matches.
    """
    combined = f"{user_input} {llm_output}".lower()
    params: dict[str, Any] = {
        "mood": None,
        "situation": None,
        "genre_pref": None,
        "seed_track": None,
        "seed_artist": None,
    }
    intent = "fallback"

    # --- Extract params regardless of intent ---

    # mood
    for keyword, mood_val in _MOOD_KEYWORDS.items():
        if keyword in combined:
            params["mood"] = mood_val
            break

    # situation
    for sit in _SITUATION_KEYWORDS:
        if sit in combined:
            params["situation"] = sit
            break

    # genre
    for keyword, genre_val in _GENRE_KEYWORDS.items():
        if keyword in combined:
            params["genre_pref"] = genre_val
            break

    # seed_track: 1) quoted text, 2) "X 비슷한/같은/유사" pattern
    # Handles straight quotes, curly quotes ('' ""), and Korean angle quotes (『』)
    quoted = re.search(r"""['\u2018\u2019\u201c\u201d\u300e\u300f"'](.+?)['\u2018\u2019\u201c\u201d\u300e\u300f"']""", user_input)
    if quoted:
        params["seed_track"] = quoted.group(1).strip()
    else:
        # "X랑/와/과/이랑 비슷한" — capture text before particle + keyword
        pattern = r"(.+?)\s*(?:랑|와|과|이랑|의)\s*(?:비슷|유사|같은)"
        m = re.search(pattern, user_input)
        if not m:
            # "X 비슷한" — no particle
            pattern = r"(\S+(?:\s+\S+){0,4}?)\s+(?:비슷|유사|같은)"
            m = re.search(pattern, user_input)
        if m:
            seed = m.group(1).strip()
            # Strip any trailing Korean particles that leaked through
            seed = re.sub(r"\s*(?:랑|와|과|이랑|의)$", "", seed).strip()
            params["seed_track"] = seed if seed else None

    # --- Determine intent ---

    # similar takes priority if seed_track is found or similar keywords present
    if params["seed_track"] or any(kw in combined for kw in _SIMILAR_KEYWORDS):
        intent = "similar"
    elif params["situation"]:
        intent = "situation"
    elif params["mood"] or any(kw in combined for kw in _EMOTION_KEYWORDS):
        intent = "emotion"

    return intent, params


# ── Template fallback for response ───────────────────────────


def _clean_response(text: str) -> str:
    """Strip CJK ideograph runs and collapse leftover whitespace.

    Qwen3.5-4B leaks Chinese characters even when prompted in Korean.
    Post-processing is more reliable than prompt-only enforcement.
    """
    cleaned = _CJK_RE.sub("", text)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def _template_response(recommendations: list[dict[str, Any]]) -> str:
    """Generate a minimal response when the LLM fails."""
    lines = []
    for i, r in enumerate(recommendations, 1):
        genres = ", ".join(r.get("playlist_genres", []))
        line = f"{i}. {r['track_name']} - {r['track_artist']}"
        if genres:
            line += f" ({genres})"
        lines.append(line)
    return "추천 곡을 골라봤어요!\n\n" + "\n".join(lines)


def _format_recommendations_for_llm(recommendations: list[dict[str, Any]]) -> str:
    """Format recommendation results into a concise string for the response LLM."""
    lines = []
    for i, r in enumerate(recommendations, 1):
        genres = ", ".join(r.get("playlist_genres", []))
        line = f"{i}. {r['track_name']} - {r['track_artist']} ({genres})"
        lines.append(line)
    return "\n".join(lines)


# ── Node factory ─────────────────────────────────────────────


def create_nodes(
    engine: RecommendationEngine,
    *,
    router_llm_factory: Callable | None = None,
    response_llm_factory: Callable | None = None,
) -> tuple[Callable, Callable, Callable]:
    """Create the three node functions with injected dependencies.

    Parameters
    ----------
    engine:
        The recommendation engine instance.
    router_llm_factory:
        Callable returning a ChatOpenAI for routing. Defaults to get_llm
        with temperature=0.1, max_completion_tokens=200.
    response_llm_factory:
        Callable returning a ChatOpenAI for response generation. Defaults to
        get_llm with temperature=0.7, max_completion_tokens=500.

    Returns
    -------
    (router_node, recommendation_node, response_node)
    """
    if router_llm_factory is None:
        router_llm_factory = lambda: get_llm(  # noqa: E731
            temperature=0.1, max_completion_tokens=200,
        )
    if response_llm_factory is None:
        response_llm_factory = lambda: get_llm(  # noqa: E731
            temperature=0.7, max_completion_tokens=500,
        )

    # ── router_node ──────────────────────────────────────────

    def router_node(state: AgentState) -> dict:
        """Classify user intent and extract recommendation parameters."""
        user_input = state["user_input"]

        # Guard: no full syllables or Latin letters → not routable content.
        if not _HAS_CONTENT_RE.search(user_input):
            return {"intent": "fallback", "params": {}}

        try:
            llm = router_llm_factory()
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]
            response = llm.invoke(messages)
            raw = response.content or ""
        except Exception:
            logger.exception("Router LLM call failed")
            raw = ""

        # Try JSON parsing
        parsed = None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = _extract_json_from_text(raw)

        if parsed and isinstance(parsed, dict):
            intent = parsed.get("intent", "")
            params = parsed.get("params", {})
            if intent in _VALID_INTENTS and isinstance(params, dict):
                return {"intent": intent, "params": params}

        # Keyword fallback
        logger.warning("Router JSON parse failed, using keyword fallback")
        intent, params = _parse_intent_fallback(user_input, raw)
        if intent != "fallback":
            return {"intent": intent, "params": params}

        return {
            "intent": "fallback",
            "params": {},
            "error": f"Router failed to parse: {raw[:200]}",
        }

    # ── recommendation_node ──────────────────────────────────

    def recommendation_node(state: AgentState) -> dict:
        """Run the recommendation engine based on classified intent."""
        intent = state.get("intent", "fallback")
        params = state.get("params", {})

        if intent == "fallback":
            return {"recommendations": []}

        try:
            results = engine.recommend(intent, params)
            return {"recommendations": results}
        except Exception as e:
            logger.exception("Recommendation engine error")
            return {"recommendations": [], "error": str(e)}

    # ── response_node ────────────────────────────────────────

    def response_node(state: AgentState) -> dict:
        """Return a short template response.

        Free LLM generation was replaced with templates for stability.
        Qwen3.5-4B produced inconsistent Korean, leaked foreign characters,
        and used awkward phrasing — templates eliminate that risk entirely.
        app.py builds a more contextual version using intent + params.
        """
        recommendations = state.get("recommendations", [])
        intent = state.get("intent", "fallback")

        if not recommendations:
            return {
                "response_text": FALLBACK_REASK,
                "error": state.get("error", "No recommendations produced"),
            }

        return {"response_text": RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["fallback"])}

    return router_node, recommendation_node, response_node
