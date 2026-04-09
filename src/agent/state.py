"""Agent state schema for the LangGraph recommendation pipeline."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the router → recommendation → response graph.

    All fields are optional (``total=False``) so each node can return only the
    keys it owns without requiring every other field.
    """

    user_input: str
    """Original user utterance."""

    intent: Literal["emotion", "situation", "similar", "fallback"]
    """Classified intent from the router node."""

    params: dict[str, Any]
    """Normalised recommendation parameters extracted by the router."""

    recommendations: list[dict[str, Any]]
    """Top-k results returned by the recommendation engine."""

    response_text: str
    """Final Korean-language response for the user."""

    error: str | None
    """Debug / logging error message (never shown to the user)."""
