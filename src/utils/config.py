"""Environment configuration and LLM factory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ── .env 로드 ────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

# ── 환경 변수 ────────────────────────────────────────────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://116.124.134.37:39193/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5-4b")
LLM_API_KEY: str = os.getenv("LLM_API_KEY") or "not-needed"

# ── 데이터 경로 ──────────────────────────────────────────────
DATA_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "Music_recommendation.csv"


def get_llm(
    *,
    temperature: float = 0.1,
    max_completion_tokens: int = 200,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for the Qwen3.5-4B server.

    Parameters
    ----------
    temperature:
        Sampling temperature. 0.1 for routing, 0.7 for response generation.
    max_completion_tokens:
        Maximum tokens in the completion. 200 for routing, 500 for responses.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
