"""Base CSS styles for Pickstape Streamlit UI.

Minimal layout styles only — retro theming is handled in PS-07.
"""

import streamlit as st

BASE_CSS: str = """
.recommendation-card {
    border: 1px solid #E8A0BF;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background-color: #FFF8F0;
    box-shadow: 0 1px 4px rgba(232, 160, 191, 0.2);
}

.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2D1B4E;
    margin-bottom: 0.15rem;
}

.card-meta {
    font-size: 0.85rem;
    color: #6B5B7B;
    margin-bottom: 0.5rem;
}

.genre-badge {
    display: inline-block;
    background-color: #F5C6DC;
    color: #2D1B4E;
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.75rem;
    margin-right: 0.3rem;
    margin-bottom: 0.5rem;
}

.feature-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
}

.feature-label {
    font-size: 0.75rem;
    color: #6B5B7B;
    width: 4.5rem;
    flex-shrink: 0;
}

.feature-track {
    flex: 1;
    background-color: #F5E6D3;
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}

.feature-fill {
    height: 100%;
    background-color: #E8A0BF;
    border-radius: 999px;
}

.feature-value {
    font-size: 0.75rem;
    color: #6B5B7B;
    width: 2.5rem;
    text-align: right;
    flex-shrink: 0;
}

.tempo-row {
    font-size: 0.75rem;
    color: #6B5B7B;
    margin-top: 0.2rem;
}
"""


def inject_base_css() -> None:
    """Inject base CSS into the Streamlit app."""
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
