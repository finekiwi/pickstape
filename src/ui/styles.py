"""CSS styles for Pickstape Streamlit UI.

PS-07 retro theme — f(x) Pink Tape motif, cream background + hot-pink accent.
Color tokens from refs/design_guide.md (v3).
"""

import streamlit as st

BASE_CSS: str = """
@import url('https://cdn.jsdelivr.net/npm/galmuri/dist/galmuri.css');

/* ── PS-07 Color Tokens ─────────────────────────────── */
:root {
    --bg-primary: #FFF8F0;
    --bg-secondary: #FFF0F5;
    --bg-sidebar: #FFF0F5;
    --bg-card: #FFFFFF;
    --text-primary: #2D1B33;
    --text-secondary: #8B2252;
    --text-on-pink: #FFFFFF;
    --accent: #FF6B9D;
    --accent-light: #FFD4E5;
    --accent-dark: #2D1B33;
    --border: #FFC8DC;
}

/* ── Global Background ──────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: var(--bg-primary) !important;
}

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: var(--bg-sidebar) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Chat User Bubble ───────────────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has([data-testid*="user"]) {
    background-color: #FFC0CB !important;
    color: #2D1B33 !important;
    border-radius: 20px 20px 4px 20px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid*="user"]) p {
    color: #2D1B33 !important;
}
.stChatMessage:nth-child(even) {
    background-color: #FFC0CB !important;
}

/* ── Chat Bot Bubble ────────────────────────────────── */
[data-testid="stChatMessage"][data-testid-type="assistant"],
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--accent-light) !important;
    border-radius: 20px 20px 20px 4px !important;
}
[data-testid="stChatMessage"][data-testid-type="assistant"] *,
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) p,
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) span {
    color: var(--text-primary) !important;
}

/* ── Chat Input ─────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-color: #F0D4DE !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background-color: var(--bg-card) !important;
    border-color: #F0D4DE !important;
    border-radius: 20px !important;
    color: #C44B78 !important;
    font-family: 'Galmuri11', monospace !important;
    font-size: 13px !important;
    padding: 12px 48px 12px 16px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #C44B78 !important;
}

/* ── Chat Send Button ───────────────────────────────── */
[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {
    background-color: var(--accent) !important;
    color: var(--text-on-pink) !important;
    border: none !important;
}
[data-testid="stChatInput"] button svg,
[data-testid="stChatInputSubmitButton"] svg {
    fill: var(--text-on-pink) !important;
    stroke: var(--text-on-pink) !important;
}

/* ── VHS Cassette Card ──────────────────────────────── */
.vhs-card {
    background-color: #FFC8DC;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 16px;
    transition: border 0.15s, background-color 0.15s;
    border: 2px solid transparent;
}
.vhs-card:hover {
    border: 2px solid #FF6B9D;
    background-color: #FFD4E5;
}

.vhs-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #FF6B9D;
    border-radius: 10px 10px 0 0;
    padding: 4px 12px;
    margin: -12px -12px 10px -12px;
}
.vhs-label {
    font-family: 'Galmuri11', monospace;
    font-size: 9px;
    color: #FFFFFF;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.vhs-reels {
    display: flex;
    gap: 6px;
}
.reel-hole {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 1.5px solid #FF6B9D;
    display: flex;
    align-items: center;
    justify-content: center;
}
.reel-hole-inner {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #FF6B9D;
}

.vhs-tape-label {
    background-color: var(--bg-card);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.card-title {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary) !important;
    margin-bottom: 4px;
}
.card-meta {
    font-size: 12px;
    color: var(--text-secondary) !important;
    margin-bottom: 8px;
}
.card-badges {
    margin-bottom: 8px;
}
.genre-badge {
    display: inline-block;
    background-color: #FF6B9D;
    color: #FFFFFF !important;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    margin-right: 4px;
}

.spotify-btn {
    display: inline-block;
    padding: 4px 12px;
    background-color: #FFF0F5;
    color: #FF6B9D !important;
    border: 1px solid #FF6B9D;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 500;
    text-decoration: none !important;
}

/* ── Find Similar Button (비슷한 곡 찾기) ────────────── */
/* Targets the st.button in the right column of the card btn row.
   Selector is intentionally loose — scoped visually by card context. */
.vhs-card + div [data-testid="stHorizontalBlock"] button[kind="secondary"] {
    font-family: 'Galmuri11', monospace !important;
    font-size: 9px !important;
    background-color: transparent !important;
    color: #DDA0B4 !important;
    border: 1px dashed #DDA0B4 !important;
    border-radius: 10px !important;
    padding: 2px 10px !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
    width: auto !important;
}
.vhs-card + div [data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    background-color: #FFF0F5 !important;
    border-color: #FF6B9D !important;
    border-style: dashed !important;
    color: #FF6B9D !important;
}

/* ── VHS Feature Bars (vertical) ────────────────────── */
.vbar-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    height: 36px;
    background-color: #FFC8DC;
    border-radius: 0 0 10px 10px;
    padding: 0 4px 4px;
}
.vbar-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 3px;
    flex: 1;
}
.vbar-fill {
    width: 100%;
    border-radius: 3px 3px 0 0;
    min-height: 2px;
}
.vbar-label {
    font-size: 8px;
    color: #C44B78;
    text-transform: uppercase;
    font-family: monospace;
}

/* ── Sidebar Custom Components ──────────────────────── */
.sidebar-logo-block {
    background: rgba(255,255,255,0.12);
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 0;
}
.sidebar-logo-row {
    display: flex;
    align-items: center;
    margin-bottom: 6px;
}
.sidebar-logo {
    font-family: 'Galmuri11', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #2D1B33;
    letter-spacing: 2px;
    text-transform: uppercase;
    vertical-align: middle;
}
.sidebar-subtitle {
    font-size: 12px;
    color: #8B2252;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}
.sidebar-tape-meta {
    font-family: 'Galmuri11', monospace;
    font-size: 9px;
    color: #C44B78;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 0;
}
.sidebar-divider {
    border-bottom: 1px solid #D4708F;
    margin: 12px 0;
}
.sidebar-desc {
    font-size: 14px;
    color: #2D1B33;
    padding: 8px 0 8px 12px;
    line-height: 1.5;
}
.sidebar-tracklist-block {
    background: rgba(255,255,255,0.22);
    padding: 12px;
    border-radius: 8px;
    margin-top: 8px;
}
.sidebar-section-label {
    font-family: 'Galmuri11', monospace;
    font-size: 11px;
    color: #8B2252;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.1);
    padding: 4px 8px 6px;
    border-radius: 4px;
    display: inline-block;
    border-bottom: 1px solid #FF6B9D;
    width: 100%;
    box-sizing: border-box;
}
.sidebar-track {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 5px 4px;
    border-bottom: 0.5px solid rgba(0,0,0,0.06);
}
.sidebar-track-even {
    background: rgba(255,255,255,0.1);
}
.sidebar-track-num {
    font-family: 'Galmuri11', monospace;
    font-size: 13px;
    font-weight: 700;
    color: #FF6B9D;
    flex-shrink: 0;
    display: inline-block;
    width: 48px;
}
.sidebar-track-title {
    font-size: 14px;
    color: #2D1B33;
    line-height: 1.4;
}
.sidebar-rec {
    font-family: 'Galmuri11', monospace;
    font-size: 9px;
    color: #C44B78;
    margin-top: 16px;
    padding-left: 12px;
    letter-spacing: 0.5px;
}
.sidebar-rec-dot {
    color: #FF6B9D;
}

/* ── Chat Avatar Image Size ─────────────────────────── */
[data-testid="stChatMessage"] img {
    width: 28px !important;
    height: 28px !important;
    object-fit: contain !important;
}

/* ── Hide User Avatar ───────────────────────────────── */
[data-testid="chatAvatarIcon-user"],
.stChatMessage [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessageAvatarUser"],
.stChatMessage:has([data-testid*="user"]) img,
.stChatMessage:has([data-testid*="user"]) svg {
    display: none !important;
}

/* ── Feedback UI ────────────────────────────────────── */
.feedback-label {
    font-family: 'Galmuri11', monospace;
    font-size: 12px;
    color: #2D1B33;
    text-align: center;
    margin: 10px 0 6px;
}
/* 버튼 세로 정렬 */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) div[data-testid="column"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) div[data-testid="column"] > div {
    width: 100%;
}
/* primary: 좋아요 — 첫 번째 버튼 */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) div:nth-child(2) button {
    background-color: #FFF0F5 !important;
    border: 1px solid #FF6B9D !important;
    border-radius: 20px !important;
    color: #FF6B9D !important;
    font-family: 'Galmuri11', monospace !important;
    font-size: 9px !important;
    padding: 2px 10px !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}
/* secondary: 다른 분위기로 — 두 번째 버튼 */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) div:nth-child(3) button {
    background-color: transparent !important;
    border: 1px solid #DDA0B4 !important;
    border-radius: 20px !important;
    color: #C44B78 !important;
    font-family: 'Galmuri11', monospace !important;
    font-size: 9px !important;
    padding: 2px 10px !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar-thumb {
    background-color: var(--accent-light) !important;
    border-radius: 4px;
}
::-webkit-scrollbar-track {
    background-color: var(--bg-primary) !important;
}
"""


def inject_base_css() -> None:
    """Inject base CSS into the Streamlit app."""
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
