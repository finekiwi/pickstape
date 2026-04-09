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
/*
 * Card is a single HTML block (st.markdown). The white inner panel
 * (.vhs-inner-panel) is an explicit HTML element — no dependency on
 * Streamlit DOM structure. Spotify button is an HTML anchor inside
 * .card-actions. The find-similar st.button() is rendered after the
 * card HTML; CSS pulls it up with negative margin to sit flush with
 * the card-actions area (right column of .card-main).
 */
.vhs-card {
    background-color: #FFC8DC;
    border-radius: 14px;
    padding: 0;
    margin-bottom: 16px;
    transition: background-color 0.15s;
    border: 2px solid transparent;
}
.vhs-card:hover {
    background-color: #FFD4E5;
    border-color: #FF6B9D;
}

.vhs-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #FF6B9D;
    border-radius: 12px 12px 0 0;
    padding: 5px 12px;
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
    border: 1.5px solid rgba(255,255,255,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
}
.reel-hole-inner {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.7);
}

/* ── White inner panel ───────────────────────────────── */
.vhs-inner-panel {
    background-color: #FFFFFF;
    border-radius: 8px;
    margin: 8px 10px 8px;
    padding: 10px 12px;
    overflow: hidden;
}

/* ── card-main: info (left) | actions (right) ─────────── */
.card-main {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
}
.card-info {
    flex: 1;
    min-width: 0;
}
.card-title {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary) !important;
    margin-bottom: 3px;
    line-height: 1.3;
    word-break: break-word;
}
.card-meta {
    font-size: 11px;
    color: var(--text-secondary) !important;
    margin-bottom: 5px;
}
.card-badges {
    margin-bottom: 0;
}
.genre-badge {
    display: inline-block;
    background-color: #FF6B9D;
    color: #FFFFFF !important;
    border-radius: 10px;
    padding: 2px 7px;
    font-size: 9px;
    margin-right: 3px;
    margin-bottom: 2px;
}

/* ── card-actions (right column, compact) ────────────── */
.card-actions {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex-shrink: 0;
    width: 84px;
    align-self: flex-start;
}

/* Spotify button */
.spotify-btn {
    display: block;
    padding: 3px 6px;
    background-color: #FFF0F5;
    color: #FF6B9D !important;
    border: 1px solid #FF6B9D;
    border-radius: 8px;
    font-size: 9px;
    font-weight: 500;
    text-decoration: none !important;
    text-align: center;
    white-space: nowrap;
    line-height: 1.4;
}
.spotify-btn:hover {
    background-color: #FFD4E5;
}

/* find-similar: secondary/dashed */
.find-similar-btn {
    display: block;
    padding: 3px 6px;
    background-color: transparent;
    color: #C44B78 !important;
    border: 1px dashed #DDA0B4;
    border-radius: 8px;
    font-size: 9px;
    font-family: 'Galmuri11', monospace;
    text-decoration: none !important;
    text-align: center;
    white-space: nowrap;
    line-height: 1.4;
}
.find-similar-btn:hover {
    background-color: #FFF0F5;
    border-color: #FF6B9D;
    color: #FF6B9D !important;
}

/* find-similar placeholder: invisible, reserves height for the real button */
.find-similar-placeholder {
    display: block;
    padding: 5px 8px;
    border: 1px dashed #DDA0B4;
    border-radius: 10px;
    font-size: 9px;
    color: transparent;
    text-align: center;
    white-space: nowrap;
    pointer-events: none;
    user-select: none;
}

/* ── VHS Feature Bars (vbar-row) ─────────────────────── */
.vbar-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    height: 22px;
    background-color: #FFC8DC;
    border-radius: 0 0 12px 12px;
    padding: 0 6px 3px;
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

/* ── Find-similar Streamlit button (below card HTML) ──── */
/*
 * The real st.button() renders after st.markdown(card_html).
 * margin-top: -34px pulls it up to overlap the invisible placeholder
 * in .card-actions. justify-content: flex-end + width: 110px aligns
 * it to the right (matching .card-actions column width).
 */
[data-testid="column"]:has(.vhs-card) .stButton {
    display: flex !important;
    justify-content: flex-end !important;
    margin-top: -34px !important;
    padding-right: 24px !important;
    position: relative;
    z-index: 10;
}
[data-testid="column"]:has(.vhs-card) .stButton > button {
    font-family: 'Galmuri11', monospace !important;
    font-size: 9px !important;
    background-color: #FFF8F0 !important;
    color: #C44B78 !important;
    border: 1px dashed #DDA0B4 !important;
    border-radius: 10px !important;
    padding: 5px 8px !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
    width: 110px !important;
}
[data-testid="column"]:has(.vhs-card) .stButton > button:hover {
    background-color: #FFF0F5 !important;
    border-color: #FF6B9D !important;
    color: #FF6B9D !important;
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
