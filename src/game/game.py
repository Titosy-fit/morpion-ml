import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# ====================== CONFIG ======================
st.set_page_config(page_title="Morpion IA - ISPM", layout="centered", initial_sidebar_state="collapsed")

# ====================== PIXEL ART CSS ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: linear-gradient(180deg, #2a1a0e 0%, #4a2e18 45%, #1a0f08 100%) !important;
    color: #f0d080 !important;
    font-family: 'Press Start 2P', monospace !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stMain"]::before {
    content: '';
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 220px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 220'%3E%3Crect x='0' y='120' width='1200' height='100' fill='%230d0704'/%3E%3Crect x='30' y='50' width='60' height='170' fill='%230d0704'/%3E%3Crect x='15' y='35' width='90' height='20' fill='%230d0704'/%3E%3Crect x='130' y='75' width='45' height='145' fill='%230d0704'/%3E%3Crect x='118' y='58' width='70' height='20' fill='%230d0704'/%3E%3Crect x='220' y='85' width='50' height='135' fill='%230d0704'/%3E%3Crect x='208' y='65' width='75' height='25' fill='%230d0704'/%3E%3Crect x='320' y='100' width='35' height='120' fill='%230d0704'/%3E%3Crect x='900' y='55' width='70' height='165' fill='%230d0704'/%3E%3Crect x='886' y='35' width='100' height='25' fill='%230d0704'/%3E%3Crect x='1000' y='80' width='50' height='140' fill='%230d0704'/%3E%3Crect x='987' y='62' width='78' height='22' fill='%230d0704'/%3E%3Crect x='1080' y='90' width='40' height='130' fill='%230d0704'/%3E%3Crect x='1140' y='70' width='55' height='150' fill='%230d0704'/%3E%3C/svg%3E") no-repeat bottom center;
    background-size: cover;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"]::after {
    content: '';
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 55px;
    background: #0d0704;
    border-top: 3px solid #2d1a0e;
    z-index: 0;
}

[data-testid="stMainBlockContainer"] {
    position: relative;
    z-index: 10;
    max-width: 780px !important;
    padding-top: 0 !important;
}

/* ====== HUD ====== */
.hud-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(0,0,0,0.75);
    border-bottom: 3px solid #3d2010;
    border-left: 3px solid #3d2010;
    border-right: 3px solid #3d2010;
    padding: 8px 16px;
}
.hud-score { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #fff; }
.hud-icon {
    width: 22px; height: 22px;
    background: #c8990a; border: 3px solid #7a5c00;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; color: #1a0f08; flex-shrink: 0;
}
.hud-turn {
    font-size: 10px; color: #f0d080;
    padding: 5px 12px;
    background: rgba(0,0,0,0.6);
    border: 2px solid #5a3a18;
    letter-spacing: 1px;
}

/* ====== FIGHTERS ====== */
.fighter-panel {
    width: 110px;
    display: flex; flex-direction: column; align-items: center;
    padding: 8px 4px 16px; flex-shrink: 0;
}
.fighter-sprite { font-size: 52px; line-height: 1; filter: drop-shadow(2px 2px 0px rgba(0,0,0,0.8)); }
.speech-bubble {
    background: #fff; border: 3px solid #333;
    padding: 5px 7px; font-size: 6px; color: #1a0f08;
    font-family: 'Press Start 2P', monospace;
    max-width: 110px; line-height: 1.5; margin-bottom: 6px;
    min-height: 36px; display: flex; align-items: center;
    justify-content: center; text-align: center; position: relative;
}
.speech-bubble::after {
    content: ''; position: absolute; bottom: -10px; left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent; border-top-color: #fff;
}
.hp-wrap { width: 90px; margin-top: 4px; }
.hp-label { font-size: 6px; color: #aaa; margin-bottom: 2px; display: flex; justify-content: space-between; }
.hp-track { height: 12px; background: #1a0f08; border: 2px solid #4a3020; overflow: hidden; }
.hp-fill-x { height: 100%; background: #c0392b; }
.hp-fill-o { height: 100%; background: #27ae60; }
.items-row { display: flex; gap: 5px; margin-top: 8px; }
.item-box { width: 20px; height: 20px; border: 2px solid #4a3020; background: #1a0f08; }

/* ====== RADIO ====== */
.stRadio > div {
    display: flex !important; gap: 6px !important;
    flex-wrap: wrap !important; justify-content: center !important; margin-bottom: 8px !important;
}
.stRadio label {
    background: rgba(0,0,0,0.5) !important; color: #c8a060 !important;
    font-family: 'Press Start 2P', monospace !important; font-size: 7px !important;
    padding: 5px 8px !important; border: 2px solid #5a3a18 !important;
    border-radius: 0 !important; cursor: pointer !important;
}
.stRadio label:has(input:checked) { background: #6a4a20 !important; color: #f0d080 !important; border-color: #c8900a !important; }
.stRadio input { display: none !important; }

/* ====== BOARD BUTTONS ====== */
.stButton button {
    font-family: 'Press Start 2P', monospace !important;
    background: #e8d8a8 !important;
    border: 3px solid #b8a070 !important;
    border-radius: 0 !important;
    color: #1a0f08 !important;
    width: 76px !important; height: 76px !important;
    font-size: 32px !important; padding: 0 !important;
    cursor: pointer !important; line-height: 1 !important;
}
.stButton button:hover { background: #f4e8c0 !important; border-color: #d4b888 !important; }
.stButton button:disabled { opacity: 1 !important; cursor: default !important; background: #e0d0a0 !important; }

/* Case gagnante : bordure or clignotante */
@keyframes winPulse {
    0%   { border-color: #b8a070; background: #e8d8a8; }
    50%  { border-color: #f0c000; background: #fff8c0; }
    100% { border-color: #b8a070; background: #e8d8a8; }
}
.win-cell button {
    animation: winPulse 0.5s ease-in-out infinite !important;
    border-color: #f0c000 !important;
}

/* ====== ACTION BUTTON ====== */
.action-btn button {
    background: #5a3a10 !important; color: #f0d080 !important;
    border: 3px solid #c8900a !important; font-size: 8px !important;
    padding: 8px 14px !important; width: auto !important; height: auto !important;
}
.action-btn button:hover { background: #7a5010 !important; }

/* ====== BOTTOM HUD ====== */
.bottom-hud {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,0,0,0.85); border: 3px solid #3d2010;
    padding: 6px 12px; margin-top: 0;
}
.bottom-hp-group { display: flex; align-items: center; gap: 8px; }
.bottom-hp-bar { width: 110px; height: 14px; background: #1a0f08; border: 2px solid #4a3020; overflow: hidden; }
.bottom-center-btn {
    width: 40px; height: 40px; background: #4a7a5a;
    border: 3px solid #2a5a3a;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; color: #a0f0a0;
}

/* ====== RESULT ====== */
.result-banner {
    background: rgba(0,0,0,0.9); border: 4px solid #c8900a;
    padding: 16px 24px; text-align: center; font-size: 14px;
    color: #f0d080; text-shadow: 2px 2px #000; margin: 8px 0; letter-spacing: 1px;
}

/* ====== TITLES ====== */
.pixel-title {
    text-align: center; font-family: 'Press Start 2P', monospace;
    font-size: 13px; color: #f0d080;
    text-shadow: 3px 3px #000, -1px -1px #c8900a;
    padding: 12px 0 6px; letter-spacing: 2px;
}
.pixel-subtitle {
    text-align: center; font-family: 'Press Start 2P', monospace;
    font-size: 7px; color: #a08040; padding-bottom: 10px; letter-spacing: 1px;
}

[data-testid="stHorizontalBlock"] { gap: 4px !important; justify-content: center !important; }
[data-testid="stColumn"] { padding: 0 2px !important; flex: none !important; width: auto !important; }

::-webkit-scrollbar { width: 6px; background: #1a0f08; }
::-webkit-scrollbar-thumb { background: #4a3020; }
div[data-baseweb="radio"] > div { gap: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ====================== CHEMINS ======================
ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
WIN_MODEL_PATH = MODELS_DIR / "model_win.pkl"
DRAW_MODEL_PATH = MODELS_DIR / "model_draw.pkl"

# ====================== CHARGEMENT OU CRÉATION DES MODÈLES ======================
@st.cache_resource
def load_or_train_models():
    if WIN_MODEL_PATH.exists() and DRAW_MODEL_PATH.exists():
        return joblib.load(WIN_MODEL_PATH), joblib.load(DRAW_MODEL_PATH)
    try:
        df = pd.read_csv(ROOT / "ressources/dataset.csv")
        X = df.iloc[:, :18]
        y_win = df['x_wins']
        y_draw = df['is_draw']
        model_win = LogisticRegression(max_iter=1000, random_state=42)
        model_draw = LogisticRegression(max_iter=1000, random_state=42)
        model_win.fit(X, y_win)
        model_draw.fit(X, y_draw)
        joblib.dump(model_win, WIN_MODEL_PATH)
        joblib.dump(model_draw, DRAW_MODEL_PATH)
        return model_win, model_draw
    except Exception:
        return None, None

model_win, model_draw = load_or_train_models()

# ====================== FONCTIONS DU JEU (logique Glow) ======================
def check_winner(board):
    """Retourne (winner, win_line) ou (None, None) — identique au code Glow"""
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a], (a, b, c)
    return None, None

def is_full(board):
    return all(x != 0 for x in board)

def encode_board(board):
    features = []
    for cell in board:
        features.extend([1 if cell == 'X' else 0, 1 if cell == 'O' else 0])
    return pd.DataFrame([features], columns=[f'c{i}_{p}' for i in range(9) for p in ['x', 'o']])

def ml_best_move(board, player):
    """ML : score = p_win - 0.5*(1 - p_draw), identique au code Glow"""
    if model_win is None:
        return minimax_best_move(board, player)
    best_score = -float('inf')
    best_move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            features = encode_board(board)
            p_win = model_win.predict_proba(features)[0][1]
            p_draw = model_draw.predict_proba(features)[0][1]
            score = p_win - 0.5 * (1 - p_draw) if player == 'X' else -(p_win - 0.5 * (1 - p_draw))
            board[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

def minimax(board, is_max, alpha=-float('inf'), beta=float('inf'), depth=0):
    winner, _ = check_winner(board)
    if winner == 'X': return 10 - depth
    if winner == 'O': return depth - 10
    if is_full(board): return 0
    if is_max:
        best = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 'X'
                best = max(best, minimax(board, False, alpha, beta, depth+1))
                board[i] = 0
                alpha = max(alpha, best)
                if beta <= alpha: break
        return best
    else:
        best = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 'O'
                best = min(best, minimax(board, True, alpha, beta, depth+1))
                board[i] = 0
                beta = min(beta, best)
                if beta <= alpha: break
        return best

def minimax_best_move(board, player):
    best_score = float('inf')
    best_move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = minimax(board, True)
            board[i] = 0
            if score < best_score:
                best_score = score
                best_move = i
    return best_move

# ====================== SESSION STATE (structure Glow) ======================
def reset_board():
    st.session_state.board = [0] * 9
    st.session_state.current_player = 'X'
    st.session_state.winner_line = None
    st.session_state.game_over = False
    st.session_state.winner_symbol = None
    st.session_state.turn = 1
    st.session_state.speech_x = "PRET AU COMBAT !"
    st.session_state.speech_o = "LA POULE SACREE ME GUIDERA."

if 'board' not in st.session_state:
    reset_board()
    st.session_state.score_x = 0
    st.session_state.score_o = 0

board = st.session_state.board

# ====================== TITLE ======================
st.markdown('<div class="pixel-title">MORPION BATTLE</div>', unsafe_allow_html=True)
st.markdown('<div class="pixel-subtitle">HACKATHON ML — MASTER 1 ISPM</div>', unsafe_allow_html=True)

# ====================== HUD TOP ======================
st.markdown(f"""
<div class="hud-bar">
    <div class="hud-score">
        <div class="hud-icon">X</div>
        <span>{st.session_state.score_x}</span>
    </div>
    <div class="hud-turn">TOUR {st.session_state.turn}</div>
    <div class="hud-score">
        <div class="hud-icon" style="background:#2d8a4a;border-color:#1a5a2a;">O</div>
        <span>{st.session_state.score_o}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== MODE ======================
mode = st.radio("",
    ["vs Humain", "vs IA (ML)", "vs IA (Hybride)"],
    horizontal=True,
    label_visibility="collapsed"
)

# ====================== HP ======================
hp_x_pct = max(0, 100 - st.session_state.score_o * 34)
hp_o_pct = max(0, 100 - st.session_state.score_x * 50)
hp_x_val = max(0, 3 - st.session_state.score_o)
hp_o_val = max(0, 2 - st.session_state.score_x)

# ====================== ARENA ======================
col_l, col_board, col_r = st.columns([1.2, 2.5, 1.2])

with col_l:
    st.markdown(f"""
    <div class="fighter-panel">
        <div class="speech-bubble">{st.session_state.speech_x}</div>
        <div class="fighter-sprite">🦆</div>
        <div class="hp-wrap">
            <div class="hp-label"><span>HP</span><span>{hp_x_val}/3</span></div>
            <div class="hp-track"><div class="hp-fill-x" style="width:{hp_x_pct}%"></div></div>
        </div>
        <div class="items-row"><div class="item-box"></div><div class="item-box"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ====================== PLATEAU (logique clics du Glow) ======================
with col_board:
    winner, win_line = check_winner(board)
    cols = st.columns(3)
    for i in range(9):
        with cols[i % 3]:
            is_win_cell = (winner and win_line and i in win_line)

            if board[i] == 0 and not st.session_state.game_over:
                # Case vide cliquable
                if st.button(" ", key=f"b{i}"):
                    if st.session_state.current_player == 'X' and not st.session_state.game_over:
                        board[i] = 'X'
                        st.session_state.speech_x = "A TOI !"
                        st.session_state.turn += 1
                        w, wl = check_winner(board)
                        if w:
                            st.session_state.game_over = True
                            st.session_state.winner_symbol = w
                            st.session_state.winner_line = wl
                            st.session_state.speech_x = "VICTOIRE !!!"
                            st.session_state.speech_o = "IMPOSSIBLE..."
                        elif is_full(board):
                            st.session_state.game_over = True
                            st.session_state.speech_x = "MATCH NUL ?"
                            st.session_state.speech_o = "LA POULE EST NEUTRE."
                        else:
                            st.session_state.current_player = 'O'
                        st.rerun()
            else:
                # Case remplie ou game over — affichage avec classe win-cell si gagnant
                label = "X" if board[i] == 'X' else ("O" if board[i] == 'O' else " ")
                if is_win_cell:
                    st.markdown('<div class="win-cell">', unsafe_allow_html=True)
                st.button(label, disabled=True, key=f"b{i}")
                if is_win_cell:
                    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown(f"""
    <div class="fighter-panel">
        <div class="speech-bubble">{st.session_state.speech_o}</div>
        <div class="fighter-sprite">🐔</div>
        <div class="hp-wrap">
            <div class="hp-label"><span>HP</span><span>{hp_o_val}/2</span></div>
            <div class="hp-track"><div class="hp-fill-o" style="width:{hp_o_pct}%"></div></div>
        </div>
        <div class="items-row"><div class="item-box"></div><div class="item-box"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ====================== BOTTOM HUD ======================
st.markdown(f"""
<div class="bottom-hud">
    <div class="bottom-hp-group">
        <span style="font-size:10px;">🦆</span>
        <div class="bottom-hp-bar"><div style="height:100%;width:{hp_x_pct}%;background:#c0392b;"></div></div>
        <span style="font-size:7px;color:#fff;font-family:'Press Start 2P',monospace;">{hp_x_val}/3</span>
    </div>
    <div class="bottom-center-btn">▶</div>
    <div class="bottom-hp-group">
        <span style="font-size:7px;color:#fff;font-family:'Press Start 2P',monospace;">{hp_o_val}/2</span>
        <div class="bottom-hp-bar"><div style="height:100%;width:{hp_o_pct}%;background:#27ae60;"></div></div>
        <span style="font-size:10px;">🐔</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== LOGIQUE IA — après affichage (pattern Glow) ======================
winner, win_line = check_winner(board)
full = is_full(board)

ai_speeches = ["CALCUL EN COURS...", "LA POULE CHOISIT !", "MOUVEMENT OPTIMAL.", "JE VOIS TOUT.", "ALGORITHME ACTIVE."]

if not winner and not full and not st.session_state.game_over:
    if mode in ["vs IA (ML)", "vs IA (Hybride)"] and st.session_state.current_player == 'O':
        with st.spinner("IA reflechit..."):
            time.sleep(0.3)
            if mode == "vs IA (ML)":
                move = ml_best_move(board, 'O')
            else:
                move = minimax_best_move(board, 'O')

            if move is not None:
                board[move] = 'O'
                st.session_state.speech_o = random.choice(ai_speeches)
                st.session_state.turn += 1
                w, wl = check_winner(board)
                if w:
                    st.session_state.game_over = True
                    st.session_state.winner_symbol = w
                    st.session_state.winner_line = wl
                    st.session_state.speech_x = "NON !!!"
                    st.session_state.speech_o = "LA POULE A PARLE !"
                elif is_full(board):
                    st.session_state.game_over = True
                    st.session_state.speech_x = "MATCH NUL ?"
                    st.session_state.speech_o = "LA POULE EST NEUTRE."
                else:
                    st.session_state.current_player = 'X'
                st.rerun()

# ====================== RÉSULTAT ======================
winner, _ = check_winner(board)
full = is_full(board)

if winner == 'X':
    st.session_state.score_x += 1
    st.markdown('<div class="result-banner">X GAGNANT !</div>', unsafe_allow_html=True)
elif winner == 'O':
    st.session_state.score_o += 1
    st.markdown('<div class="result-banner">O GAGNANT !</div>', unsafe_allow_html=True)
elif full:
    st.markdown('<div class="result-banner">MATCH NUL !</div>', unsafe_allow_html=True)

if winner or full or st.session_state.game_over:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        if st.button("NOUVELLE PARTIE", key="new_game"):
            reset_board()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)