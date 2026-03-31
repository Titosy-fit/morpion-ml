import streamlit as st
import streamlit.components.v1 as components
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
    overflow: hidden !important;
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

/* ====== ROUND SELECTOR LABEL ====== */
[data-testid="stRadio"] > label {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    color: #f0d080 !important;
    letter-spacing: 1px !important;
    margin-bottom: 4px !important;
}

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

/* ====== HIDDEN NEW GAME BUTTON (triggered via JS popup) ====== */
#hidden-new-game { display: none !important; }
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

# ====================== FONCTIONS DU JEU ======================
def check_winner(board):
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

# ====================== SESSION STATE ======================
def reset_board():
    """Réinitialise le plateau pour une nouvelle manche — conserve les scores du duel."""
    st.session_state.board = [0] * 9
    st.session_state.current_player = 'X'
    st.session_state.winner_line = None
    st.session_state.game_over = False
    st.session_state.winner_symbol = None
    st.session_state.turn = 1
    st.session_state.score_counted = False
    st.session_state.speech_x = "PRET AU COMBAT !"
    st.session_state.speech_o = "LA POULE SACREE ME GUIDERA."

def reset_duel():
    """Réinitialise tout : plateau + scores + manches."""
    st.session_state.score_x = 0
    st.session_state.score_o = 0
    st.session_state.max_rounds = st.session_state.get('max_rounds', 3)
    reset_board()

if 'board' not in st.session_state:
    st.session_state.score_x = 0
    st.session_state.score_o = 0
    st.session_state.max_rounds = 3
    st.session_state.score_counted = False
    reset_board()

board = st.session_state.board
max_rounds = st.session_state.get('max_rounds', 3)
wins_to_win = (max_rounds + 1) // 2  # manches nécessaires pour gagner le duel

# ====================== TITLE ======================
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'loading'


def render_splash():
    st.markdown("""
    <style>
    div[data-testid="stButton"] { display: none !important; }
    .splash-wrapper {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        z-index: 50; pointer-events: none;
        background: radial-gradient(circle at center, rgba(200, 144, 10, 0.4) 0%, transparent 60%);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }
    .splash-logo {
        font-family: 'Press Start 2P', monospace; font-size: 40px; color: #f0d080;
        text-shadow: 4px 4px #000, -2px -2px #c8900a; margin-bottom: 40px; text-align: center; line-height: 1.5;
    }
    .loading-container { width: 300px; height: 20px; border: 3px solid #5a3a18; background: #1a0f08; position: relative; margin-bottom: 10px; }
    .loading-bar { height: 100%; background: #c8900a; width: 0%; animation: loadBar 3s linear forwards; }
    @keyframes loadBar { 0% { width: 0%; } 20% { width: 15%; } 50% { width: 45%; } 80% { width: 80%; } 100% { width: 100%; } }
    .loading-text { font-family: 'Press Start 2P', monospace; font-size: 10px; color: #a08040; text-align: center; }
    .loading-text::after { content: "Loading..."; animation: textChange 3s linear forwards; }
    @keyframes textChange { 0% { content: "Loading."; } 33% { content: "Loading.."; } 66% { content: "Loading..."; } 99% { content: "Loading..."; } 100% { content: "Ready!"; color: #55ff55; } }
    .particles {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(#f0d080 3px, transparent 3px), radial-gradient(#f0d080 2px, transparent 2px);
        background-size: 60px 60px; background-position: 0 0, 30px 30px;
        animation: particleMove 20s linear infinite; opacity: 0.15; z-index: -1;
    }
    @keyframes particleMove { 0% { background-position: 0 0, 30px 30px; } 100% { background-position: 600px 600px, 630px 630px; } }
    </style>
    <div class="splash-wrapper">
        <div class="particles"></div>
        <div class="splash-logo">MORPION<br>BATTLE</div>
        <div class="loading-container"><div class="loading-bar"></div></div>
        <div class="loading-text"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("TransitionSplash", key="hidden_splash_btn"):
        st.session_state.app_state = 'menu'
        st.rerun()

    components.html("""
    <script>
    setTimeout(function() {
        var doc = window.parent.document;
        // The text is "TransitionSplash" but Streamlit wraps it in <p>
        var btns = Array.from(doc.querySelectorAll('button p')).filter(el => el.textContent === 'TransitionSplash');
        if (btns.length > 0) {
            btns[0].parentElement.click();
        } else {
            // fallback generic if structure changed
            var allBtns = doc.querySelectorAll('button');
            if(allBtns.length > 0) allBtns[0].click();
        }
    }, 3200);
    </script>
    """, height=0)


def render_menu():
    st.markdown("""
    <style>
    .menu-wrapper {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 50vh; animation: fadeIn 0.5s ease-out; margin-top: 10vh;
    }
    .menu-logo {
        font-family: 'Press Start 2P', monospace; font-size: 50px; color: #f0d080;
        text-shadow: 5px 5px #000, -2px -2px #c8900a; margin-bottom: 60px; text-align: center; line-height: 1.2;
    }
    .stButton button {
        font-size: 16px !important; padding: 15px 30px !important; width: 250px !important; height: 60px !important;
        background: #5a3a10 !important; border: 4px solid #c8900a !important; color: #f0d080 !important;
        border-radius: 0 !important; font-family: 'Press Start 2P', monospace !important;
        transition: all 0.2s !important; display: block !important; margin: 0 auto !important; line-height: 1.5 !important;
    }
    .stButton button:hover {
        background: #7a5010 !important; transform: scale(1.05) !important;
        box-shadow: 0 0 15px rgba(200, 144, 10, 0.6) !important; border-color: #f0d080 !important;
    }
    .stButton { justify-content: center; margin-bottom: 20px; }
    </style>
    <div class="menu-wrapper">
        <div class="menu-logo">MORPION<br>BATTLE</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("JOUER", key="btn_jouer"):
            st.session_state.app_state = 'game'
            st.rerun()
        if st.button("OPTIONS", key="btn_options"):
            st.session_state.app_state = 'options'
            st.rerun()

def render_options():
    st.markdown("""
    <style>
    .options-wrapper {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 60vh; text-align: center; color: #f0d080; font-family: 'Press Start 2P', monospace;
    }
    .stButton button {
        font-size: 14px !important; padding: 15px 30px !important; width: 200px !important; height: 50px !important;
        background: #5a3a10 !important; border: 4px solid #c8900a !important; color: #f0d080 !important;
        border-radius: 0 !important; font-family: 'Press Start 2P', monospace !important;
        transition: all 0.2s !important; display: block !important; margin: 30px auto 0 !important;
    }
    .stButton button:hover {
        background: #7a5010 !important; border-color: #f0d080 !important;
    }
    .stButton { justify-content: center; }
    </style>
    <div class="options-wrapper">
        <h2 style="font-size: 24px; margin-bottom: 20px;">OPTIONS</h2>
        <p style="font-size: 12px; color: #a08040;">⚙️ En construction...</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("RETOUR", key="btn_retour"):
            st.session_state.app_state = 'menu'
            st.rerun()

def render_bg_animation():
    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        if (doc.getElementById('bg-morpion-container')) return;
        
        var container = doc.createElement('div');
        container.id = 'bg-morpion-container';
        container.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:2;opacity:0.4;pointer-events:none;overflow:hidden;';
        
        function createGrid() {
            if (!doc.getElementById('bg-morpion-container')) return;
            var grid = doc.createElement('div');
            var size = 80 + Math.random() * 60;
            grid.style.cssText = 'position:absolute;width:'+size+'px;height:'+size+'px;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);gap:4px;background:#c8900a;border:4px solid #c8900a;opacity:0;transition:opacity 1s, transform 5s linear;transform:scale('+(0.6+Math.random()*0.8)+') rotate('+(Math.random()*40-20)+'deg);';
            
            grid.style.left = (Math.random() * 90) + '%';
            grid.style.top = (Math.random() * 90) + '%';
            container.appendChild(grid);
            
            var cells = [];
            for(var i=0; i<9; i++){
                var cell = doc.createElement('div');
                cell.style.cssText = 'background:#1a0f08;display:flex;align-items:center;justify-content:center;font-family:"Press Start 2P",monospace;font-size:'+(size/4)+'px;';
                grid.appendChild(cell);
                cells.push(cell);
            }
            setTimeout(function(){ grid.style.opacity = '1'; grid.style.transform += ' translateY(-20px)'; }, 50);
            
            var turn = 0;
            var symbols = ['X', 'O'];
            var intv = setInterval(function() {
                var empty = cells.filter(c => c.innerHTML === '');
                if(empty.length === 0 || turn >= 9) {
                    clearInterval(intv);
                    grid.style.opacity = '0';
                    setTimeout(function(){ grid.remove(); }, 1000);
                    return;
                }
                var cell = empty[Math.floor(Math.random()*empty.length)];
                cell.innerHTML = symbols[turn%2];
                cell.style.color = symbols[turn%2] === 'X' ? '#c0392b' : '#27ae60';
                turn++;
            }, 300 + Math.random()*400);
        }
        
        createGrid(); createGrid(); createGrid();
        
        var spawnIntv = setInterval(function() {
            if (!doc.getElementById('bg-morpion-container')) {
                clearInterval(spawnIntv); return;
            }
            createGrid();
        }, 1200);
        
        var stMain = doc.querySelector('[data-testid="stMain"]');
        if(stMain) stMain.appendChild(container);
        else doc.body.appendChild(container);
    })();
    </script>
    """, height=0)

# ====================== ROUTAGE ======================
render_bg_animation()

if st.session_state.app_state == 'loading':
    render_splash()
elif st.session_state.app_state == 'menu':
    render_menu()
elif st.session_state.app_state == 'options':
    render_options()
elif st.session_state.app_state == 'game':
    st.markdown('<div class="pixel-title">MORPION BATTLE</div>', unsafe_allow_html=True)


    # ====================== SÉLECTEUR DE MANCHES ======================
    # Affiché seulement si le duel n'a pas commencé (scores = 0 et plateau vide)
    duel_started = (st.session_state.score_x > 0 or st.session_state.score_o > 0 or any(x != 0 for x in st.session_state.board))

    if not duel_started:
        round_options = {"BO3": 3, "BO5": 5, "BO7": 7}
        selected_label = st.radio(
            "NOMBRE DE MANCHES",
            list(round_options.keys()),
            index=list(round_options.values()).index(st.session_state.get('max_rounds', 3)),
            horizontal=True,
            label_visibility="visible"
        )
        st.session_state.max_rounds = round_options[selected_label]
        max_rounds = st.session_state.max_rounds
        wins_to_win = (max_rounds + 1) // 2

    # ====================== HUD TOP — victoires sous forme d'étoiles/icônes ======================
    def render_wins(score, wins_needed, color_filled, color_empty):
        icons = ""
        for i in range(wins_needed):
            if i < score:
                icons += f'<span style="color:{color_filled};font-size:14px;text-shadow:0 0 6px {color_filled};">★</span>'
            else:
                icons += f'<span style="color:{color_empty};font-size:14px;">☆</span>'
        return icons

    stars_x = render_wins(st.session_state.score_x, wins_to_win, "#f0c000", "#3a2a10")
    stars_o = render_wins(st.session_state.score_o, wins_to_win, "#f0c000", "#3a2a10")

    st.markdown(f"""
    <div class="hud-bar">
        <div class="hud-score">
            <div class="hud-icon">🦆</div>
            <div style="display:flex;flex-direction:column;gap:2px;">
                <span style="font-size:9px;color:#f0d080;">CANARD</span>
                <div style="display:flex;gap:2px;">{stars_x}</div>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
            <div class="hud-turn">MANCHE {st.session_state.turn}</div>
            <div style="font-size:7px;color:#a08040;letter-spacing:1px;">BO{max_rounds} · {wins_to_win} VICTOIRE{'S' if wins_to_win > 1 else ''}</div>
        </div>
        <div class="hud-score" style="flex-direction:row-reverse;">
            <div class="hud-icon" style="background:#2d8a4a;border-color:#1a5a2a;">🐔</div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;">
                <span style="font-size:9px;color:#f0d080;">POULE</span>
                <div style="display:flex;gap:2px;">{stars_o}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== MODE ======================
    mode = st.radio("",
        ["vs Humain", "vs IA (ML)", "vs IA (Hybride)"],
        horizontal=True,
        label_visibility="collapsed",
        disabled=duel_started
    )

    # ====================== HP ======================
    hp_x_pct = max(0, int(100 * (1 - st.session_state.score_o / wins_to_win)))
    hp_o_pct = max(0, int(100 * (1 - st.session_state.score_x / wins_to_win)))
    hp_x_val = max(0, wins_to_win - st.session_state.score_o)
    hp_o_val = max(0, wins_to_win - st.session_state.score_x)

    # ====================== ARENA ======================
    col_l, col_board, col_r = st.columns([1.2, 2.5, 1.2])

    with col_l:
        st.markdown(f"""
        <div class="fighter-panel">
            <div class="speech-bubble">{st.session_state.speech_x}</div>
            <div class="fighter-sprite">🦆</div>
            <div class="hp-wrap">
                <div class="hp-label"><span>HP</span><span>{hp_x_val}/{wins_to_win}</span></div>
                <div class="hp-track"><div class="hp-fill-x" style="width:{hp_x_pct}%"></div></div>
            </div>
            <div class="items-row"><div class="item-box"></div><div class="item-box"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # ====================== PLATEAU ======================
    with col_board:
        winner, win_line = check_winner(board)
        cols = st.columns(3)
        for i in range(9):
            with cols[i % 3]:
                is_win_cell = (winner and win_line and i in win_line)

                if board[i] == 0 and not st.session_state.game_over:
                    can_click = True
                    if mode != "vs Humain" and st.session_state.current_player == 'O':
                        can_click = False

                    if can_click and st.button(" ", key=f"b{i}"):
                        player = st.session_state.current_player
                        board[i] = player
                        next_player = 'O' if player == 'X' else 'X'
                        st.session_state.turn += 1

                        if player == 'X':
                            st.session_state.speech_x = "A TOI !"
                        else:
                            st.session_state.speech_o = "A TOI !"

                        w, wl = check_winner(board)
                        if w:
                            st.session_state.game_over = True
                            st.session_state.winner_symbol = w
                            st.session_state.winner_line = wl
                            if w == 'X':
                                st.session_state.speech_x = "VICTOIRE !!!"
                                st.session_state.speech_o = "IMPOSSIBLE..."
                            else:
                                st.session_state.speech_o = "VICTOIRE !!!"
                                st.session_state.speech_x = "IMPOSSIBLE..."
                        elif is_full(board):
                            st.session_state.game_over = True
                            st.session_state.speech_x = "MATCH NUL ?"
                            st.session_state.speech_o = "LA POULE EST NEUTRE."
                        else:
                            st.session_state.current_player = next_player
                        st.rerun()
                    elif not can_click:
                        st.button(" ", disabled=True, key=f"b{i}")
                else:
                    label = "X" if board[i] == 'X' else ("O" if board[i] == 'O' else " ")
                    st.button(label, disabled=True, key=f"b{i}")

    # ====================== WIN ANIMATION — CONFETTIS ONLY (no line) ======================
    winner, win_line = check_winner(board)

    if winner and win_line:
        win_line_list = list(win_line)
        components.html(f"""
        <script>
        (function() {{
            var winCells = {win_line_list};
            var COLORS = ['#f0c000','#ff6b6b','#48dbfb','#ff9ff3','#feca57','#54a0ff','#5f27cd','#01a3a4','#ff4757','#2ed573'];

            function run() {{
                try {{
                    var doc = window.parent.document;
                    var boardBtns = doc.querySelectorAll('[data-testid="stHorizontalBlock"] .stButton button');
                    if (!boardBtns || boardBtns.length < 9) return;

                    function getDomIdx(gridIdx) {{
                        return (gridIdx % 3) * 3 + Math.floor(gridIdx / 3);
                    }}

                    // Highlight winning cells
                    winCells.forEach(function(ci) {{
                        var di = getDomIdx(ci);
                        if (boardBtns[di]) {{
                            boardBtns[di].style.animation = 'winPulse 0.5s ease-in-out infinite';
                            boardBtns[di].style.borderColor = '#f0c000';
                        }}
                    }});

                    // ====== LIGNE DE VICTOIRE SVG ======
                    var centers = [];
                    winCells.forEach(function(ci) {{
                        var di = getDomIdx(ci);
                        var btn = boardBtns[di];
                        if (btn) {{
                            var r = btn.getBoundingClientRect();
                            centers.push({{ x: r.left + r.width/2, y: r.top + r.height/2 }});
                        }}
                    }});

                    if (centers.length === 3) {{
                        var old = doc.getElementById('win-line-svg');
                        if (old) old.remove();

                        var svg = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        svg.id = 'win-line-svg';
                        svg.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:9999;';

                        var len = Math.sqrt(Math.pow(centers[2].x-centers[0].x,2)+Math.pow(centers[2].y-centers[0].y,2));

                        // Glow line behind
                        var glow = doc.createElementNS('http://www.w3.org/2000/svg', 'line');
                        glow.setAttribute('x1', centers[0].x);
                        glow.setAttribute('y1', centers[0].y);
                        glow.setAttribute('x2', centers[2].x);
                        glow.setAttribute('y2', centers[2].y);
                        glow.setAttribute('stroke', '#f0c000');
                        glow.setAttribute('stroke-width', '12');
                        glow.setAttribute('stroke-linecap', 'square');
                        glow.setAttribute('opacity', '0.3');
                        glow.setAttribute('stroke-dasharray', len);
                        glow.setAttribute('stroke-dashoffset', len);
                        glow.style.animation = 'drawLine 0.4s ease-out forwards';

                        // Main golden line
                        var line = doc.createElementNS('http://www.w3.org/2000/svg', 'line');
                        line.setAttribute('x1', centers[0].x);
                        line.setAttribute('y1', centers[0].y);
                        line.setAttribute('x2', centers[2].x);
                        line.setAttribute('y2', centers[2].y);
                        line.setAttribute('stroke', '#f0c000');
                        line.setAttribute('stroke-width', '5');
                        line.setAttribute('stroke-linecap', 'square');
                        line.setAttribute('stroke-dasharray', len);
                        line.setAttribute('stroke-dashoffset', len);
                        line.style.animation = 'drawLine 0.4s ease-out forwards';

                        svg.appendChild(glow);
                        svg.appendChild(line);
                        doc.body.appendChild(svg);

                        // Add keyframes
                        if (!doc.getElementById('win-line-style')) {{
                            var style = doc.createElement('style');
                            style.id = 'win-line-style';
                            style.textContent = '@keyframes drawLine {{ to {{ stroke-dashoffset: 0; }} }} @keyframes winPulse {{ 0% {{ border-color: #b8a070; background: #e8d8a8; }} 50% {{ border-color: #f0c000; background: #fff8c0; }} 100% {{ border-color: #b8a070; background: #e8d8a8; }} }} @keyframes confettiFall {{ 0% {{ opacity:1; transform: translateY(0) rotate(0deg); }} 100% {{ opacity:0; transform: translateY(120px) rotate(720deg); }} }}';
                            doc.head.appendChild(style);
                        }}
                    }}

                }} catch(e) {{ console.log('win anim error', e); }}
            }}

            setTimeout(run, 200);
            setTimeout(run, 600);
        }})();
        </script>
        """, height=0)
    else:
        # Cleanup on new game
        components.html("""
        <script>
        (function() {
            try {
                var doc = window.parent.document;
                var conf = doc.getElementById('confetti-container');
                if (conf) conf.remove();
                var overlay = doc.getElementById('game-over-popup');
                if (overlay) overlay.remove();
            } catch(e) {}
        })();
        </script>
        """, height=0)

    with col_r:
        st.markdown(f"""
        <div class="fighter-panel">
            <div class="speech-bubble">{st.session_state.speech_o}</div>
            <div class="fighter-sprite">🐔</div>
            <div class="hp-wrap">
                <div class="hp-label"><span>HP</span><span>{hp_o_val}/{wins_to_win}</span></div>
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
            <span style="font-size:7px;color:#fff;font-family:'Press Start 2P',monospace;">{hp_x_val}/{wins_to_win}</span>
        </div>
        <div class="bottom-center-btn">▶</div>
        <div class="bottom-hp-group">
            <span style="font-size:7px;color:#fff;font-family:'Press Start 2P',monospace;">{hp_o_val}/{wins_to_win}</span>
            <div class="bottom-hp-bar"><div style="height:100%;width:{hp_o_pct}%;background:#27ae60;"></div></div>
            <span style="font-size:10px;">🐔</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== LOGIQUE IA ======================
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

    # ====================== RÉSULTAT + POPUP ======================
    winner, _ = check_winner(board)
    full = is_full(board)

    # Track score increment once per manche
    if st.session_state.game_over and not st.session_state.get('score_counted', False):
        if winner == 'X':
            st.session_state.score_x += 1
            st.session_state.score_counted = True
        elif winner == 'O':
            st.session_state.score_o += 1
            st.session_state.score_counted = True
        elif full:
            st.session_state.score_counted = True

    # Check if duel is won
    duel_winner = None
    if st.session_state.score_x >= wins_to_win:
        duel_winner = 'X'
    elif st.session_state.score_o >= wins_to_win:
        duel_winner = 'O'

    # Result messages
    if duel_winner == 'X':
        result_msg = "🦆 REMPORTE LE DUEL !"
        result_emoji = "🦆"
        result_color = "#c0392b"
        result_sub = f"VICTOIRE EN {st.session_state.score_x} MANCHES"
    elif duel_winner == 'O':
        result_msg = "🐔 REMPORTE LE DUEL !"
        result_emoji = "🐔"
        result_color = "#27ae60"
        result_sub = f"VICTOIRE EN {st.session_state.score_o} MANCHES"
    elif winner == 'X':
        result_msg = "MANCHE X !"
        result_emoji = "🦆"
        result_color = "#c0392b"
        result_sub = f"SCORE : {st.session_state.score_x} - {st.session_state.score_o}"
    elif winner == 'O':
        result_msg = "MANCHE O !"
        result_emoji = "🐔"
        result_color = "#27ae60"
        result_sub = f"SCORE : {st.session_state.score_x} - {st.session_state.score_o}"
    elif full:
        result_msg = "MATCH NUL !"
        result_emoji = "⚔️"
        result_color = "#c8900a"
        result_sub = f"SCORE : {st.session_state.score_x} - {st.session_state.score_o}"
    else:
        result_msg = None
        result_sub = ""

    # CSS to hide Streamlit buttons
    st.markdown("""
    <style>
    .new-game-hidden, .new-game-hidden * {
        display: none !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if winner or full or st.session_state.game_over:
        # Hidden "Manche suivante" button
        st.markdown('<div class="new-game-hidden">', unsafe_allow_html=True)
        if st.button("MANCHE SUIVANTE", key="next_round"):
            reset_board()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Hidden "Nouveau duel" button
        st.markdown('<div class="new-game-hidden">', unsafe_allow_html=True)
        if st.button("NOUVEAU DUEL", key="new_duel"):
            reset_duel()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ====== POPUP OVERLAY + CONFETTIS via JS ======
        is_duel_over = "true" if duel_winner else "false"
        components.html(f"""
        <script>
        (function() {{
            var doc = window.parent.document;
            var resultMsg = "{result_msg or ''}";
            var resultEmoji = "{result_emoji if result_msg else ''}";
            var resultColor = "{result_color if result_msg else '#c8900a'}";
            var resultSub = "{result_sub}";
            var isDuelOver = {is_duel_over};
            var COLORS = ['#f0c000','#ff6b6b','#48dbfb','#ff9ff3','#feca57','#54a0ff','#5f27cd','#01a3a4','#ff4757','#2ed573'];

            function spawnConfetti(parent) {{
                for (var c = 0; c < 60; c++) {{
                    var conf = doc.createElement('div');
                    var size = 4 + Math.floor(Math.random() * 8);
                    var startX = 5 + Math.random() * 90;
                    var startY = -10 + Math.random() * 30;
                    var color = COLORS[Math.floor(Math.random() * COLORS.length)];
                    var delay = Math.random() * 0.8;
                    var dur = 1.8 + Math.random() * 2;
                    conf.style.cssText = [
                        'position:absolute',
                        'left:' + startX + '%',
                        'top:' + startY + '%',
                        'width:' + size + 'px',
                        'height:' + size + 'px',
                        'background:' + color,
                        'opacity:0',
                        'pointer-events:none',
                        'animation:confettiFall ' + dur + 's ease-out ' + delay + 's forwards'
                    ].join(';');
                    parent.appendChild(conf);
                }}
            }}

            function clickHidden(btnText) {{
                var hiddenWrappers = doc.querySelectorAll('.new-game-hidden');
                hiddenWrappers.forEach(function(w) {{
                    w.style.cssText = 'display:block!important;height:auto!important;';
                }});
                var allBtns = doc.querySelectorAll('button');
                for (var i = 0; i < allBtns.length; i++) {{
                    if (allBtns[i].textContent.trim() === btnText) {{
                        allBtns[i].click();
                        break;
                    }}
                }}
            }}

            function showPopup() {{
                if (!resultMsg) return;
                if (doc.getElementById('game-over-popup')) return;

                if (!doc.getElementById('popup-style')) {{
                    var st = doc.createElement('style');
                    st.id = 'popup-style';
                    st.textContent = [
                        '@keyframes popupIn {{ 0% {{ opacity:0; transform:scale(0.6) translateY(30px); }} 100% {{ opacity:1; transform:scale(1) translateY(0); }} }}',
                        '@keyframes confettiFall {{ 0% {{ opacity:1; transform:translateY(0) rotate(0deg); }} 100% {{ opacity:0; transform:translateY(200px) rotate(720deg); }} }}'
                    ].join(' ');
                    doc.head.appendChild(st);
                }}

                // Overlay
                var overlay = doc.createElement('div');
                overlay.id = 'game-over-popup';
                overlay.style.cssText = [
                    'position:fixed','top:0','left:0','width:100vw','height:100vh',
                    'background:rgba(10,5,2,0.78)',
                    'backdrop-filter:blur(6px)','-webkit-backdrop-filter:blur(6px)',
                    'display:flex','align-items:center','justify-content:center',
                    'z-index:99999','font-family:\\'Press Start 2P\\',monospace',
                    'overflow:hidden'
                ].join(';');

                if (isDuelOver) spawnConfetti(overlay);

                // Box
                var box = doc.createElement('div');
                box.style.cssText = [
                    'position:relative','z-index:2','background:#1a0f08',
                    'border:4px solid ' + resultColor,
                    'box-shadow:0 0 50px ' + resultColor + '88, inset 0 0 20px rgba(0,0,0,0.8)',
                    'padding:32px 44px','text-align:center','min-width:320px',
                    'animation:popupIn 0.35s cubic-bezier(0.34,1.56,0.64,1) forwards'
                ].join(';');

                // Emoji
                var emojiEl = doc.createElement('div');
                emojiEl.style.cssText = 'font-size:' + (isDuelOver ? '60px' : '44px') + ';margin-bottom:12px;line-height:1;';
                emojiEl.textContent = resultEmoji;

                // Title
                var title = doc.createElement('div');
                title.style.cssText = 'font-size:' + (isDuelOver ? '13px' : '12px') + ';color:' + resultColor + ';text-shadow:2px 2px #000;margin-bottom:6px;letter-spacing:1px;line-height:1.5;';
                title.textContent = resultMsg;

                // Sub
                var sub = doc.createElement('div');
                sub.style.cssText = 'font-size:7px;color:#a08040;margin-bottom:24px;letter-spacing:1px;';
                sub.textContent = resultSub;

                // Button(s)
                var btnRow = doc.createElement('div');
                btnRow.style.cssText = 'display:flex;gap:10px;justify-content:center;flex-wrap:wrap;';

                function makeBtn(label, action, primary) {{
                    var btn = doc.createElement('button');
                    btn.textContent = label;
                    btn.style.cssText = [
                        'font-family:\\'Press Start 2P\\',monospace','font-size:8px',
                        'background:' + (primary ? '#5a3a10' : '#1a2a10'),
                        'color:#f0d080',
                        'border:3px solid ' + (primary ? '#c8900a' : '#4a8a20'),
                        'padding:10px 16px','cursor:pointer','letter-spacing:1px','transition:background 0.15s'
                    ].join(';');
                    btn.onmouseover = function() {{ this.style.opacity = '0.8'; }};
                    btn.onmouseout  = function() {{ this.style.opacity = '1'; }};
                    btn.onclick = function() {{
                        var pop = doc.getElementById('game-over-popup');
                        if (pop) pop.remove();
                        var winLine = doc.getElementById('win-line-svg');
                        if (winLine) winLine.remove();
                        action();
                    }};
                    return btn;
                }}

                if (isDuelOver) {{
                    btnRow.appendChild(makeBtn('► NOUVEAU DUEL', function() {{ clickHidden('NOUVEAU DUEL'); }}, true));
                }} else {{
                    btnRow.appendChild(makeBtn('► MANCHE SUIVANTE', function() {{ clickHidden('MANCHE SUIVANTE'); }}, true));
                    btnRow.appendChild(makeBtn('↺ ABANDON', function() {{ clickHidden('NOUVEAU DUEL'); }}, false));
                }}

                box.appendChild(emojiEl);
                box.appendChild(title);
                box.appendChild(sub);
                box.appendChild(btnRow);
                overlay.appendChild(box);
                doc.body.appendChild(overlay);
            }}

            setTimeout(showPopup, 600);
        }})();
        </script>
        """, height=0)