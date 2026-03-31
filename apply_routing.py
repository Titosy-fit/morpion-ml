import os

game_file = r"e:\work\morpion-ml\src\game\game.py"
with open(game_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

title_line_index = -1
for i, line in enumerate(lines):
    if "st.markdown('<div class=\"pixel-title\">MORPION BATTLE</div>" in line:
        title_line_index = i
        break

if title_line_index == -1:
    print("Could not find the target split line")
    exit(1)

top_lines = lines[:title_line_index]
bottom_lines = lines[title_line_index:]

new_code = []

new_code.append("if 'app_state' not in st.session_state:\n")
new_code.append("    st.session_state.app_state = 'loading'\n\n")

functions = '''
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

# ====================== ROUTAGE ======================
if st.session_state.app_state == 'loading':
    render_splash()
elif st.session_state.app_state == 'menu':
    render_menu()
elif st.session_state.app_state == 'options':
    render_options()
elif st.session_state.app_state == 'game':
'''
new_code.append(functions)

for line in bottom_lines:
    if line.strip() == "":
        new_code.append("\\n")
    else:
        new_code.append("    " + line)

final_content = "".join(top_lines) + "".join(new_code)
with open(game_file, "w", encoding='utf-8') as f:
    f.write(final_content)

print("Refactoring applied successfully.")
