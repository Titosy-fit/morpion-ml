import os

game_file = r"e:\work\morpion-ml\src\game\game.py"
with open(game_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "def render_splash():" in line:
        start_idx = i
    if "if st.session_state.app_state == 'loading':" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + ["from interface import render_splash, render_menu, render_quit, render_bg_animation\n\nrender_bg_animation(st.session_state.app_state)\n\n"] + lines[end_idx:]
    with open(game_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Split applied.")
else:
    print("Patterns not found.")
