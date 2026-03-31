import os

game_file = r"e:\work\morpion-ml\src\game\game.py"
with open(game_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("\\n"):
        new_lines.append(line.replace("\\n", "\n"))
    else:
        new_lines.append(line)

with open(game_file, "w", encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fix applied successfully.")
