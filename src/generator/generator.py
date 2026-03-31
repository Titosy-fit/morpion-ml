import pandas as pd
import itertools

# ====================== FONCTIONS TIC-TAC-TOE ======================
def check_winner(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0

def is_full(board):
    return all(x != 0 for x in board)

def minimax(board, is_max, alpha=-float('inf'), beta=float('inf')):
    winner = check_winner(board)
    if winner == 'X': return 1
    if winner == 'O': return -1
    if is_full(board): return 0

    if is_max:  # tour de X
        best = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 'X'
                best = max(best, minimax(board, False, alpha, beta))
                board[i] = 0
                alpha = max(alpha, best)
                if beta <= alpha: break
        return best
    else:  # tour de O
        best = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 'O'
                best = min(best, minimax(board, True, alpha, beta))
                board[i] = 0
                beta = min(beta, best)
                if beta <= alpha: break
        return best

# ====================== GÉNÉRATION DE TOUS LES ÉTATS ======================
data = []
empty = [0]*9

# On génère tous les états possibles valides (X commence, nombre X = nombre O ou +1)
for nb_moves in range(0, 10):  # 0 à 9 coups
    # Toutes les combinaisons de positions occupées
    for positions in itertools.combinations(range(9), nb_moves):
        for assignment in itertools.product(['X','O'], repeat=nb_moves):
            board = [0]*9
            x_count = o_count = 0
            for i, pos in enumerate(positions):
                board[pos] = assignment[i]
                if assignment[i] == 'X': x_count += 1
                else: o_count += 1
            
            if x_count != o_count and x_count != o_count + 1:
                continue  # état invalide
            if check_winner(board) != 0:
                continue  # déjà fini
            
            # On ne garde que les états où c'est au tour de X
            if x_count == o_count:  # tour de X
                # On encode
                features = []
                for i in range(9):
                    features.extend([1 if board[i]=='X' else 0, 1 if board[i]=='O' else 0])
                
                # Label avec Minimax
                board_copy = board[:]
                score = minimax(board_copy, True)  # True = X to move
                
                x_wins = 1 if score == 1 else 0
                is_draw = 1 if score == 0 else 0
                
                data.append(features + [x_wins, is_draw])

# ====================== EXPORT CSV ======================
columns = [f'c{i}_{p}' for i in range(9) for p in ['x','o']] + ['x_wins', 'is_draw']
df = pd.DataFrame(data, columns=columns)
df.to_csv('ressources/dataset.csv', index=False)
print(f"Dataset généré : {len(df)} lignes")